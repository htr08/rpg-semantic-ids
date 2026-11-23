"""So sánh k-means, phân cụm phân cấp (Ward) và DBSCAN trên embedding sản phẩm (RQ1, T3 tuần 2).

Phụ trách: C. Dùng lại hàm của src/clustering.py; quy ước đầu vào giống file đó (hàng j ứng với item_id j + 1).

Quy trình:
- Chuẩn hóa L2, PCA xuống 50 chiều trên toàn bộ item (cả khi dùng mẫu, để cùng không gian với k-means thử ở T3 tuần 1).
- Phân cụm phân cấp (Ward) và k-means trên cùng mẫu 5.000 item, cùng k ∈ {64, 256, 1024}, vì mọi độ đo phụ thuộc số cụm.
- DBSCAN (cosine) trên toàn bộ item: eps lấy tại điểm gãy của đường k-distance và tại vài phân vị của k-distance để
  thấy độ nhạy; MinPts ∈ {5, 10, 20} (theo quy ước scikit-learn, MinPts tính cả điểm đang xét); chạy trên PCA-50 và PCA-20.
- Silhouette (cosine, tối đa 5.000 điểm) và Davies-Bouldin chỉ tính trên điểm không bị gán nhiễu; tỉ lệ nhiễu ghi riêng.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn import config_context
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.metrics import davies_bouldin_score
from sklearn.neighbors import NearestNeighbors

from src.clustering import (
    DEFAULT_EMBEDDINGS,
    ROOT,
    cluster_size_summary,
    fit_pca,
    l2_normalize,
    load_embeddings,
    run_kmeans,
    sample_indices,
    silhouette_on_sample,
)

DEFAULT_OUTPUT = ROOT / "reports" / "clustering_hier_dbscan.json"
DEFAULT_MARKDOWN = ROOT / "reports" / "clustering_hier_dbscan.md"
DEFAULT_FIGURE = ROOT / "reports" / "figures" / "dbscan_kdistance.png"
NOISE = -1


def k_distances(matrix, min_samples, metric="cosine"):
    """Khoảng cách tới láng giềng thứ min_samples (tính cả điểm đang xét, như DBSCAN), sắp tăng dần."""
    if not 2 <= min_samples <= matrix.shape[0]:
        raise ValueError(f"min_samples phải trong khoảng 2..{matrix.shape[0]}, nhận được {min_samples}")
    neighbors = NearestNeighbors(n_neighbors=min_samples, metric=metric).fit(matrix)
    # Truy vấn theo khối: một lần gọi cho toàn bộ item cấp phát ~1 GiB (mảng chỉ số 11.000 × 12.101) và gây MemoryError.
    values = [neighbors.kneighbors(matrix[start:start + 1000], return_distance=True)[0][:, min_samples - 1]
              for start in range(0, matrix.shape[0], 1000)]
    return np.sort(np.concatenate(values))


def knee_index(sorted_values):
    """Chỉ số điểm gãy của đường cong tăng dần: điểm xa đường nối hai đầu nhất, sau khi chuẩn hóa hai trục về [0, 1].

    Với đường cong lồi (phẳng rồi dốc dần), điểm gãy là chỗ x - y lớn nhất.
    """
    values = np.asarray(sorted_values, dtype=float)
    if len(values) < 3:
        raise ValueError("cần ít nhất 3 giá trị để tìm điểm gãy")
    span = values[-1] - values[0]
    if span <= 0:
        raise ValueError("đường k-distance phẳng, không có điểm gãy")
    x = np.linspace(0.0, 1.0, len(values))
    y = (values - values[0]) / span
    return int(np.argmax(x - y))


def evaluate_labels(matrix, labels, sample_size=5000, seed=42):
    """Số cụm, tỉ lệ nhiễu, cụm lớn nhất, silhouette (cosine) và Davies-Bouldin trên các điểm không phải nhiễu.

    Davies-Bouldin tính trên mọi điểm không nhiễu; silhouette tính trên mẫu tối đa sample_size điểm không nhiễu nên
    cụm rất nhỏ có thể không vào mẫu. Cả hai là None khi số cụm không nằm trong 2..(số điểm không nhiễu - 1);
    silhouette còn là None khi mẫu chứa dưới 2 cụm.
    Tỉ lệ cụm lớn nhất tính trên toàn bộ điểm, kể cả nhiễu.
    """
    labels = np.asarray(labels)
    keep = labels != NOISE
    kept_labels = labels[keep]
    result = {"noise_ratio": round(float(1.0 - keep.mean()), 6)}
    if kept_labels.size == 0:
        return {**result, "n_clusters": 0, "size_min": None, "size_median": None, "size_max": None, "n_singletons": None,
                "largest_cluster_fraction": 0.0, "silhouette": None, "davies_bouldin": None}
    kept = matrix[keep]
    summary = cluster_size_summary(kept_labels)
    result.update(summary)
    result["largest_cluster_fraction"] = round(summary["size_max"] / len(labels), 6)
    result["silhouette"] = None
    result["davies_bouldin"] = None
    if 2 <= summary["n_clusters"] <= len(kept) - 1:
        indices = sample_indices(len(kept), sample_size, seed)
        try:
            result["silhouette"] = round(silhouette_on_sample(kept, kept_labels, indices), 6)
        except ValueError:  # mẫu rơi vào dưới 2 cụm
            pass
        result["davies_bouldin"] = round(float(davies_bouldin_score(kept, kept_labels)), 6)
    return result


def run_hierarchical(matrix, k):
    """Phân cụm phân cấp Ward (Euclid) cắt ở k cụm; trả về (nhãn, giây)."""
    if not 2 <= k <= matrix.shape[0]:
        raise ValueError(f"k phải trong khoảng 2..{matrix.shape[0]}, nhận được {k}")
    start = time.perf_counter()
    labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(matrix)
    return labels, time.perf_counter() - start


def run_dbscan(matrix, eps, min_samples, metric="cosine"):
    """DBSCAN; trả về (nhãn với -1 là nhiễu, giây)."""
    start = time.perf_counter()
    # working_memory nhỏ buộc scikit-learn truy vấn láng giềng theo khối ~64 MiB thay vì ~1 GiB mỗi khối.
    with config_context(working_memory=64):
        labels = DBSCAN(eps=eps, min_samples=min_samples, metric=metric).fit_predict(matrix)
    return labels, time.perf_counter() - start


def eps_candidates(curve, quantiles):
    """eps tại điểm gãy và tại các phân vị của đường k-distance; trả về danh sách (tên quy tắc, eps)."""
    candidates = [("knee", float(curve[knee_index(curve)]))]
    candidates += [(f"q{q:.2f}", float(np.quantile(curve, q))) for q in quantiles]
    return candidates


def run_comparison(embeddings, n_components=50, ks=(64, 256, 1024), min_samples_list=(5, 10, 20),
                   eps_quantiles=(0.5, 0.75, 0.9), extra_dbscan_dims=(20,), sample_size=5000, n_init=5, seed=42):
    """Chạy toàn bộ so sánh; trả về dict kết quả (có thể ghi JSON) và các đường k-distance (cho hình)."""
    normalized = l2_normalize(embeddings)
    reduced, variance_kept = fit_pca(normalized, n_components, seed)
    indices = sample_indices(len(normalized), sample_size, seed)
    sample = reduced[indices]
    result = {
        "n_items": int(embeddings.shape[0]),
        "input_dim": int(embeddings.shape[1]),
        "pca_dim": int(n_components),
        "pca_explained_variance_ratio": round(variance_kept, 6),
        "seed": int(seed),
        "sample_size": int(len(indices)),
        "silhouette_metric": "cosine",
        "hierarchical_linkage": "ward",
        "kmeans_n_init": int(n_init),
        "sample_algorithms": [],
        "dbscan": [],
    }
    for k in ks:
        labels, seconds = run_hierarchical(sample, k)
        result["sample_algorithms"].append(
            {"algorithm": "ward", "k": int(k), "seconds": round(seconds, 3), **evaluate_labels(sample, labels, sample_size, seed)})
        labels, _, seconds = run_kmeans(sample, k, n_init, seed)
        result["sample_algorithms"].append(
            {"algorithm": "kmeans", "k": int(k), "seconds": round(seconds, 3), **evaluate_labels(sample, labels, sample_size, seed)})

    curves = {}
    for dim in dict.fromkeys((n_components, *extra_dbscan_dims)):
        matrix = reduced if dim == n_components else fit_pca(normalized, dim, seed)[0]
        for min_samples in min_samples_list:
            curve = k_distances(matrix, min_samples)
            curves[(dim, min_samples)] = curve
            for rule, eps in eps_candidates(curve, eps_quantiles):
                labels, seconds = run_dbscan(matrix, eps, min_samples)
                result["dbscan"].append({
                    "pca_dim": int(dim), "min_samples": int(min_samples), "eps_rule": rule, "eps": round(eps, 6),
                    "seconds": round(seconds, 3), **evaluate_labels(matrix, labels, sample_size, seed)})
    return result, curves


def _fmt(value, digits=4):
    return "—" if value is None else f"{value:.{digits}f}"


def format_markdown(result):
    """Bảng Markdown từ kết quả run_comparison."""
    lines = [
        "# So sánh phân cụm: Ward, k-means và DBSCAN trên embedding",
        "",
        f"Embedding {result['n_items']} item × {result['input_dim']} chiều, chuẩn hóa L2, PCA {result['pca_dim']} chiều "
        f"(giữ {result['pca_explained_variance_ratio']:.4f} phương sai), seed {result['seed']}. "
        "Silhouette dùng khoảng cách cosine; Davies–Bouldin dùng khoảng cách Euclid. "
        "Sinh bởi `python -m src.clustering_compare`.",
        "",
        f"## Ward và k-means trên cùng mẫu {result['sample_size']} item",
        "",
        "| Thuật toán | k | Silhouette | Davies–Bouldin | Kích thước cụm nhỏ nhất | Cụm đơn lẻ | Giây |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["sample_algorithms"]:
        lines.append(f"| {row['algorithm']} | {row['k']} | {_fmt(row['silhouette'])} | {_fmt(row['davies_bouldin'])} "
                     f"| {row['size_min']} | {row['n_singletons']} | {row['seconds']:.2f} |")
    lines += [
        "",
        "## DBSCAN (cosine) trên toàn bộ item",
        "",
        "eps `knee` lấy tại điểm gãy của đường k-distance; `qX` là phân vị X của đường đó. "
        "Silhouette và Davies–Bouldin chỉ tính trên điểm không nhiễu (— khi dưới 2 cụm). Silhouette tính trên mẫu tối đa "
        f"{result['sample_size']} điểm không nhiễu nên các cụm rất nhỏ có thể không vào mẫu; giá trị giữa các dòng chỉ so sánh được một cách tương đối.",
        "",
        "| PCA | MinPts | eps (quy tắc) | eps | Số cụm | Tỉ lệ nhiễu | Cụm lớn nhất / mọi item | Silhouette | Davies–Bouldin |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["dbscan"]:
        lines.append(f"| {row['pca_dim']} | {row['min_samples']} | {row['eps_rule']} | {row['eps']:.4f} | {row['n_clusters']} "
                     f"| {row['noise_ratio']:.4f} | {row['largest_cluster_fraction']:.4f} "
                     f"| {_fmt(row['silhouette'])} | {_fmt(row['davies_bouldin'])} |")
    knee_rows = [r for r in result["dbscan"] if r["eps_rule"] == "knee"]
    useful = [r for r in result["dbscan"] if r["n_clusters"] >= 2 and r["largest_cluster_fraction"] < 0.5]
    lines += [
        "",
        "## Tóm tắt DBSCAN",
        "",
        f"- Tại eps ở điểm gãy ({len(knee_rows)} cấu hình): số cụm {min(r['n_clusters'] for r in knee_rows)}–"
        f"{max(r['n_clusters'] for r in knee_rows)}, cụm lớn nhất chiếm "
        f"{min(r['largest_cluster_fraction'] for r in knee_rows):.1%}–{max(r['largest_cluster_fraction'] for r in knee_rows):.1%} "
        f"số item, nhiễu {min(r['noise_ratio'] for r in knee_rows):.1%}–{max(r['noise_ratio'] for r in knee_rows):.1%}.",
        f"- Số cấu hình có ít nhất 2 cụm và cụm lớn nhất dưới 50% số item: {len(useful)} / {len(result['dbscan'])}.",
        "",
    ]
    return "\n".join(lines)


def plot_kdistance(curves, rows, path, dim=50):
    """Vẽ đường k-distance (đã sắp tăng dần) cho mọi MinPts ở PCA `dim` chiều, đánh dấu eps tại điểm gãy."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7, 4.5))
    for (curve_dim, min_samples), curve in sorted(curves.items()):
        if curve_dim != dim:
            continue
        line, = axis.plot(np.arange(len(curve)), curve, label=f"MinPts = {min_samples}")
        knee = next(r for r in rows if r["pca_dim"] == dim and r["min_samples"] == min_samples and r["eps_rule"] == "knee")
        axis.scatter([knee_index(curve)], [knee["eps"]], color=line.get_color(), zorder=3)
    axis.set_xlabel("Item (sắp theo k-distance tăng dần)")
    axis.set_ylabel("Khoảng cách cosine tới điểm thứ MinPts (tính cả chính nó)")
    axis.set_title(f"Đường k-distance, PCA {dim} chiều (chấm: điểm gãy)")
    axis.legend()
    figure.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main(argv=None):
    parser = argparse.ArgumentParser(description="So sánh Ward, k-means và DBSCAN trên embedding sản phẩm (T3 tuần 2).")
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--components", type=int, default=50)
    parser.add_argument("--ks", type=int, nargs="+", default=[64, 256, 1024])
    parser.add_argument("--min-samples", type=int, nargs="+", default=[5, 10, 20])
    parser.add_argument("--extra-dbscan-dims", type=int, nargs="*", default=[20], help="số chiều PCA khác để chạy thêm DBSCAN")
    parser.add_argument("--sample-size", type=int, default=5000)
    parser.add_argument("--n-init", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--figure", type=lambda value: Path(value) if value else None, default=DEFAULT_FIGURE, help="đặt rỗng ('') để bỏ qua hình")
    args = parser.parse_args(argv)
    try:
        embeddings = load_embeddings(args.embeddings)
        result, curves = run_comparison(embeddings, args.components, tuple(args.ks), tuple(args.min_samples),
                                        extra_dbscan_dims=tuple(args.extra_dbscan_dims), sample_size=args.sample_size, n_init=args.n_init, seed=args.seed)
        if args.figure is not None:
            plot_kdistance(curves, result["dbscan"], args.figure, dim=args.components)
    except (OSError, ValueError, ImportError, MemoryError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    for path, text in ((args.output, json.dumps(result, ensure_ascii=False, indent=2) + "\n"),
                       (args.markdown, format_markdown(result))):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")
    # Chỉ in đường dẫn: bảng mã cp1252 của console Windows không in được tiếng Việt (UnicodeEncodeError).
    print(f"wrote {args.output} {args.markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
