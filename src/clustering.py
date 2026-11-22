"""Phân cụm trên embedding sản phẩm: PCA, k-means, phân cụm phân cấp (Ward), DBSCAN; silhouette, Davies-Bouldin, ARI.

Phụ trách: C (RQ1). File tự viết, dùng scikit-learn.

Tuần 1 (T3): PCA 768 -> 50 chiều, k-means k = 256, silhouette trên mẫu 5.000 item (khoảng cách cosine).
Quy ước đầu vào: ma trận (n_items, dim) kiểu float; hàng j ứng với item_id j + 1 (cùng quy ước với src/patterns.py).
Embedding được chuẩn hóa L2 trước khi giảm chiều (kế hoạch mục 3.5e).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EMBEDDINGS = ROOT / "data" / "processed" / "item_embeddings.npy"
DEFAULT_OUTPUT = ROOT / "reports" / "kmeans_trial.json"


def load_embeddings(path):
    """Đọc ma trận embedding từ file .npy và kiểm tra hình dạng, giá trị hữu hạn."""
    embeddings = np.load(path)
    if embeddings.ndim != 2:
        raise ValueError(f"embedding phải là ma trận 2 chiều, nhận được {embeddings.ndim} chiều")
    if not np.isfinite(embeddings).all():
        raise ValueError("embedding chứa NaN hoặc vô cùng")
    return embeddings.astype(np.float32, copy=False)


def l2_normalize(matrix):
    """Chuẩn hóa từng hàng về độ dài 1; hàng toàn 0 được giữ nguyên."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms == 0, 1.0, norms)


def fit_pca(matrix, n_components=50, seed=42):
    """PCA xuống n_components chiều; trả về (ma trận đã giảm chiều, tỉ lệ phương sai giữ lại)."""
    if not 1 <= n_components <= min(matrix.shape):
        raise ValueError(f"n_components phải trong khoảng 1..{min(matrix.shape)}, nhận được {n_components}")
    pca = PCA(n_components=n_components, random_state=seed)
    reduced = pca.fit_transform(matrix)
    return reduced, float(pca.explained_variance_ratio_.sum())


def run_kmeans(matrix, k, n_init=5, seed=42):
    """k-means (n_init lần khởi tạo); trả về (nhãn cụm, inertia, giây)."""
    if not 1 <= k <= matrix.shape[0]:
        raise ValueError(f"k phải trong khoảng 1..{matrix.shape[0]}, nhận được {k}")
    model = KMeans(n_clusters=k, n_init=n_init, random_state=seed)
    start = time.perf_counter()
    labels = model.fit_predict(matrix)
    elapsed = time.perf_counter() - start
    return labels, float(model.inertia_), elapsed


def sample_indices(n_items, sample_size=5000, seed=42):
    """Chỉ số mẫu cố định theo seed (đã sắp xếp); lấy toàn bộ nếu n_items không lớn hơn sample_size."""
    if n_items <= sample_size:
        return np.arange(n_items)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_items, size=sample_size, replace=False))


def silhouette_on_sample(matrix, labels, indices, metric="cosine"):
    """Silhouette trung bình trên các chỉ số mẫu; nhãn cụm lấy từ phân cụm toàn bộ item.

    Cụm chỉ có một phần tử trong mẫu có silhouette bằng 0 (quy ước của scikit-learn).
    Cần ít nhất 2 cụm khác nhau trong mẫu.
    """
    sample_labels = np.asarray(labels)[indices]
    n_clusters = len(np.unique(sample_labels))
    if n_clusters < 2 or n_clusters > len(indices) - 1:
        raise ValueError(f"silhouette cần 2..{len(indices) - 1} cụm trong mẫu, nhận được {n_clusters}")
    return float(silhouette_score(matrix[indices], sample_labels, metric=metric))


def cluster_size_summary(labels):
    """Số cụm khác rỗng và thống kê kích thước cụm."""
    sizes = np.bincount(labels)
    sizes = sizes[sizes > 0]
    return {
        "n_clusters": int(len(sizes)),
        "size_min": int(sizes.min()),
        "size_median": float(np.median(sizes)),
        "size_max": int(sizes.max()),
        "n_singletons": int((sizes == 1).sum()),
    }


def run_trial(embeddings, n_components=50, k=256, n_init=5, sample_size=5000, seed=42):
    """Chạy thử T3: PCA, k-means trên PCA (chính) và trên không gian gốc (đối chiếu), silhouette trên cùng mẫu.

    Silhouette của mỗi phép phân cụm được đo trong chính không gian mà k-means chạy (cosine, cùng chỉ số mẫu).
    """
    normalized = l2_normalize(embeddings)
    reduced, variance_kept = fit_pca(normalized, n_components, seed)
    indices = sample_indices(len(normalized), sample_size, seed)
    result = {
        "n_items": int(embeddings.shape[0]),
        "input_dim": int(embeddings.shape[1]),
        "pca_dim": int(n_components),
        "pca_explained_variance_ratio": round(variance_kept, 6),
        "k": int(k),
        "n_init": int(n_init),
        "seed": int(seed),
        "silhouette_sample_size": int(len(indices)),
        "silhouette_metric": "cosine",
    }
    for name, matrix in (("pca", reduced), ("original", normalized)):
        labels, inertia, elapsed = run_kmeans(matrix, k, n_init, seed)
        result[f"kmeans_{name}"] = {
            "dim": int(matrix.shape[1]),
            "inertia": round(inertia, 4),
            "seconds": round(elapsed, 3),
            "silhouette": round(silhouette_on_sample(matrix, labels, indices), 6),
            **cluster_size_summary(labels),
        }
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="PCA, k-means và silhouette thử trên embedding sản phẩm (T3 tuần 1).")
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--components", type=int, default=50)
    parser.add_argument("--k", type=int, default=256)
    parser.add_argument("--n-init", type=int, default=5)
    parser.add_argument("--sample-size", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    try:
        embeddings = load_embeddings(args.embeddings)
        result = run_trial(embeddings, args.components, args.k, args.n_init, args.sample_size, args.seed)
    except (OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
