"""Phân tích thêm cho RQ1: hình phân tán độ đo cụm ↔ NDCG@10, kiểm tra theo tầng (vì sao RQ thua), ID va chạm, hình 2D.

Bốn phần, đều chạy trên CPU, không train:

- ``scatter``: từ ``reports/rq1/rq1_cluster_summary.csv`` và ``reports/rq1/rq1_runs.csv`` vẽ NDCG@10 (trung bình ± độ
  lệch chuẩn 3 seed) theo từng độ đo cụm, kèm hệ số Spearman. Không cần dữ liệu nào khác.
- ``levels``: theo từng vị trí mã j của mỗi biến thể: entropy codebook, AMI với danh mục lá, AMI giữa mã ở vị trí j và
  j − 1 (mức phụ thuộc giữa hai vị trí liền nhau), và độ thuần láng giềng @k khi chỉ dùng L mã đầu. Giả thuyết cần kiểm
  tra: mã RQ ở tầng sau phụ thuộc tầng trước và mang ít thông tin, không hợp với 32 đầu dự đoán độc lập của RPG.
- ``collisions``: các nhóm item trùng nguyên ID, kèm tên, thương hiệu, danh mục, độ dài mô tả, embedding có trùng hẳn
  không, để tìm nguyên nhân.
- ``project``: chiếu embedding PCA 128 chiều xuống 2 chiều (PCA, và UMAP nếu cài ``umap-learn``), tô màu theo danh mục
  lá và theo mã ở vị trí đầu của từng biến thể.

Quy ước như ``src/cluster_metrics.py``: mảng ID (N, 32), hàng i ↔ item_id i + 1. Dòng lệnh::

    python -m src.rq1_analysis scatter --cluster-csv reports/rq1/rq1_cluster_summary.csv \\
        --runs-csv reports/rq1/rq1_runs.csv --out-dir reports/rq1/figures
    python -m src.rq1_analysis all --processed-dir <cache>/processed --items <items.jsonl> \\
        --ids opq=opq.npy pq=pq.npy rq=rq.npy random=random.npy --out-dir <thư mục>
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np

from src import cluster_metrics as M

VARIANT_ORDER = ("opq", "pq", "rq", "random")
VARIANT_NAME = {"opq": "OPQ", "pq": "PQ", "rq": "RQ", "random": "Ngẫu nhiên"}
SCATTER_METRICS = (
    ("leaf_ami_pos_mean", "AMI danh mục lá"),
    ("leaf_neighbor_purity@10", "Độ thuần láng giềng @10, danh mục lá"),
    ("brand_neighbor_purity@10", "Độ thuần láng giềng @10, thương hiệu"),
)
PREFIX_LENGTHS = (1, 2, 4, 8, 16, 32)


def _utf8_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError, OSError):
        pass


def _read_csv(path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["empty"])
        writer.writeheader()
        writer.writerows(rows)


# --- 1. Hình phân tán độ đo cụm ↔ NDCG@10 ---

def scatter_table(cluster_rows: list[dict], run_rows: list[dict],
                  metrics: Sequence[tuple[str, str]] = SCATTER_METRICS) -> tuple[list[dict], dict]:
    """Ghép độ đo cụm với NDCG@10 theo biến thể. Trả về (bảng, Spearman theo độ đo).

    Bảng: một dòng mỗi biến thể có cả hai loại số, gồm ndcg_mean, ndcg_std (ddof = 1), n_seed và các cột độ đo.
    """
    from scipy.stats import spearmanr

    by_variant = defaultdict(list)
    for r in run_rows:
        by_variant[r["variant"]].append(float(r["test_ndcg10"]))
    cluster = {r["variant"]: r for r in cluster_rows}
    table = []
    for v in [x for x in VARIANT_ORDER if x in cluster and x in by_variant] + \
             sorted(set(cluster) & set(by_variant) - set(VARIANT_ORDER)):
        vals = np.array(by_variant[v])
        row = {"variant": v, "ndcg_mean": float(vals.mean()),
               "ndcg_std": float(vals.std(ddof=1)) if len(vals) > 1 else float("nan"), "n_seed": len(vals)}
        for col, _ in metrics:
            row[col] = float(cluster[v][col])
        table.append(row)
    rho = {}
    for col, _ in metrics:
        if len(table) >= 3:
            rho[col] = float(spearmanr([r[col] for r in table], [r["ndcg_mean"] for r in table])[0])
        else:
            rho[col] = float("nan")
    return table, rho


def plot_scatter(table: list[dict], rho: dict, out_png: Path,
                 metrics: Sequence[tuple[str, str]] = SCATTER_METRICS) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 3.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (col, label) in zip(axes, metrics):
        for r in table:
            accent = r["variant"] == "rq"
            ax.errorbar(r[col], r["ndcg_mean"], yerr=r["ndcg_std"], fmt="o", capsize=3,
                        color="#185FA5" if accent else "#5F5E5A", ms=6)
            ax.annotate(VARIANT_NAME.get(r["variant"], r["variant"]), (r[col], r["ndcg_mean"]),
                        textcoords="offset points", xytext=(6, 4), fontsize=9)
        ax.margins(x=0.15, y=0.12)
        ax.set_xlabel(label, fontsize=9)
        ax.set_title(f"Spearman = {rho[col]:+.2f} (n = {len(table)})", fontsize=9)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("NDCG@10 test (trung bình ± độ lệch chuẩn, 3 seed)", fontsize=9)
    fig.suptitle("ID gom cụm tốt hơn thì NDCG cao hơn, trừ RQ thấp hơn cả token ngẫu nhiên", fontsize=11)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    return out_png


# --- 2. Kiểm tra theo tầng ---

def position_dependence(ids) -> np.ndarray:
    """AMI giữa mã ở vị trí j − 1 và j, j = 1..m − 1 (phần tử 0 là NaN). Cao = vị trí sau phụ thuộc vị trí trước."""
    arr = np.asarray(ids)
    out = np.full(arr.shape[1], np.nan)
    for j in range(1, arr.shape[1]):
        out[j] = M.ami(arr[:, j - 1].astype(str), arr[:, j])
    return out


def level_diagnostics(ids_by_variant: dict, labels: Sequence, codebook_size: int = M.CODEBOOK_SIZE,
                      k: int = M.K_NEIGHBORS, prefix_lengths: Sequence[int] = PREFIX_LENGTHS,
                      seed: int = 0) -> tuple[list[dict], list[dict]]:
    """(theo vị trí, theo số mã đầu) cho mỗi biến thể.

    Theo vị trí: entropy codebook chuẩn hóa, AMI với ``labels``, AMI với vị trí liền trước.
    Theo số mã đầu L: độ thuần láng giềng @k khi chỉ dùng L mã đầu (bỏ qua L lớn hơn số vị trí).
    """
    per_pos, per_prefix = [], []
    chance = M.chance_neighbor_purity(labels)
    for name, ids in ids_by_variant.items():
        arr = np.asarray(ids)
        entropy = M.codebook_entropy(arr, codebook_size)
        dep = position_dependence(arr)
        for j in range(arr.shape[1]):
            per_pos.append({"variant": name, "position": j, "entropy": float(entropy[j]),
                            "label_ami": M.ami(labels, arr[:, j]), "ami_with_previous": float(dep[j])})
        for length in prefix_lengths:
            if length > arr.shape[1]:
                continue
            nb = M.nearest_by_shared_codes(arr[:, :length], k=k, seed=seed)
            per_prefix.append({"variant": name, "n_codes": length,
                               "neighbor_purity": M.neighbor_purity(nb, labels), "chance": chance})
    return per_pos, per_prefix


def plot_levels(per_pos: list[dict], per_prefix: list[dict], out_png: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"opq": "#888780", "pq": "#5F5E5A", "rq": "#185FA5", "random": "#B4B2A9"}
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.6))
    panels = [("entropy", "Entropy codebook (chuẩn hóa)"), ("label_ami", "AMI với danh mục lá"),
              ("ami_with_previous", "AMI với vị trí liền trước")]
    variants = list(dict.fromkeys(r["variant"] for r in per_pos))
    for ax, (col, title) in zip(axes, panels):
        for v in variants:
            rows = [r for r in per_pos if r["variant"] == v]
            ax.plot([r["position"] for r in rows], [r[col] for r in rows], marker=".", lw=1.5 if v == "rq" else 1,
                    color=colors.get(v), label=VARIANT_NAME.get(v, v))
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Vị trí mã (0–31)", fontsize=9)
        ax.grid(alpha=0.3)
    ax = axes[3]
    for v in variants:
        rows = [r for r in per_prefix if r["variant"] == v]
        ax.plot([r["n_codes"] for r in rows], [r["neighbor_purity"] for r in rows], marker="o",
                color=colors.get(v), lw=1.5 if v == "rq" else 1, label=VARIANT_NAME.get(v, v))
    if per_prefix:
        ax.axhline(per_prefix[0]["chance"], color="#B4B2A9", ls="--", lw=1)
    ax.set_xscale("log", base=2)
    ax.set_title("Độ thuần láng giềng @10 khi dùng L mã đầu", fontsize=10)
    ax.set_xlabel("Số mã đầu L", fontsize=9)
    ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    return out_png


# --- 3. ID va chạm ---

def load_items(items_path) -> list[dict]:
    rows = []
    with open(items_path, encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    if [int(r["item_id"]) for r in rows] != list(range(1, len(rows) + 1)):
        raise ValueError("items.jsonl phải có item_id liên tục 1..N theo thứ tự tăng dần")
    return rows


def _text(value) -> str:
    if isinstance(value, (list, tuple)):
        return " ".join(_text(v) for v in value)
    return "" if value is None else str(value).strip()


def collision_groups(ids_by_variant: dict, items: list[dict], emb=None) -> list[dict]:
    """Một dòng mỗi item nằm trong nhóm trùng nguyên ID, kèm thông tin để tìm nguyên nhân.

    ``same_embedding`` = mọi item trong nhóm có embedding (trước PCA) giống hệt nhau; ``same_title`` tương tự cho tiêu đề.
    """
    out = []
    for name, ids in ids_by_variant.items():
        arr = np.asarray(ids)
        if arr.shape[0] != len(items):
            raise ValueError(f"{name}: {arr.shape[0]} hàng ID nhưng items.jsonl có {len(items)} item")
        _, inverse, counts = np.unique(arr, axis=0, return_inverse=True, return_counts=True)
        inverse = inverse.ravel()
        group_no = 0
        for g in np.flatnonzero(counts > 1):
            members = np.flatnonzero(inverse == g)
            group_no += 1
            titles = [_text((items[i].get("metadata") or {}).get("title")) for i in members]
            same_emb = bool(emb is not None and all(np.array_equal(emb[members[0]], emb[i]) for i in members[1:]))
            for i in members:
                meta = items[i].get("metadata") or {}
                paths = [p for p in (meta.get("categories") or []) if isinstance(p, (list, tuple)) and p]
                leaf, _ = M.category_labels(paths)
                out.append({"variant": name, "group": group_no, "item_id": i + 1, "asin": items[i].get("asin", ""),
                            "title": _text(meta.get("title"))[:120], "brand": _text(meta.get("brand")),
                            "leaf": leaf or "", "description_chars": len(_text(meta.get("description"))),
                            "metadata_missing": bool(items[i].get("metadata_missing", False)),
                            "same_title": len(set(titles)) == 1, "same_embedding": same_emb})
    return out


# --- 4. Hình 2D ---

def top_categories(values: Sequence, n: int) -> list:
    """n giá trị thường gặp nhất (bỏ None), thứ tự giảm dần theo tần suất, hòa thì theo giá trị."""
    counts = Counter(v for v in values if v is not None)
    return [v for v, _ in sorted(counts.items(), key=lambda kv: (-kv[1], str(kv[0])))[:n]]


def project_2d(x, method: str = "pca", seed: int = 0) -> np.ndarray:
    """Tọa độ 2D. ``pca``: hai thành phần chính đầu của ``x``; ``umap``: UMAP (cần umap-learn)."""
    x = np.asarray(x, dtype=np.float32)
    if method == "pca":
        centered = x - x.mean(0)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        return centered @ vt[:2].T
    if method == "umap":
        import umap  # noqa: PLC0415  (chỉ cần khi vẽ UMAP)
        return umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=seed).fit_transform(x)
    raise ValueError(f"method phải là 'pca' hoặc 'umap', nhận {method!r}")


def plot_projection(coords: np.ndarray, leaf: Sequence, first_codes: dict, out_png: Path, title: str,
                    n_top: int = 8) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [("Danh mục lá", list(leaf))] + [(f"Mã đầu của {VARIANT_NAME.get(v, v)}", list(c))
                                              for v, c in first_codes.items()]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.2))
    cmap = plt.get_cmap("tab10")
    for ax, (name, values) in zip(np.atleast_1d(axes), panels):
        top = top_categories(values, n_top)
        rank = {v: i for i, v in enumerate(top)}
        other = np.array([v not in rank for v in values])
        ax.scatter(coords[other, 0], coords[other, 1], s=2, color="#D3D1C7", alpha=0.5, linewidths=0)
        for v in top:
            mask = np.array([u == v for u in values])
            ax.scatter(coords[mask, 0], coords[mask, 1], s=3, color=cmap(rank[v] % 10), alpha=0.8, linewidths=0,
                       label=str(v)[:22])
        ax.set_title(f"{name} ({n_top} nhóm lớn nhất, còn lại xám)", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        if name == "Danh mục lá":
            ax.legend(fontsize=6, markerscale=3, frameon=False, loc="lower left")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    return out_png


# --- CLI ---

def _parse_ids(specs: Sequence[str]) -> dict:
    out = {}
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(f"--ids cần dạng tên=đường_dẫn.npy, nhận {spec!r}")
        name, path = spec.split("=", 1)
        out[name] = np.load(path)
    return out


def run_scatter(cluster_csv, runs_csv, out_dir: Path) -> dict:
    table, rho = scatter_table(_read_csv(cluster_csv), _read_csv(runs_csv))
    _write_csv(out_dir / "rq1_scatter_table.csv", table)
    png = plot_scatter(table, rho, out_dir / "rq1_scatter_cluster_ndcg.png")
    print(f"Đã ghi {png}; Spearman: " + ", ".join(f"{k} {v:+.2f}" for k, v in rho.items()))
    return rho


def main(argv: Sequence[str] | None = None) -> int:
    _utf8_stdout()
    parser = argparse.ArgumentParser(prog="python -m src.rq1_analysis", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_sc = sub.add_parser("scatter", help="hình phân tán độ đo cụm ↔ NDCG@10 (chỉ cần 2 file CSV)")
    p_all = sub.add_parser("all", help="cả bốn phần (cần cache code gốc, items.jsonl và các file ID .npy)")
    for p in (p_sc, p_all):
        p.add_argument("--cluster-csv", default="reports/rq1/rq1_cluster_summary.csv")
        p.add_argument("--runs-csv", default="reports/rq1/rq1_runs.csv")
        p.add_argument("--out-dir", required=True)
    p_all.add_argument("--processed-dir", required=True, help="<cache>/AmazonReviews2014/Beauty/processed")
    p_all.add_argument("--items", required=True, help="items.jsonl của pipeline B")
    p_all.add_argument("--ids", nargs="+", required=True, metavar="TÊN=FILE.npy")
    p_all.add_argument("--emb-name", default="sentence-t5-base")
    p_all.add_argument("--emb-dim", type=int, default=768)
    p_all.add_argument("--k", type=int, default=M.K_NEIGHBORS)
    p_all.add_argument("--seed", type=int, default=0)
    p_all.add_argument("--no-umap", action="store_true", help="bỏ hình UMAP (không cần umap-learn)")
    args = parser.parse_args(argv)
    out = Path(args.out_dir)

    run_scatter(args.cluster_csv, args.runs_csv, out)
    if args.cmd == "scatter":
        return 0

    from src.tokenizer import load_embeddings, pca_whiten

    ids = _parse_ids(args.ids)
    items = load_items(args.items)
    labels, _ = M.load_item_labels(args.items)
    leaf = labels["leaf"]

    semantic = {k: v for k, v in ids.items() if k != "random"}
    per_pos, per_prefix = level_diagnostics(ids, leaf, k=args.k, seed=args.seed)
    _write_csv(out / "rq1_levels_per_position.csv", per_pos)
    _write_csv(out / "rq1_levels_prefix.csv", per_prefix)
    print(f"Đã ghi {plot_levels(per_pos, per_prefix, out / 'rq1_levels.png')}")
    for v in ids:
        dep = [r["ami_with_previous"] for r in per_pos if r["variant"] == v and r["position"] > 0]
        amis = [r["label_ami"] for r in per_pos if r["variant"] == v]
        print(f"  {v}: AMI với vị trí trước, trung bình {np.mean(dep):.4f}; AMI danh mục lá vị trí 0 {amis[0]:.4f}, "
              f"vị trí 31 {amis[-1]:.4f}")

    proc = Path(args.processed_dir)
    emb = load_embeddings(proc, args.emb_name, args.emb_dim)
    rows = collision_groups(ids, items, emb)
    _write_csv(out / "rq1_collisions.csv", rows)
    print(f"Đã ghi {out / 'rq1_collisions.csv'}: {len(rows)} dòng")

    x = pca_whiten(emb)
    first = {v: np.asarray(a)[:, 0] for v, a in semantic.items()}
    methods = ["pca"] if args.no_umap else ["pca", "umap"]
    for method in methods:
        try:
            # PCA trên embedding gốc (chưa whiten): sau whiten mọi hướng có cùng phương sai nên PCA lần hai vô nghĩa.
            # Hai thành phần đầu này chính là hai chiều đầu của không gian tạo ID (đoạn 0 của PQ). UMAP chạy trên x.
            coords = project_2d(emb if method == "pca" else x, method, seed=args.seed)
        except ImportError:
            print("Bỏ qua UMAP: chưa cài umap-learn (pip install umap-learn)")
            continue
        np.save(out / f"rq1_coords_{method}.npy", coords.astype(np.float32))
        png = plot_projection(coords, leaf, first, out / f"rq1_{method}_2d.png",
                              "Hai thành phần chính đầu của embedding sentence-t5" if method == "pca" else "UMAP trên embedding PCA 128 chiều có whiten")
        print(f"Đã ghi {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
