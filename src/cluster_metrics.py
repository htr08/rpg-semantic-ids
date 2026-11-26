"""Đánh giá semantic ID như một phép phân cụm: purity, NMI theo danh mục lá và thương hiệu, tỉ lệ va chạm ID, entropy codebook.

Dùng cho RQ1: semantic ID "càng ngữ nghĩa" (item cùng danh mục, cùng thương hiệu có mã giống nhau) thì RPG có
gợi ý tốt hơn không. Mọi mảng ID theo quy ước của nhóm: hàng i ↔ item_id i + 1, mã thô 0..codebook_size-1.

Ba nhóm độ đo cho mỗi loại nhãn (danh mục lá, danh mục cấp 3, thương hiệu):

- **Từng vị trí:** coi mã ở codebook j là nhãn cụm (256 cụm), tính purity, NMI và AMI, rồi lấy trung bình 32 vị trí.
  Với 256 cụm và hàng trăm nhãn, purity và NMI cao ngay cả khi mã gán ngẫu nhiên (thử trên 12.101 item: NMI
  khoảng 0,31 với danh mục lá, 0,58 với thương hiệu). AMI trừ phần trùng hợp ngẫu nhiên nên ID ngẫu nhiên cho
  AMI quanh 0; dùng AMI (và độ thuần láng giềng) làm số chính, purity/NMI chỉ để đối chiếu với kế hoạch.
- **Toàn ID:** coi cả 32 mã là một cụm. Gần như mỗi item có ID riêng nên purity toàn ID luôn gần 1 và NMI cao
  (kể cả ID ngẫu nhiên); AMI toàn ID quanh 0. Giữ lại vì kế hoạch yêu cầu, nhưng không dùng để so sánh.
- **Độ thuần láng giềng @k:** với mỗi item, lấy k item có nhiều vị trí trùng mã nhất (hòa thì chọn ngẫu nhiên
  có seed), đếm tỉ lệ láng giềng cùng nhãn. Đây là cách đo "toàn ID" có ý nghĩa, vì RPG dựng đồ thị giải mã
  từ độ giống giữa các token. Mốc ngẫu nhiên: xác suất hai item bất kỳ cùng nhãn.

Quy tắc nhãn (đề xuất cho câu hỏi mở G4 trong project charter, 25/11):

- **Đường danh mục chính:** đường đầu tiên trong ``metadata.categories`` bắt đầu bằng ``Beauty``; nếu không có
  thì lấy đường đầu tiên. Nhãn được giải mã HTML và bỏ khoảng trắng hai đầu.
- **Danh mục lá:** phần tử cuối của đường chính. **Cấp 3:** phần tử thứ 3 (``Beauty > Makeup > Nails``);
  đường ngắn hơn thì lấy phần tử cuối.
- **Thương hiệu:** giống ``src.patterns.load_item_metadata``: rỗng hoặc ``Unknown`` là thiếu.
- Item thiếu nhãn bị loại khỏi phép tính của loại nhãn đó (không gộp thành một nhãn "thiếu" chung).

Ví dụ::

    python -m src.cluster_metrics --items data/processed/items.jsonl \\
        --ids opq=semantic_ids_opq.npy pq=semantic_ids_pq.npy rq=semantic_ids_rq.npy random=semantic_ids_random.npy \\
        --out-dir reports

Phụ trách: A (RQ1).
"""
from __future__ import annotations

import argparse
import csv
import html
import json
from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np

LABEL_TYPES = ("leaf", "level3", "brand")
PREFERRED_ROOT = "Beauty"
CODEBOOK_SIZE = 256
K_NEIGHBORS = 10


# --- Nhãn từ items.jsonl ---

def _clean(label) -> str | None:
    if not isinstance(label, str):
        return None
    label = html.unescape(label).strip()
    return label or None


def primary_category_path(paths, prefer: str = PREFERRED_ROOT) -> tuple[str, ...] | None:
    """Đường danh mục chính: đường đầu tiên bắt đầu bằng ``prefer``, nếu không có thì đường đầu tiên hợp lệ."""
    clean = []
    for path in paths or []:
        if isinstance(path, (list, tuple)):
            labels = tuple(c for c in (_clean(x) for x in path) if c)
            if labels:
                clean.append(labels)
    if not clean:
        return None
    for labels in clean:
        if labels[0] == prefer:
            return labels
    return clean[0]


def category_labels(paths, prefer: str = PREFERRED_ROOT) -> tuple[str | None, str | None]:
    """(danh mục lá, danh mục cấp 3) theo quy tắc ở đầu module."""
    path = primary_category_path(paths, prefer)
    if path is None:
        return None, None
    return path[-1], path[2] if len(path) >= 3 else path[-1]


def brand_label(brand) -> str | None:
    label = _clean(brand)
    return None if label is None or label.lower() == "unknown" else label


def load_item_labels(items_path: str | Path, prefer: str = PREFERRED_ROOT) -> tuple[dict, dict]:
    """Đọc ``items.jsonl`` của B. Trả về (nhãn, thống kê).

    ``nhãn[loại]`` là list dài N, phần tử i là nhãn của item_id i + 1 (None nếu thiếu).
    Báo lỗi nếu item_id không liên tục 1..N.
    """
    rows = []
    with open(items_path, encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    ids = [int(r["item_id"]) for r in rows]
    if ids != list(range(1, len(rows) + 1)):
        raise ValueError("items.jsonl phải có item_id liên tục 1..N theo thứ tự tăng dần")

    labels = {t: [] for t in LABEL_TYPES}
    n_multi = n_no_pref = 0
    for r in rows:
        meta = r.get("metadata") or {}
        paths = [p for p in (meta.get("categories") or []) if isinstance(p, (list, tuple)) and p]
        n_multi += len(paths) > 1
        path = primary_category_path(paths, prefer)
        n_no_pref += path is not None and path[0] != prefer
        leaf, level3 = category_labels(paths, prefer)
        labels["leaf"].append(leaf)
        labels["level3"].append(level3)
        labels["brand"].append(brand_label(meta.get("brand")))

    stats = {"n_items": len(rows), "n_multi_path": n_multi, "n_primary_not_preferred": n_no_pref}
    for t in LABEL_TYPES:
        present = [x for x in labels[t] if x is not None]
        stats[f"{t}_coverage"] = len(present) / len(rows) if rows else 0.0
        stats[f"{t}_n_classes"] = len(set(present))
    return labels, stats


# --- Độ đo cơ bản ---

def _labeled(labels, clusters):
    labels = np.asarray(labels, dtype=object)
    clusters = np.asarray(clusters)
    if labels.shape[0] != clusters.shape[0]:
        raise ValueError(f"labels có {labels.shape[0]} phần tử, clusters có {clusters.shape[0]}")
    mask = np.array([x is not None for x in labels], dtype=bool)
    return labels[mask], clusters[mask]


def purity(labels, clusters) -> float:
    """Tổng (số item của nhãn chiếm đa số trong mỗi cụm) / số item có nhãn. Item thiếu nhãn bị bỏ qua."""
    lab, clu = _labeled(labels, clusters)
    if len(lab) == 0:
        return float("nan")
    by_cluster: dict = {}
    for c, l in zip(clu.tolist(), lab.tolist()):
        by_cluster.setdefault(c, Counter())[l] += 1
    return sum(max(cnt.values()) for cnt in by_cluster.values()) / len(lab)


def nmi(labels, clusters) -> float:
    """NMI (chuẩn hóa trung bình cộng, như scikit-learn mặc định). Item thiếu nhãn bị bỏ qua."""
    from sklearn.metrics import normalized_mutual_info_score

    lab, clu = _labeled(labels, clusters)
    if len(lab) == 0:
        return float("nan")
    return float(normalized_mutual_info_score(lab.astype(str), clu))


def ami(labels, clusters) -> float:
    """Adjusted mutual information: NMI đã trừ kỳ vọng khi gán cụm ngẫu nhiên (0 = như ngẫu nhiên, 1 = khớp hoàn toàn)."""
    from sklearn.metrics import adjusted_mutual_info_score

    lab, clu = _labeled(labels, clusters)
    if len(lab) == 0:
        return float("nan")
    return float(adjusted_mutual_info_score(lab.astype(str), clu))


def codebook_entropy(ids, codebook_size: int = CODEBOOK_SIZE) -> np.ndarray:
    """Entropy chuẩn hóa của phân bố mã ở từng vị trí: H / log2(codebook_size), trong [0, 1]."""
    arr = np.asarray(ids)
    out = np.empty(arr.shape[1])
    for j in range(arr.shape[1]):
        p = np.bincount(arr[:, j], minlength=codebook_size) / arr.shape[0]
        p = p[p > 0]
        out[j] = float(-(p * np.log2(p)).sum() / np.log2(codebook_size))
    return out


def collision_rate(ids) -> float:
    """Tỉ lệ item có ID (cả 32 mã) trùng với ít nhất một item khác."""
    _, inverse, counts = np.unique(np.asarray(ids), axis=0, return_inverse=True, return_counts=True)
    return float((counts[inverse.ravel()] > 1).mean())


def full_id_clusters(ids) -> np.ndarray:
    """Chỉ số cụm khi coi cả ID là một cụm (item trùng ID hoàn toàn chung một cụm)."""
    _, inverse = np.unique(np.asarray(ids), axis=0, return_inverse=True)
    return inverse.ravel()


# --- Láng giềng theo số vị trí trùng mã ---

def nearest_by_shared_codes(ids, k: int = K_NEIGHBORS, seed: int = 0, chunk: int = 512) -> np.ndarray:
    """(N, k) chỉ số k item có nhiều vị trí trùng mã nhất với mỗi item (không tính chính nó).

    Hòa số mã trùng thì chọn ngẫu nhiên theo ``seed`` (cộng nhiễu trong [0, 1) vào số nguyên).
    """
    arr = np.asarray(ids)
    n = arr.shape[0]
    if not 1 <= k < n:
        raise ValueError(f"k phải trong [1, {n - 1}], nhận {k}")
    rng = np.random.default_rng(seed)
    out = np.empty((n, k), dtype=np.int64)
    for start in range(0, n, chunk):
        rows = np.arange(start, min(start + chunk, n))
        shared = np.zeros((len(rows), n), dtype=np.float32)
        for j in range(arr.shape[1]):
            shared += arr[rows, j][:, None] == arr[None, :, j]
        shared += rng.random(shared.shape, dtype=np.float32)
        shared[np.arange(len(rows)), rows] = -1.0
        top = np.argpartition(-shared, k - 1, axis=1)[:, :k]
        order = np.argsort(-np.take_along_axis(shared, top, axis=1), axis=1)
        out[rows] = np.take_along_axis(top, order, axis=1)
    return out


def neighbor_purity(neighbors, labels) -> float:
    """Trung bình tỉ lệ láng giềng cùng nhãn, chỉ tính item có nhãn và láng giềng có nhãn."""
    lab = np.asarray(labels, dtype=object)
    scores = []
    for i, nb in enumerate(np.asarray(neighbors)):
        if lab[i] is None:
            continue
        known = [lab[j] for j in nb if lab[j] is not None]
        if known:
            scores.append(sum(x == lab[i] for x in known) / len(known))
    return float(np.mean(scores)) if scores else float("nan")


def chance_neighbor_purity(labels) -> float:
    """Mốc ngẫu nhiên: xác suất hai item khác nhau (cùng có nhãn) có chung nhãn = Σ n_c(n_c − 1) / (n(n − 1))."""
    counts = np.array(list(Counter(x for x in labels if x is not None).values()), dtype=float)
    n = counts.sum()
    return float((counts * (counts - 1)).sum() / (n * (n - 1))) if n > 1 else float("nan")


# --- Tổng hợp ---

def evaluate(ids, labels: dict, k: int = K_NEIGHBORS, codebook_size: int = CODEBOOK_SIZE,
             seed: int = 0) -> tuple[dict, list[dict]]:
    """Trả về (một dòng tóm tắt, danh sách dòng theo từng vị trí codebook)."""
    arr = np.asarray(ids)
    for t, lab in labels.items():
        if len(lab) != arr.shape[0]:
            raise ValueError(f"Nhãn {t} có {len(lab)} phần tử, ID có {arr.shape[0]} hàng")
    entropy = codebook_entropy(arr, codebook_size)
    full = full_id_clusters(arr)
    neighbors = nearest_by_shared_codes(arr, k=k, seed=seed)

    summary = {"n_items": int(arr.shape[0]), "n_codebook": int(arr.shape[1]),
               "collision_rate": collision_rate(arr),
               "entropy_mean": float(entropy.mean()), "entropy_min": float(entropy.min())}
    per_position = [{"position": j, "entropy": float(entropy[j])} for j in range(arr.shape[1])]
    for t, lab in labels.items():
        pur = [purity(lab, arr[:, j]) for j in range(arr.shape[1])]
        mi = [nmi(lab, arr[:, j]) for j in range(arr.shape[1])]
        adj = [ami(lab, arr[:, j]) for j in range(arr.shape[1])]
        for j in range(arr.shape[1]):
            per_position[j][f"{t}_purity"] = pur[j]
            per_position[j][f"{t}_nmi"] = mi[j]
            per_position[j][f"{t}_ami"] = adj[j]
        summary.update({
            f"{t}_purity_pos_mean": float(np.mean(pur)), f"{t}_nmi_pos_mean": float(np.mean(mi)),
            f"{t}_ami_pos_mean": float(np.mean(adj)),
            f"{t}_purity_full_id": purity(lab, full), f"{t}_nmi_full_id": nmi(lab, full),
            f"{t}_ami_full_id": ami(lab, full),
            f"{t}_neighbor_purity@{k}": neighbor_purity(neighbors, lab),
            f"{t}_neighbor_chance": chance_neighbor_purity(lab),
        })
    return summary, per_position


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.cluster_metrics", description=__doc__.split("\n")[0])
    parser.add_argument("--items", required=True, help="data/processed/items.jsonl của pipeline B")
    parser.add_argument("--ids", nargs="+", required=True, metavar="TÊN=FILE.npy",
                        help="mỗi biến thể một cặp tên=đường dẫn, mảng (N, 32) theo item_id")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--k", type=int, default=K_NEIGHBORS)
    parser.add_argument("--codebook-size", type=int, default=CODEBOOK_SIZE)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    labels, stats = load_item_labels(args.items)
    print("Nhãn:", json.dumps(stats, ensure_ascii=False))
    summaries, positions = [], []
    for spec in args.ids:
        name, sep, path = spec.partition("=")
        if not sep or not name or not path:
            parser.error(f"--ids cần dạng tên=file.npy, nhận {spec!r}")
        ids = np.load(path, allow_pickle=False)
        summary, per_pos = evaluate(ids, labels, k=args.k, codebook_size=args.codebook_size, seed=args.seed)
        summaries.append({"variant": name, **summary})
        positions += [{"variant": name, **row} for row in per_pos]
        print(f"{name}: " + ", ".join(f"{key}={val:.4f}" for key, val in summary.items()
                                      if isinstance(val, float) and ("neighbor" in key or "ami_pos" in key)))
    out = Path(args.out_dir)
    _write_csv(out / "rq1_cluster_metrics.csv", summaries)
    _write_csv(out / "rq1_cluster_metrics_per_position.csv", positions)
    (out / "rq1_label_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n",
                                              encoding="utf-8")
    print(f"Đã ghi {out / 'rq1_cluster_metrics.csv'} và {out / 'rq1_cluster_metrics_per_position.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
