"""Tạo semantic ID tự viết cho RQ1: PQ và token ngẫu nhiên (hoán vị mã OPQ). RQ và ID có tín hiệu hành vi thêm sau.

Mọi mảng theo quy ước của nhóm: hàng i ↔ item_id i + 1 trong ``id_mapping.json``; mã thô
0..codebook_size-1, chưa cộng offset (code gốc tự cộng khi nạp). Ghi vào code gốc bằng
``adapters.load_ids.write_semantic_ids`` (chế độ b).

Hai biến thể và lý do thiết kế (quyết định 23/11):

- **PQ**: giống hệt đường đi của code gốc trước bước OPQ, tức embedding sentence-t5 768 chiều →
  PCA 128 chiều có whiten, fit trên toàn bộ item (``tokenizer.py`` dòng 278–282) → chia 32 đoạn liên
  tiếp × 4 chiều → k-means 256 cụm trên từng đoạn, chỉ fit trên item có trong phần train (như
  ``index.train(sent_embs[train_mask])``, dòng 202), rồi gán mã cho mọi item. Khác OPQ đúng một điểm:
  không có phép xoay học được, nên chênh lệch PQ ↔ OPQ đo được tác dụng của phép xoay.
- **Ngẫu nhiên**: hoán vị mã OPQ giữa các item, độc lập theo từng codebook. Mỗi codebook giữ nguyên
  phân bố mã (số item dùng mỗi mã), nhưng mã của một item không còn liên quan tới nội dung của nó.
  Đảm bảo không có hai item trùng ID hoàn toàn.

Ví dụ trên Colab::

    python -m src.tokenizer pq --processed-dir <processed> --out pq.npy
    python -m src.tokenizer random --processed-dir <processed> --out random.npy

Phụ trách: A.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

N_CODEBOOK = 32
CODEBOOK_SIZE = 256
PCA_DIM = 128
SEED = 42


# --- Dữ liệu đầu vào lấy từ cache của code gốc ---

def _items(proc_dir: Path) -> tuple[list[str], dict[str, int]]:
    mapping = json.loads((proc_dir / "id_mapping.json").read_text(encoding="utf-8"))
    id2item = mapping["id2item"]
    if not id2item or id2item[0] not in ("[PAD]", None):
        raise ValueError("Phần tử 0 của id2item phải là padding")
    return id2item[1:], {a: i for i, a in enumerate(id2item) if i}


def train_item_mask(proc_dir: str | Path) -> np.ndarray:
    """Mặt nạ (N,) đánh dấu item xuất hiện trong phần train, giống ``_get_items_for_training`` của code gốc.

    Code gốc chia leave-one-out: train = chuỗi bỏ 2 item cuối, chỉ lấy user có hơn 2 item
    (``genrec/dataset.py`` dòng 104–128); mặt nạ là tập item trong các chuỗi train đó
    (``tokenizer.py`` dòng 158–176). Đọc từ ``all_item_seqs.json`` và ``id_mapping.json``.
    """
    proc_dir = Path(proc_dir)
    items, item2id = _items(proc_dir)
    seqs = json.loads((proc_dir / "all_item_seqs.json").read_text(encoding="utf-8"))
    mask = np.zeros(len(items), dtype=bool)
    for seq in seqs.values():
        if len(seq) > 2:
            for asin in seq[:-2]:
                mask[item2id[asin] - 1] = True
    return mask


def load_embeddings(proc_dir: str | Path, name: str = "sentence-t5-base", dim: int = 768) -> np.ndarray:
    """Đọc ``<name>.sent_emb`` của code gốc (float32 nhị phân, hàng i ↔ item_id i + 1)."""
    proc_dir = Path(proc_dir)
    items, _ = _items(proc_dir)
    raw = np.fromfile(proc_dir / f"{name}.sent_emb", dtype=np.float32)
    if raw.size != len(items) * dim:
        raise ValueError(f"{name}.sent_emb có {raw.size} số, cần {len(items)} x {dim}")
    return raw.reshape(len(items), dim)


# --- PCA giống code gốc ---

def pca_whiten(emb, n_components: int = PCA_DIM) -> np.ndarray:
    """PCA có whiten, fit trên toàn bộ item, giống ``tokenizer.py`` dòng 278–282 của code gốc."""
    from sklearn.decomposition import PCA

    x = np.asarray(emb, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] < n_components:
        raise ValueError(f"Cần mảng 2 chiều có ít nhất {n_components} cột, nhận shape {x.shape}")
    return PCA(n_components=n_components, whiten=True).fit_transform(x).astype(np.float32)


# --- k-means cho từng đoạn ---

def _kmeans(x: np.ndarray, k: int, seed: int, niter: int, backend: str) -> np.ndarray:
    """Trả về tâm cụm (k, d) học trên ``x``."""
    if backend in ("auto", "faiss"):
        try:
            import faiss
        except ImportError:
            if backend == "faiss":
                raise
        else:
            km = faiss.Kmeans(x.shape[1], k, niter=niter, seed=seed, verbose=False)
            km.train(np.ascontiguousarray(x, dtype=np.float32))
            return np.asarray(km.centroids, dtype=np.float32)
    if backend not in ("auto", "sklearn"):
        raise ValueError(f"backend phải là auto, faiss hoặc sklearn, nhận {backend!r}")
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=k, n_init=1, max_iter=niter, random_state=seed).fit(x)
    return km.cluster_centers_.astype(np.float32)


def _assign(x: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Chỉ số tâm cụm gần nhất (khoảng cách L2) cho từng hàng của ``x``."""
    d2 = (x * x).sum(1, keepdims=True) - 2.0 * x @ centroids.T + (centroids * centroids).sum(1)
    return d2.argmin(1)


def pq_ids(x, n_codebook: int = N_CODEBOOK, codebook_size: int = CODEBOOK_SIZE, *,
           train_mask=None, seed: int = SEED, niter: int = 20, backend: str = "auto",
           return_codebooks: bool = False):
    """Product quantization không xoay: chia cột thành ``n_codebook`` đoạn liên tiếp bằng nhau.

    ``x``: (N, D) với D chia hết cho ``n_codebook`` (sau PCA 128: 32 đoạn × 4 chiều).
    ``train_mask``: (N,) bool; tâm cụm chỉ học trên các hàng True, mã gán cho mọi hàng.
    Trả về mảng int16 (N, n_codebook); kèm danh sách tâm cụm nếu ``return_codebooks``.
    """
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"x phải là mảng 2 chiều, nhận ndim={x.ndim}")
    n, dim = x.shape
    if dim % n_codebook:
        raise ValueError(f"Số chiều {dim} không chia hết cho n_codebook={n_codebook}")
    if not 2 <= codebook_size <= 32767:
        raise ValueError(f"codebook_size phải trong [2, 32767], nhận {codebook_size}")
    mask = np.ones(n, dtype=bool) if train_mask is None else np.asarray(train_mask, dtype=bool)
    if mask.shape != (n,):
        raise ValueError(f"train_mask phải có shape ({n},), nhận {mask.shape}")
    if mask.sum() < codebook_size:
        raise ValueError(f"Cần ít nhất {codebook_size} hàng train để học {codebook_size} tâm cụm, có {mask.sum()}")

    dsub = dim // n_codebook
    codes = np.empty((n, n_codebook), dtype=np.int16)
    codebooks = []
    for j in range(n_codebook):
        sub = x[:, j * dsub:(j + 1) * dsub]
        centroids = _kmeans(sub[mask], codebook_size, seed + j, niter, backend)
        codes[:, j] = _assign(sub, centroids)
        codebooks.append(centroids)
    return (codes, codebooks) if return_codebooks else codes


# --- Token ngẫu nhiên ---

def permuted_ids(ids, seed: int = SEED, max_tries: int = 100) -> np.ndarray:
    """Hoán vị mã giữa các item, độc lập theo từng codebook, đến khi không còn ID trùng.

    Mỗi cột của kết quả là một hoán vị của cột tương ứng trong ``ids``, nên số item dùng mỗi mã
    trong từng codebook giữ nguyên. Báo lỗi nếu sau ``max_tries`` lần vẫn còn ID trùng.
    """
    arr = np.asarray(ids)
    if arr.ndim != 2:
        raise ValueError(f"ids phải là mảng 2 chiều, nhận ndim={arr.ndim}")
    rng = np.random.default_rng(seed)
    for _ in range(max_tries):
        out = np.stack([rng.permutation(arr[:, j]) for j in range(arr.shape[1])], axis=1)
        if len(np.unique(out, axis=0)) == len(out):
            return out.astype(np.int16)
    raise ValueError(f"Không tạo được ID không trùng sau {max_tries} lần hoán vị")


# --- Thống kê nhanh (độ đo đầy đủ nằm trong src/cluster_metrics.py) ---

def summary(ids, codebook_size: int = CODEBOOK_SIZE) -> dict:
    """Số mã được dùng mỗi codebook và số item trùng ID hoàn toàn."""
    arr = np.asarray(ids)
    _, counts = np.unique(arr, axis=0, return_counts=True)
    used = [int(len(np.unique(arr[:, j]))) for j in range(arr.shape[1])]
    return {
        "shape": list(arr.shape),
        "codes_used_min": min(used), "codes_used_max": max(used), "codebook_size": codebook_size,
        "n_colliding_items": int(counts[counts > 1].sum()),
    }


# --- CLI ---

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.tokenizer", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="method", required=True)

    p_pq = sub.add_parser("pq", help="PQ trên PCA 128 của embedding code gốc")
    p_pq.add_argument("--processed-dir", required=True)
    p_pq.add_argument("--out", required=True)
    p_pq.add_argument("--emb-name", default="sentence-t5-base")
    p_pq.add_argument("--emb-dim", type=int, default=768)
    p_pq.add_argument("--seed", type=int, default=SEED)
    p_pq.add_argument("--niter", type=int, default=20)
    p_pq.add_argument("--backend", default="auto", choices=["auto", "faiss", "sklearn"])
    p_pq.add_argument("--pca-dim", type=int, default=PCA_DIM)

    p_rd = sub.add_parser("random", help="hoán vị mã OPQ của code gốc giữa các item")
    p_rd.add_argument("--processed-dir", required=True)
    p_rd.add_argument("--out", required=True)
    p_rd.add_argument("--source-name", default="sentence-t5-base",
                      help="tên file .sem_ids nguồn (mặc định: ID OPQ của code gốc)")
    p_rd.add_argument("--seed", type=int, default=SEED)
    for p in (p_pq, p_rd):
        p.add_argument("--n-codebook", type=int, default=N_CODEBOOK)
        p.add_argument("--codebook-size", type=int, default=CODEBOOK_SIZE)

    args = parser.parse_args(argv)
    proc = Path(args.processed_dir)
    if args.method == "pq":
        x = pca_whiten(load_embeddings(proc, args.emb_name, args.emb_dim), args.pca_dim)
        mask = train_item_mask(proc)
        print(f"PCA: {x.shape} | item dùng để học tâm cụm: {int(mask.sum()):,} / {len(mask):,}")
        ids = pq_ids(x, args.n_codebook, args.codebook_size, train_mask=mask, seed=args.seed,
                     niter=args.niter, backend=args.backend)
    else:
        from adapters.load_ids import read_semantic_ids

        src = read_semantic_ids(proc, args.source_name, n_codebook=args.n_codebook,
                                codebook_size=args.codebook_size)
        ids = permuted_ids(src, seed=args.seed)
    np.save(args.out, ids)
    print(f"Đã ghi {args.out}: {summary(ids, args.codebook_size)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
