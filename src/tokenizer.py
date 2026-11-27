"""Tạo semantic ID tự viết cho RQ1: PQ, RQ và token ngẫu nhiên (hoán vị mã OPQ). ID có tín hiệu hành vi thêm sau.

Mọi mảng theo quy ước của nhóm: hàng i ↔ item_id i + 1 trong ``id_mapping.json``; mã thô
0..codebook_size-1, chưa cộng offset (code gốc tự cộng khi nạp). Ghi vào code gốc bằng
``adapters.load_ids.write_semantic_ids`` (chế độ b).

Đầu vào chung của PQ và RQ: embedding sentence-t5 768 chiều → PCA 128 chiều có whiten, fit trên toàn bộ item,
giống ``genrec/models/RPG/tokenizer.py`` dòng 278–282 của code gốc. Khác code gốc ở một điểm: truyền
``random_state`` cho PCA. Với dữ liệu thật (12.101 × 768 → 128), scikit-learn ≥ 1.5 chọn solver
``covariance_eigh`` tất định nên kết quả trùng code gốc; với dữ liệu nhỏ, sklearn có thể chọn solver
``randomized`` và khi đó cần ``random_state`` để chạy lại ra cùng kết quả (review PR rq1/pq-random, 25/11).

Ba biến thể (quyết định 23/11, sửa 25/11 sau review):

- **PQ:** chạy đúng quy trình FAISS của code gốc (``tokenizer.py`` dòng 178–229: ``index_factory`` với
  ``METRIC_INNER_PRODUCT``, train trên item có trong phần train, add mọi item, đọc mã từ danh sách IVF duy
  nhất), chỉ bỏ phần ``OPQ32,`` khỏi chuỗi ``OPQ32,IVF1,PQ32x8``. Mọi tham số còn lại (số vòng k-means 25,
  seed 1234 của FAISS, số luồng) giữ mặc định như OPQ, nên PQ và OPQ chỉ khác nhau ở phép xoay học được.
  ``opq_ids`` chạy chuỗi đầy đủ để kiểm tra code này tái tạo đúng semantic ID của code gốc.
- **RQ** (residual quantization): 32 tầng k-means 256 cụm nối tiếp trên toàn bộ 128 chiều; tầng j học trên phần
  dư sau j tầng trước (chỉ item train), gán mã cho mọi item, rồi trừ tâm cụm đã gán. Dùng ``faiss.Kmeans``
  với tham số mặc định của FAISS (25 vòng, seed 1234, cùng seed cho mọi tầng) để thống nhất với PQ/OPQ.
- **Ngẫu nhiên:** hoán vị mã OPQ của code gốc giữa các item, độc lập theo từng codebook (seed 42). Mỗi codebook
  giữ nguyên phân bố mã, nhưng mã của một item không còn liên quan tới nội dung. Đảm bảo không có ID trùng.

Mỗi biến thể có **một** bộ ID; các lần train seed 42, 43, 44 đều dùng chung bộ đó, nên độ lệch chuẩn giữa
các seed chỉ phản ánh nhiễu khi train, không phản ánh nhiễu khi tạo ID (hạn chế đã ghi trong decisions.md).

Ví dụ trên Kaggle/Colab::

    python -m src.tokenizer opq --processed-dir <processed> --out opq_check.npy   # tự kiểm tra
    python -m src.tokenizer pq --processed-dir <processed> --out pq.npy
    python -m src.tokenizer rq --processed-dir <processed> --out rq.npy
    python -m src.tokenizer random --processed-dir <processed> --out random.npy

Phụ trách: A.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

N_CODEBOOK = 32
CODEBOOK_SIZE = 256
PCA_DIM = 128
SEED = 42            # seed của PCA và của hoán vị (token ngẫu nhiên)
FAISS_NITER = 25     # mặc định ClusteringParameters.niter của FAISS (dùng cho bộ lượng tử hóa PQ trong OPQ gốc)
FAISS_SEED = 1234    # mặc định ClusteringParameters.seed của FAISS
FAISS_THREADS = os.cpu_count() or 2   # scripts/run_rpg.py đặt faiss_omp_num_threads = số CPU khi tạo OPQ


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


# --- PCA giống code gốc, có seed ---

def pca_whiten(emb, n_components: int = PCA_DIM, seed: int = SEED) -> np.ndarray:
    """PCA có whiten, fit trên toàn bộ item (như code gốc), thêm ``random_state`` để chạy lại ra cùng kết quả."""
    from sklearn.decomposition import PCA

    x = np.asarray(emb, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] < n_components:
        raise ValueError(f"Cần mảng 2 chiều có ít nhất {n_components} cột, nhận shape {x.shape}")
    pca = PCA(n_components=n_components, whiten=True, random_state=seed)
    return pca.fit_transform(x).astype(np.float32)


# --- Kiểm tra đầu vào chung ---

def _bits(codebook_size: int) -> int:
    bits = math.log2(codebook_size)
    if codebook_size < 2 or not bits.is_integer():
        raise ValueError(f"codebook_size phải là lũy thừa của 2, nhận {codebook_size}")
    return int(bits)


def _check_quant_input(x, codebook_size, train_mask):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"x phải là mảng 2 chiều, nhận ndim={x.ndim}")
    if not 2 <= codebook_size <= 32767:
        raise ValueError(f"codebook_size phải trong [2, 32767], nhận {codebook_size}")
    n = x.shape[0]
    mask = np.ones(n, dtype=bool) if train_mask is None else np.asarray(train_mask, dtype=bool)
    if mask.shape != (n,):
        raise ValueError(f"train_mask phải có shape ({n},), nhận {mask.shape}")
    if mask.sum() < codebook_size:
        raise ValueError(f"Cần ít nhất {codebook_size} hàng train để học {codebook_size} tâm cụm, có {mask.sum()}")
    return x, mask


# --- PQ và OPQ bằng đúng quy trình FAISS của code gốc ---

def faiss_factory(n_codebook: int = N_CODEBOOK, codebook_size: int = CODEBOOK_SIZE, rotation: bool = False) -> str:
    """Chuỗi ``index_factory``: có phép xoay là chuỗi của code gốc (OPQ), không có là PQ thường."""
    pq = f"IVF1,PQ{n_codebook}x{_bits(codebook_size)}"
    return f"OPQ{n_codebook},{pq}" if rotation else pq


def faiss_quantize(x, factory: str, n_codebook: int = N_CODEBOOK, codebook_size: int = CODEBOOK_SIZE, *,
                   train_mask=None, threads: int | None = None) -> np.ndarray:
    """Lượng tử hóa bằng FAISS theo đúng các bước của code gốc (``tokenizer.py`` dòng 192–222).

    ``index_factory(D, factory, METRIC_INNER_PRODUCT)`` → train trên hàng ``train_mask`` → add mọi hàng →
    đọc mã của danh sách IVF duy nhất bằng ``BitstringReader``. Khác code gốc: sắp xếp lại theo id trong
    danh sách IVF thay vì giả định thứ tự add (kết quả như nhau, nhưng không phụ thuộc giả định đó).
    Trả về mảng int16 (N, n_codebook).
    """
    import faiss

    x, mask = _check_quant_input(x, codebook_size, train_mask)
    bits = _bits(codebook_size)
    if threads:
        faiss.omp_set_num_threads(int(threads))
    index = faiss.index_factory(x.shape[1], factory, faiss.METRIC_INNER_PRODUCT)
    index.train(np.ascontiguousarray(x[mask]))
    index.add(np.ascontiguousarray(x))

    invlists = faiss.extract_index_ivf(index).invlists
    n_list = invlists.list_size(0)
    code_size = invlists.code_size
    if n_list != x.shape[0]:
        raise RuntimeError(f"Danh sách IVF có {n_list} mã, cần {x.shape[0]}")
    raw = faiss.rev_swig_ptr(invlists.get_codes(0), n_list * code_size).reshape(n_list, code_size).copy()
    ids = faiss.rev_swig_ptr(invlists.get_ids(0), n_list).copy()
    if not np.array_equal(np.sort(ids), np.arange(n_list)):
        raise RuntimeError("Id trong danh sách IVF không phải 0..N-1")

    codes = np.empty((n_list, n_codebook), dtype=np.int16)
    for r in range(n_list):
        reader = faiss.BitstringReader(faiss.swig_ptr(raw[r]), code_size)
        codes[r] = [reader.read(bits) for _ in range(n_codebook)]
    out = np.empty_like(codes)
    out[ids] = codes
    return out


def pq_ids(x, n_codebook: int = N_CODEBOOK, codebook_size: int = CODEBOOK_SIZE, *,
           train_mask=None, threads: int | None = None) -> np.ndarray:
    """PQ thường: quy trình FAISS của code gốc với chuỗi ``IVF1,PQ{m}x{bits}`` (bỏ ``OPQ{m},``)."""
    return faiss_quantize(x, faiss_factory(n_codebook, codebook_size, rotation=False), n_codebook, codebook_size,
                          train_mask=train_mask, threads=threads)


def opq_ids(x, n_codebook: int = N_CODEBOOK, codebook_size: int = CODEBOOK_SIZE, *,
            train_mask=None, threads: int | None = None) -> np.ndarray:
    """OPQ như code gốc (chuỗi ``OPQ{m},IVF1,PQ{m}x{bits}``); dùng để kiểm tra ``faiss_quantize`` tái tạo đúng."""
    return faiss_quantize(x, faiss_factory(n_codebook, codebook_size, rotation=True), n_codebook, codebook_size,
                          train_mask=train_mask, threads=threads)


# --- k-means cho RQ ---

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


def rq_ids(x, n_codebook: int = N_CODEBOOK, codebook_size: int = CODEBOOK_SIZE, *,
           train_mask=None, seed: int = FAISS_SEED, niter: int = FAISS_NITER, backend: str = "faiss",
           return_codebooks: bool = False):
    """Residual quantization: ``n_codebook`` tầng k-means nối tiếp trên toàn bộ số chiều của ``x``.

    Tầng j học ``codebook_size`` tâm cụm trên phần dư (chỉ các hàng ``train_mask``), gán mã cho mọi hàng, rồi
    trừ tâm cụm đã gán khỏi phần dư của hàng đó. Mặc định dùng FAISS với tham số mặc định của FAISS (25 vòng,
    seed 1234 cho mọi tầng); ``backend='sklearn'`` chỉ dùng khi không có faiss (test trên CI).
    Trả về mảng int16 (N, n_codebook); kèm danh sách tâm cụm nếu ``return_codebooks``.
    """
    x, mask = _check_quant_input(x, codebook_size, train_mask)
    residual = x.copy()
    codes = np.empty((x.shape[0], n_codebook), dtype=np.int16)
    codebooks = []
    for j in range(n_codebook):
        centroids = _kmeans(residual[mask], codebook_size, seed, niter, backend)
        assigned = _assign(residual, centroids)
        codes[:, j] = assigned
        residual -= centroids[assigned]
        codebooks.append(centroids)
    return (codes, codebooks) if return_codebooks else codes


def rq_residual_energy(x, codes, codebooks) -> np.ndarray:
    """Tỉ lệ năng lượng còn lại sau mỗi tầng RQ: phần tử j = mean ||phần dư sau j tầng||² / mean ||x||².

    Phần tử 0 bằng 1; giá trị giảm nhanh nghĩa là các tầng sau gần như không còn thông tin để mã hóa.
    """
    x = np.asarray(x, dtype=np.float32)
    residual = x.copy()
    base = float((x * x).sum(1).mean())
    energy = [1.0]
    for j, centroids in enumerate(codebooks):
        residual -= centroids[np.asarray(codes)[:, j]]
        energy.append(float((residual * residual).sum(1).mean()) / base)
    return np.array(energy)


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


def library_versions() -> dict:
    """Phiên bản thư viện để ghi vào nhật ký (tái lập ID)."""
    import sklearn

    out = {"numpy": np.__version__, "sklearn": sklearn.__version__}
    try:
        import faiss
        out["faiss"] = faiss.__version__
    except ImportError:
        out["faiss"] = None
    return out


# --- CLI ---

def _utf8_stdout() -> None:
    """In được tiếng Việt cả trên console Windows (cp1252/cp437); bỏ qua nếu luồng không đổi được mã hóa."""
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError, OSError):
        pass


def main(argv: Sequence[str] | None = None) -> int:
    _utf8_stdout()
    parser = argparse.ArgumentParser(prog="python -m src.tokenizer", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="method", required=True)
    parsers = {}
    for name, help_text in [("pq", "PQ thường bằng FAISS (chuỗi IVF1,PQ32x8) trên PCA 128"),
                            ("opq", "OPQ như code gốc, để kiểm tra tái tạo đúng semantic ID của code gốc"),
                            ("rq", "RQ 32 tầng k-means trên PCA 128"),
                            ("random", "hoán vị mã OPQ của code gốc giữa các item")]:
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--processed-dir", required=True)
        p.add_argument("--out", required=True)
        p.add_argument("--seed", type=int, default=SEED, help="seed của PCA (pq, opq, rq) hoặc của hoán vị (random)")
        p.add_argument("--n-codebook", type=int, default=N_CODEBOOK)
        p.add_argument("--codebook-size", type=int, default=CODEBOOK_SIZE)
        parsers[name] = p
    for name in ("pq", "opq", "rq"):
        p = parsers[name]
        p.add_argument("--emb-name", default="sentence-t5-base")
        p.add_argument("--emb-dim", type=int, default=768)
        p.add_argument("--pca-dim", type=int, default=PCA_DIM)
    for name in ("pq", "opq"):
        parsers[name].add_argument("--threads", type=int, default=FAISS_THREADS,
                                   help="số luồng FAISS (scripts/run_rpg.py dùng số CPU của máy khi code gốc tạo OPQ)")
    parsers["rq"].add_argument("--kmeans-seed", type=int, default=FAISS_SEED)
    parsers["rq"].add_argument("--niter", type=int, default=FAISS_NITER)
    parsers["rq"].add_argument("--backend", default="faiss", choices=["faiss", "sklearn", "auto"])
    parsers["random"].add_argument("--source-name", default="sentence-t5-base",
                                   help="tên file .sem_ids nguồn (mặc định: ID OPQ của code gốc)")

    args = parser.parse_args(argv)
    proc = Path(args.processed_dir)
    m, k = args.n_codebook, args.codebook_size
    if args.method in ("pq", "opq", "rq"):
        x = pca_whiten(load_embeddings(proc, args.emb_name, args.emb_dim), args.pca_dim, seed=args.seed)
        mask = train_item_mask(proc)
        print(f"PCA: {x.shape} | item dùng để học tâm cụm: {int(mask.sum()):,} / {len(mask):,}")
        if args.method == "pq":
            ids = pq_ids(x, m, k, train_mask=mask, threads=args.threads)
        elif args.method == "opq":
            ids = opq_ids(x, m, k, train_mask=mask, threads=args.threads)
        else:
            ids, books = rq_ids(x, m, k, train_mask=mask, seed=args.kmeans_seed, niter=args.niter,
                                backend=args.backend, return_codebooks=True)
            energy = rq_residual_energy(x, ids, books)
            marks = [j for j in (1, 2, 4, 8, 16, 32) if j < len(energy)]
            print("Năng lượng phần dư còn lại sau tầng " + ", ".join(f"{j}: {energy[j]:.1%}" for j in marks))
    else:
        from adapters.load_ids import read_semantic_ids

        ids = permuted_ids(read_semantic_ids(proc, args.source_name, n_codebook=m, codebook_size=k), seed=args.seed)
    np.save(args.out, ids)
    print(f"Đã ghi {args.out}: {summary(ids, k)}")
    print(f"Thư viện: {library_versions()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
