"""Adapter nạp embedding hoặc semantic ID do nhóm tạo vào code gốc RPG_KDD2025, không sửa code gốc.

Code gốc (commit 7dcf95c, ``genrec/models/RPG/tokenizer.py``) đọc hai file cache trong
``<cache_dir>/AmazonReviews2014/<category>/processed/``:

- ``<tên>.sent_emb``: embedding nhị phân float32 không header, hàng i ↔ item_id i + 1
  (``_encode_sent_emb``, dòng 104–106 và 155; đọc lại ở dòng 271–273).
- ``<tên>_OPQ32,IVF1,PQ32x8.sem_ids``: JSON ``{asin: [32 mã thô 0..255]}`` (dòng 223–229).
  Nếu file này đã có, code gốc bỏ qua encode, PCA và OPQ (dòng 265) rồi tự cộng offset
  ``256 * digit + 1`` để ra token (``_sem_ids_to_tokens``, dòng 241–247).

``<tên>`` là ``os.path.basename(sent_emb_model)`` (dòng 260–263), nên chạy code gốc với
``sent_emb_model = 'x/<tên>'`` sẽ khiến nó dùng file của nhóm thay cho file của mình.

Hai chế độ:

- (a) ``write_embeddings``: ghi embedding tự tạo thành ``<tên>.sent_emb``; code gốc tự chạy
  PCA (``sent_emb_pca``) và OPQ trên embedding đó. Dùng cho nâng cấp ghép embedding hành vi.
- (b) ``write_semantic_ids``: ghi semantic ID tự tạo (PQ, RQ, ngẫu nhiên...) thành
  ``<tên>_OPQ32,IVF1,PQ32x8.sem_ids``; code gốc nạp thẳng, không chạy OPQ. Tên file vẫn chứa
  ``OPQ32,IVF1,PQ32x8`` vì code gốc ghép chuỗi này cố định; nội dung là ID của nhóm.

Quy ước dữ liệu của nhóm: hàng i của mọi mảng ↔ ``item_id`` i + 1 trong ``id_mapping.json``
(đã kiểm tra khớp từng vị trí giữa code gốc và pipeline của B, xem ``reports/decisions.md``).
Mã trong mảng semantic ID là mã thô 0..codebook_size-1, chưa cộng offset.

Ví dụ trên Colab::

    python -m adapters.load_ids ids --processed-dir <processed> --npy pq.npy --name pq
    python /content/drive/MyDrive/rpg/scripts/run_rpg.py --sent_emb_model=x/pq --seed=42 --run_id=pq_s42

Phụ trách: A.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np

N_CODEBOOK = 32
CODEBOOK_SIZE = 256
# Tên file cache mà code gốc tự tạo; ghi đè sẽ làm hỏng cấu hình chính của nhóm.
RESERVED_NAMES = frozenset({"sentence-t5-base", "text-embedding-3-large"})
_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


# --- Đường dẫn và tên file theo đúng quy ước của code gốc ---

def index_factory(n_codebook: int = N_CODEBOOK, codebook_size: int = CODEBOOK_SIZE) -> str:
    """Chuỗi FAISS mà code gốc ghép vào tên file semantic ID (tokenizer.py dòng 41–42, 85–88)."""
    bits = math.log2(codebook_size)
    if codebook_size < 2 or not bits.is_integer():
        raise ValueError(f"codebook_size phải là lũy thừa của 2, nhận {codebook_size}")
    if n_codebook < 1:
        raise ValueError(f"n_codebook phải >= 1, nhận {n_codebook}")
    return f"OPQ{n_codebook},IVF1,PQ{n_codebook}x{int(bits)}"


def processed_dir(cache_dir: str | Path, category: str = "Beauty") -> Path:
    """Thư mục ``processed`` mà code gốc dùng cho Amazon Reviews 2014."""
    return Path(cache_dir) / "AmazonReviews2014" / category / "processed"


def sem_ids_path(proc_dir: str | Path, name: str,
                 n_codebook: int = N_CODEBOOK, codebook_size: int = CODEBOOK_SIZE) -> Path:
    return Path(proc_dir) / f"{name}_{index_factory(n_codebook, codebook_size)}.sem_ids"


def sent_emb_path(proc_dir: str | Path, name: str) -> Path:
    return Path(proc_dir) / f"{name}.sent_emb"


def sidecar_path(proc_dir: str | Path, name: str) -> Path:
    """File ghi nguồn gốc do adapter tạo; code gốc không đọc file này."""
    return Path(proc_dir) / f"{name}.adapter.json"


def run_args(name: str, sent_emb_dim: int | None = None) -> dict:
    """Tham số truyền cho code gốc (qua scripts/run_rpg.py) để dùng file mang tên ``name``."""
    _check_name(name, allow_reserved=True)
    args = {"sent_emb_model": f"x/{name}"}
    if sent_emb_dim is not None:
        args["sent_emb_dim"] = int(sent_emb_dim)
    return args


# --- Đọc thứ tự item ---

def load_id2item(proc_dir: str | Path) -> list[str]:
    """Danh sách asin theo item_id 1..N (bỏ phần tử 0 là padding) từ ``id_mapping.json``.

    Chấp nhận cả định dạng của code gốc (phần tử 0 là ``'[PAD]'``) và của B (phần tử 0 là ``null``).
    """
    path = Path(proc_dir) / "id_mapping.json"
    if not path.exists():
        raise FileNotFoundError(f"Không thấy {path}; cần chạy code gốc một lần để tạo cache dữ liệu")
    id2item = json.loads(path.read_text(encoding="utf-8"))["id2item"]
    if not id2item or id2item[0] not in ("[PAD]", None):
        raise ValueError(f"Phần tử 0 của id2item phải là padding, nhận {id2item[:1]}")
    items = id2item[1:]
    if len(set(items)) != len(items):
        raise ValueError("id2item có asin trùng nhau")
    return items


# --- Kiểm tra dữ liệu đầu vào ---

def validate_semantic_ids(ids, n_items: int, n_codebook: int = N_CODEBOOK,
                          codebook_size: int = CODEBOOK_SIZE) -> np.ndarray:
    """Trả về mảng int64 (n_items, n_codebook) nếu hợp lệ, ngược lại báo lỗi rõ ràng."""
    arr = np.asarray(ids)
    if arr.ndim != 2:
        raise ValueError(f"Semantic ID phải là mảng 2 chiều, nhận ndim={arr.ndim}")
    if arr.shape != (n_items, n_codebook):
        raise ValueError(f"Semantic ID phải có shape ({n_items}, {n_codebook}), nhận {arr.shape}")
    if not np.issubdtype(arr.dtype, np.integer):
        raise TypeError(f"Semantic ID phải là số nguyên, nhận dtype {arr.dtype}")
    lo, hi = int(arr.min()), int(arr.max())
    if lo < 0 or hi >= codebook_size:
        raise ValueError(f"Mã phải nằm trong [0, {codebook_size}), nhận min={lo}, max={hi}")
    return arr.astype(np.int64)


def validate_embeddings(emb, n_items: int) -> np.ndarray:
    """Trả về mảng float32 C-contiguous (n_items, dim) nếu hợp lệ."""
    arr = np.asarray(emb)
    if arr.ndim != 2:
        raise ValueError(f"Embedding phải là mảng 2 chiều, nhận ndim={arr.ndim}")
    if arr.shape[0] != n_items:
        raise ValueError(f"Embedding phải có {n_items} hàng (một hàng mỗi item), nhận {arr.shape[0]}")
    if arr.shape[1] < 1:
        raise ValueError("Embedding phải có ít nhất 1 chiều")
    if not np.issubdtype(arr.dtype, np.number):
        raise TypeError(f"Embedding phải là số, nhận dtype {arr.dtype}")
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    if not np.isfinite(arr).all():
        raise ValueError("Embedding chứa NaN hoặc vô cực")
    return arr


def _check_name(name: str, allow_reserved: bool = False) -> None:
    if not isinstance(name, str) or not _NAME_RE.fullmatch(name):
        raise ValueError(f"Tên không hợp lệ: {name!r}; chỉ dùng chữ, số, '_', '-', '.', không có '/'")
    if not allow_reserved and name in RESERVED_NAMES:
        raise ValueError(f"Tên {name!r} là file cache của code gốc; chọn tên khác để không ghi đè")


def _check_overwrite(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} đã tồn tại; truyền overwrite=True (CLI: --overwrite) nếu muốn ghi đè")


# --- Thống kê ---

def to_tokens(ids, codebook_size: int = CODEBOOK_SIZE) -> np.ndarray:
    """Token mà code gốc sẽ dùng: mã + codebook_size * digit + 1 (tokenizer.py dòng 241–247)."""
    arr = np.asarray(ids, dtype=np.int64)
    return arr + codebook_size * np.arange(arr.shape[1], dtype=np.int64) + 1


def collision_stats(ids) -> dict:
    """Số item có semantic ID trùng hoàn toàn với ít nhất một item khác."""
    arr = np.asarray(ids)
    counts = Counter(map(tuple, arr.tolist()))
    groups = [c for c in counts.values() if c > 1]
    return {
        "n_items": int(arr.shape[0]),
        "n_unique_ids": len(counts),
        "n_colliding_items": int(sum(groups)),
        "n_collision_groups": len(groups),
        "collision_rate": float(sum(groups) / arr.shape[0]) if arr.shape[0] else 0.0,
    }


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_sidecar(proc_dir: Path, name: str, info: dict) -> None:
    info = {"name": name, "created_at": time.strftime("%Y-%m-%d %H:%M:%S"), **info}
    sidecar_path(proc_dir, name).write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n",
                                            encoding="utf-8")


# --- Chế độ (b): nạp thẳng semantic ID ---

def write_semantic_ids(ids, proc_dir: str | Path, name: str, *, n_codebook: int = N_CODEBOOK,
                       codebook_size: int = CODEBOOK_SIZE, overwrite: bool = False,
                       source: str | None = None) -> Path:
    """Ghi semantic ID của nhóm theo định dạng code gốc đọc được. Trả về đường dẫn file ``.sem_ids``.

    ``ids``: mảng (N, n_codebook) mã thô 0..codebook_size-1, hàng i ↔ item_id i + 1.
    """
    _check_name(name)
    proc_dir = Path(proc_dir)
    items = load_id2item(proc_dir)
    arr = validate_semantic_ids(ids, len(items), n_codebook, codebook_size)
    out = sem_ids_path(proc_dir, name, n_codebook, codebook_size)
    _check_overwrite(out, overwrite)

    payload = {asin: [int(v) for v in row] for asin, row in zip(items, arr.tolist())}
    text = json.dumps(payload)
    out.write_text(text, encoding="utf-8")
    _write_sidecar(proc_dir, name, {
        "mode": "semantic_ids", "file": out.name, "shape": list(arr.shape),
        "n_codebook": n_codebook, "codebook_size": codebook_size, "source": source,
        "sha256": _sha256(text.encode("utf-8")), **collision_stats(arr),
    })
    return out


def read_semantic_ids(proc_dir: str | Path, name: str, *, n_codebook: int = N_CODEBOOK,
                      codebook_size: int = CODEBOOK_SIZE) -> np.ndarray:
    """Đọc file ``.sem_ids`` (của code gốc hoặc do adapter ghi) thành mảng (N, n_codebook) theo item_id."""
    proc_dir = Path(proc_dir)
    items = load_id2item(proc_dir)
    path = sem_ids_path(proc_dir, name, n_codebook, codebook_size)
    data = json.loads(path.read_text(encoding="utf-8"))
    missing = [a for a in items if a not in data]
    if missing:
        raise ValueError(f"{path.name} thiếu {len(missing)} item, ví dụ {missing[:3]}")
    extra = set(data) - set(items)
    if extra:
        raise ValueError(f"{path.name} có {len(extra)} asin không nằm trong id_mapping, ví dụ {sorted(extra)[:3]}")
    arr = np.array([data[a] for a in items], dtype=np.int64)
    return validate_semantic_ids(arr, len(items), n_codebook, codebook_size)


# --- Chế độ (a): thay file embedding để code gốc tự PCA + OPQ ---

def write_embeddings(emb, proc_dir: str | Path, name: str, *, overwrite: bool = False,
                     n_codebook: int = N_CODEBOOK, codebook_size: int = CODEBOOK_SIZE,
                     source: str | None = None) -> Path:
    """Ghi embedding của nhóm thành ``<name>.sent_emb``. Trả về đường dẫn file.

    Nếu đã có file ``.sem_ids`` cùng tên (từ lần chạy trước), code gốc sẽ bỏ qua PCA và OPQ;
    vì vậy file đó bị xóa khi ``overwrite=True``, còn không thì báo lỗi.
    Khi chạy code gốc phải truyền ``sent_emb_dim`` bằng số cột của ``emb`` (xem ``run_args``).
    """
    _check_name(name)
    proc_dir = Path(proc_dir)
    items = load_id2item(proc_dir)
    arr = validate_embeddings(emb, len(items))
    out = sent_emb_path(proc_dir, name)
    stale = sem_ids_path(proc_dir, name, n_codebook, codebook_size)
    _check_overwrite(out, overwrite)
    if stale.exists():
        if not overwrite:
            raise FileExistsError(f"{stale} đã tồn tại nên code gốc sẽ không chạy lại OPQ; "
                                  "truyền overwrite=True để xóa")
        stale.unlink()

    data = arr.tobytes(order="C")
    out.write_bytes(data)
    _write_sidecar(proc_dir, name, {
        "mode": "embeddings", "file": out.name, "shape": list(arr.shape), "dtype": "float32",
        "sent_emb_dim": int(arr.shape[1]), "source": source, "sha256": _sha256(data),
    })
    return out


def read_embeddings(proc_dir: str | Path, name: str, dim: int) -> np.ndarray:
    """Đọc ``<name>.sent_emb`` giống code gốc (tokenizer.py dòng 273): float32, reshape(-1, dim)."""
    proc_dir = Path(proc_dir)
    path = sent_emb_path(proc_dir, name)
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size % dim:
        raise ValueError(f"{path.name} có {raw.size} số, không chia hết cho dim={dim}")
    arr = raw.reshape(-1, dim)
    n_items = len(load_id2item(proc_dir))
    if arr.shape[0] != n_items:
        raise ValueError(f"{path.name} có {arr.shape[0]} hàng với dim={dim}, cần {n_items}")
    return arr


# --- Kiểm tra trước khi chạy code gốc ---

def read_sidecar(proc_dir: str | Path, name: str) -> dict | None:
    """Nội dung ``<name>.adapter.json`` nếu có."""
    path = sidecar_path(proc_dir, name)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def resolve_run(proc_dir: str | Path, name: str, sent_emb_dim: int | None = None, *,
                n_codebook: int = N_CODEBOOK, codebook_size: int = CODEBOOK_SIZE) -> dict:
    """Xác định code gốc sẽ dùng file nào cho ``sent_emb_model = x/<name>`` và kiểm tra trước khi train.

    - Có ``.sem_ids``: chế độ (b), code gốc nạp thẳng ID; kiểm tra file đủ item và mã hợp lệ.
    - Chỉ có ``.sent_emb``: chế độ (a). ``sent_emb_dim`` lấy từ tham số hoặc từ ``.adapter.json``;
      báo lỗi nếu hai nguồn khác nhau hoặc kích thước file khác ``số item × dim × 4`` byte. Code gốc
      đọc bằng ``reshape(-1, dim)`` (tokenizer.py dòng 273) nên dim sai mà chia hết vẫn chạy được, ra
      embedding sai mà không báo lỗi.
    - Không có file nào: báo lỗi ngay, thay vì để code gốc thử tải model tên ``x/<name>``.

    Trả về ``{"mode": "semantic_ids" | "embeddings", "sent_emb_dim": int | None, "sidecar": dict | None}``.
    """
    _check_name(name, allow_reserved=True)
    proc_dir = Path(proc_dir)
    sidecar = read_sidecar(proc_dir, name)
    if sem_ids_path(proc_dir, name, n_codebook, codebook_size).exists():
        read_semantic_ids(proc_dir, name, n_codebook=n_codebook, codebook_size=codebook_size)
        return {"mode": "semantic_ids", "sent_emb_dim": sent_emb_dim, "sidecar": sidecar}

    emb = sent_emb_path(proc_dir, name)
    if not emb.exists():
        raise FileNotFoundError(
            f"Không có {sem_ids_path(proc_dir, name, n_codebook, codebook_size).name} hay {emb.name} trong "
            f"{proc_dir}; ghi trước bằng python -m adapters.load_ids ids|emb --name {name} ...")
    recorded = sidecar.get("sent_emb_dim") if sidecar else None
    if sent_emb_dim is not None and recorded is not None and int(sent_emb_dim) != int(recorded):
        raise ValueError(f"--sent_emb_dim={sent_emb_dim} khác {recorded} ghi trong {sidecar_path(proc_dir, name).name}")
    dim = sent_emb_dim if sent_emb_dim is not None else recorded
    if dim is None:
        raise ValueError(f"Không biết số chiều của {emb.name}: truyền --sent_emb_dim hoặc ghi file bằng adapter")
    n_items = len(load_id2item(proc_dir))
    expected = n_items * int(dim) * 4
    if emb.stat().st_size != expected:
        raise ValueError(f"{emb.name} có {emb.stat().st_size} byte, cần {n_items} item x {dim} chiều x 4 = {expected}")
    return {"mode": "embeddings", "sent_emb_dim": int(dim), "sidecar": sidecar}


# --- CLI ---

def _load_array(path: str) -> np.ndarray:
    return np.load(path, allow_pickle=False)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m adapters.load_ids", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    def common(p, need_name=True):
        p.add_argument("--processed-dir", required=True, help="thư mục processed trong cache của code gốc")
        if need_name:
            p.add_argument("--name", required=True, help="tên dùng trong sent_emb_model = x/<tên>")

    p_ids = sub.add_parser("ids", help="chế độ (b): ghi semantic ID (.npy, shape (N, 32), mã 0..255)")
    common(p_ids)
    p_ids.add_argument("--npy", required=True)
    p_ids.add_argument("--overwrite", action="store_true")

    p_emb = sub.add_parser("emb", help="chế độ (a): ghi embedding (.npy, shape (N, dim))")
    common(p_emb)
    p_emb.add_argument("--npy", required=True)
    p_emb.add_argument("--overwrite", action="store_true")

    p_exp = sub.add_parser("export", help="đọc file .sem_ids (vd. của code gốc) ra .npy theo item_id")
    common(p_exp)
    p_exp.add_argument("--out", required=True)

    args = parser.parse_args(argv)
    if args.cmd == "ids":
        arr = _load_array(args.npy)
        out = write_semantic_ids(arr, args.processed_dir, args.name, overwrite=args.overwrite,
                                 source=str(args.npy))
        stats = collision_stats(arr)
        print(f"Đã ghi {out}")
        print(f"  {stats['n_items']:,} item | trùng ID: {stats['n_colliding_items']} item")
        print(f"  Chạy code gốc với: {run_args(args.name)}")
    elif args.cmd == "emb":
        arr = _load_array(args.npy)
        out = write_embeddings(arr, args.processed_dir, args.name, overwrite=args.overwrite,
                               source=str(args.npy))
        print(f"Đã ghi {out}  shape={arr.shape}")
        print(f"  Chạy code gốc với: {run_args(args.name, arr.shape[1])}")
    elif args.cmd == "export":
        arr = read_semantic_ids(args.processed_dir, args.name)
        np.save(args.out, arr.astype(np.int16))
        print(f"Đã xuất {args.out}  shape={arr.shape}  trùng ID: {collision_stats(arr)['n_colliding_items']} item")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
