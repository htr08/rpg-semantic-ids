"""Nâng cấp chính: semantic ID có tín hiệu hành vi (thử nhanh, chưa train; rồi tạo bộ ID để train).

Theo kế hoạch (mục 3.3): ghép embedding văn bản (sentence-t5, 768 chiều) với embedding đồng xuất hiện của C
(item2vec hoặc SVD trên PPMI, 128 chiều, ``src/cooccurrence.py``) theo trọng số α rồi mới chạy OPQ; mô hình giữ nguyên,
nạp qua adapter (chế độ b).

Cách ghép: chuẩn hóa L2 từng phần rồi nối ``[√(1 − α)·văn bản, √α·đồng xuất hiện]``, nên tích vô hướng của hai item
= (1 − α)·cos văn bản + α·cos đồng xuất hiện. Sau đó đi đúng đường của code gốc: PCA 128 có whiten (seed 42) → FAISS
``OPQ32,IVF1,PQ32x8``, tâm cụm học trên item của phần train. Lưu ý: whiten đưa mọi thành phần chính về cùng phương sai,
nên α tác động qua việc chọn không gian con 128 chiều được giữ lại chứ không qua độ lớn; vì vậy cần đo, không suy luận.
33 item không có trong train có vector đồng xuất hiện bằng 0 (C ghi rõ), nên chỉ còn phần văn bản.

Cách thứ hai (``--mode split``, thêm sau lần thử 27/11 cho thấy cách ghép theo α làm mất ngữ nghĩa vì whiten đưa mọi
hướng về cùng phương sai): **chia codebook**, b trong 32 vị trí mã dành cho hành vi, (32 − b) cho văn bản, mỗi phần PCA
có whiten riêng rồi PQ, nên tỉ lệ văn bản / hành vi đúng bằng số vị trí mã. Mốc so sánh là ID PQ của RQ1 (b = 0).

Thử nhanh đo, cho mỗi cấu hình:

- độ đo cụm như RQ1 (``src/cluster_metrics.evaluate``): ID còn giữ ngữ nghĩa danh mục, thương hiệu không;
- độ khớp hành vi: số mã trùng trung bình giữa hai item mua liền nhau trong phần train, so với cặp ngẫu nhiên;
- NDCG@10 proxy trên validation: xếp hạng theo số mã trùng với item cuối của lịch sử (hòa thì theo độ phổ biến trong
  train). Không phải RPG, chỉ để biết ID mới có giúp đoán item kế tiếp hơn không, trước khi tốn GPU; chênh lệch so với
  mốc tính ghép cặp theo user, kèm sai số chuẩn và z.

Dòng lệnh::

    python -m src.behavior_ids trial --processed-dir <cache>/processed --items <items.jsonl> \\
        --cooc item2vec=item_emb_cooc_item2vec.npy svd=item_emb_cooc_svd.npy --alphas 0 0.25 0.5 --out-dir <thư mục>
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from src import cluster_metrics as M
from src import tokenizer as T

DEFAULT_ALPHAS = (0.0, 0.25, 0.5)
N_PROXY_USERS = 5000
N_RANDOM_PAIRS = 200_000


def _utf8_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError, OSError):
        pass


# --- Dữ liệu ---

def load_cooc(path, n_items: int) -> np.ndarray:
    """Đọc embedding đồng xuất hiện (N, d) float32 của C; kiểm số hàng và giá trị hữu hạn."""
    arr = np.load(path, allow_pickle=False)
    if arr.ndim != 2 or arr.shape[0] != n_items:
        raise ValueError(f"{path}: cần (N={n_items}, d), nhận {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{path}: có giá trị không hữu hạn")
    return arr.astype(np.float32)


def train_sequences(proc_dir) -> tuple[list[list[int]], int]:
    """Chuỗi item (chỉ số 0-based theo ``id_mapping.json``) của từng user và số item."""
    proc_dir = Path(proc_dir)
    mapping = json.loads((proc_dir / "id_mapping.json").read_text(encoding="utf-8"))
    item2id = mapping["item2id"]
    seqs = json.loads((proc_dir / "all_item_seqs.json").read_text(encoding="utf-8"))
    return [[item2id[a] - 1 for a in s] for s in seqs.values()], len(mapping["id2item"]) - 1


def consecutive_train_pairs(seqs: Sequence[Sequence[int]]) -> np.ndarray:
    """(P, 2) các cặp item liền nhau trong phần train (``seq[:-2]``, user có hơn 2 item), bỏ cặp trùng item."""
    pairs = [(a, b) for s in seqs if len(s) > 2 for a, b in zip(s[:-3], s[1:-2]) if a != b]
    return np.array(pairs, dtype=np.int64).reshape(-1, 2)


def train_popularity(seqs: Sequence[Sequence[int]], n_items: int) -> np.ndarray:
    counts = np.zeros(n_items, dtype=np.int64)
    for s in seqs:
        if len(s) > 2:
            np.add.at(counts, np.asarray(s[:-2]), 1)
    return counts


# --- Ghép và tạo ID ---

def _l2(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return np.divide(x, norm, out=np.zeros_like(x), where=norm > 0)


def combine(text, cooc, alpha: float) -> np.ndarray:
    """``[√(1 − α)·l2(text), √α·l2(cooc)]``; hàng cooc toàn 0 giữ nguyên 0. α = 0 trả về văn bản đã chuẩn hóa."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha phải trong [0, 1], nhận {alpha}")
    text, cooc = np.asarray(text), np.asarray(cooc)
    if text.shape[0] != cooc.shape[0]:
        raise ValueError(f"text có {text.shape[0]} hàng, cooc có {cooc.shape[0]}")
    if alpha == 0.0:
        return _l2(text).astype(np.float32)
    return np.hstack([math.sqrt(1 - alpha) * _l2(text), math.sqrt(alpha) * _l2(cooc)]).astype(np.float32)


def behavior_ids(text, cooc, alpha: float, train_mask, *, pca_dim: int = T.PCA_DIM, seed: int = T.SEED,
                 quantize: Callable | None = None, threads: int | None = None) -> np.ndarray:
    """Semantic ID (N, 32) từ embedding đã ghép: PCA có whiten rồi OPQ như code gốc (``quantize`` để thay khi test)."""
    z = T.pca_whiten(combine(text, cooc, alpha), n_components=pca_dim, seed=seed)
    if quantize is None:
        return T.opq_ids(z, train_mask=train_mask, threads=threads)
    return quantize(z, train_mask)


def split_features(text, cooc, n_behavior_codes: int, *, n_codebook: int = T.N_CODEBOOK, pca_dim: int = T.PCA_DIM,
                   seed: int = T.SEED) -> np.ndarray:
    """Chia codebook: ``n_behavior_codes`` vị trí mã cuối dành cho hành vi, phần còn lại cho văn bản.

    Mỗi vị trí mã ứng với ``pca_dim / n_codebook`` chiều liền nhau (128 / 32 = 4). Văn bản: PCA có whiten xuống
    ``pca_dim − d`` chiều; đồng xuất hiện: PCA có whiten xuống ``d = n_behavior_codes · pca_dim / n_codebook`` chiều; nối
    văn bản trước, hành vi sau. Lượng tử hóa bằng PQ (không xoay) thì mỗi vị trí mã chỉ chứa một loại tín hiệu, và
    tỉ lệ văn bản / hành vi đúng bằng số vị trí mã, không phụ thuộc phương sai như cách ghép theo α.
    """
    if pca_dim % n_codebook:
        raise ValueError(f"pca_dim ({pca_dim}) phải chia hết cho n_codebook ({n_codebook})")
    if not 0 < n_behavior_codes < n_codebook:
        raise ValueError(f"n_behavior_codes phải trong 1..{n_codebook - 1}, nhận {n_behavior_codes}")
    d = n_behavior_codes * (pca_dim // n_codebook)
    text_part = T.pca_whiten(np.asarray(text, dtype=np.float32), n_components=pca_dim - d, seed=seed)
    cooc_part = T.pca_whiten(_l2(cooc).astype(np.float32), n_components=d, seed=seed)
    return np.hstack([text_part, cooc_part]).astype(np.float32)


def split_ids(text, cooc, n_behavior_codes: int, train_mask, *, n_codebook: int = T.N_CODEBOOK,
              codebook_size: int = T.CODEBOOK_SIZE, pca_dim: int = T.PCA_DIM, seed: int = T.SEED,
              quantize: Callable | None = None, threads: int | None = None) -> np.ndarray:
    """Semantic ID lai: (32 − b) mã văn bản + b mã hành vi, PQ của FAISS như biến thể PQ của RQ1."""
    z = split_features(text, cooc, n_behavior_codes, n_codebook=n_codebook, pca_dim=pca_dim, seed=seed)
    if quantize is None:
        return T.pq_ids(z, n_codebook=n_codebook, codebook_size=codebook_size, train_mask=train_mask, threads=threads)
    return quantize(z, train_mask)


# --- Kiểm tra thứ tự hàng của embedding đồng xuất hiện ---

def order_check(cooc, train_mask, pairs, n_random: int = N_RANDOM_PAIRS, seed: int = 0) -> dict:
    """Hai phép thử độc lập với file: hàng 0 ↔ item ngoài train; cặp mua liền nhau giống nhau hơn cặp ngẫu nhiên."""
    cooc = _l2(cooc)
    zero = np.linalg.norm(cooc, axis=1) == 0
    mask = np.asarray(train_mask, dtype=bool)
    rng = np.random.default_rng(seed)
    rand = rng.integers(0, len(cooc), size=(n_random, 2))
    rand = rand[rand[:, 0] != rand[:, 1]]
    cos = lambda p: float((cooc[p[:, 0]] * cooc[p[:, 1]]).sum(1).mean()) if len(p) else float("nan")
    out = {"zero_rows": int(zero.sum()), "items_outside_train": int((~mask).sum()),
           "zero_rows_match_outside_train": bool(np.array_equal(zero, ~mask)),
           "cos_consecutive_pairs": cos(pairs), "cos_random_pairs": cos(rand)}
    out["passed"] = bool(out["zero_rows_match_outside_train"]
                         and out["cos_consecutive_pairs"] > out["cos_random_pairs"] + 0.05)
    return out


# --- Độ đo hành vi ---

def pair_shared_codes(ids, pairs, n_random: int = N_RANDOM_PAIRS, seed: int = 0) -> dict:
    """Số mã trùng trung bình và tỉ lệ có ≥ 1 mã trùng: cặp cho trước so với cặp ngẫu nhiên."""
    arr = np.asarray(ids)
    rng = np.random.default_rng(seed)
    rand = rng.integers(0, len(arr), size=(n_random, 2))
    rand = rand[rand[:, 0] != rand[:, 1]]
    shared = lambda p: (arr[p[:, 0]] == arr[p[:, 1]]).sum(1)
    sp, sr = shared(pairs), shared(rand)
    return {"pair_shared_mean": float(sp.mean()), "random_shared_mean": float(sr.mean()),
            "pair_any_shared": float((sp > 0).mean()), "random_any_shared": float((sr > 0).mean()),
            "pair_lift": float(sp.mean() / sr.mean()) if sr.mean() > 0 else float("nan")}


def last_item_overlap_proxy(ids, seqs: Sequence[Sequence[int]], popularity, n_users: int = N_PROXY_USERS,
                            k: int = 10, seed: int = 0, chunk: int = 256, per_user: bool = False) -> dict:
    """NDCG@k, Recall@k trên validation khi xếp hạng theo số mã trùng với item cuối của lịch sử train.

    Lịch sử = ``seq[:-2]``, đích = ``seq[-2]`` (như leave-one-out của code gốc), chỉ user có hơn 2 item; loại chính
    item truy vấn; hòa điểm thì item phổ biến hơn (trong train) đứng trước, hòa tiếp thì theo thứ tự ngẫu nhiên có seed. Lấy mẫu ``n_users`` user theo ``seed``.
    """
    arr = np.asarray(ids)
    eligible = [s for s in seqs if len(s) > 2]
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(eligible), size=min(n_users, len(eligible)), replace=False)
    queries = np.array([eligible[i][-3] for i in pick])
    targets = np.array([eligible[i][-2] for i in pick])
    pop = np.asarray(popularity, dtype=np.float64)
    # hòa số mã trùng → độ phổ biến; hòa tiếp → thứ tự ngẫu nhiên cố định theo seed (không để hòa, vì đếm "điểm cao hơn
    # đích" khi còn hòa sẽ xếp đích lên đầu nhóm hòa và thổi phồng kết quả)
    tie = (pop + rng.random(len(pop))) / (pop.max() + 1.0)
    user_ndcg = np.zeros(len(queries))
    for start in range(0, len(queries), chunk):
        q, t = queries[start:start + chunk], targets[start:start + chunk]
        score = (arr[q][:, None, :] == arr[None, :, :]).sum(2) + tie[None, :]
        score[np.arange(len(q)), q] = -np.inf
        target_score = score[np.arange(len(q)), t]
        rank = (score > target_score[:, None]).sum(1)
        user_ndcg[start:start + len(q)] = np.where(rank < k, 1.0 / np.log2(rank + 2), 0.0)
    out = {f"proxy_ndcg@{k}": float(user_ndcg.mean()), f"proxy_recall@{k}": float((user_ndcg > 0).mean()),
           "proxy_users": int(len(queries))}
    if per_user:
        out["per_user_ndcg"] = user_ndcg
    return out


def paired_difference(a, b) -> dict:
    """Chênh lệch trung bình a − b trên cùng tập user, kèm sai số chuẩn và z (ghép cặp theo user)."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan")
    return {"diff": float(d.mean()), "se": se, "z": float(d.mean() / se) if se and se > 0 else float("nan")}


# --- Thử nhanh ---

def run_trial(text, coocs: dict, alphas: Sequence[float], proc_dir, labels: dict, out_dir: Path, *,
              mode: str = "blend", behavior_codes: Sequence[int] = (8, 16), baseline: str | None = None,
              quantize: Callable | None = None, threads: int | None = None, extra_ids: dict | None = None,
              n_proxy_users: int = N_PROXY_USERS, pca_dim: int = T.PCA_DIM, n_codebook: int = T.N_CODEBOOK,
              seed: int = 0) -> list[dict]:
    """Tạo ID cho từng cấu hình, đo, ghi ``trial_ids/<tên>.npy`` và trả về bảng kết quả.

    ``mode='blend'``: ghép theo α (α = 0 chỉ tạo một lần, tên ``text``). ``mode='split'``: chia codebook, b mã hành vi
    cho mỗi b trong ``behavior_codes`` (tên ``<emb>_b<b>``). ``extra_ids`` = ID có sẵn để đối chiếu, chỉ đo, không tạo.
    ``baseline`` = tên cấu hình làm mốc cho chênh lệch proxy ghép cặp theo user (mặc định ``text`` khi blend).
    """
    if mode not in ("blend", "split"):
        raise ValueError(f"mode phải là 'blend' hoặc 'split', nhận {mode!r}")
    out_dir = Path(out_dir)
    (out_dir / "trial_ids").mkdir(parents=True, exist_ok=True)
    seqs, n_items = train_sequences(proc_dir)
    mask = T.train_item_mask(proc_dir)
    pairs = consecutive_train_pairs(seqs)
    pop = train_popularity(seqs, n_items)

    configs = []                                            # (tên, embedding, tham số, có sẵn)
    configs += [(n, None, None, ids) for n, ids in (extra_ids or {}).items()]
    if mode == "blend":
        if any(a == 0 for a in alphas):
            configs.append(("text", None, 0.0, None))
        configs += [(f"{e}_a{int(round(a * 100)):02d}", e, a, None) for e in coocs for a in alphas if a > 0]
    else:
        configs += [(f"{e}_b{b}", e, b, None) for e in coocs for b in behavior_codes]
    baseline = baseline or ("text" if mode == "blend" else None)

    rows, per_user = [], {}
    for name, emb, param, given in configs:
        if given is not None:
            ids = np.asarray(given)
        elif mode == "split":
            ids = split_ids(text, coocs[emb], int(param), mask, n_codebook=n_codebook, pca_dim=pca_dim,
                            quantize=quantize, threads=threads)
        else:
            cooc = coocs[emb] if emb else np.zeros((n_items, 1), dtype=np.float32)
            ids = behavior_ids(text, cooc, param, mask, pca_dim=pca_dim, quantize=quantize, threads=threads)
        if given is None:
            np.save(out_dir / "trial_ids" / f"{name}.npy", ids.astype(np.int16))
        summary, _ = M.evaluate(ids, labels, k=10, seed=seed)
        row = {"config": name, "cooc": emb or ("" if given is None else "có sẵn"),
               "alpha": param if mode == "blend" and given is None else "",
               "behavior_codes": param if mode == "split" and given is None else ""}
        row.update({key: summary[key] for key in ("collision_rate", "entropy_min", "leaf_ami_pos_mean",
                                                  "brand_ami_pos_mean", "leaf_neighbor_purity@10",
                                                  "brand_neighbor_purity@10")})
        row.update(pair_shared_codes(ids, pairs, seed=seed))
        proxy = last_item_overlap_proxy(ids, seqs, pop, n_users=n_proxy_users, seed=seed, per_user=True)
        per_user[name] = proxy.pop("per_user_ndcg")
        row.update(proxy)
        rows.append(row)
        print(f"{name}: proxy NDCG@10 {row['proxy_ndcg@10']:.4f} | mã trùng cặp mua liền nhau "
              f"{row['pair_shared_mean']:.3f} (ngẫu nhiên {row['random_shared_mean']:.3f}) | độ thuần láng giềng lá "
              f"{row['leaf_neighbor_purity@10']:.3f} | va chạm {row['collision_rate']:.4%}")
    if baseline in per_user:
        for row in rows:
            d = paired_difference(per_user[row["config"]], per_user[baseline])
            row.update({"proxy_diff_vs_baseline": d["diff"], "proxy_diff_se": d["se"], "proxy_diff_z": d["z"],
                        "baseline": baseline})
    return rows


def recommend(rows: list[dict], baseline: str = "text", max_collision: float = 0.001, min_entropy: float = 0.95,
              min_purity_ratio: float = 0.9, min_z: float | None = None) -> dict:
    """Chọn cấu hình mới (không phải mốc, không phải ID có sẵn) có proxy NDCG@10 cao nhất trong các cấu hình đạt ba
    ngưỡng so với ``baseline``: tỉ lệ va chạm ≤ ``max_collision``, entropy thấp nhất ≥ ``min_entropy``, độ thuần láng
    giềng danh mục lá ≥ ``min_purity_ratio`` × của mốc. Phải cao hơn mốc; ``min_z`` (nếu có) đòi chênh lệch ghép cặp
    đạt z tối thiểu. Không cấu hình nào đạt → ``config`` = None.
    """
    base = next((r for r in rows if r["config"] == baseline), None)
    if base is None:
        raise ValueError(f"Không có cấu hình mốc {baseline!r}")
    ok = [r for r in rows if r["config"] != baseline and r.get("cooc") not in ("", "có sẵn")
          and r["collision_rate"] <= max_collision and r["entropy_min"] >= min_entropy
          and r["leaf_neighbor_purity@10"] >= min_purity_ratio * base["leaf_neighbor_purity@10"]
          and (min_z is None or r.get("proxy_diff_z", float("-inf")) >= min_z)]
    best = max(ok, key=lambda r: r["proxy_ndcg@10"], default=None)
    if best is None or best["proxy_ndcg@10"] <= base["proxy_ndcg@10"]:
        return {"config": None, "baseline": baseline,
                "reason": f"Không cấu hình nào đạt ngưỡng và có proxy NDCG@10 cao hơn {baseline}",
                "baseline_proxy_ndcg@10": base["proxy_ndcg@10"]}
    return {"config": best["config"], "cooc": best["cooc"], "alpha": best.get("alpha", ""),
            "behavior_codes": best.get("behavior_codes", ""), "baseline": baseline,
            "proxy_ndcg@10": best["proxy_ndcg@10"], "baseline_proxy_ndcg@10": base["proxy_ndcg@10"],
            "proxy_gain": best["proxy_ndcg@10"] / base["proxy_ndcg@10"] - 1,
            "proxy_diff_z": best.get("proxy_diff_z"),
            "thresholds": {"max_collision": max_collision, "min_entropy": min_entropy,
                           "min_purity_ratio": min_purity_ratio, "min_z": min_z}}


def _write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    _utf8_stdout()
    parser = argparse.ArgumentParser(prog="python -m src.behavior_ids", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("trial", help="tạo ID cho mỗi (embedding, α) và so sánh, chưa train")
    p.add_argument("--processed-dir", required=True)
    p.add_argument("--items", required=True, help="items.jsonl của pipeline B (nhãn danh mục, thương hiệu)")
    p.add_argument("--cooc", nargs="+", required=True, metavar="TÊN=FILE.npy")
    p.add_argument("--mode", choices=["blend", "split"], default="blend",
                   help="blend: ghép theo α rồi OPQ; split: chia codebook (b mã hành vi) rồi PQ")
    p.add_argument("--alphas", nargs="+", type=float, default=list(DEFAULT_ALPHAS))
    p.add_argument("--behavior-codes", nargs="+", type=int, default=[8, 16], help="số mã hành vi (mode split)")
    p.add_argument("--baseline", default=None, help="cấu hình mốc (mặc định text khi blend; khi split nên là ID PQ có sẵn)")
    p.add_argument("--reference", nargs="*", default=[], metavar="TÊN=FILE.npy",
                   help="ID có sẵn để đối chiếu (ví dụ opq_goc=semantic_ids_opq.npy)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--emb-name", default="sentence-t5-base")
    p.add_argument("--emb-dim", type=int, default=768)
    p.add_argument("--threads", type=int, default=T.FAISS_THREADS)
    p.add_argument("--pca-dim", type=int, default=T.PCA_DIM)
    p.add_argument("--proxy-users", type=int, default=N_PROXY_USERS)
    p.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    if args.mode == "split" and not (args.baseline or args.reference):
        raise SystemExit("mode split cần --reference (ví dụ pq_rq1=semantic_ids_pq.npy) làm mốc")
    proc, out = Path(args.processed_dir), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    text = T.load_embeddings(proc, args.emb_name, args.emb_dim)
    n = text.shape[0]
    coocs = {}
    for spec in args.cooc:
        name, path = spec.split("=", 1)
        coocs[name] = load_cooc(path, n)
    seqs, _ = train_sequences(proc)
    mask = T.train_item_mask(proc)
    pairs = consecutive_train_pairs(seqs)
    checks = {name: order_check(c, mask, pairs, seed=args.seed) for name, c in coocs.items()}
    (out / "order_check.json").write_text(json.dumps(checks, indent=2, ensure_ascii=False), encoding="utf-8")
    for name, c in checks.items():
        print(f"Kiểm tra thứ tự {name}: hàng 0 = item ngoài train: {c['zero_rows_match_outside_train']} | cos cặp mua "
              f"liền nhau {c['cos_consecutive_pairs']:.3f}, ngẫu nhiên {c['cos_random_pairs']:.3f}")
    if not all(c["passed"] for c in checks.values()):
        print("DỪNG: thứ tự hàng của embedding đồng xuất hiện không khớp id_mapping.json, hỏi C trước khi dùng")
        return 2

    labels, _ = M.load_item_labels(args.items)
    reference = {spec.split("=", 1)[0]: np.load(spec.split("=", 1)[1]) for spec in args.reference}
    baseline = args.baseline or ("text" if args.mode == "blend" else next(iter(reference), None))
    if baseline is None:
        raise SystemExit("mode split cần --reference (ví dụ pq_rq1=semantic_ids_pq.npy) làm mốc")
    rows = run_trial(text, coocs, args.alphas, proc, labels, out, mode=args.mode, behavior_codes=args.behavior_codes,
                     baseline=baseline, threads=args.threads, extra_ids=reference, n_proxy_users=args.proxy_users,
                     pca_dim=args.pca_dim, seed=args.seed)
    tag = "" if args.mode == "blend" else "_split"
    _write_csv(out / f"behavior_trial{tag}.csv", rows)
    rec = recommend(rows, baseline=baseline)
    rec["versions"] = T.library_versions()
    (out / f"behavior_recommendation{tag}.json").write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Đề xuất:", json.dumps({k: v for k, v in rec.items() if k != "versions"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
