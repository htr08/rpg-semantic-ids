"""Embedding đồng xuất hiện từ phần train: item2vec (gensim) và SVD trên ma trận PPMI.

Phụ trách: C (RQ2, T7 21/11). Đầu ra bàn giao cho A để ghép với embedding văn bản theo trọng số α rồi chạy OPQ
(kế hoạch mục 3.5, nâng cấp chính; việc ghép nằm ở phía A).

Quy ước:
- Giao dịch là tập item duy nhất trong phần train của mỗi user (patterns.load_train_transactions), cùng giao dịch với
  FP-Growth; validation và test không được đọc để huấn luyện.
- Hàng j của ma trận ứng với item_id j + 1 (docs/data_contract.md). Mỗi hàng được chuẩn hóa L2. Item không có vector
  (không xuất hiện trong train, hoặc không có cặp đồng xuất hiện với PMI dương ở SVD) là hàng toàn 0, không phải giá trị
  suy đoán; bên ghép cần xử lý các hàng này (số lượng ghi trong báo cáo).
- item2vec: skip-gram với negative sampling, cửa sổ phủ cả giao dịch (không co cửa sổ ngẫu nhiên) vì giao dịch là tập, thứ tự
  trong giao dịch không mang nghĩa (Barkan & Koenigstein, 2016).
- SVD: ma trận đồng xuất hiện C = XᵀX (X là user × item nhị phân, bỏ đường chéo) được đổi sang PPMI rồi phân rã SVD cắt cụt;
  embedding = U·√Σ. Dùng PPMI thay cho số đếm thô vì số đếm thô bị chi phối bởi item phổ biến.
- Kiểm tra chất lượng: xếp hạng theo cosine giữa vector item và trung bình vector lịch sử, chấm bằng src/evaluate.py trên
  validation, loại item đã xem (đối chiếu có lọc), so với Popularity cùng cách lọc. Đây là phép thử "có tín hiệu hay không",
  không phải baseline của dự án.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

from src.clustering import l2_normalize
from src.evaluate import evaluate, load_split
from src.patterns import DEFAULT_INTERACTIONS, DEFAULT_SPLITS, build_transaction_matrix, load_train_transactions

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
DEFAULT_VALIDATION = PROCESSED / "validation.jsonl"
DEFAULT_ITEM2VEC = PROCESSED / "item_emb_cooc_item2vec.npy"
DEFAULT_SVD = PROCESSED / "item_emb_cooc_svd.npy"
DEFAULT_OUTPUT = ROOT / "reports" / "cooccurrence_embeddings.json"
DEFAULT_MARKDOWN = ROOT / "reports" / "cooccurrence_embeddings.md"
K = 10


def item2vec_embeddings(transactions, n_items, dim=128, epochs=100, negative=5, seed=42):
    """Ma trận (n_items, dim) float32 chưa chuẩn hóa; hàng của item không có trong giao dịch là 0.

    workers = 1 và seed cố định để cùng đầu vào, cùng phiên bản gensim cho cùng kết quả (đã kiểm không phụ thuộc
    PYTHONHASHSEED với gensim 4.3.3).
    """
    from gensim.models import Word2Vec

    sentences = [[str(int(i)) for i in t] for t in transactions if len(t) > 0]
    if not sentences:
        raise ValueError("không có giao dịch nào để huấn luyện item2vec")
    window = max(len(s) for s in sentences)
    model = Word2Vec(sentences, vector_size=dim, window=window, min_count=1, sg=1, hs=0, negative=negative, epochs=epochs,
                     shrink_windows=False, workers=1, seed=seed)
    matrix = np.zeros((n_items, dim), dtype=np.float32)
    for token in model.wv.index_to_key:
        item_id = int(token)
        if not 1 <= item_id <= n_items:
            raise ValueError(f"item_id {item_id} nằm ngoài 1..{n_items}")
        matrix[item_id - 1] = model.wv[token]
    return matrix


def cooccurrence_matrix(user_item):
    """C = XᵀX với đường chéo bằng 0: số user đã mua cả hai item; ma trận thưa (n_items, n_items) kiểu int32."""
    x = user_item.astype(np.int32)
    c = (x.T @ x).tolil()
    c.setdiag(0)
    c = c.tocsr()
    c.eliminate_zeros()
    return c


def ppmi_matrix(cooccurrence):
    """PPMI = max(0, log(C_ij · S / (r_i · r_j))) với r_i là tổng hàng i và S = Σ r_i; ô không dương bị loại."""
    c = cooccurrence.tocoo()
    if c.nnz == 0:
        return sparse.csr_matrix(c.shape, dtype=np.float64)
    row_sum = np.asarray(cooccurrence.sum(axis=1)).ravel().astype(np.float64)
    total = row_sum.sum()
    pmi = np.log(c.data * total / (row_sum[c.row] * row_sum[c.col]))
    keep = pmi > 0
    return sparse.csr_matrix((pmi[keep], (c.row[keep], c.col[keep])), shape=c.shape)


def svd_embeddings(user_item, dim=128, seed=42):
    """SVD cắt cụt của PPMI; trả về (embedding chưa chuẩn hóa (n_items, dim) float32, tổng phương sai giải thích)."""
    ppmi = ppmi_matrix(cooccurrence_matrix(user_item))
    if ppmi.nnz == 0:
        raise ValueError("ma trận PPMI rỗng: không có cặp item đồng xuất hiện với PMI dương")
    if not 1 <= dim < min(ppmi.shape):
        raise ValueError(f"dim phải trong 1..{min(ppmi.shape) - 1}, nhận {dim}")
    svd = TruncatedSVD(n_components=dim, algorithm="randomized", n_iter=7, random_state=seed)
    projected = svd.fit_transform(ppmi)  # U·Σ
    embedding = projected / np.sqrt(np.maximum(svd.singular_values_, 1e-12))  # U·√Σ
    return embedding.astype(np.float32), float(svd.explained_variance_ratio_.sum())


def finalize(matrix):
    """Chuẩn hóa L2 từng hàng, giữ hàng toàn 0 là 0; trả về (ma trận float32, số hàng toàn 0)."""
    normalized = l2_normalize(matrix.astype(np.float64)).astype(np.float32)
    return normalized, int((np.linalg.norm(normalized, axis=1) == 0).sum())


def centroid_rankings(embeddings, histories, seen, k=K, batch=2000):
    """Top-k item theo cosine với trung bình vector lịch sử (item đã xem bị loại); trả về list các list item_id.

    embeddings đã chuẩn hóa L2. Lịch sử có tâm bằng 0 (mọi item không có vector) cho ra danh sách rỗng.
    """
    n_items = embeddings.shape[0]
    k = min(k, n_items)
    rankings = []
    for start in range(0, len(histories), batch):
        chunk = range(start, min(start + batch, len(histories)))
        centroid = np.stack([embeddings[np.asarray(histories[u], dtype=np.int64) - 1].mean(axis=0) for u in chunk])
        scores = centroid @ embeddings.T
        for row, u in enumerate(chunk):
            scores[row, np.asarray(list(seen[u]), dtype=np.int64) - 1] = -np.inf
        top = np.argpartition(-scores, k - 1, axis=1)[:, :k]
        for row in range(len(chunk)):
            order = top[row][np.argsort(-scores[row, top[row]], kind="stable")]
            rankings.append([int(i) + 1 for i in order if np.isfinite(scores[row, i])] if np.any(centroid[row]) else [])
    return rankings


def popularity_ranking(train_counts, seen_items, k=K):
    """Top-k item phổ biến nhất trong train, bỏ item đã xem; hòa số đếm thì item_id nhỏ hơn đứng trước."""
    order = np.lexsort((np.arange(len(train_counts)), -np.asarray(train_counts)))
    seen = set(seen_items)
    out = []
    for j in order:
        if int(j) + 1 not in seen:
            out.append(int(j) + 1)
            if len(out) == k:
                break
    return out


def load_validation_rows(path):
    """(user_id, lịch sử, tập seen_item_ids) theo thứ tự file; lịch sử tối đa 50 item gần nhất."""
    users, histories, seen = [], [], []
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                users.append(int(row["user_id"]))
                histories.append(row["history"])
                seen.append(set(row["seen_item_ids"]))
    return users, histories, seen


def quality_check(embeddings, validation_path, k=K):
    """Recall@k và NDCG@k trên validation của xếp hạng theo trung bình vector lịch sử (item đã xem bị loại)."""
    users, histories, seen = load_validation_rows(validation_path)
    split = load_split(validation_path)
    rankings = dict(zip(users, centroid_rankings(embeddings, histories, seen, k)))
    scores = evaluate(rankings, split, ks=(k,))
    return {f"recall@{k}": float(scores[f"recall@{k}"]), f"ndcg@{k}": float(scores[f"ndcg@{k}"])}


def popularity_check(validation_path, train_counts, k=K):
    users, _, seen = load_validation_rows(validation_path)
    split = load_split(validation_path)
    rankings = {u: popularity_ranking(train_counts, s, k) for u, s in zip(users, seen)}
    scores = evaluate(rankings, split, ks=(k,))
    return {f"recall@{k}": float(scores[f"recall@{k}"]), f"ndcg@{k}": float(scores[f"ndcg@{k}"])}


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def format_markdown(payload):
    p = payload
    lines = [
        "# Embedding đồng xuất hiện từ phần train",
        "",
        f"{p['n_transactions']:,} giao dịch train, {p['n_items']:,} item, {p['dim']} chiều. Sinh bởi `python -m src.cooccurrence`.",
        "",
        "- Giao dịch là tập item duy nhất trong phần train của mỗi user; validation và test không dùng để huấn luyện.",
        "- SHA-256 của file SVD chỉ tái lập khi cùng số luồng BLAS: khác số luồng làm các phần tử lệch cỡ 1e-11 (đã đo trên một mẫu 8.000 giao dịch). "
        "File item2vec tái lập theo seed với `workers = 1`.",
        "- Hàng j ứng với `item_id` j + 1; mỗi hàng chuẩn hóa L2; float32. Hàng toàn 0 nghĩa là item không có vector (không phải giá trị suy đoán).",
        f"- item2vec: skip-gram, negative = {p['item2vec']['negative']}, {p['item2vec']['epochs']} epoch, cửa sổ phủ cả giao dịch "
        f"(window = {p['item2vec']['window']}), min_count = 1, seed {p['seed']}.",
        f"- SVD: PPMI của XᵀX (bỏ đường chéo), SVD cắt cụt {p['dim']} chiều, embedding U·√Σ; "
        f"phương sai giải thích của ma trận PPMI {p['svd']['explained_variance']:.3f}.",
        "",
        "| Embedding | File | Hàng toàn 0 | SHA-256 |",
        "|---|---|---:|---|",
    ]
    for name in ("item2vec", "svd"):
        e = p[name]
        lines.append(f"| {name} | `{e['file']}` | {e['n_zero_rows']} | `{e['sha256']}` |")
    lines += [
        "",
        f"Kiểm tra tín hiệu trên {p['n_validation_users']:,} user validation: xếp hạng theo cosine giữa vector item và trung bình vector "
        f"lịch sử, loại item đã xem, bộ chấm `src/evaluate.py`. Popularity cũng loại item đã xem. Đây chỉ là phép thử có tín hiệu hay không; "
        "số liệu không so sánh trực tiếp với bảng kết quả chính (bảng chính giữ item đã xem).",
        "",
        f"| Phương pháp | Recall@{K} | NDCG@{K} |",
        "|---|---:|---:|",
        f"| Popularity (train) | {p['quality']['popularity']['recall@10']:.4f} | {p['quality']['popularity']['ndcg@10']:.4f} |",
        f"| item2vec | {p['quality']['item2vec']['recall@10']:.4f} | {p['quality']['item2vec']['ndcg@10']:.4f} |",
        f"| SVD (PPMI) | {p['quality']['svd']['recall@10']:.4f} | {p['quality']['svd']['ndcg@10']:.4f} |",
    ]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Embedding đồng xuất hiện (item2vec, SVD) từ phần train (T7 21/11).")
    parser.add_argument("--interactions", type=Path, default=DEFAULT_INTERACTIONS)
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--validation", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--negative", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--item2vec-out", type=Path, default=DEFAULT_ITEM2VEC)
    parser.add_argument("--svd-out", type=Path, default=DEFAULT_SVD)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args(argv)
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    try:
        n_items = int(pd.read_csv(args.interactions, usecols=["item_id"])["item_id"].max())
        _, transactions = load_train_transactions(args.interactions, args.splits)
        if not 1 <= args.dim < n_items:
            raise ValueError(f"dim phải trong 1..{n_items - 1}, nhận {args.dim}")
        user_item = build_transaction_matrix(transactions, n_items)
        train_counts = np.asarray(user_item.sum(axis=0)).ravel()
        v2v, v2v_zero = finalize(item2vec_embeddings(transactions, n_items, args.dim, args.epochs, args.negative, args.seed))
        svd_raw, variance = svd_embeddings(user_item, args.dim, args.seed)
        svd, svd_zero = finalize(svd_raw)
        quality = {
            "popularity": popularity_check(args.validation, train_counts),
            "item2vec": quality_check(v2v, args.validation),
            "svd": quality_check(svd, args.validation),
        }
        for path, matrix in ((args.item2vec_out, v2v), (args.svd_out, svd)):
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, matrix)
        payload = {
            "n_transactions": len(transactions), "n_items": n_items, "dim": int(args.dim), "seed": int(args.seed),
            "n_validation_users": len(load_split(args.validation)),
            "n_items_without_train": int((train_counts == 0).sum()),
            "item2vec": {"file": args.item2vec_out.name, "sha256": file_sha256(args.item2vec_out), "n_zero_rows": v2v_zero,
                         "epochs": int(args.epochs), "negative": int(args.negative),
                         "window": int(max(len(t) for t in transactions))},
            "svd": {"file": args.svd_out.name, "sha256": file_sha256(args.svd_out), "n_zero_rows": svd_zero,
                    "explained_variance": variance},
            "quality": quality,
        }
        for path in (args.output, args.markdown):
            path.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        args.markdown.write_text(format_markdown(payload), encoding="utf-8")
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"Error: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(f"Wrote {args.item2vec_out}, {args.svd_out}, {args.output} and {args.markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
