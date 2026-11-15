"""Khai phá mẫu mua kèm: FP-Growth (mlxtend), PrefixSpan, đối chiếu luật với semantic ID, embedding đồng xuất hiện (item2vec, SVD).

Phụ trách: C (RQ2).

Tuần 1 (T2): dựng tập giao dịch từ file chỉ số chia dữ liệu và chạy FP-Growth thử.
Giao dịch = tập item duy nhất trong phần train của mỗi user (không thứ tự, không có validation/test).
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INTERACTIONS = ROOT / "data" / "processed" / "interactions.csv"
DEFAULT_SPLITS = ROOT / "data" / "splits" / "leave_last_out.csv"
DEFAULT_OUTPUT = ROOT / "reports" / "fpgrowth_trial.json"


def load_train_transactions(interactions_path=DEFAULT_INTERACTIONS, splits_path=DEFAULT_SPLITS):
    """Trả về (user_ids, transactions): mỗi giao dịch là mảng item_id đã sắp xếp, không trùng.

    Train của user là khoảng nửa mở [train_start, train_stop) theo interaction_id
    (chỉ số dòng dữ liệu trong interactions.csv, không tính header). Không đọc phần validation/test.
    """
    interactions = pd.read_csv(interactions_path, usecols=["interaction_id", "item_id"])
    if not np.array_equal(interactions["interaction_id"].to_numpy(), np.arange(len(interactions))):
        raise ValueError("interaction_id phải là 0..n-1 theo thứ tự dòng để dùng làm chỉ số chia")
    items = interactions["item_id"].to_numpy()
    splits = pd.read_csv(splits_path)
    user_ids, transactions = [], []
    for row in splits.itertuples(index=False):
        if not row.train_start <= row.train_stop == row.validation_index:
            raise ValueError(f"user {row.user_id}: chỉ số chia không hợp lệ")
        user_ids.append(int(row.user_id))
        transactions.append(np.unique(items[row.train_start:row.train_stop]))
    return user_ids, transactions


def build_transaction_matrix(transactions, n_items):
    """Ma trận thưa nhị phân (số giao dịch x n_items); cột j ứng với item_id j + 1."""
    lengths = np.fromiter((len(t) for t in transactions), dtype=np.int64, count=len(transactions))
    indptr = np.concatenate(([0], np.cumsum(lengths)))
    if len(transactions) == 0 or indptr[-1] == 0:
        indices = np.empty(0, dtype=np.int64)
    else:
        indices = np.concatenate([np.asarray(t, dtype=np.int64) for t in transactions]) - 1
    if indices.size and (indices.min() < 0 or indices.max() >= n_items):
        raise ValueError("item_id nằm ngoài khoảng 1..n_items")
    data = np.ones(len(indices), dtype=bool)
    return sparse.csr_matrix((data, indices, indptr), shape=(len(transactions), n_items))


def _to_sparse_bool_frame(matrix, names):
    """DataFrame thưa kiểu bool (fill_value False), mỗi cột ứng với một item."""
    dtype = pd.SparseDtype(bool, False)
    columns = matrix.tocsc()
    return pd.DataFrame(
        {name: pd.arrays.SparseArray(columns[:, j].toarray().ravel(), dtype=dtype) for j, name in enumerate(names)}
    )


def run_fpgrowth(matrix, min_support):
    """Chạy mlxtend.fpgrowth trên DataFrame thưa; trả về (tập phổ biến, giây).

    itemsets là frozenset các item_id (cột + 1). Chỉ đo thời gian khai phá, không tính dựng DataFrame.
    Pandas yêu cầu tên cột của DataFrame thưa là chuỗi hoặc bắt đầu từ 0, nên dùng tên chuỗi rồi đổi lại thành số.
    Cột dùng SparseDtype(bool, False): mlxtend khuyến nghị kiểu bool, còn from_spmatrix với ma trận bool
    (fill_value 0) bị pandas cảnh báo FutureWarning.
    """
    from mlxtend.frequent_patterns import fpgrowth

    names = [str(j) for j in range(1, matrix.shape[1] + 1)]
    frame = _to_sparse_bool_frame(matrix, names)
    start = time.perf_counter()
    itemsets = fpgrowth(frame, min_support=min_support, use_colnames=True)
    elapsed = time.perf_counter() - start
    itemsets["itemsets"] = itemsets["itemsets"].map(lambda s: frozenset(int(name) for name in s))
    return itemsets, elapsed


def summarize_itemsets(itemsets):
    """Số tập phổ biến theo kích thước."""
    sizes = itemsets["itemsets"].map(len)
    return {str(int(size)): int(count) for size, count in sizes.value_counts().sort_index().items()}


def main(argv=None):
    parser = argparse.ArgumentParser(description="FP-Growth thử trên tập giao dịch train (T2 tuần 1).")
    parser.add_argument("--interactions", type=Path, default=DEFAULT_INTERACTIONS)
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--min-support", type=float, default=0.002)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    try:
        n_items = int(pd.read_csv(args.interactions, usecols=["item_id"])["item_id"].max())
        _, transactions = load_train_transactions(args.interactions, args.splits)
        matrix = build_transaction_matrix(transactions, n_items)
        itemsets, elapsed = run_fpgrowth(matrix, args.min_support)
    except (OSError, ValueError, KeyError, csv.Error) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    sizes = matrix.sum(axis=1).A1
    result = {
        "min_support": args.min_support,
        "min_count": int(np.ceil(args.min_support * matrix.shape[0])),
        "n_transactions": int(matrix.shape[0]),
        "n_items": int(matrix.shape[1]),
        "n_nonzero": int(matrix.nnz),
        "mean_transaction_size": float(sizes.mean()),
        "n_frequent_itemsets": int(len(itemsets)),
        "itemsets_by_size": summarize_itemsets(itemsets),
        "fpgrowth_seconds": round(elapsed, 3),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
