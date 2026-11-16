"""Kiểm tra dựng giao dịch từ file chỉ số chia và FP-Growth trên ví dụ tính tay."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.patterns import (
    build_transaction_matrix,
    load_train_transactions,
    run_fpgrowth,
    summarize_itemsets,
)


def write_fixture(root):
    """Hai user: u1 có 5 tương tác (train 1,2,2 | val 3 | test 4), u2 có 4 (train 1,3 | val 2 | test 4)."""
    interactions = [
        (0, 1, 1), (1, 1, 2), (2, 1, 2), (3, 1, 3), (4, 1, 4),
        (5, 2, 1), (6, 2, 3), (7, 2, 2), (8, 2, 4),
    ]
    lines = ["interaction_id,user_id,item_id,timestamp,source_line"]
    lines += [f"{i},{u},{item},0,{i}" for i, u, item in interactions]
    (root / "interactions.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (root / "splits.csv").write_text(
        "user_id,train_start,train_stop,validation_index,test_index\n"
        "1,0,3,3,4\n"
        "2,5,7,7,8\n",
        encoding="utf-8",
    )
    return root / "interactions.csv", root / "splits.csv"


class PatternsTests(unittest.TestCase):
    def test_transactions_use_train_only_and_unique_items(self):
        with tempfile.TemporaryDirectory() as directory:
            interactions, splits = write_fixture(Path(directory))
            users, transactions = load_train_transactions(interactions, splits)
        self.assertEqual(users, [1, 2])
        self.assertEqual([t.tolist() for t in transactions], [[1, 2], [1, 3]])
        # Item validation/test (3 và 4 của u1; 2 và 4 của u2) không lọt vào giao dịch tương ứng.
        self.assertNotIn(3, transactions[0])
        self.assertNotIn(4, transactions[0])
        self.assertNotIn(2, transactions[1])

    def test_invalid_split_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            interactions, splits = write_fixture(Path(directory))
            splits.write_text(
                "user_id,train_start,train_stop,validation_index,test_index\n1,0,3,4,5\n", encoding="utf-8"
            )
            with self.assertRaises(ValueError):
                load_train_transactions(interactions, splits)

    def test_matrix_shape_and_columns(self):
        matrix = build_transaction_matrix([np.array([1, 2]), np.array([1, 3]), np.array([], dtype=int)], n_items=4)
        self.assertEqual(matrix.shape, (3, 4))
        self.assertEqual(matrix.toarray().astype(int).tolist(), [[1, 1, 0, 0], [1, 0, 1, 0], [0, 0, 0, 0]])
        with self.assertRaises(ValueError):
            build_transaction_matrix([np.array([5])], n_items=4)

    def test_fpgrowth_matches_hand_count(self):
        # 4 giao dịch: {1,2}, {1,2}, {1,3}, {2}. support: 1 = 3/4, 2 = 3/4, 3 = 1/4, {1,2} = 2/4.
        transactions = [np.array([1, 2]), np.array([1, 2]), np.array([1, 3]), np.array([2])]
        matrix = build_transaction_matrix(transactions, n_items=3)
        itemsets, elapsed = run_fpgrowth(matrix, min_support=0.5)
        found = {frozenset(row.itemsets): row.support for row in itemsets.itertuples()}
        self.assertEqual(found, {frozenset({1}): 0.75, frozenset({2}): 0.75, frozenset({1, 2}): 0.5})
        self.assertEqual(summarize_itemsets(itemsets), {"1": 2, "2": 1})
        self.assertGreaterEqual(elapsed, 0.0)


if __name__ == "__main__":
    unittest.main()
