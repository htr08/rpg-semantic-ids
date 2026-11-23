"""Kiểm tra dựng giao dịch từ file chỉ số chia và FP-Growth trên ví dụ tính tay."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.patterns import (
    build_transaction_matrix,
    describe_rule,
    generate_rules,
    load_item_metadata,
    load_train_transactions,
    pair_rules_only,
    run_fpgrowth,
    summarize_itemsets,
    summarize_rules,
    top_rules_by_lift,
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


def rule_fixture():
    """4 giao dịch {1,2,3}, {1,2,3}, {1,2}, {3,4}; min_support 0,5 giữ các tập có số đếm >= 2.

    Số đếm: 1 = 3, 2 = 3, 3 = 3, {1,2} = 3, {1,3} = 2, {2,3} = 2, {1,2,3} = 2; item 4 chỉ có 1 giao dịch nên bị loại.
    """
    transactions = [np.array([1, 2, 3]), np.array([1, 2, 3]), np.array([1, 2]), np.array([3, 4])]
    matrix = build_transaction_matrix(transactions, n_items=4)
    itemsets, _ = run_fpgrowth(matrix, min_support=0.5)
    return itemsets, len(transactions)


def find_rule(rules, antecedents, consequents):
    mask = [a == frozenset(antecedents) and c == frozenset(consequents)
            for a, c in zip(rules["antecedents"], rules["consequents"])]
    found = rules[mask]
    assert len(found) == 1, (antecedents, consequents, len(found))
    return found.iloc[0]


class RuleTests(unittest.TestCase):
    def test_rule_metrics_match_hand_count(self):
        itemsets, n = rule_fixture()
        rules = generate_rules(itemsets, n, min_confidence=0.1)
        # {1} => {2}: conf = 3/3 = 1, lift = 1 / (3/4) = 4/3, kulczynski = (1 + 1) / 2.
        rule = find_rule(rules, {1}, {2})
        self.assertEqual((rule["count"], rule["antecedent_count"], rule["consequent_count"]), (3, 3, 3))
        self.assertAlmostEqual(rule["confidence"], 1.0)
        self.assertAlmostEqual(rule["lift"], 4 / 3)
        self.assertAlmostEqual(rule["kulczynski"], 1.0)
        # {1,3} => {2}: X có 2 giao dịch, Y có 3; conf = 2/2 = 1, lift = 4/3, kulczynski = (1 + 2/3) / 2 = 5/6.
        rule = find_rule(rules, {1, 3}, {2})
        self.assertEqual((rule["count"], rule["antecedent_count"], rule["consequent_count"]), (2, 2, 3))
        self.assertAlmostEqual(rule["confidence"], 1.0)
        self.assertAlmostEqual(rule["lift"], 4 / 3)
        self.assertAlmostEqual(rule["kulczynski"], 5 / 6)
        # {1,2} => {3}: conf = 2/3, lift = (2/3) / (3/4) = 8/9 < 1 (tương quan âm).
        rule = find_rule(rules, {1, 2}, {3})
        self.assertAlmostEqual(rule["confidence"], 2 / 3)
        self.assertAlmostEqual(rule["lift"], 8 / 9)
        # 3 tập 2 item cho 3 * 2 luật, 1 tập 3 item cho 2^3 - 2 luật: tổng 12.
        self.assertEqual(len(rules), 12)

    def test_min_confidence_and_max_rule_len(self):
        itemsets, n = rule_fixture()
        # conf: 1 => 2 và 2 => 1 bằng 1; luật {1,2} => {3} chỉ đạt 2/3 nên bị loại ở ngưỡng 0,9.
        strict = generate_rules(itemsets, n, min_confidence=0.9)
        self.assertTrue((strict["confidence"] >= 0.9).all())
        self.assertIn((frozenset({1}), frozenset({2})), set(zip(strict["antecedents"], strict["consequents"])))
        self.assertNotIn((frozenset({1, 2}), frozenset({3})), set(zip(strict["antecedents"], strict["consequents"])))
        pairs = generate_rules(itemsets, n, min_confidence=0.1, max_rule_len=2)
        self.assertEqual(len(pairs), 6)
        self.assertTrue(all(len(a | c) == 2 for a, c in zip(pairs["antecedents"], pairs["consequents"])))
        # Giới hạn độ dài không làm đổi số đo của luật còn lại.
        self.assertAlmostEqual(find_rule(pairs, {1}, {3})["confidence"], 2 / 3)

    def test_rule_with_confidence_exactly_at_threshold_is_kept(self):
        # 24 giao dịch: item 1 có 10 giao dịch, trong đó đúng 1 giao dịch có thêm item 2; 14 giao dịch còn lại chỉ có item 3.
        # conf(1 => 2) = 1 / 10 = 0,1 đúng bằng ngưỡng. Với support dạng float, (1/24) / (10/24) = 0,0999... nên
        # nếu so sánh trực tiếp thì mlxtend loại luật này.
        transactions = [np.array([1, 2])] + [np.array([1])] * 9 + [np.array([3])] * 14
        matrix = build_transaction_matrix(transactions, n_items=3)
        itemsets, _ = run_fpgrowth(matrix, min_support=0.04)
        self.assertLess((1 / 24) / (10 / 24), 0.1)
        rules = generate_rules(itemsets, len(transactions), min_confidence=0.1)
        rule = find_rule(rules, {1}, {2})
        self.assertEqual((rule["count"], rule["antecedent_count"]), (1, 10))
        self.assertAlmostEqual(rule["confidence"], 0.1)
        # Dung sai chỉ cứu luật nằm đúng ngưỡng: luật 2 => 1 (conf 1) có mặt, còn không có luật nào dưới 0,1.
        self.assertTrue((rules["count"] * 10 >= rules["antecedent_count"]).all())

    def test_no_rules_when_only_singletons(self):
        itemsets, n = rule_fixture()
        rules = generate_rules(itemsets, n, min_confidence=0.1, max_rule_len=1)
        self.assertTrue(rules.empty)
        self.assertEqual(summarize_rules(rules)["n_rules"], 0)
        self.assertTrue(top_rules_by_lift(rules).empty)

    def test_top_rules_order_and_unique_itemsets(self):
        itemsets, n = rule_fixture()
        rules = generate_rules(itemsets, n, min_confidence=0.1)
        top = top_rules_by_lift(rules, n=2)
        # Lift 4/3 gồm {1,2} (đếm 3) và {1,2,3} (đếm 2); mỗi tập X ∪ Y chỉ giữ một luật.
        union = [tuple(sorted(a | c)) for a, c in zip(top["antecedents"], top["consequents"])]
        self.assertEqual(union, [(1, 2), (1, 2, 3)])
        # Sáu luật tách {1,2,3} đều có lift 4/3 (vd {1} => {2,3}: conf 2/3, lift (2/3) / (2/4)) và cùng số đếm 2;
        # hòa thì chọn tiền đề nhỏ nhất theo thứ tự item, tức {1} => {2,3}.
        self.assertEqual(top.iloc[1]["antecedents"], frozenset({1}))
        self.assertEqual(top.iloc[1]["consequents"], frozenset({2, 3}))
        self.assertAlmostEqual(top.iloc[0]["lift"], 4 / 3)
        # Không gộp theo tập: cả hai chiều của {1,2} đứng trước các luật có đếm thấp hơn.
        both = top_rules_by_lift(rules, n=3, unique_itemsets=False)
        self.assertEqual(
            [(tuple(sorted(a)), tuple(sorted(c))) for a, c in zip(both["antecedents"], both["consequents"])],
            [((1,), (2,)), ((2,), (1,)), ((1,), (2, 3))],
        )

    def test_pair_rules_only(self):
        itemsets, n = rule_fixture()
        pairs = pair_rules_only(generate_rules(itemsets, n, min_confidence=0.1))
        self.assertEqual(len(pairs), 6)
        self.assertTrue(all(len(a) == 1 and len(c) == 1 for a, c in zip(pairs["antecedents"], pairs["consequents"])))

    def test_summarize_rules(self):
        itemsets, n = rule_fixture()
        summary = summarize_rules(generate_rules(itemsets, n, min_confidence=0.1))
        self.assertEqual(summary["n_rules"], 12)
        self.assertEqual(summary["rules_by_size"], {"2": 6, "3": 6})
        self.assertEqual(summary["n_items_in_rules"], 3)
        self.assertAlmostEqual(summary["lift_max"], 4 / 3)


class MetadataTests(unittest.TestCase):
    def load(self):
        rows = [
            {"item_id": 1, "metadata": {"title": "A &amp; B", "brand": "OPI", "categories": [["Beauty", "Nails", "Polish"]]}},
            {"item_id": 2, "metadata": {"title": "C", "brand": "opi",
                                        "categories": [["Beauty", "Nails", "Polish"], ["Beauty", "Gifts"]]}},
            {"item_id": 3, "metadata": {"title": "D", "brand": "Unknown", "categories": [["Beauty", "Hair"]]}},
            {"item_id": 4, "metadata": {"title": "E", "categories": [["Beauty", "Gifts"]]}},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "items.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            return load_item_metadata(path)

    def test_load_item_metadata(self):
        metadata = self.load()
        self.assertEqual(metadata[1]["title"], "A & B")
        self.assertEqual(metadata[1]["brand"], "OPI")
        self.assertIsNone(metadata[3]["brand"])
        self.assertIsNone(metadata[4]["brand"])
        self.assertEqual(metadata[2]["leaf_categories"], frozenset({"Polish", "Gifts"}))

    def test_describe_rule_flags(self):
        metadata = self.load()
        # Brand khác nhau về chữ hoa thường vẫn tính là cùng brand; chung danh mục lá "Polish".
        info = describe_rule(frozenset({1}), frozenset({2}), metadata)
        self.assertTrue(info["same_brand"])
        self.assertTrue(info["same_leaf_category"])
        self.assertEqual(info["antecedent_titles"], ["A & B"])
        # Brand thiếu không bao giờ tính là cùng brand, kể cả khi cả hai cùng thiếu.
        info = describe_rule(frozenset({3}), frozenset({4}), metadata)
        self.assertFalse(info["same_brand"])
        self.assertFalse(info["same_leaf_category"])
        # Danh mục lá chung phải có mặt ở mọi item trong X ∪ Y: item 1 không có "Gifts".
        self.assertFalse(describe_rule(frozenset({2, 4}), frozenset({1}), metadata)["same_leaf_category"])
        self.assertTrue(describe_rule(frozenset({2}), frozenset({4}), metadata)["same_leaf_category"])


if __name__ == "__main__":
    unittest.main()
