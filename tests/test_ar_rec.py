"""Kiểm tra AR-Rec trên ví dụ tính tay và trên dữ liệu giả lập qua CLI."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path

import pandas as pd

from src.ar_rec import ARRecommender, check_train_sequences, load_train_counts, popularity_order, rank_users, write_rankings
from src.ar_rec import main as ar_rec_main
from src.evaluate import evaluate

N_ITEMS = 6
# Tần suất train: item 6 phổ biến nhất, 5 và 4 hòa (4 đứng trước vì item_id nhỏ hơn), 1, 2, 3 tiếp theo.
COUNTS = {1: 3, 2: 2, 3: 1, 4: 5, 5: 5, 6: 9}
POP = [6, 4, 5, 1, 2, 3]


def make_rules():
    rows = [
        # (X, Y, confidence, lift) -> điểm = confidence * lift
        ({1}, {2}, 0.5, 4.0),        # 2.0
        ({1, 3}, {4}, 1.0, 3.0),     # 3.0
        ({5}, {2, 4}, 0.4, 10.0),    # 4.0 cho cả 2 và 4
    ]
    return pd.DataFrame({
        "antecedents": [frozenset(x) for x, _, _, _ in rows],
        "consequents": [frozenset(y) for _, y, _, _ in rows],
        "confidence": [c for _, _, c, _ in rows],
        "lift": [lift for _, _, _, lift in rows],
    })


class ARRecommenderTests(unittest.TestCase):
    def setUp(self):
        self.model = ARRecommender(make_rules(), N_ITEMS, popularity_order(COUNTS, N_ITEMS))

    def test_popularity_order_breaks_ties_by_item_id_and_keeps_zero_count_items(self):
        self.assertEqual(popularity_order(COUNTS, N_ITEMS).tolist(), POP)
        self.assertEqual(popularity_order({3: 2}, 3).tolist(), [3, 1, 2])
        with self.assertRaises(ValueError):
            popularity_order({6: 2}, 3)

    def test_antecedent_must_be_subset_of_history(self):
        # Lịch sử {1}: chỉ luật {1} => {2} khớp; luật {1,3} => {4} thiếu item 3.
        ranked, n_rule = self.model.recommend({1}, 6)
        self.assertEqual(n_rule, 1)
        self.assertEqual(ranked[0], 2)
        self.assertNotIn(4, ranked[:1])
        self.assertEqual(self.model.matched_rules({1}).tolist(), [0])

    def test_rule_score_is_confidence_times_lift(self):
        # Lịch sử {1,3}: item 4 có điểm 3,0 và item 2 có điểm 2,0 nên 4 đứng trước 2.
        ranked, n_rule = self.model.recommend({1, 3}, 6)
        self.assertEqual(n_rule, 2)
        self.assertEqual(ranked[:2], [4, 2])

    def test_equal_scores_break_ties_by_item_id(self):
        ranked, n_rule = self.model.recommend({5}, 6)
        self.assertEqual(n_rule, 2)
        self.assertEqual(ranked[:2], [2, 4])

    def test_item_in_several_rules_uses_max_not_sum(self):
        # Lịch sử {1,3,5} khớp cả 3 luật. Item 2: 2,0 (luật 0) và 4,0 (luật 2); item 4: 3,0 (luật 1) và 4,0 (luật 2).
        # Lấy max: cả hai đều 4,0, hòa nên item_id tăng dần -> [2, 4]. Nếu cộng dồn: 2 có 6,0 và 4 có 7,0 -> [4, 2].
        ranked, _ = self.model.recommend({1, 3, 5}, 6)
        self.assertEqual(ranked[:2], [2, 4])

    def test_padding_uses_popularity_without_duplicates(self):
        ranked, n_rule = self.model.recommend({1}, 6)
        self.assertEqual(ranked, [2, 6, 4, 5, 1, 3])  # 2 từ luật, rồi Popularity bỏ qua item 2
        self.assertEqual(n_rule, 1)
        self.assertEqual(len(set(ranked)), len(ranked))

    def test_no_matching_rule_equals_popularity(self):
        ranked, n_rule = self.model.recommend({3}, 4)
        self.assertEqual((ranked, n_rule), (POP[:4], 0))
        self.assertEqual(self.model.recommend(set(), 3), (POP[:3], 0))

    def test_length_is_capped_by_catalog_and_truncates_rule_items(self):
        self.assertEqual(len(self.model.recommend({5}, 100)[0]), N_ITEMS)
        ranked, n_rule = self.model.recommend({5}, 1)
        self.assertEqual((ranked, n_rule), ([2], 1))

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            self.model.recommend({7}, 3)
        with self.assertRaises(ValueError):
            self.model.recommend({0}, 3)
        with self.assertRaises(ValueError):
            self.model.recommend({1}, 0)
        with self.assertRaises(ValueError):
            ARRecommender(make_rules(), N_ITEMS, [1, 2, 3])
        with self.assertRaises(ValueError):
            ARRecommender(make_rules().drop(columns=["lift"]), N_ITEMS, POP)
        bad = make_rules()
        bad.loc[0, "lift"] = 0.0
        with self.assertRaises(ValueError):
            ARRecommender(bad, N_ITEMS, POP)
        out_of_range = make_rules()
        out_of_range.at[0, "consequents"] = frozenset({9})
        with self.assertRaises(ValueError):
            ARRecommender(out_of_range, N_ITEMS, POP)

    def test_scores_equal_in_exact_arithmetic_tie_by_item_id(self):
        # 0,1 x 3,0 = 0,30000000000000004 trong dấu phẩy động nhưng bằng 0,3 x 1,0; làm tròn điểm phải coi là hòa.
        rules = pd.DataFrame({"antecedents": [frozenset({1}), frozenset({1})],
                              "consequents": [frozenset({5}), frozenset({2})],
                              "confidence": [0.1, 0.3], "lift": [3.0, 1.0]})
        model = ARRecommender(rules, N_ITEMS, POP)
        self.assertEqual(model.recommend({1}, 2)[0], [2, 5])

    def test_matching_uses_full_seen_list_beyond_50_items(self):
        # Item 1 là item cũ nhất trong 60 item đã thấy; nếu chỉ lấy 50 item cuối thì luật {1} => {70} không khớp.
        rules = pd.DataFrame({"antecedents": [frozenset({1})], "consequents": [frozenset({70})],
                              "confidence": [0.5], "lift": [2.0]})
        model = ARRecommender(rules, 70, list(range(70, 0, -1)))
        split = {1: {"target": 70, "seen_item_ids": list(range(1, 61))}}
        rankings, stats = rank_users(model, split, max_k=3)
        self.assertEqual(rankings[1][:3], [70, 69, 68])
        self.assertEqual(stats["users_with_rule"], 1)

    def test_rule_item_statistics_are_capped_at_top_k_and_split_seen_from_unseen(self):
        # Lịch sử {1, 5}: item từ luật là 2, 4 (luật 2) và 2 (luật 0) -> [2, 4]; thêm luật {1} => {5}, item 5 đã thấy.
        rules = make_rules()
        rules = pd.concat([rules, pd.DataFrame({"antecedents": [frozenset({1})], "consequents": [frozenset({5})],
                                                "confidence": [1.0], "lift": [1.0]})], ignore_index=True)
        model = ARRecommender(rules, N_ITEMS, POP)
        split = {1: {"target": 6, "seen_item_ids": [1, 5]}}
        rankings, stats = rank_users(model, split, max_k=2)
        self.assertEqual(rankings[1][:3], [2, 4, 5])
        # 3 item từ luật nhưng top-K = 2: chỉ đếm 2 (cả hai chưa thấy).
        self.assertEqual(stats["mean_rule_items_in_top_k"], 2)
        self.assertEqual(stats["rule_items_already_seen_ratio"], 0.0)
        rankings, stats = rank_users(model, split, max_k=3)
        self.assertEqual(stats["mean_rule_items_in_top_k"], 3)
        self.assertAlmostEqual(stats["rule_items_already_seen_ratio"], 1 / 3)
        self.assertEqual(stats["users_with_unseen_rule_item"], 1)

    def test_rank_users_exports_max_k_plus_seen_and_evaluates_by_hand(self):
        split = {
            1: {"target": 4, "seen_item_ids": [1, 3]},   # ranking [4, 2, 6, 5, 1, 3]: nhãn ở hạng 1
            2: {"target": 5, "seen_item_ids": [3]},      # không luật: Popularity [6, 4, 5, 1]: nhãn ở hạng 3
        }
        rankings, stats = rank_users(self.model, split, max_k=2)
        self.assertEqual(len(rankings[1]), 2 + 2)
        self.assertEqual(len(rankings[2]), 2 + 1)
        self.assertEqual(rankings[1][:2], [4, 2])
        self.assertEqual(rankings[2], [6, 4, 5])
        self.assertEqual(stats["users_with_rule"], 1)
        self.assertAlmostEqual(stats["mean_rule_items_in_top_k"], (2 + 0) / 2)
        scores = evaluate(rankings, split, ks=(2,))
        # K = 2: user 1 trúng ở hạng 1 (recall 1, NDCG 1); user 2 nhãn ở hạng 3, ngoài top-2 (0).
        self.assertAlmostEqual(scores["recall@2"], 0.5)
        self.assertAlmostEqual(scores["ndcg@2"], 0.5)


# Dữ liệu giả lập 7 user, catalog 8 item. Mỗi user: (train, validation, test).
# Item 7 (validation) và 8 (test) không xuất hiện ở train nên không được vào bất kỳ tập phổ biến nào;
# nếu có rò rỉ validation/test, {2,3,7} hoặc {2,3,8} sẽ thành tập phổ biến.
USERS = {
    1: ([2, 3], 7, 8), 2: ([2, 3], 7, 8), 3: ([2, 3], 7, 8),
    4: ([1, 4], 7, 8), 5: ([1, 5], 7, 8), 6: ([1, 6], 7, 8),
    7: ([2], 5, 8),
}


def write_dataset(root):
    """Giao dịch train: {2,3} x3, {1,4}, {1,5}, {1,6}, {2}. Đếm: 1 -> 3, 2 -> 4, 3 -> 3, {2,3} -> 3.

    Với min_support 0,4 (số đếm >= 3): tập phổ biến {1}, {2}, {3}, {2,3}.
    Luật 2 => 3: confidence 3/4, lift 1,75; luật 3 => 2: confidence 1, lift 1,75.
    Popularity (tần suất mọi tương tác train): 2 -> 4, rồi 1 và 3 (hòa, id nhỏ trước), rồi 4, 5, 6, 7, 8.
    """
    rows, split_lines, train, validation = [], ["user_id,train_start,train_stop,validation_index,test_index"], [], []
    for user, (items, val, test) in USERS.items():
        start = len(rows)
        rows += [(user, item) for item in items] + [(user, val), (user, test)]
        stop = start + len(items)
        split_lines.append(f"{user},{start},{stop},{stop},{stop + 1}")
        train.append({"user_id": user, "item_ids": items})
        validation.append({"user_id": user, "history": items, "target": val, "seen_item_ids": items,
                           "target_in_history": False})
    lines = ["interaction_id,user_id,item_id,timestamp,source_line"]
    lines += [f"{i},{u},{item},0,{i}" for i, (u, item) in enumerate(rows)]
    (root / "interactions.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (root / "splits.csv").write_text("\n".join(split_lines) + "\n", encoding="utf-8")
    for name, records in (("train_sequences.jsonl", train), ("validation.jsonl", validation)):
        (root / name).write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")


class ARRecCliTests(unittest.TestCase):
    def run_cli(self, root, *extra):
        write_dataset(root)
        return ar_rec_main([
            "--processed-dir", str(root), "--interactions", str(root / "interactions.csv"),
            "--splits", str(root / "splits.csv"), "--output-dir", str(root / "out"), "--ks", "1", "2", *extra,
        ])

    def test_train_counts_count_every_interaction(self):
        with tempfile.TemporaryDirectory() as directory:
            write_dataset(Path(directory))
            counts = load_train_counts(Path(directory) / "train_sequences.jsonl")
            self.assertEqual(counts, {2: 4, 3: 3, 1: 3, 4: 1, 5: 1, 6: 1})
            self.assertEqual(popularity_order(counts, 8).tolist(), [2, 1, 3, 4, 5, 6, 7, 8])

    def test_cli_uses_train_only_rules_and_writes_shared_format(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(self.run_cli(root, "--min-support", "0.4", "--min-confidence", "0.5", "--max-len", "2"), 0)
            rows = [json.loads(line) for line in (root / "out" / "rankings.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual([r["user_id"] for r in rows], list(range(1, 8)))
            by_user = {r["user_id"]: r["ranked_items"] for r in rows}
            # User 7 có lịch sử {2}: luật 2 => 3 đưa item 3 lên đầu, còn Popularity thuần sẽ đặt item 2 đầu tiên.
            self.assertEqual(by_user[7], [3, 2, 1])
            # Độ dài = max_k + len(seen) = 2 + len(train của user).
            self.assertEqual([len(by_user[u]) for u in range(1, 8)], [4, 4, 4, 4, 4, 4, 3])
            metrics = json.loads((root / "out" / "ar_rec_metrics.json").read_text(encoding="utf-8"))
            # Chính xác 4 tập phổ biến: nếu nhãn validation/test (item 7, 8) bị lẫn vào train thì có thêm {7}, {8}, {2,7}, ...
            self.assertEqual(metrics["mining"]["n_frequent_itemsets"], 4)
            self.assertEqual(metrics["mining"]["n_transactions"], 7)
            self.assertEqual(metrics["split"], "validation")
            self.assertEqual(set(metrics["metrics"]), {"keep_seen", "exclude_seen"})
            # Chỉ user 7 (lịch sử {2}) có luật khớp; user 1-3 (lịch sử {2,3}) có luật 2 => 3 và 3 => 2 nhưng hậu quả
            # đã nằm trong lịch sử, vẫn tính là có luật khớp vì mô hình không tự lọc item đã thấy.
            self.assertEqual(metrics["ranking"]["users_with_rule"], 4)

    def test_cli_test_split_matches_on_validation_free_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_dataset(root)
            # Bộ test: lịch sử gồm cả item validation (7); luật và Popularity vẫn chỉ từ train.
            rows = [{"user_id": u, "history": items + [val], "target": test, "seen_item_ids": items + [val],
                     "target_in_history": False} for u, (items, val, test) in USERS.items()]
            lines = [json.dumps(r) for r in rows]
            (root / "test.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
            code = ar_rec_main(["--split", "test", "--processed-dir", str(root), "--splits", str(root / "splits.csv"),
                                "--output-dir", str(root / "out"), "--ks", "1", "2",
                                "--min-support", "0.4", "--min-confidence", "0.5", "--max-len", "2"])
            self.assertEqual(code, 0)
            metrics = json.loads((root / "out" / "ar_rec_metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(metrics["split"], "test")
            self.assertEqual(metrics["mining"]["n_frequent_itemsets"], 4)  # interactions.csv suy ra từ --processed-dir

    def test_cli_rejects_out_of_range_thresholds(self):
        for flag, value in (("--min-support", "1.5"), ("--min-support", "0"), ("--min-confidence", "nan")):
            with tempfile.TemporaryDirectory() as directory, self.assertRaises(SystemExit) as raised:
                ar_rec_main([flag, value, "--output-dir", directory])
            self.assertEqual(raised.exception.code, 2)

    def test_cli_without_rules_falls_back_to_popularity_for_every_user(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with redirect_stderr(io.StringIO()) as err:
                code = self.run_cli(root, "--min-support", "0.99", "--min-confidence", "0.99")
            self.assertEqual(code, 0)
            self.assertIn("Cảnh báo", err.getvalue())
            metrics = json.loads((root / "out" / "ar_rec_metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(metrics["mining"]["n_rules"], 0)
            self.assertEqual(metrics["ranking"]["users_with_rule"], 0)
            popularity = [2, 1, 3, 4, 5, 6, 7, 8]
            for line in (root / "out" / "rankings.jsonl").read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                length = min(2 + len(USERS[row["user_id"]][0]), len(popularity))
                self.assertEqual(row["ranked_items"], popularity[:length])

    def test_check_train_sequences_accepts_consistent_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_dataset(root)
            check_train_sequences(root / "train_sequences.jsonl", root / "interactions.csv", root / "splits.csv")

    def test_check_train_sequences_rejects_mixed_splits(self):
        # Một item validation lọt vào train_sequences (như khi file đến từ lần chia khác) sẽ kéo item đó lên Popularity.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_dataset(root)
            path = root / "train_sequences.jsonl"
            records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            variants = {
                "extra item": [dict(records[0], item_ids=records[0]["item_ids"] + [USERS[1][1]])] + records[1:],
                "swapped item, same length": [dict(records[0], item_ids=records[0]["item_ids"][:-1] + [USERS[1][1]])]
                                             + records[1:],
                "missing user": records[:-1],
                "unknown user": records + [{"user_id": 99, "item_ids": [1]}],
                "duplicate user": records + [records[0]],
            }
            for name, changed in variants.items():
                with self.subTest(name):
                    path.write_text("".join(json.dumps(r) + "\n" for r in changed), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        check_train_sequences(path, root / "interactions.csv", root / "splits.csv")
                    self.assertEqual(ar_rec_main([
                        "--processed-dir", str(root), "--interactions", str(root / "interactions.csv"),
                        "--splits", str(root / "splits.csv"), "--output-dir", str(root / "out")]), 1)

    def test_written_rankings_are_sorted_by_user(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "r.jsonl"
            write_rankings(path, {2: [3, 1], 1: [2, 1]})
            self.assertEqual([json.loads(line)["user_id"] for line in path.read_text().splitlines()], [1, 2])


if __name__ == "__main__":
    unittest.main()
