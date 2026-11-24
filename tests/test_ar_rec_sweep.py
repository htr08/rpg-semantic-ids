"""Kiểm tra quét AR-Rec: lọc confidence, thống kê cặp, chọn cấu hình và chạy đầu-cuối trên dữ liệu giả lập."""

import json
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from src import ar_rec_sweep
from src.ar_rec import ARRecommender, load_train_counts, popularity_order
from src.ar_rec import main as ar_rec_main
from src.ar_rec_sweep import (
    bucket_label,
    cooccurrence_stats,
    diagnose,
    filter_by_confidence,
    format_markdown,
    history_bucket,
    paired_difference,
    pair_counts,
    parse_grid,
    run_sweep,
    selection_key,
)
from src.ar_rec_sweep import main as sweep_main
from src.evaluate import evaluate, load_split
from src.patterns import build_transaction_matrix, generate_rules, run_fpgrowth
from tests.test_ar_rec import write_dataset


def make_run(ndcg, n_rules=10, support=0.001, confidence=0.1, max_len=2):
    return {"min_support": support, "min_confidence": confidence, "max_len": max_len, "n_rules": n_rules,
            "metrics": {"keep_seen": {"ndcg@10": ndcg}}}


class HelperTests(unittest.TestCase):
    def test_filter_by_confidence_equals_direct_generation(self):
        transactions = [[1, 2], [1, 2], [1, 3], [2, 3], [1, 2, 3], [1, 2, 3], [4]]
        matrix = build_transaction_matrix([np.array(t) for t in transactions], 4)
        itemsets, _ = run_fpgrowth(matrix, 0.2)
        low = generate_rules(itemsets, len(transactions), min_confidence=0.2)
        for threshold in (0.5, 0.6, 2 / 3, 0.75):
            direct = generate_rules(itemsets, len(transactions), min_confidence=threshold)
            filtered = filter_by_confidence(low, threshold)
            key = lambda frame: sorted((tuple(sorted(a)), tuple(sorted(c)), round(conf, 9))
                                       for a, c, conf in zip(frame["antecedents"], frame["consequents"], frame["confidence"]))
            self.assertEqual(key(filtered), key(direct), f"threshold {threshold}")
        # Ngưỡng bằng đúng một confidence có thật (2/3) phải giữ luật đó, không loại do sai số làm tròn.
        self.assertTrue((filter_by_confidence(low, 2 / 3)["confidence"] > 0.66).any())

    def test_pair_counts_by_hand(self):
        matrix = build_transaction_matrix([np.array([1, 2, 3]), np.array([1, 2]), np.array([2, 3])], 3)
        # (1,2): 2, (1,3): 1, (2,3): 2
        self.assertEqual(sorted(pair_counts(matrix).tolist()), [1, 2, 2])
        self.assertEqual(cooccurrence_stats(pair_counts(matrix), 2),
                         {"pairs_cooccurring": 3, "pairs_reaching_min_count": 2, "pair_count_median": 2.0})
        empty = build_transaction_matrix([np.array([1]), np.array([2])], 2)
        self.assertEqual(cooccurrence_stats(pair_counts(empty), 1)["pairs_cooccurring"], 0)

    def test_history_buckets(self):
        self.assertEqual([history_bucket(n) for n in (1, 4, 5, 9, 10, 19, 20, 500)], [0, 0, 1, 1, 2, 2, 3, 3])
        self.assertEqual([bucket_label(i) for i in range(4)], ["<=4", "5-9", "10-19", ">=20"])

    def test_selection_prefers_ndcg_then_fewer_rules_then_larger_thresholds(self):
        runs = [make_run(0.02, n_rules=50), make_run(0.03, n_rules=90), make_run(0.03, n_rules=60, support=0.0005),
                make_run(0.03, n_rules=60, support=0.001, confidence=0.2), make_run(0.03, n_rules=60, support=0.001, confidence=0.1),
                make_run(0.03, n_rules=60, support=0.001, confidence=0.1, max_len=3)]
        best = min(runs, key=selection_key)
        self.assertEqual((best["min_support"], best["min_confidence"], best["max_len"]), (0.001, 0.2, 2))

    def test_paired_difference(self):
        diff = paired_difference([1.0, 2.0, 3.0], [0.0, 0.0, 0.0])
        self.assertAlmostEqual(diff["mean"], 2.0)
        self.assertAlmostEqual(diff["standard_error"], 1 / np.sqrt(3))

    def test_parse_grid(self):
        self.assertEqual(parse_grid(["2:0.0001,0.0002", "3:0.001"]), {2: (0.0001, 0.0002), 3: (0.001,)})
        for bad in (["2"], ["x:0.1"], ["1:0.1"], ["2:"], ["2:0.1", "2:0.2"]):
            with self.assertRaises(ValueError, msg=str(bad)):
                parse_grid(bad)

    def test_diagnose_counts_rule_users_target_coverage_and_buckets(self):
        rules = pd.DataFrame({"antecedents": [frozenset({1})], "consequents": [frozenset({2})],
                              "confidence": [0.5], "lift": [4.0]})
        model = ARRecommender(rules, 30, popularity_order({}, 30))
        split = {
            1: {"target": 2, "seen_item_ids": [1]},                      # luật khớp, nhãn là hậu quả
            2: {"target": 9, "seen_item_ids": [1, 3]},                   # luật khớp, nhãn không phải hậu quả
            3: {"target": 2, "seen_item_ids": [3, 4]},                   # không luật khớp
            4: {"target": 5, "seen_item_ids": [1] + list(range(10, 29))},  # 20 item khác nhau, luật khớp
        }
        result = diagnose(model, split)
        self.assertAlmostEqual(result["users_with_rule_ratio"], 3 / 4)
        self.assertAlmostEqual(result["target_covered_ratio"], 1 / 4)
        buckets = result["by_history_length"]
        self.assertEqual({k: v["users"] for k, v in buckets.items()}, {"<=4": 3, "5-9": 0, "10-19": 0, ">=20": 1})
        self.assertAlmostEqual(buckets["<=4"]["users_with_rule_ratio"], 2 / 3)
        self.assertIsNone(buckets["5-9"]["users_with_rule_ratio"])
        self.assertAlmostEqual(buckets[">=20"]["users_with_rule_ratio"], 1.0)


class SweepTests(unittest.TestCase):
    def sweep(self, root, grid, confidences, **kwargs):
        write_dataset(root)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # catalog 8 item < K = 10 nên danh sách ngắn hơn K
            return run_sweep(root / "interactions.csv", root / "splits.csv", root, grid, confidences, **kwargs)

    def test_end_to_end_matches_ar_rec_cli_and_hand_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload, rankings = self.sweep(root, {2: (0.4,)}, (0.5,))
            self.assertEqual(payload["selected"], {"max_len": 2, "min_support": 0.4, "min_confidence": 0.5})
            level, run = payload["levels"][0], payload["runs"][0]
            # Item phổ biến (đếm >= 3): 1, 2, 3. Cặp cùng xuất hiện: {2,3}: 3 lần, {1,4}, {1,5}, {1,6}: 1 lần.
            self.assertEqual((level["min_count"], level["n_frequent_items"], level["pairs_cooccurring"],
                              level["pairs_reaching_min_count"]), (3, 3, 4, 1))
            self.assertEqual(level["users_with_frequent_item_ratio"], 1.0)
            self.assertEqual(level["target_is_frequent_item_ratio"], 0.0)  # nhãn là item 7 và 5, chưa từng phổ biến
            self.assertAlmostEqual(run["diagnostics"]["users_with_rule_ratio"], 4 / 7)
            self.assertEqual(run["diagnostics"]["target_covered_ratio"], 0.0)
            self.assertEqual(run["n_rules"], 2)
            # Trùng khớp với CLI ar_rec cùng cấu hình.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ar_rec_main(["--processed-dir", str(root), "--splits", str(root / "splits.csv"), "--output-dir", str(root / "out"),
                             "--min-support", "0.4", "--min-confidence", "0.5", "--max-len", "2"])
            expected = json.loads((root / "out" / "ar_rec_metrics.json").read_text(encoding="utf-8"))["metrics"]
            for protocol in ("keep_seen", "exclude_seen"):
                for name, value in run["metrics"][protocol].items():
                    if name in expected[protocol]:
                        self.assertAlmostEqual(value, expected[protocol][name], msg=f"{protocol} {name}")
            on_disk = [json.loads(line) for line in (root / "out" / "rankings.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual({r["user_id"]: r["ranked_items"] for r in on_disk}, rankings)

    def test_memory_error_marks_level_and_continues(self):
        real = ar_rec_sweep.run_fpgrowth
        calls = []

        def flaky(matrix, support, max_len=None):
            calls.append(support)
            if len(calls) == 1:
                raise MemoryError
            return real(matrix, support, max_len=max_len)

        with tempfile.TemporaryDirectory() as directory, mock.patch.object(ar_rec_sweep, "run_fpgrowth", flaky):
            payload, _ = self.sweep(Path(directory), {2: (0.2, 0.4)}, (0.5,))
        self.assertEqual([level["status"] for level in payload["levels"]], ["memory_error", "ok"])
        self.assertEqual(payload["selected"]["min_support"], 0.4)
        self.assertIn("memory_error", format_markdown(payload))

    def test_raises_when_nothing_runs(self):
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(ValueError):
            self.sweep(Path(directory), {2: (0.99,)}, (0.99,))

    def test_rankings_and_markdown_cover_every_level(self):
        with tempfile.TemporaryDirectory() as directory:
            payload, rankings = self.sweep(Path(directory), {2: (0.2, 0.4), 3: (0.4,)}, (0.3, 0.5))
            self.assertEqual(len(payload["levels"]), 3)
            self.assertEqual(len(payload["runs"]), 6)
            self.assertEqual(sorted(rankings), list(range(1, 8)))
            text = format_markdown(payload)
            self.assertEqual(text.count("**(chọn)**"), 2)  # một dòng ở bảng tốt nhất, một dòng ở bảng min_confidence
            self.assertEqual(set(payload["best_by_max_len"]), {"2", "3"})

    def test_cli_writes_outputs_and_rejects_bad_grid(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_dataset(root)
            args = ["--processed-dir", str(root), "--splits", str(root / "splits.csv"), "--grid", "2:0.4",
                    "--confidences", "0.5", "--output", str(root / "o.json"), "--markdown", str(root / "o.md"),
                    "--rankings-output", str(root / "r.jsonl")]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.assertEqual(sweep_main(args), 0)
            self.assertEqual(json.loads((root / "o.json").read_text(encoding="utf-8"))["selected"]["min_support"], 0.4)
            self.assertTrue((root / "o.md").read_text(encoding="utf-8").startswith("# Quét"))
            self.assertEqual(len((root / "r.jsonl").read_text(encoding="utf-8").splitlines()), 7)
            for bad in (["--grid", "2"], ["--grid", "2:1.5"], ["--confidences", "0"]):
                with self.assertRaises(SystemExit) as raised:
                    sweep_main(bad)
                self.assertEqual(raised.exception.code, 2)


# Dữ liệu cho test chọn cấu hình: các cấu hình khác nhau cho điểm khác nhau.
# user -> (train, validation, test). Item 1..8, N = 8 giao dịch.
#   {1,2} x4 (u1-u4), {3,4} x2 (u5-u6), {1} (u7), {3} (u8). Đếm item: 1 -> 5, 2 -> 4, 3 -> 3, 4 -> 2; cặp {1,2} -> 4, {3,4} -> 2.
#   min_support 0,25 (đếm >= 2): luật 1=>2 (conf 0,8), 2=>1 (1), 3=>4 (2/3), 4=>3 (1).
#   min_support 0,5 (đếm >= 4): chỉ 1=>2 và 2=>1.
#   Nhãn của u7 (item 2) khớp luật 1=>2 ở cả hai mức; nhãn của u8 (item 4) chỉ khớp luật 3=>4 ở mức 0,25.
USERS_TWO_LEVELS = {
    1: ([1, 2], 5, 8), 2: ([1, 2], 5, 8), 3: ([1, 2], 5, 8), 4: ([1, 2], 5, 8),
    5: ([3, 4], 5, 8), 6: ([3, 4], 5, 8), 7: ([1], 2, 8), 8: ([3], 4, 8),
}


def write_two_level_dataset(root):
    rows, split_lines, train, validation = [], ["user_id,train_start,train_stop,validation_index,test_index"], [], []
    for user, (items, val, test) in USERS_TWO_LEVELS.items():
        start = len(rows)
        rows += [(user, item) for item in items] + [(user, val), (user, test)]
        stop = start + len(items)
        split_lines.append(f"{user},{start},{stop},{stop},{stop + 1}")
        train.append({"user_id": user, "item_ids": items})
        validation.append({"user_id": user, "history": items, "target": val, "seen_item_ids": items, "target_in_history": False})
    lines = ["interaction_id,user_id,item_id,timestamp,source_line"] + [f"{i},{u},{item},0,{i}" for i, (u, item) in enumerate(rows)]
    (root / "interactions.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (root / "splits.csv").write_text("\n".join(split_lines) + "\n", encoding="utf-8")
    for name, records in (("train_sequences.jsonl", train), ("validation.jsonl", validation)):
        (root / name).write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")


class SelectionTests(unittest.TestCase):
    def run_grid(self, grid, confidences):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        write_two_level_dataset(root)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return run_sweep(root / "interactions.csv", root / "splits.csv", root, grid, confidences), root

    def test_selection_diagnostics_and_summary_fields_by_hand(self):
        (payload, rankings), root = self.run_grid({2: (0.25, 0.5), 3: (0.25,)}, (0.5, 0.9))
        self.assertEqual(payload["selected"], {"max_len": 2, "min_support": 0.25, "min_confidence": 0.5})
        low, high = (next(lv for lv in payload["levels"] if lv["max_len"] == 2 and lv["min_support"] == s) for s in (0.25, 0.5))
        self.assertEqual((low["min_count"], low["n_frequent_items"], low["users_with_frequent_item_ratio"],
                          low["target_is_frequent_item_ratio"]), (2, 4, 1.0, 0.25))
        self.assertEqual((high["min_count"], high["n_frequent_items"], high["users_with_frequent_item_ratio"],
                          high["target_is_frequent_item_ratio"]), (4, 2, 5 / 8, 1 / 8))
        self.assertEqual((low["pairs_cooccurring"], low["pairs_reaching_min_count"], high["pairs_reaching_min_count"]), (2, 2, 1))

        def run(max_len, support, confidence):
            return next(r for r in payload["runs"] if (r["max_len"], r["min_support"], r["min_confidence"]) == (max_len, support, confidence))

        self.assertEqual(run(2, 0.25, 0.5)["n_rules"], 4)
        self.assertEqual(run(2, 0.25, 0.9)["n_rules"], 2)  # chỉ 2=>1 và 4=>3 có confidence 1
        self.assertEqual((run(2, 0.25, 0.5)["diagnostics"]["users_with_rule_ratio"], run(2, 0.25, 0.5)["diagnostics"]["target_covered_ratio"]), (1.0, 2 / 8))
        self.assertEqual((run(2, 0.5, 0.5)["diagnostics"]["users_with_rule_ratio"], run(2, 0.5, 0.5)["diagnostics"]["target_covered_ratio"]), (5 / 8, 1 / 8))
        self.assertEqual(run(2, 0.25, 0.5)["n_items_in_rules"], 4)
        self.assertEqual(run(2, 0.5, 0.5)["n_items_in_rules"], 2)
        # Xếp hạng được trả về là của cấu hình được chọn: u7 (lịch sử {1}) có item 2 đầu tiên, u8 (lịch sử {3}) có item 4 đầu tiên.
        self.assertEqual((rankings[7][0], rankings[8][0]), (2, 4))
        # Cấu hình support 0,5 xếp item 4 ở hạng 4 cho u8 (Popularity 1, 2, 3, 4), nên NDCG@10 thấp hơn.
        ok = sorted((r for r in payload["runs"] if r["status"] == "ok"), key=selection_key)
        self.assertGreater(ok[0]["metrics"]["keep_seen"]["ndcg@10"], run(2, 0.5, 0.5)["metrics"]["keep_seen"]["ndcg@10"])
        self.assertEqual(payload["comparison"]["runner_up"], {k: ok[1][k] for k in ("max_len", "min_support", "min_confidence")})
        for max_len in (2, 3):
            best = min((r for r in ok if r["max_len"] == max_len), key=selection_key)
            self.assertEqual(payload["best_by_max_len"][str(max_len)]["ndcg@10"], best["metrics"]["keep_seen"]["ndcg@10"])
            self.assertEqual(payload["best_by_max_len"][str(max_len)]["min_support"], best["min_support"])

    def test_popularity_reference_and_paired_difference(self):
        (payload, _), root = self.run_grid({2: (0.25,)}, (0.5,))
        counts = load_train_counts(root / "train_sequences.jsonl")
        order = popularity_order(counts, 8).tolist()
        self.assertEqual(order[:4], [1, 2, 3, 4])
        split = load_split(root / "validation.jsonl")
        expected = evaluate({u: order for u in split}, split, ks=(5, 10), exclude_seen=False)
        self.assertEqual(payload["reference_popularity"]["keep_seen"], expected)
        best = payload["runs"][0]["metrics"]["keep_seen"]["ndcg@10"]
        self.assertAlmostEqual(payload["comparison"]["selected_vs_popularity"]["mean"], best - expected["ndcg@10"])

    def test_markdown_survives_level_without_rules(self):
        (payload, _), _ = self.run_grid({2: (0.25, 0.99)}, (0.5,))
        self.assertEqual([r["status"] for r in payload["runs"]], ["ok", "no_rules"])
        self.assertIn("không có luật nào thỏa ngưỡng", format_markdown(payload))

    def test_rejects_train_sequences_from_another_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_two_level_dataset(root)
            path = root / "train_sequences.jsonl"
            records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            val_item = USERS_TWO_LEVELS[records[0]["user_id"]][1]
            records[0]["item_ids"].append(val_item)
            path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "không khớp"):
                run_sweep(root / "interactions.csv", root / "splits.csv", root, {2: (0.25,)}, (0.5,))

    def test_rejects_empty_grid_or_confidences(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_two_level_dataset(root)
            for grid, confidences in (({2: ()}, (0.5,)), ({2: (0.25,)}, ())):
                with self.assertRaises(ValueError):
                    run_sweep(root / "interactions.csv", root / "splits.csv", root, grid, confidences)


if __name__ == "__main__":
    unittest.main()
