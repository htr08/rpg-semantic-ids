"""Hồi quy cho các hành vi mà báo cáo và tệp kết quả của AR-Rec dựa vào (bộ chia, cấu hình, chọn cấu hình, bảng Markdown)."""

import io
import json
import tempfile
import unittest
import warnings
from contextlib import redirect_stderr
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from src import ar_rec
from src.ar_rec import ARRecommender, check_split_file, load_train_counts, popularity_order, rank_users
from src.ar_rec import main as ar_rec_main
from src.ar_rec_sweep import filter_by_confidence, format_markdown, run_sweep, selection_key
from tests import test_ar_rec
from tests.test_ar_rec import N_ITEMS, POP, USERS, make_rules, write_dataset
from tests.test_ar_rec_sweep import make_run, write_two_level_dataset


def quiet(function, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return function(*args, **kwargs)


class ARRecReportTests(unittest.TestCase):
    def test_test_split_reads_test_jsonl(self):
        # Lịch sử của bộ test có thêm item validation, nên độ dài ranking dài hơn bộ validation đúng một item.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_dataset(root)
            rows = [{"user_id": u, "history": items + [val], "target": test, "seen_item_ids": items + [val],
                     "target_in_history": False} for u, (items, val, test) in USERS.items()]
            (root / "test.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            code = ar_rec_main(["--split", "test", "--processed-dir", str(root), "--splits", str(root / "splits.csv"),
                                "--output-dir", str(root / "out"), "--ks", "1", "2",
                                "--min-support", "0.4", "--min-confidence", "0.5", "--max-len", "2"])
            self.assertEqual(code, 0)
            written = [json.loads(line) for line in (root / "out" / "rankings.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual({r["user_id"]: len(r["ranked_items"]) for r in written},
                             {u: min(2 + len(USERS[u][0]) + 1, 8) for u in USERS})

    def test_rank_users_counts_users_with_rule_and_with_unseen_rule_item(self):
        # User 1 không có luật khớp; user 2 có item luật chưa thấy; user 3 có luật nhưng mọi item luật đã nằm trong lịch sử.
        model = ARRecommender(make_rules(), N_ITEMS, popularity_order(test_ar_rec.COUNTS, N_ITEMS))
        split = {1: {"target": 6, "seen_item_ids": [3]}, 2: {"target": 6, "seen_item_ids": [1]},
                 3: {"target": 6, "seen_item_ids": [1, 2]}}
        _, stats = rank_users(model, split, max_k=2)
        self.assertEqual(stats["users_with_rule"], 2)
        self.assertAlmostEqual(stats["users_with_rule_ratio"], 2 / 3)
        self.assertEqual(stats["users_with_unseen_rule_item"], 1)

    def test_train_counts_count_repeated_items_within_a_user(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.jsonl"
            path.write_text(json.dumps({"user_id": 1, "item_ids": [5, 5, 2]}) + "\n", encoding="utf-8")
            self.assertEqual(load_train_counts(path), {5: 2, 2: 1})

    def test_duplicate_items_in_history_do_not_complete_an_antecedent(self):
        # Lịch sử [1, 1] chưa chứa item 3 nên luật {1, 3} => {4} không khớp.
        model = ARRecommender(make_rules(), N_ITEMS, POP)
        self.assertEqual(model.matched_rules([1, 1]).tolist(), [0])

    def test_max_len_limits_itemsets_and_rules(self):
        # Bốn giao dịch: {2, 3, 4} x 3 và {1}; min_support 0,5 giữ item 2, 3, 4.
        users = {1: ([2, 3, 4], 7, 8), 2: ([2, 3, 4], 7, 8), 3: ([2, 3, 4], 7, 8), 4: ([1], 5, 8)}
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(USERS, users, clear=True):
            root = Path(directory)
            write_dataset(root)
            for max_len, itemsets, rules in ((2, 6, 6), (3, 7, 12)):
                with self.subTest(max_len=max_len):
                    code = quiet(ar_rec_main, ["--processed-dir", str(root), "--splits", str(root / "splits.csv"),
                                               "--output-dir", str(root / "out"), "--ks", "1", "2", "--min-support", "0.5",
                                               "--min-confidence", "0.5", "--max-len", str(max_len)])
                    self.assertEqual(code, 0)
                    mining = json.loads((root / "out" / "ar_rec_metrics.json").read_text(encoding="utf-8"))["mining"]
                    self.assertEqual((mining["n_frequent_itemsets"], mining["n_rules"]), (itemsets, rules))

    def test_metrics_file_reports_config_and_rule_count(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_dataset(root)
            code = quiet(ar_rec_main, ["--processed-dir", str(root), "--splits", str(root / "splits.csv"),
                                       "--output-dir", str(root / "out"), "--min-support", "0.4",
                                       "--min-confidence", "0.5", "--max-len", "2"])
            self.assertEqual(code, 0)
            result = json.loads((root / "out" / "ar_rec_metrics.json").read_text(encoding="utf-8"))
            self.assertEqual((result["config"]["max_len"], result["config"]["min_support"]), (2, 0.4))
            self.assertEqual(result["mining"]["n_rules"], 2)

    def test_memory_error_exits_with_code_1(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_dataset(root)
            with mock.patch.object(ar_rec, "build_recommender", side_effect=MemoryError), redirect_stderr(io.StringIO()) as err:
                code = ar_rec_main(["--processed-dir", str(root), "--splits", str(root / "splits.csv"),
                                    "--output-dir", str(root / "out")])
            self.assertEqual(code, 1)
            self.assertIn("hết bộ nhớ", err.getvalue())

    def write_test_file(self, root):
        rows = [{"user_id": u, "history": items + [val], "target": test, "seen_item_ids": sorted(items + [val]),
                 "target_in_history": False} for u, (items, val, test) in USERS.items()]
        path = root / "test.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        return rows, path

    def test_check_split_file_accepts_consistent_validation_and_test_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_dataset(root)
            _, test_path = self.write_test_file(root)
            check_split_file(root / "validation.jsonl", "validation", root / "interactions.csv", root / "splits.csv")
            check_split_file(test_path, "test", root / "interactions.csv", root / "splits.csv")

    def test_check_split_file_rejects_files_from_another_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_dataset(root)
            test_rows, test_path = self.write_test_file(root)
            validation_path = root / "validation.jsonl"
            validation_rows = [json.loads(line) for line in validation_path.read_text(encoding="utf-8").splitlines()]
            cases = {
                # Lịch sử của bộ test nằm trong file validation: điểm đổi dù luật và Popularity không đổi.
                "test history in validation file": (validation_path, "validation", test_rows),
                "validation history in test file": (test_path, "test", validation_rows),
                "wrong target": (validation_path, "validation", [dict(validation_rows[0], target=6)] + validation_rows[1:]),
                "missing user": (validation_path, "validation", validation_rows[:-1]),
                "duplicate user": (validation_path, "validation", validation_rows + [validation_rows[0]]),
            }
            for name, (path, split_name, rows) in cases.items():
                with self.subTest(name):
                    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        check_split_file(path, split_name, root / "interactions.csv", root / "splits.csv")
            with self.assertRaises(ValueError):
                check_split_file(validation_path, "train", root / "interactions.csv", root / "splits.csv")

    def test_cli_rejects_validation_file_that_does_not_match_the_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_dataset(root)
            test_rows, _ = self.write_test_file(root)
            (root / "validation.jsonl").write_text("".join(json.dumps(r) + "\n" for r in test_rows), encoding="utf-8")
            with redirect_stderr(io.StringIO()) as err:
                code = ar_rec_main(["--processed-dir", str(root), "--splits", str(root / "splits.csv"),
                                    "--output-dir", str(root / "out")])
            self.assertEqual(code, 1)
            self.assertIn("không khớp", err.getvalue())
            self.assertFalse((root / "out").exists())


class SweepReportTests(unittest.TestCase):
    def run_two_level(self, grid, confidences):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        write_two_level_dataset(root)
        return quiet(run_sweep, root / "interactions.csv", root / "splits.csv", root, grid, confidences)

    def run_seven_users(self, grid, confidences):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        write_dataset(root)
        return quiet(run_sweep, root / "interactions.csv", root / "splits.csv", root, grid, confidences)

    def test_sweep_rejects_validation_file_that_does_not_match_the_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_dataset(root)
            rows = [json.loads(line) for line in (root / "validation.jsonl").read_text(encoding="utf-8").splitlines()]
            rows[0]["seen_item_ids"] = rows[0]["seen_item_ids"] + [8]
            (root / "validation.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "không khớp"):
                run_sweep(root / "interactions.csv", root / "splits.csv", root, {2: (0.4,)}, (0.5,))

    def test_each_tie_break_decides_on_its_own(self):
        pick = lambda runs: min(runs, key=selection_key)
        self.assertEqual(pick([make_run(0.03, support=0.0005), make_run(0.03, support=0.001)])["min_support"], 0.001)
        self.assertEqual(pick([make_run(0.03, confidence=0.1), make_run(0.03, confidence=0.2)])["min_confidence"], 0.2)
        # min_support được xét trước min_confidence.
        best = pick([make_run(0.03, support=0.0005, confidence=0.2), make_run(0.03, support=0.001, confidence=0.1)])
        self.assertEqual((best["min_support"], best["min_confidence"]), (0.001, 0.1))
        self.assertLess(selection_key(make_run(0.03, max_len=2)), selection_key(make_run(0.03, max_len=3)))

    def test_filter_keeps_rule_one_ulp_below_threshold(self):
        # mlxtend chia hai float nên confidence đúng bằng ngưỡng có thể thấp hơn ngưỡng một ulp.
        rules = pd.DataFrame({"confidence": [np.nextafter(0.3, 0), 0.29]})
        self.assertEqual(len(filter_by_confidence(rules, 0.3)), 1)

    def test_best_rankings_belong_to_selected_config_when_it_runs_second(self):
        payload, rankings = self.run_two_level({2: (0.5, 0.25)}, (0.5,))
        self.assertEqual(payload["selected"]["min_support"], 0.25)
        self.assertEqual((rankings[7][0], rankings[8][0]), (2, 4))

    def test_min_count_rounds_up(self):
        # 0,3 x 7 giao dịch = 2,1: cần ít nhất 3 giao dịch.
        payload, _ = self.run_seven_users({2: (0.3,)}, (0.5,))
        self.assertEqual(payload["levels"][0]["min_count"], 3)

    def test_payload_declares_validation_split(self):
        payload, _ = self.run_seven_users({2: (0.4,)}, (0.5,))
        self.assertEqual(payload["split"], "validation")

    def test_confidence_order_does_not_change_results(self):
        ascending, _ = self.run_two_level({2: (0.25,)}, (0.5, 0.9))
        descending, _ = self.run_two_level({2: (0.25,)}, (0.9, 0.5))
        key = lambda payload: sorted((r["min_confidence"], r["n_rules"]) for r in payload["runs"])
        self.assertEqual(key(descending), key(ascending))
        self.assertEqual(key(ascending), [(0.5, 4), (0.9, 2)])

    def test_comparison_uses_the_selected_run_and_appears_in_markdown(self):
        payload, _ = self.run_two_level({2: (0.25, 0.5)}, (0.5,))
        ranked = sorted((r for r in payload["runs"] if r["status"] == "ok"), key=selection_key)
        ndcg = lambda run: run["metrics"]["keep_seen"]["ndcg@10"]
        self.assertNotAlmostEqual(ndcg(ranked[0]), ndcg(ranked[1]))
        popularity = payload["reference_popularity"]["keep_seen"]["ndcg@10"]
        comparison = payload["comparison"]
        self.assertAlmostEqual(comparison["selected_vs_popularity"]["mean"], ndcg(ranked[0]) - popularity)
        self.assertAlmostEqual(comparison["selected_vs_runner_up"]["mean"], ndcg(ranked[0]) - ndcg(ranked[1]))
        text = format_markdown(payload)
        self.assertIn(f"{comparison['selected_vs_popularity']['mean']:+.5f}", text)
        self.assertIn(f"{comparison['selected_vs_runner_up']['mean']:+.5f}", text)

    def test_markdown_marks_selected_rows_and_reads_columns_from_the_right_fields(self):
        payload, _ = self.run_seven_users({2: (0.2, 0.4)}, (0.5,))
        selected = payload["selected"]
        self.assertEqual(selected["min_support"], 0.4)
        text = format_markdown(payload)
        marked = [line for line in text.splitlines() if "(chọn)" in line]
        self.assertEqual(len(marked), 2)
        self.assertTrue(marked[0].startswith(f"| {selected['max_len']} | {selected['min_support']} **(chọn)** |"), marked[0])
        self.assertTrue(marked[1].startswith(f"| {selected['min_confidence']} **(chọn)** |"), marked[1])
        self.assertIn(f"max_len = {selected['max_len']}, min_support = {selected['min_support']}, "
                      f"min_confidence = {selected['min_confidence']}.", text)
        run = next(r for r in payload["runs"] if r["min_support"] == selected["min_support"])
        diagnostics, excluded = run["diagnostics"], run["metrics"]["exclude_seen"]
        percent = lambda value: f"{100 * value:.1f}%"
        self.assertIn(f"| {percent(diagnostics['users_with_rule_ratio'])} | {percent(diagnostics['target_covered_ratio'])} |", marked[0])
        self.assertTrue(marked[0].endswith(f"| {excluded['ndcg@10']:.5f} |"), marked[0])


if __name__ == "__main__":
    unittest.main()
