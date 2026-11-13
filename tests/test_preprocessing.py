"""Kiểm chứng k-core, thời gian, ranh giới holdout và khả năng tái lập."""

import csv
import gzip
import json
import tempfile
import unittest
from pathlib import Path

from src.preprocessing import (
    METADATA_FILE,
    REVIEWS_FILE,
    Review,
    build_dataset,
    filter_k_core,
    iter_training_examples,
    load_aligned_metadata,
    load_reviews,
    preprocess,
    read_records,
    sha256_file,
    verify_preprocessed,
)


class PreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_k_core_repeats_until_cascade_stabilizes(self):
        pairs = [
            ("u1", "a"), ("u1", "b"), ("u2", "b"), ("u2", "c"),
            ("u3", "c"), ("u3", "d"), ("u4", "d"),
            ("u5", "x"), ("u5", "y"), ("u6", "x"), ("u6", "y"),
        ]
        reviews = [Review(user, item, i, i + 1) for i, (user, item) in enumerate(pairs)]
        retained, rounds = filter_k_core(reviews, 2)
        self.assertGreater(len(rounds), 1)
        self.assertEqual({(row.user, row.item) for row in retained}, set(pairs[-4:]))
        self.assertEqual(filter_k_core(retained, 2), (retained, []))

    def test_timestamp_ties_keep_source_order_and_mapping_roundtrips(self):
        reviews = [
            Review("u", "c", 20, 1), Review("u", "a", 10, 2),
            Review("u", "b", 10, 3), Review("u", "d", 30, 4), Review("u", "e", 40, 5),
        ]
        data = build_dataset(reviews)
        mapping = data["id_mapping"]
        self.assertEqual(mapping["id2item"], [None, "a", "b", "c", "d", "e"])
        self.assertEqual([row["source_line"] for row in data["interactions"]], [2, 3, 1, 4, 5])
        self.assertEqual(data["train_sequences"][0]["item_ids"], [1, 2, 3])
        self.assertEqual(data["validation"][0]["target"], 4)
        self.assertEqual(data["test"][0]["history"], [1, 2, 3, 4])
        self.assertEqual(data["test"][0]["target"], 5)
        for raw_id, numeric_id in mapping["item2id"].items():
            self.assertEqual(mapping["id2item"][numeric_id], raw_id)
        self.assertEqual(mapping["id2user"], [None, "u"])

    def test_repeated_item_is_preserved_but_heldout_events_are_disjoint(self):
        reviews = [Review("u", item, i, i + 1) for i, item in enumerate(["a", "b", "a", "a", "c"])]
        data = build_dataset(reviews)
        split = data["splits"][0]
        self.assertEqual(data["train_sequences"][0]["item_ids"], [1, 2, 1])
        self.assertEqual((split["train_start"], split["train_stop"]), (0, 3))
        self.assertEqual((split["validation_index"], split["test_index"]), (3, 4))
        self.assertTrue(data["validation"][0]["target_in_history"])
        self.assertFalse(data["test"][0]["target_in_history"])
        self.assertEqual(len(data["interactions"]), 5)

    def test_history_window_does_not_truncate_mining_data_or_leak_targets(self):
        data = build_dataset([Review("u", str(i), i, i + 1) for i in range(60)])
        train = data["train_sequences"][0]["item_ids"]
        self.assertEqual(train, list(range(1, 59)))
        self.assertEqual(data["validation"][0]["history"], list(range(9, 59)))
        self.assertEqual(data["validation"][0]["seen_item_ids"], list(range(1, 59)))
        self.assertEqual(data["test"][0]["history"], list(range(10, 60)))
        examples = list(iter_training_examples(train))
        self.assertEqual(len(examples), 57)
        self.assertEqual(examples[-1], {"history": list(range(8, 58)), "target": 58})
        self.assertTrue(all(len(row["history"]) <= 50 for row in examples))
        self.assertTrue(all(row["target"] not in (59, 60) for row in examples))

    def test_invalid_and_empty_split_inputs_fail(self):
        with self.assertRaisesRegex(ValueError, "No interactions"):
            build_dataset([])
        with self.assertRaisesRegex(ValueError, "at least 3"):
            build_dataset([Review("u", "a", 1, 1)])
        with self.assertRaisesRegex(ValueError, "max_history"):
            build_dataset([], max_history=0)
        with self.assertRaisesRegex(ValueError, "min_core"):
            filter_k_core([], min_core=True)

    def test_review_parser_preserves_physical_line_numbers(self):
        path = self.root / "reviews.jsonl"
        path.write_text(
            '\n{"reviewerID":"u","asin":"a","unixReviewTime":1}\n', encoding="utf-8"
        )
        self.assertEqual(load_reviews(path), [Review("u", "a", 1, 2)])
        path.write_text('{"reviewerID":"u","asin":"a","unixReviewTime":true}\n', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "unixReviewTime"):
            load_reviews(path)

    def test_parser_accepts_legacy_literals_without_evaluating_expressions(self):
        path = self.root / "meta.jsonl"
        path.write_text("{'asin': 'a', 'title': 'Mỹ phẩm', 'flag': True}\n", encoding="utf-8")
        self.assertEqual(list(read_records(path))[0][1]["title"], "Mỹ phẩm")
        path.write_text("{'asin': str(123)}\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "invalid record"):
            list(read_records(path))

    def test_metadata_order_missing_rows_and_excluded_fields(self):
        path = self.root / "meta.jsonl"
        path.write_text(
            "{'asin': 'b', 'brand': 'B', 'imUrl': 'unused', 'related': {'also_bought': ['a']}}\n"
            "{'asin': 'a', 'title': 'A'}\n", encoding="utf-8"
        )
        mapping = {"item2id": {"a": 1, "b": 2, "c": 3}, "id2item": [None, "a", "b", "c"]}
        items = load_aligned_metadata(path, mapping)
        self.assertEqual([row["asin"] for row in items], ["a", "b", "c"])
        self.assertEqual(items[1]["metadata"], {"brand": "B"})
        self.assertTrue(items[2]["metadata_missing"])

    def test_duplicate_metadata_is_reported(self):
        path = self.root / "meta.jsonl"
        path.write_text("{'asin':'a'}\n{'asin':'a'}\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Duplicate metadata"):
            load_aligned_metadata(path, {"item2id": {"a": 1}, "id2item": [None, "a"]})

    def make_raw_fixture(self):
        raw = self.root / "raw"
        raw.mkdir()
        reviews = [
            {"reviewerID": f"u{user}", "asin": f"i{item}", "unixReviewTime": 5 - item}
            for user in range(3) for item in range(5)
        ]
        metadata = [{"asin": f"i{item}", "title": f"Product {item}"} for item in range(5)]
        specs = []
        for name, rows in ((REVIEWS_FILE, reviews), (METADATA_FILE, metadata)):
            content = "".join(json.dumps(row) + "\n" for row in rows).encode("utf-8")
            (raw / name).write_bytes(gzip.compress(content, mtime=0))
            specs.append({"filename": name, "sha256": sha256_file(raw / name)})
        manifest = self.root / "sources.json"
        manifest.write_text(json.dumps({"files": specs}), encoding="utf-8")
        return raw, manifest

    def test_pipeline_roundtrip_all_events_partitioned_and_reproducible(self):
        raw, manifest = self.make_raw_fixture()
        output, splits = self.root / "processed", self.root / "splits"
        stats = preprocess(raw, output, splits, min_core=3, max_history=2, source_manifest=manifest)
        self.assertEqual((stats["users"], stats["items"], stats["train_interactions"]), (3, 5, 9))
        self.assertEqual(stats["retained_interactions"], 15)
        self.assertEqual(stats["missing_metadata_items"], 0)
        with (splits / "leave_last_out.csv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        covered = []
        for row in rows:
            start, stop, val, test = [int(row[key]) for key in SPLIT_KEYS]
            self.assertEqual(stop, val)
            self.assertEqual(test, val + 1)
            covered.extend([*range(start, stop), val, test])
        self.assertEqual(sorted(covered), list(range(15)))
        self.assertEqual(len(set(covered)), 15)
        saved_manifest = json.loads((splits / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(verify_preprocessed(output, splits), saved_manifest)
        for name, reference in saved_manifest["artifacts"].items():
            directory = splits if name == "leave_last_out.csv" else output
            self.assertEqual(sha256_file(directory / name), reference["sha256"])

        other_output, other_splits = self.root / "processed2", self.root / "splits2"
        preprocess(raw, other_output, other_splits, min_core=3, max_history=2, source_manifest=manifest)
        for original, repeated in ((output, other_output), (splits, other_splits)):
            for path in original.iterdir():
                self.assertEqual(path.read_bytes(), (repeated / path.name).read_bytes(), path.name)

        with (output / "train_sequences.jsonl").open("a", encoding="utf-8") as stream:
            stream.write("{}\n")
        with self.assertRaisesRegex(ValueError, "does not match split manifest"):
            verify_preprocessed(output, splits)

    def test_changed_raw_data_rejected_before_outputs_are_written(self):
        raw, manifest = self.make_raw_fixture()
        with (raw / REVIEWS_FILE).open("ab") as stream:
            stream.write(b"changed")
        output, splits = self.root / "processed", self.root / "splits"
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            preprocess(raw, output, splits, min_core=3, source_manifest=manifest)
        self.assertFalse(output.exists())
        self.assertFalse(splits.exists())


SPLIT_KEYS = ("train_start", "train_stop", "validation_index", "test_index")


if __name__ == "__main__":
    unittest.main()
