"""Kiểm tra thống kê EDA trên tập nhỏ có đáp án biết trước."""

import gzip
import json
import tempfile
import unittest
from pathlib import Path

from src.eda import analyze_data, clean_text, describe
from src.preprocessing import METADATA_FILE, REVIEWS_FILE, preprocess, sha256_file


class EdaTests(unittest.TestCase):
    def test_percentiles_missing_text_and_html(self):
        self.assertEqual(describe([1, 2, 3, 4])["median"], 2.5)
        self.assertAlmostEqual(describe([1, 2, 3, 4])["p90"], 3.7)
        self.assertIsNone(describe([])["mean"])
        self.assertEqual(clean_text(["<b>A</b>", " B &amp; C "]), "A B & C")
        self.assertEqual(clean_text(None), "")

    def test_popularity_uses_train_and_examples_match_catalog(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            raw.mkdir()
            reviews = [
                {"reviewerID": f"u{user}", "asin": item, "unixReviewTime": time}
                for user in range(3) for time, item in enumerate(("a", "b", "c"), start=1)
            ]
            metadata = [
                {"asin": "a", "title": "A"},
                {"asin": "b", "title": "B", "brand": "Brand", "description": " "},
                {"asin": "c", "title": "C", "categories": [["Beauty", "Skin"]], "description": "Two words"},
            ]
            specs = []
            for name, rows in ((REVIEWS_FILE, reviews), (METADATA_FILE, metadata)):
                (raw / name).write_bytes(gzip.compress("".join(json.dumps(row) + "\n" for row in rows).encode(), mtime=0))
                specs.append({"filename": name, "sha256": sha256_file(raw / name)})
            sources = root / "sources.json"
            sources.write_text(json.dumps({"files": specs}), encoding="utf-8")
            preprocess(raw, root / "processed", root / "splits", min_core=3, source_manifest=sources)
            result = analyze_data(root / "processed", root / "splits")
            stats = result["summary"]
            self.assertEqual(result["train_counts"], {1: 3, 2: 0, 3: 0})
            self.assertEqual(stats["head_train_interaction_share"], 1)
            self.assertEqual(stats["train_interactions"], 3)
            self.assertEqual(stats["head_items"], 1)
            self.assertEqual({row["asin"] for row in result["examples"]}, {"a", "b", "c"})
            self.assertEqual(stats["sparsity"], 0)
            self.assertEqual(stats["history_window"], {
                "max_history": 50, "validation_users_truncated": 0, "test_users_truncated": 0,
            })
            self.assertEqual(stats["metadata_missing_items"], 0)
            self.assertEqual(stats["missing_fields"], {"title": 0, "brand": 2, "categories": 2, "description": 2})
            self.assertAlmostEqual(stats["missing_field_ratios"]["brand"], 2 / 3)
            self.assertEqual(stats["missing_title_and_description"], 0)
            self.assertEqual(stats["missing_description_with_title"], 2)


if __name__ == "__main__":
    unittest.main()
