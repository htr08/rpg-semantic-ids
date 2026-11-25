"""Kiểm tra các kiểm tra thứ tự hàng đầu vào của src.semantic_alignment (semantic ID theo asin, embedding theo semantic ID)."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.semantic_alignment import (
    check_input_alignment,
    format_markdown,
    verify_embedding_alignment,
)


def clustered_data(n_groups=20, per_group=15, dim=16, seed=0):
    """Embedding quanh n_groups tâm; semantic ID của mỗi item là mã nhóm của nó ở mọi vị trí (như lượng tử hóa)."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(n_groups, dim)) * 5
    labels = np.repeat(np.arange(n_groups), per_group)
    embeddings = centers[labels] + rng.normal(scale=0.3, size=(len(labels), dim))
    ids = np.column_stack([labels, (labels * 7) % 256, (labels * 13) % 256])
    return embeddings.astype(np.float32), ids


class EmbeddingAlignmentTests(unittest.TestCase):
    def test_aligned_rows_pass_with_large_ratio(self):
        embeddings, ids = clustered_data()
        result = verify_embedding_alignment(embeddings, ids, n_sample=100)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["n_sample"], 100)
        self.assertAlmostEqual(result["neighbour_mean_shared"], 3.0)  # láng giềng gần nhất cùng nhóm: trùng cả 3 vị trí
        self.assertGreater(result["ratio"], 5)

    def test_shuffled_embedding_rows_are_rejected(self):
        embeddings, ids = clustered_data()
        shuffled = embeddings[np.random.default_rng(1).permutation(len(ids))]
        with self.assertRaisesRegex(ValueError, "không khớp thứ tự hàng"):
            verify_embedding_alignment(shuffled, ids, n_sample=100)

    def test_result_is_deterministic_for_a_seed(self):
        embeddings, ids = clustered_data()
        first = verify_embedding_alignment(embeddings, ids, n_sample=50, seed=3)
        self.assertEqual(first, verify_embedding_alignment(embeddings, ids, n_sample=50, seed=3))

    def test_ids_that_share_no_token_are_rejected_even_without_random_baseline(self):
        embeddings, _ = clustered_data()
        distinct = np.column_stack([np.arange(len(embeddings)) % 256, np.arange(len(embeddings)) // 256])
        with self.assertRaises(ValueError):
            verify_embedding_alignment(embeddings, distinct, n_sample=50)

    def test_rejects_row_mismatch_and_tiny_input(self):
        embeddings, ids = clustered_data()
        with self.assertRaises(ValueError):
            verify_embedding_alignment(embeddings[:-1], ids)
        with self.assertRaises(ValueError):
            verify_embedding_alignment(embeddings[:2], ids[:2])


class CheckInputAlignmentTests(unittest.TestCase):
    def setUp(self):
        self.embeddings, self.ids = clustered_data(n_groups=10, per_group=10)
        self.asins = [f"B{i:03d}" for i in range(len(self.ids))]
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name) / "ref.sem_ids"
        self.write({asin: row.tolist() for asin, row in zip(self.asins, self.ids)})

    def write(self, data):
        self.path.write_text(json.dumps(data), encoding="utf-8")

    def test_default_runs_both_checks(self):
        order, coupling = check_input_alignment(self.ids, self.embeddings, self.asins, self.path)
        self.assertEqual(order["status"], "verified")
        self.assertEqual(coupling["status"], "passed")

    def test_missing_reference_file_is_an_error_not_a_silent_skip(self):
        for missing in (None, Path(self.folder.name) / "absent.sem_ids"):
            with self.subTest(missing=missing), self.assertRaisesRegex(ValueError, "--no-row-order-check"):
                check_input_alignment(self.ids, self.embeddings, self.asins, missing)

    def test_reordered_semantic_ids_are_rejected_by_default(self):
        reordered = self.ids[np.random.default_rng(2).permutation(len(self.ids))]
        with self.assertRaisesRegex(ValueError, "hàng semantic ID khác"):
            check_input_alignment(reordered, self.embeddings, self.asins, self.path)

    def test_reordered_embeddings_are_rejected_even_when_semantic_ids_match(self):
        shuffled = self.embeddings[np.random.default_rng(2).permutation(len(self.ids))]
        with self.assertRaisesRegex(ValueError, "embedding không khớp"):
            check_input_alignment(self.ids, shuffled, self.asins, self.path)

    def test_explicit_skip_bypasses_both_checks(self):
        shuffled = self.embeddings[::-1]
        self.assertEqual(check_input_alignment(self.ids[::-1], shuffled, self.asins, None, skip=True), (None, None))


class InputsMarkdownTests(unittest.TestCase):
    BASE = {"n_items": 3, "n_positions": 2, "codebook_size": 256, "rules": [], "inputs": {
        "semantic_ids": {"file": "s.npy", "sha256": "00"}},
        "kmeans": {"k": 3, "pca_dim": 2, "pca_explained_variance_ratio": 0.5, "n_init": 1, "seeds": [1],
                   "ari_first_token_by_seed": [0.1], "ari_first_token_mean": 0.1, "ari_other_positions": None,
                   "ari_first_token_shuffled": {"n_shuffle": 2, "mean": 0.0, "std": 0.0}}}

    def test_unchecked_run_is_labelled_unchecked(self):
        text = format_markdown(self.BASE)
        self.assertIn("chưa được kiểm", text)
        self.assertNotIn("không truyền", text)

    def test_checked_run_reports_both_checks(self):
        result = {**self.BASE,
                  "semantic_id_row_order": {"file": "ref.sem_ids", "n_items_checked": 3, "status": "verified"},
                  "embedding_alignment": {"n_sample": 3, "neighbour_mean_shared": 3.08, "random_mean_shared": 0.14,
                                          "ratio": 21.9, "min_ratio": 5.0, "status": "passed"}}
        text = format_markdown(result)
        self.assertIn("3 hàng khớp từng hàng với `ref.sem_ids`", text)
        self.assertIn("láng giềng gần nhất theo cosine của 3 item", text)
        self.assertIn("tỉ số 21.9, ngưỡng 5", text)
        self.assertNotIn("chưa được kiểm", text)


if __name__ == "__main__":
    unittest.main()
