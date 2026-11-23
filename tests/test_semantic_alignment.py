"""Kiểm tra đối chiếu luật ↔ semantic ID và k-means ↔ token trên ví dụ tính tay và dữ liệu tổng hợp."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy import sparse

from src.semantic_alignment import (
    format_markdown,
    kmeans_token_alignment,
    load_semantic_ids,
    null_threshold,
    overlap_summary,
    permutation_test,
    random_item_pairs,
    rule_alignment,
    select_top_pairs,
    shared_token_counts,
    within_brand_permutation_test,
)

IDS = np.array([
    [1, 2, 3, 4],  # item 1
    [1, 2, 9, 9],  # item 2: trùng 2 vị trí với item 1
    [5, 6, 7, 8],  # item 3: không trùng item 1
    [1, 2, 3, 4],  # item 4: trùng hoàn toàn item 1
])


class SharedTokenTests(unittest.TestCase):
    def test_shared_token_counts_by_hand(self):
        counts = shared_token_counts(IDS, [[1, 2], [1, 3], [1, 4], [2, 3]])
        np.testing.assert_array_equal(counts, [2, 0, 4, 0])

    def test_only_same_position_counts(self):
        ids = np.array([[1, 2], [2, 1]])
        self.assertEqual(shared_token_counts(ids, [[1, 2]])[0], 0)

    def test_rejects_item_id_out_of_range(self):
        for bad in ([[0, 1]], [[1, 5]]):
            with self.assertRaises(ValueError):
                shared_token_counts(IDS, bad)

    def test_empty_pairs(self):
        self.assertEqual(shared_token_counts(IDS, np.empty((0, 2))).shape, (0,))


class RandomPairsTests(unittest.TestCase):
    def test_pairs_distinct_inside_pool_and_deterministic(self):
        pool = [10, 20, 30]
        pairs = random_item_pairs(pool, 1000, seed=1)
        self.assertTrue((pairs[:, 0] != pairs[:, 1]).all())
        self.assertTrue(set(pairs.ravel().tolist()) <= set(pool))
        np.testing.assert_array_equal(pairs, random_item_pairs(pool, 1000, seed=1))

    def test_two_item_pool_always_swaps(self):
        pairs = random_item_pairs([5, 7], 50)
        self.assertTrue(((pairs[:, 0] + pairs[:, 1]) == 12).all())

    def test_duplicates_in_pool_ignored(self):
        pairs = random_item_pairs([4, 4, 9], 100)
        self.assertTrue((pairs[:, 0] != pairs[:, 1]).all())

    def test_rejects_single_item_pool(self):
        with self.assertRaises(ValueError):
            random_item_pairs([3, 3], 5)

    def test_uniform_over_ordered_pairs(self):
        pairs = random_item_pairs([1, 2, 3, 4], 60000, seed=3)
        frequency = np.bincount(pairs[:, 0] * 10 + pairs[:, 1], minlength=50)
        used = frequency[[12, 13, 14, 21, 23, 24, 31, 32, 34, 41, 42, 43]]
        self.assertTrue((np.abs(used / 60000 - 1 / 12) < 0.01).all())


class ThresholdTests(unittest.TestCase):
    def test_null_threshold_by_hand(self):
        # 100 cặp: 80 có 0, 15 có 1, 5 có 2 token chung. ≥1: 0,20 > 0,05; ≥2: 0,05 ≤ 0,05.
        counts = np.array([0] * 80 + [1] * 15 + [2] * 5)
        self.assertEqual(null_threshold(counts), 2)
        self.assertEqual(null_threshold(counts, alpha=0.2), 1)

    def test_null_threshold_when_all_pairs_share(self):
        self.assertEqual(null_threshold(np.full(10, 3)), 4)

    def test_overlap_summary_rates_by_hand(self):
        null = np.array([0] * 90 + [1] * 8 + [2] * 2)
        counts = np.array([0, 1, 2, 2, 3])
        summary = overlap_summary(counts, null, threshold=2)
        self.assertEqual(summary["n_pairs"], 5)
        self.assertAlmostEqual(summary["mean_shared"], 1.6)
        self.assertEqual(summary["median_shared"], 2.0)
        self.assertAlmostEqual(summary["zero_shared_rate"], 0.2)
        self.assertAlmostEqual(summary["null_zero_shared_rate"], 0.9)
        self.assertAlmostEqual(summary["null_mean_shared"], 0.12)
        self.assertAlmostEqual(summary["rate_at_threshold"], 0.6)
        self.assertAlmostEqual(summary["null_rate_at_threshold"], 0.02)
        self.assertAlmostEqual(summary["rate_ratio"], 30.0)
        self.assertEqual(summary["rate_by_threshold"]["1"], 0.8)
        self.assertEqual(summary["rate_by_threshold"]["3"], 0.2)

    def test_overlap_summary_with_zero_null_rate(self):
        summary = overlap_summary(np.array([1, 2]), np.zeros(10, dtype=int), threshold=1)
        self.assertIsNone(summary["rate_ratio"])


class PermutationTests(unittest.TestCase):
    def test_pairs_that_share_everything_have_small_p(self):
        # 40 item xếp thành 20 cặp; hai item trong cặp có ID giống hệt, giữa các cặp ID khác nhau.
        ids = np.repeat(np.arange(20)[:, None] * np.ones((1, 4), dtype=int), 2, axis=0)
        pairs = [[2 * i + 1, 2 * i + 2] for i in range(20)]
        result = permutation_test(ids, pairs, threshold=2, n_perm=500, seed=1)
        self.assertEqual(result["observed_mean_shared"], 4.0)
        self.assertLess(result["permuted_mean_shared"], 1.0)
        self.assertAlmostEqual(result["p_value_mean"], 1 / 501)
        self.assertAlmostEqual(result["p_value_rate"], 1 / 501)
        self.assertEqual(result["n_items"], 40)

    def test_constant_ids_give_p_one(self):
        ids = np.zeros((6, 3), dtype=int)
        result = permutation_test(ids, [[1, 2], [3, 4], [1, 5]], threshold=1, n_perm=50)
        self.assertEqual(result["p_value_mean"], 1.0)
        self.assertEqual(result["p_value_rate"], 1.0)
        self.assertEqual(result["n_items"], 5)

    def test_permuted_mean_matches_random_pairs_in_pool(self):
        # Pool 3 item (1, 2, 3) có ID 0, 0, 1: mọi cặp khác nhau trùng token với xác suất 1/3.
        ids = np.array([[0], [0], [1], [1]])
        result = permutation_test(ids, [[1, 2], [1, 3]], threshold=1, n_perm=2000, seed=0)
        self.assertAlmostEqual(result["permuted_mean_shared"], 1 / 3, delta=0.04)

    def test_rejects_empty_pairs_and_zero_permutations(self):
        with self.assertRaises(ValueError):
            permutation_test(IDS, np.empty((0, 2)), 1)
        with self.assertRaises(ValueError):
            permutation_test(IDS, [[1, 2]], 1, n_perm=0)


class WithinBrandPermutationTests(unittest.TestCase):
    def setUp(self):
        # Brand X: 6 item (1..6), ID giống nhau theo cặp (1, 2), (3, 4), (5, 6); brand Y: 2 item (7, 8) ID khác nhau.
        self.metadata = {i: {"brand": "X" if i <= 6 else "Y"} for i in range(1, 9)}
        self.ids = np.array([[0, 0], [0, 0], [1, 1], [1, 1], [2, 2], [2, 2], [5, 5], [6, 6]])

    def test_pairs_matching_their_twin_beat_within_brand_shuffle(self):
        pairs = [[1, 2], [3, 4], [5, 6]]
        result = within_brand_permutation_test(self.ids, pairs, self.metadata, threshold=2, n_perm=2000, seed=0)
        self.assertEqual(result["observed_mean_shared"], 2.0)
        # Hoán vị trong brand X (6 item, 3 cặp ID giống nhau): xác suất một cặp bất kỳ trùng hoàn toàn là 1/5.
        self.assertAlmostEqual(result["permuted_mean_shared"], 2 * 0.2, delta=0.05)
        # P(cả ba cặp cùng trùng hoàn toàn khi hoán vị) = 3! · 2^3 / 6! = 1/15; p-value ước lượng bằng (1 + số lần) / (1 + n_perm).
        self.assertAlmostEqual(result["p_value_mean"], 1 / 15, delta=0.02)
        self.assertEqual(result["n_brands"], 1)

    def test_permutation_stays_inside_brand(self):
        # Brand Y chỉ có item 7, 8 với ID khác nhau: hoán vị trong brand không bao giờ làm cặp (7, 8) trùng token.
        result = within_brand_permutation_test(self.ids, [[7, 8]], self.metadata, threshold=1, n_perm=100)
        self.assertEqual(result["permuted_mean_shared"], 0.0)
        self.assertEqual(result["observed_mean_shared"], 0.0)
        self.assertEqual(result["p_value_mean"], 1.0)

    def test_rejects_pairs_with_different_or_missing_brand(self):
        metadata = {1: {"brand": "X"}, 2: {"brand": "Y"}, 3: {"brand": None}, 4: {"brand": "x"}}
        for pair in ([1, 2], [1, 3]):
            with self.subTest(pair=pair), self.assertRaises(ValueError):
                within_brand_permutation_test(np.zeros((4, 2), dtype=int), [pair], metadata, 1, n_perm=5)
        within_brand_permutation_test(np.zeros((4, 2), dtype=int), [[1, 4]], metadata, 1, n_perm=5)  # casefold

    def test_rejects_empty_pairs(self):
        with self.assertRaises(ValueError):
            within_brand_permutation_test(self.ids, np.empty((0, 2)), self.metadata, 1)


class LoadSemanticIdsTests(unittest.TestCase):
    def _load(self, array, **kwargs):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "ids.npy"
            np.save(path, array)
            return load_semantic_ids(path, **kwargs)

    def test_valid(self):
        self.assertEqual(self._load(IDS, n_items=4).shape, (4, 4))

    def test_rejects_bad_inputs(self):
        for array, kwargs in (
            (np.zeros(4, dtype=int), {}),
            (np.zeros((2, 2), dtype=float), {}),
            (np.array([[0, 256]]), {}),
            (np.array([[-1, 0]]), {}),
            (IDS, {"n_items": 5}),
            (np.empty((0, 4), dtype=int), {}),
        ):
            with self.subTest(array=array.shape, kwargs=kwargs), self.assertRaises(ValueError):
                self._load(array, **kwargs)


def make_matrix(rows, n_items):
    """Ma trận thưa giao dịch x item; cột j ứng với item_id j + 1."""
    data = [(r, item - 1) for r, items in enumerate(rows) for item in items]
    return sparse.csr_matrix(
        (np.ones(len(data), dtype=bool), ([r for r, _ in data], [c for _, c in data])), shape=(len(rows), n_items)
    )


class RuleAlignmentTests(unittest.TestCase):
    def setUp(self):
        # 20 giao dịch: {1, 2} cùng nhau 6 lần; {3, 4} 4 lần; item 5 một mình. min_support 0,1 tương ứng count ≥ 2.
        rows = [[1, 2]] * 6 + [[3, 4]] * 4 + [[1], [2], [3], [5]] * 2 + [[5]] * 2
        self.matrix = make_matrix(rows, 5)
        self.ids = np.array([[1, 1], [1, 1], [2, 3], [2, 4], [9, 9]])

    def test_select_top_pairs_unique_pairs_sorted_by_lift(self):
        rules, n_available = select_top_pairs(self.matrix, 0.1, 0.1, 10)
        self.assertEqual(n_available, 2)
        self.assertEqual({frozenset(a | c) for a, c in zip(rules["antecedents"], rules["consequents"])},
                         {frozenset({1, 2}), frozenset({3, 4})})
        self.assertTrue((rules["lift"].diff().dropna() <= 1e-12).all())

    def test_select_top_pairs_respects_top_n(self):
        rules, n_available = select_top_pairs(self.matrix, 0.1, 0.1, 1)
        self.assertEqual((len(rules), n_available), (1, 2))

    def test_select_top_pairs_without_rules(self):
        rules, n_available = select_top_pairs(self.matrix, 0.9, 0.1, 5)
        self.assertEqual((len(rules), n_available), (0, 0))

    def test_rule_alignment_counts_by_hand(self):
        metadata = {i: {"brand": b} for i, b in {1: "A", 2: "a", 3: "B", 4: None, 5: "C"}.items()}
        result = rule_alignment(self.ids, self.matrix, metadata, 0.1, top_n=10, n_null=500, n_perm=50)
        self.assertEqual(result["n_pairs_used"], 2)
        self.assertEqual(result["n_items_in_pairs"], 4)
        self.assertEqual(result["n_identical_id_pairs"], 1)  # cặp (1, 2) giống hệt ID
        # Cặp (1, 2) có 2 token chung, cặp (3, 4) có 1.
        self.assertAlmostEqual(result["vs_random_catalog"]["mean_shared"], 1.5)
        self.assertEqual(result["permutation"]["n_perm"], 50)
        by_brand = result["by_brand"]
        self.assertEqual(by_brand["same_brand"]["n_pairs"], 1)  # (1, 2): "A" và "a"
        self.assertEqual(by_brand["missing_brand"]["n_pairs"], 1)  # (3, 4): item 4 thiếu brand
        self.assertEqual(by_brand["different_brand"]["n_pairs"], 0)
        self.assertIsNone(by_brand["different_brand"]["mean_shared"])
        self.assertAlmostEqual(by_brand["same_brand"]["mean_shared"], 2.0)
        self.assertEqual(by_brand["same_brand"]["permutation_within_brand"]["n_perm"], 50)
        self.assertNotIn("permutation_within_brand", by_brand["missing_brand"])

    def test_rule_alignment_without_rules_and_without_metadata(self):
        empty = rule_alignment(self.ids, self.matrix, None, 0.9, top_n=10, n_null=10, n_perm=5)
        self.assertEqual(empty["n_pairs_used"], 0)
        self.assertNotIn("vs_random_catalog", empty)
        result = rule_alignment(self.ids, self.matrix, None, 0.1, top_n=10, n_null=50, n_perm=5)
        self.assertNotIn("by_brand", result)

    def test_rule_alignment_rejects_invalid_arguments(self):
        for kwargs in ({"top_n": 0}, {"top_n": -1}, {"n_null": 0}, {"n_perm": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                rule_alignment(self.ids, self.matrix, None, 0.1, **{"top_n": 5, "n_null": 5, "n_perm": 5, **kwargs})
        with self.assertRaises(ValueError):
            rule_alignment(self.ids[:4], self.matrix, None, 0.1, top_n=5, n_null=5, n_perm=5)


class KMeansAlignmentTests(unittest.TestCase):
    def test_ari_one_when_token_equals_cluster_and_zero_for_constant_token(self):
        rng = np.random.default_rng(0)
        centers = np.eye(6)[:3] * 10
        embeddings = np.vstack([c + rng.normal(scale=0.1, size=(30, 6)) for c in centers])
        truth = np.repeat(np.arange(3), 30)
        ids = np.column_stack([truth, np.zeros_like(truth), rng.integers(0, 3, size=len(truth))])
        result = kmeans_token_alignment(embeddings, ids, k=3, n_components=3, n_init=3, seeds=(1, 2))
        self.assertAlmostEqual(result["ari_first_token_mean"], 1.0)
        self.assertAlmostEqual(result["ari_by_position_mean"][1], 0.0)  # token hằng
        self.assertLess(abs(result["ari_by_position_mean"][2]), 0.1)  # token ngẫu nhiên
        self.assertAlmostEqual(result["ari_between_kmeans_seeds"], 1.0)
        self.assertLess(abs(result["ari_first_token_shuffled"]["mean"]), 0.05)
        self.assertLess(result["ari_first_token_vs_other_tokens"]["max"], 0.1)
        self.assertEqual(result["ari_other_positions"]["argmax_position"], 2)

    def test_rejects_row_mismatch(self):
        with self.assertRaises(ValueError):
            kmeans_token_alignment(np.zeros((4, 3)), np.zeros((5, 2), dtype=int), k=2, n_components=2)

    def test_rejects_empty_seeds(self):
        with self.assertRaises(ValueError):
            kmeans_token_alignment(np.zeros((4, 3)), np.zeros((4, 2), dtype=int), k=2, n_components=2, seeds=())

    def test_markdown_contains_both_sections(self):
        rules = [{"min_support": 0.001, "n_pairs_used": 0, "n_unique_pairs_available": 0}]
        kmeans = {"k": 3, "pca_dim": 2, "pca_explained_variance_ratio": 0.5, "n_init": 1, "seeds": [1],
                  "ari_first_token_by_seed": [0.1], "ari_first_token_mean": 0.1, "ari_other_positions": None,
                  "ari_first_token_shuffled": {"n_shuffle": 2, "mean": 0.0, "std": 0.0}}
        text = format_markdown({"n_items": 5, "n_positions": 2, "codebook_size": 256, "rules": rules, "kmeans": kmeans})
        self.assertIn("ARI với token đầu, trung bình", text)
        self.assertIn("| 0.001 | 0 / 0 |", text)


if __name__ == "__main__":
    unittest.main()
