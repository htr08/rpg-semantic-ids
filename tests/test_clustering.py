"""Kiểm tra PCA, k-means, silhouette trên ví dụ tính tay và dữ liệu tổng hợp có cụm rõ ràng."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score

from src.clustering import (
    cluster_size_summary,
    fit_pca,
    l2_normalize,
    load_embeddings,
    main,
    run_kmeans,
    run_trial,
    sample_indices,
    silhouette_on_sample,
)


def make_blobs(n_per_cluster=40, dim=20, n_clusters=3, seed=0):
    """Các cụm quanh tâm ở các trục tọa độ khác nhau, nhiễu nhỏ; trả về (ma trận, nhãn thật)."""
    rng = np.random.default_rng(seed)
    centers = np.eye(dim)[:n_clusters] * 10.0
    points = np.vstack([c + rng.normal(scale=0.3, size=(n_per_cluster, dim)) for c in centers])
    labels = np.repeat(np.arange(n_clusters), n_per_cluster)
    return points, labels


class ClusteringTests(unittest.TestCase):
    def test_l2_normalize_unit_rows_and_zero_row_kept(self):
        result = l2_normalize(np.array([[3.0, 4.0], [0.0, 0.0]]))
        np.testing.assert_allclose(result, [[0.6, 0.8], [0.0, 0.0]])

    def test_pca_shape_and_variance_ratio(self):
        points, _ = make_blobs()
        reduced, kept = fit_pca(points, n_components=5)
        self.assertEqual(reduced.shape, (len(points), 5))
        self.assertTrue(0.0 < kept <= 1.0)
        # Ba cụm nằm trên ba trục nên hai thành phần đầu đã giữ gần hết phương sai.
        _, kept_full = fit_pca(points, n_components=20)
        self.assertAlmostEqual(kept_full, 1.0, places=6)

    def test_pca_rejects_invalid_components(self):
        with self.assertRaises(ValueError):
            fit_pca(np.zeros((10, 4)), n_components=5)

    def test_kmeans_recovers_separated_clusters(self):
        points, truth = make_blobs()
        labels, inertia, _ = run_kmeans(points, k=3, n_init=5, seed=1)
        self.assertEqual(adjusted_rand_score(truth, labels), 1.0)
        self.assertGreater(inertia, 0.0)

    def test_kmeans_rejects_k_larger_than_n(self):
        with self.assertRaises(ValueError):
            run_kmeans(np.zeros((3, 2)), k=4)

    def test_sample_indices_deterministic_sorted_unique(self):
        first = sample_indices(100, 10, seed=7)
        second = sample_indices(100, 10, seed=7)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(np.unique(first)), 10)
        np.testing.assert_array_equal(first, np.sort(first))
        np.testing.assert_array_equal(sample_indices(5, 10), np.arange(5))

    def test_silhouette_perfect_clusters_equals_one(self):
        # Trong cụm trùng nhau (cosine 0), giữa hai cụm vuông góc (cosine 1) nên s = (1 - 0) / 1 = 1.
        matrix = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
        labels = np.array([0, 0, 1, 1])
        self.assertAlmostEqual(silhouette_on_sample(matrix, labels, np.arange(4)), 1.0)

    def test_silhouette_hand_computed_four_points(self):
        # Bốn vector đơn vị ở góc 0, 60, 90, 150 độ; cụm {0, 60} và {90, 150}; khoảng cách = 1 - cos(góc).
        # Điểm 0 và 150: a = 0,5, b = (1 + 1 + sqrt(3)/2) / 2 ≈ 1,4330 -> s ≈ 0,6511.
        # Điểm 60 và 90: a = 0,5, b = (0,1340 + 1) / 2 = 0,5670 -> s ≈ 0,1181.
        # Trung bình ≈ (2 * 0,6511 + 2 * 0,1181) / 4 = 0,3846.
        angles = np.deg2rad([0, 60, 90, 150])
        matrix = np.column_stack([np.cos(angles), np.sin(angles)])
        labels = np.array([0, 0, 1, 1])
        self.assertAlmostEqual(silhouette_on_sample(matrix, labels, np.arange(4)), 0.3846, places=3)

    def test_silhouette_true_labels_beat_random_labels(self):
        points, truth = make_blobs()
        indices = np.arange(len(points))
        random_labels = np.random.default_rng(3).integers(0, 3, size=len(points))
        good = silhouette_on_sample(points, truth, indices, metric="euclidean")
        bad = silhouette_on_sample(points, random_labels, indices, metric="euclidean")
        self.assertGreater(good, 0.8)
        self.assertLess(bad, 0.1)

    def test_silhouette_requires_two_clusters_in_sample(self):
        matrix = np.random.default_rng(0).normal(size=(6, 3))
        with self.assertRaises(ValueError):
            silhouette_on_sample(matrix, np.zeros(6, dtype=int), np.arange(6))

    def test_cluster_size_summary(self):
        summary = cluster_size_summary(np.array([0, 0, 0, 2, 2, 4]))
        self.assertEqual(summary["n_clusters"], 3)  # nhãn 1 và 3 rỗng nên không tính
        self.assertEqual(summary["size_min"], 1)
        self.assertEqual(summary["size_max"], 3)
        self.assertEqual(summary["n_singletons"], 1)

    def test_run_trial_on_synthetic_embeddings(self):
        points, _ = make_blobs()
        result = run_trial(points, n_components=5, k=3, n_init=3, sample_size=50, seed=0)
        self.assertEqual(result["n_items"], 120)
        self.assertEqual(result["input_dim"], 20)
        self.assertEqual(result["silhouette_sample_size"], 50)
        self.assertGreater(result["pca_explained_variance_ratio"], 0.9)
        for name in ("kmeans_pca", "kmeans_original"):
            self.assertEqual(result[name]["n_clusters"], 3)
            self.assertGreater(result[name]["silhouette"], 0.8)

    def test_load_embeddings_rejects_bad_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.save(root / "vector.npy", np.zeros(5))
            np.save(root / "nan.npy", np.array([[1.0, np.nan]]))
            with self.assertRaises(ValueError):
                load_embeddings(root / "vector.npy")
            with self.assertRaises(ValueError):
                load_embeddings(root / "nan.npy")

    def test_main_writes_json_and_reports_missing_file(self):
        points, _ = make_blobs()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.save(root / "emb.npy", points)
            output = root / "out" / "trial.json"
            code = main(["--embeddings", str(root / "emb.npy"), "--components", "5", "--k", "3",
                         "--sample-size", "50", "--output", str(output)])
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["k"], 3)
            self.assertEqual(main(["--embeddings", str(root / "missing.npy"), "--output", str(output)]), 1)


if __name__ == "__main__":
    unittest.main()
