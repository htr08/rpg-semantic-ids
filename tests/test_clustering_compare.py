"""Kiểm tra Ward, DBSCAN, điểm gãy k-distance và các độ đo trên ví dụ tính tay và dữ liệu tổng hợp."""

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score

from src.clustering_compare import (
    NOISE,
    eps_candidates,
    evaluate_labels,
    format_markdown,
    k_distances,
    knee_index,
    main,
    run_comparison,
    run_dbscan,
    run_hierarchical,
)


def make_blobs(n_per_cluster=40, dim=20, n_clusters=3, seed=0):
    """Các cụm quanh tâm ở các trục tọa độ khác nhau, nhiễu nhỏ; trả về (ma trận, nhãn thật)."""
    rng = np.random.default_rng(seed)
    centers = np.eye(dim)[:n_clusters] * 10.0
    points = np.vstack([c + rng.normal(scale=0.3, size=(n_per_cluster, dim)) for c in centers])
    return points, np.repeat(np.arange(n_clusters), n_per_cluster)


class KDistanceTests(unittest.TestCase):
    def test_k_distances_hand_computed(self):
        # Ba điểm trên đường thẳng ở 0, 1, 3 (Euclid). MinPts = 2 tính cả điểm đang xét: láng giềng thứ 2 là điểm gần nhất khác.
        matrix = np.array([[0.0], [1.0], [3.0]])
        np.testing.assert_allclose(k_distances(matrix, 2, metric="euclidean"), [1.0, 1.0, 2.0])
        # MinPts = 3: điểm xa nhất trong 3 điểm: 3, 2, 3 -> sắp tăng dần.
        np.testing.assert_allclose(k_distances(matrix, 3, metric="euclidean"), [2.0, 3.0, 3.0])

    def test_k_distances_chunking_matches_full_computation(self):
        # 2.500 điểm vượt kích thước một khối (1.000), so với tính trực tiếp bằng ma trận khoảng cách đầy đủ.
        matrix = np.random.default_rng(0).normal(size=(2500, 4))
        squared = (matrix ** 2).sum(axis=1)
        full = np.sqrt(np.maximum(squared[:, None] + squared[None, :] - 2 * matrix @ matrix.T, 0.0))
        expected = np.sort(np.sort(full, axis=1)[:, 2])  # MinPts = 3: điểm đang xét + 2 láng giềng
        np.testing.assert_allclose(k_distances(matrix, 3, metric="euclidean"), expected, atol=1e-6)

    def test_core_points_equal_points_with_kdistance_at_most_eps(self):
        # Quy ước MinPts của DBSCAN (tính cả điểm đang xét) khớp cột min_samples - 1 của k-distance, với metric cosine.
        from sklearn.cluster import DBSCAN

        matrix = np.random.default_rng(1).normal(size=(300, 5))
        min_samples = 6
        curve = k_distances(matrix, min_samples)
        eps = float(np.quantile(curve, 0.6))
        model = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine").fit(matrix)
        self.assertEqual(len(model.core_sample_indices_), int((curve <= eps).sum()))

    def test_k_distances_rejects_bad_min_samples(self):
        with self.assertRaises(ValueError):
            k_distances(np.zeros((3, 2)), 4)
        with self.assertRaises(ValueError):
            k_distances(np.zeros((3, 2)), 1)

    def test_knee_index_on_hockey_stick(self):
        # 90 giá trị gần phẳng rồi 10 giá trị dốc: điểm gãy nằm quanh vị trí chuyển.
        curve = np.concatenate([np.linspace(0.0, 0.1, 90), np.linspace(0.2, 5.0, 10)])
        self.assertTrue(85 <= knee_index(curve) <= 92)

    def test_knee_index_straight_line_is_not_error_but_flat_curve_is(self):
        self.assertIsInstance(knee_index(np.linspace(0, 1, 10)), int)
        with self.assertRaises(ValueError):
            knee_index(np.ones(10))
        with self.assertRaises(ValueError):
            knee_index(np.array([1.0, 2.0]))

    def test_eps_candidates_names_and_order(self):
        curve = np.concatenate([np.linspace(0.0, 0.1, 90), np.linspace(0.2, 5.0, 10)])
        names = [name for name, _ in eps_candidates(curve, (0.5, 0.9))]
        self.assertEqual(names, ["knee", "q0.50", "q0.90"])


class EvaluateLabelsTests(unittest.TestCase):
    def test_noise_excluded_from_metrics_and_reported(self):
        # Hai cụm đúng vuông góc nhau + 2 điểm nhiễu: silhouette cosine của hai cụm = 1, tỉ lệ nhiễu = 2/6.
        matrix = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [1.0, 1.0], [-1.0, -1.0]])
        labels = np.array([0, 0, 1, 1, NOISE, NOISE])
        result = evaluate_labels(matrix, labels)
        self.assertEqual(result["n_clusters"], 2)
        self.assertAlmostEqual(result["noise_ratio"], 2 / 6, places=5)
        self.assertAlmostEqual(result["silhouette"], 1.0)
        self.assertAlmostEqual(result["largest_cluster_fraction"], 2 / 6, places=5)

    def test_non_contiguous_labels_after_noise_removal(self):
        # Nhãn 0 và 2 (nhãn 1 vắng): không được đếm cụm rỗng.
        matrix = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
        result = evaluate_labels(matrix, np.array([0, 0, 2, 2]))
        self.assertEqual(result["n_clusters"], 2)
        self.assertEqual(result["size_min"], 2)

    def test_single_cluster_gives_none_metrics(self):
        matrix = np.random.default_rng(0).normal(size=(10, 3))
        result = evaluate_labels(matrix, np.zeros(10, dtype=int))
        self.assertEqual(result["n_clusters"], 1)
        self.assertIsNone(result["silhouette"])
        self.assertIsNone(result["davies_bouldin"])
        self.assertEqual(result["largest_cluster_fraction"], 1.0)

    def test_all_noise(self):
        matrix = np.random.default_rng(0).normal(size=(5, 3))
        result = evaluate_labels(matrix, np.full(5, NOISE))
        self.assertEqual(result["n_clusters"], 0)
        self.assertEqual(result["noise_ratio"], 1.0)
        self.assertIsNone(result["silhouette"])

    def test_all_singletons_gives_none_metrics(self):
        # n cụm trên n điểm: silhouette không xác định, không được ném lỗi.
        matrix = np.random.default_rng(0).normal(size=(5, 3))
        result = evaluate_labels(matrix, np.arange(5))
        self.assertEqual(result["n_clusters"], 5)
        self.assertIsNone(result["silhouette"])
        self.assertIsNone(result["davies_bouldin"])


class AlgorithmTests(unittest.TestCase):
    def test_ward_recovers_separated_clusters(self):
        points, truth = make_blobs()
        labels, _ = run_hierarchical(points, 3)
        self.assertEqual(adjusted_rand_score(truth, labels), 1.0)

    def test_ward_rejects_bad_k(self):
        with self.assertRaises(ValueError):
            run_hierarchical(np.zeros((3, 2)), 4)
        with self.assertRaises(ValueError):
            run_hierarchical(np.zeros((3, 2)), 1)

    def test_dbscan_finds_blobs_and_marks_outlier_as_noise(self):
        points, truth = make_blobs()
        points = np.vstack([points, np.full((1, points.shape[1]), 50.0)])  # điểm lạc xa
        labels, _ = run_dbscan(points, eps=2.0, min_samples=5, metric="euclidean")
        self.assertEqual(labels[-1], NOISE)
        self.assertEqual(adjusted_rand_score(truth, labels[:-1]), 1.0)


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        points, _ = make_blobs(n_per_cluster=30, dim=12)
        self.points = points

    def test_run_comparison_structure_and_determinism(self):
        kwargs = dict(n_components=5, ks=(3, 6), min_samples_list=(4,), eps_quantiles=(0.5,),
                      extra_dbscan_dims=(3,), sample_size=60, n_init=2, seed=0)
        first, curves = run_comparison(self.points, **kwargs)
        second, _ = run_comparison(self.points, **kwargs)
        for result in (first, second):
            for row in result["sample_algorithms"] + result["dbscan"]:
                row.pop("seconds")
        self.assertEqual(first, second)
        self.assertEqual(first["sample_size"], 60)
        self.assertEqual([(r["algorithm"], r["k"]) for r in first["sample_algorithms"]],
                         [("ward", 3), ("kmeans", 3), ("ward", 6), ("kmeans", 6)])
        self.assertEqual({r["pca_dim"] for r in first["dbscan"]}, {5, 3})
        self.assertEqual(set(curves), {(5, 4), (3, 4)})
        for row in first["sample_algorithms"]:
            self.assertEqual(row["noise_ratio"], 0.0)

    def test_extra_dim_equal_to_main_dim_is_not_repeated(self):
        result, curves = run_comparison(self.points, n_components=5, ks=(3,), min_samples_list=(4,), eps_quantiles=(),
                                        extra_dbscan_dims=(5,), sample_size=60, n_init=1, seed=0)
        self.assertEqual(len(result["dbscan"]), 1)  # chỉ "knee"
        self.assertEqual(set(curves), {(5, 4)})

    def test_format_markdown_has_both_tables(self):
        result, _ = run_comparison(self.points, n_components=5, ks=(3,), min_samples_list=(4,), eps_quantiles=(0.5,),
                                   extra_dbscan_dims=(), sample_size=60, n_init=1, seed=0)
        text = format_markdown(result)
        self.assertIn("| ward | 3 |", text)
        self.assertIn("| kmeans | 3 |", text)
        self.assertIn("| 5 | 4 | knee |", text)

    def test_main_writes_outputs_and_reports_missing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.save(root / "emb.npy", self.points)
            args = ["--embeddings", str(root / "emb.npy"), "--components", "5", "--ks", "3", "--min-samples", "4",
                    "--extra-dbscan-dims", "3",
                    "--sample-size", "60", "--n-init", "1", "--output", str(root / "out" / "r.json"),
                    "--markdown", str(root / "out" / "r.md"), "--figure", ""]
            self.assertEqual(main(args), 0)
            self.assertEqual(json.loads((root / "out" / "r.json").read_text(encoding="utf-8"))["pca_dim"], 5)
            self.assertTrue((root / "out" / "r.md").read_text(encoding="utf-8").startswith("# So sánh phân cụm"))
            self.assertEqual(main(["--embeddings", str(root / "missing.npy"), "--figure", ""]), 1)
            self.assertEqual(sorted(os.listdir(root / "out")), ["r.json", "r.md"])  # --figure "" không sinh hình

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "CI không cài matplotlib")
    def test_main_writes_figure_when_requested(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.save(root / "emb.npy", self.points)
            figure = root / "fig" / "kdist.png"
            args = ["--embeddings", str(root / "emb.npy"), "--components", "5", "--ks", "3", "--min-samples", "4",
                    "--extra-dbscan-dims", "3", "--sample-size", "60", "--n-init", "1", "--output", str(root / "r.json"),
                    "--markdown", str(root / "r.md"), "--figure", str(figure)]
            self.assertEqual(main(args), 0)
            self.assertGreater(figure.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
