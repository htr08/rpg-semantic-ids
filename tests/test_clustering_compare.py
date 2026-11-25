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
    GROUP_A_EMBEDDING_SHA256,
    NOISE,
    eps_candidates,
    evaluate_labels,
    file_sha256,
    format_markdown,
    k_distances,
    knee_index,
    main,
    plot_kdistance,
    run_comparison,
    run_dbscan,
    run_hierarchical,
    skip_reason,
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

    def test_file_sha256_matches_hashlib(self):
        import hashlib
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "x.bin"
            data = os.urandom(3 * (1 << 20) + 17)  # lớn hơn một khối đọc
            path.write_bytes(data)
            self.assertEqual(file_sha256(path), hashlib.sha256(data).hexdigest())

    def test_skip_reason(self):
        self.assertIsNone(skip_reason(0.1))
        self.assertIn("eps xấp xỉ 0", skip_reason(0.0))
        self.assertIn("eps xấp xỉ 0", skip_reason(5.96e-8))  # sai số float32 của hai bản sao
        self.assertIsNone(skip_reason(1e-3))
        self.assertIn("phẳng", skip_reason(None))

    def test_eps_candidates_flat_curve_gives_no_knee_but_keeps_quantiles(self):
        candidates = eps_candidates(np.full(10, 0.3), (0.5,))
        self.assertEqual(candidates, [("knee", None), ("q0.50", 0.3)])

    def test_duplicate_embeddings_skip_only_zero_eps_configs(self):
        # 3 điểm gốc, mỗi điểm lặp 30 lần: k-distance bằng 0 cho 90/180 điểm, nên phân vị 0.25 có eps = 0.
        rng = np.random.default_rng(0)
        base = rng.normal(size=(3, 16))
        points = np.vstack([np.repeat(base, 30, axis=0), rng.normal(size=(90, 16))])
        result, curves = run_comparison(points, n_components=8, ks=(4,), min_samples_list=(5,), eps_quantiles=(0.25, 0.9),
                                        extra_dbscan_dims=(), sample_size=100, n_init=1, seed=0)
        by_rule = {r["eps_rule"]: r for r in result["dbscan"]}
        self.assertEqual(set(by_rule), {"knee", "q0.25", "q0.90"})
        self.assertIn("eps xấp xỉ 0", by_rule["q0.25"]["skipped"])
        self.assertEqual(by_rule["q0.25"]["eps"], 0.0)
        self.assertIsNone(by_rule["q0.90"]["skipped"])  # cấu hình có eps hợp lệ vẫn chạy
        self.assertIn("n_clusters", by_rule["q0.90"])
        text = format_markdown(result)
        self.assertIn("Cấu hình bị bỏ qua", text)
        self.assertIn("eps `q0.25`", text)
        self.assertNotIn("| 8 | 5 | q0.25 |", text)  # dòng bị bỏ qua không nằm trong bảng
        self.assertIn("| 8 | 5 | q0.90 |", text)

    def test_float32_duplicates_are_skipped_despite_rounding_noise(self):
        rng = np.random.default_rng(1)
        base = rng.normal(size=(4, 32)).astype(np.float32)
        points = np.vstack([np.repeat(base, 40, axis=0), rng.normal(size=(100, 32)).astype(np.float32)])
        result, _ = run_comparison(points, n_components=10, ks=(4,), min_samples_list=(5,), eps_quantiles=(0.25, 0.5),
                                   extra_dbscan_dims=(), sample_size=100, n_init=1, seed=0)
        for row in result["dbscan"]:
            if row["eps"] is not None and row["eps"] < 1e-6:
                self.assertIsNotNone(row["skipped"], row)  # không chạy DBSCAN với eps nhiễu số

    def test_too_short_curve_skips_only_the_knee(self):
        candidates = eps_candidates(np.array([0.1, 0.2]), (0.5,))
        self.assertEqual(candidates[0], ("knee", None))
        self.assertAlmostEqual(candidates[1][1], 0.15)
        self.assertIn("dưới 3 điểm", skip_reason(None))

    def test_two_items_with_min_samples_two_still_run_quantile_configs(self):
        points = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        result, _ = run_comparison(points, n_components=1, ks=(2,), min_samples_list=(2,), eps_quantiles=(0.5,),
                                   extra_dbscan_dims=(), sample_size=10, n_init=1, seed=0)
        by_rule = {r["eps_rule"]: r for r in result["dbscan"]}
        self.assertIn("dưới 3 điểm", by_rule["knee"]["skipped"])
        self.assertIsNone(by_rule["q0.50"]["skipped"])
        self.assertIn("n_clusters", by_rule["q0.50"])

    def test_main_succeeds_with_two_items(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.save(root / "two.npy", np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32))
            args = ["--embeddings", str(root / "two.npy"), "--components", "1", "--ks", "2", "--min-samples", "2",
                    "--extra-dbscan-dims", "--sample-size", "10", "--n-init", "1", "--output", str(root / "r.json"),
                    "--markdown", str(root / "r.md"), "--figure", ""]
            self.assertEqual(main(args), 0)
            self.assertIn("dưới 3 điểm", (root / "r.md").read_text(encoding="utf-8"))

    def test_flat_k_distance_curve_does_not_stop_the_run(self):
        points = np.vstack([np.zeros((40, 6)) + np.array([1, 0, 0, 0, 0, 0]), np.zeros((40, 6)) + np.array([0, 1, 0, 0, 0, 0])])
        result, curves = run_comparison(points, n_components=1, ks=(2,), min_samples_list=(5,), eps_quantiles=(0.5,),
                                        extra_dbscan_dims=(), sample_size=50, n_init=1, seed=0)
        self.assertEqual(len(result["dbscan"]), 2)
        by_rule = {r["eps_rule"]: r for r in result["dbscan"]}
        self.assertIn("phẳng", by_rule["knee"]["skipped"])
        self.assertIn("eps xấp xỉ 0", by_rule["q0.50"]["skipped"])
        self.assertIn("Tóm tắt DBSCAN", format_markdown(result))

    def test_all_dbscan_configs_skipped_still_formats_markdown(self):
        result = {"n_items": 10, "input_dim": 4, "pca_dim": 2, "pca_explained_variance_ratio": 0.5, "seed": 0, "sample_size": 10,
                  "sample_algorithms": [], "dbscan": [
                      {"pca_dim": 2, "min_samples": 5, "eps_rule": "knee", "eps": None, "skipped": "đường k-distance phẳng"}]}
        text = format_markdown(result)
        self.assertIn("Không cấu hình nào có eps ở điểm gãy chạy được", text)
        self.assertIn("0 / 0 cấu hình đã chạy", text)

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "CI không cài matplotlib")
    def test_plot_tolerates_skipped_knee(self):
        curve = np.zeros(20)
        rows = [{"pca_dim": 5, "min_samples": 4, "eps_rule": "knee", "eps": None, "skipped": "phẳng"}]
        with tempfile.TemporaryDirectory() as directory:
            plot_kdistance({(5, 4): curve}, rows, Path(directory) / "k.png", dim=5)
            self.assertTrue((Path(directory) / "k.png").exists())

    def test_package_provenance_only_when_checksum_matches(self):
        base = {"n_items": 10, "input_dim": 4, "pca_dim": 2, "pca_explained_variance_ratio": 0.5, "seed": 0, "sample_size": 10,
                "sample_algorithms": [], "dbscan": [], "embedding_file": "item_embeddings.npy"}
        known = format_markdown({**base, "embedding_sha256": GROUP_A_EMBEDDING_SHA256})
        self.assertIn("sentence-t5-base", known)
        self.assertIn("share-20261125T050341Z-1-001.zip", known)
        other = format_markdown({**base, "embedding_sha256": "0" * 64})
        self.assertIn("0" * 64, other)
        self.assertNotIn("sentence-t5-base", other)
        self.assertNotIn("share-20261125T050341Z-1-001.zip", other)
        self.assertIn("không khớp", other)

    def test_main_writes_outputs_and_reports_missing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.save(root / "emb.npy", self.points)
            args = ["--embeddings", str(root / "emb.npy"), "--components", "5", "--ks", "3", "--min-samples", "4",
                    "--extra-dbscan-dims", "3",
                    "--sample-size", "60", "--n-init", "1", "--output", str(root / "out" / "r.json"),
                    "--markdown", str(root / "out" / "r.md"), "--figure", ""]
            self.assertEqual(main(args), 0)
            written = json.loads((root / "out" / "r.json").read_text(encoding="utf-8"))
            self.assertEqual(written["pca_dim"], 5)
            self.assertEqual(written["embedding_file"], "emb.npy")
            self.assertEqual(written["embedding_sha256"], file_sha256(root / "emb.npy"))
            markdown = (root / "out" / "r.md").read_text(encoding="utf-8")
            self.assertTrue(markdown.startswith("# So sánh phân cụm"))
            self.assertIn(written["embedding_sha256"], markdown)
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
