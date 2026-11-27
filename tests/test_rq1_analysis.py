"""Unit test cho src/rq1_analysis.py (dữ liệu nhỏ dựng tay; không cần faiss, UMAP bỏ qua khi thiếu)."""
import csv
import json

import numpy as np
import pytest

from src import rq1_analysis as R


def write_csv(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return path


CLUSTER = [
    {"variant": "opq", "leaf_ami_pos_mean": "0.059", "leaf_neighbor_purity@10": "0.142", "brand_neighbor_purity@10": "0.128"},
    {"variant": "pq", "leaf_ami_pos_mean": "0.067", "leaf_neighbor_purity@10": "0.197", "brand_neighbor_purity@10": "0.152"},
    {"variant": "rq", "leaf_ami_pos_mean": "0.031", "leaf_neighbor_purity@10": "0.127", "brand_neighbor_purity@10": "0.120"},
    {"variant": "random", "leaf_ami_pos_mean": "0.000", "leaf_neighbor_purity@10": "0.018", "brand_neighbor_purity@10": "0.005"},
]
RUNS = [{"variant": v, "seed": s, "test_ndcg10": x} for v, xs in
        [("opq", [0.0399, 0.0427, 0.0433]), ("pq", [0.0441, 0.0444, 0.0436]),
         ("rq", [0.0343, 0.0334, 0.0355]), ("random", [0.0373, 0.0363, 0.0369])]
        for s, x in zip([42, 43, 44], xs)]


# --- scatter ---

def test_scatter_table_means_std_and_spearman():
    table, rho = R.scatter_table(CLUSTER, [{k: str(v) for k, v in r.items()} for r in RUNS])
    assert [r["variant"] for r in table] == ["opq", "pq", "rq", "random"]
    pq = table[1]
    assert pq["n_seed"] == 3 and pq["ndcg_mean"] == pytest.approx(np.mean([0.0441, 0.0444, 0.0436]))
    assert pq["ndcg_std"] == pytest.approx(np.std([0.0441, 0.0444, 0.0436], ddof=1))
    # thứ tự NDCG: pq > opq > random > rq; thứ tự AMI: pq > opq > rq > random -> chỉ đổi chỗ rq/random -> rho = 0.8
    assert rho["leaf_ami_pos_mean"] == pytest.approx(0.8)


def test_scatter_skips_variants_without_runs():
    table, rho = R.scatter_table(CLUSTER, [r for r in RUNS if r["variant"] != "random"])
    assert [r["variant"] for r in table] == ["opq", "pq", "rq"]
    assert rho["leaf_ami_pos_mean"] == pytest.approx(1.0)


def test_cli_scatter_writes_png_and_table(tmp_path, capsys):
    c = write_csv(tmp_path / "c.csv", CLUSTER)
    r = write_csv(tmp_path / "r.csv", RUNS)
    assert R.main(["scatter", "--cluster-csv", str(c), "--runs-csv", str(r), "--out-dir", str(tmp_path / "out")]) == 0
    assert (tmp_path / "out" / "rq1_scatter_cluster_ndcg.png").stat().st_size > 1000
    rows = list(csv.DictReader(open(tmp_path / "out" / "rq1_scatter_table.csv", encoding="utf-8")))
    assert len(rows) == 4 and "Spearman" in capsys.readouterr().out


# --- theo tầng ---

def test_position_dependence_detects_copied_position():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 8, 400)
    ids = np.stack([a, a, rng.integers(0, 8, 400)], axis=1)        # vị trí 1 chép vị trí 0, vị trí 2 độc lập
    dep = R.position_dependence(ids)
    assert np.isnan(dep[0]) and dep[1] == pytest.approx(1.0) and abs(dep[2]) < 0.05


def test_level_diagnostics_shapes_and_prefix_effect():
    rng = np.random.default_rng(1)
    labels = list(np.repeat(["a", "b", "c", "d"], 50))
    lab_idx = np.repeat(np.arange(4), 50)
    informative = np.stack([lab_idx] + [rng.integers(0, 4, 200) for _ in range(3)], axis=1)  # chỉ vị trí 0 có nghĩa
    per_pos, per_prefix = R.level_diagnostics({"x": informative}, labels, codebook_size=4, k=5,
                                              prefix_lengths=(1, 4, 8))
    assert len(per_pos) == 4 and {r["n_codes"] for r in per_prefix} == {1, 4}      # bỏ L = 8 > 4 vị trí
    assert per_pos[0]["label_ami"] > 0.9 and abs(per_pos[3]["label_ami"]) < 0.05
    pur = {r["n_codes"]: r["neighbor_purity"] for r in per_prefix}
    assert pur[1] > 0.9 and pur[4] < pur[1]                         # thêm mã nhiễu làm láng giềng kém thuần đi
    assert per_prefix[0]["chance"] == pytest.approx(49 / 199)


def test_plot_levels_writes_png(tmp_path):
    per_pos = [{"variant": "rq", "position": j, "entropy": 1.0, "label_ami": 0.1, "ami_with_previous": 0.0} for j in range(4)]
    per_prefix = [{"variant": "rq", "n_codes": L, "neighbor_purity": 0.1 * L, "chance": 0.02} for L in (1, 2, 4)]
    assert R.plot_levels(per_pos, per_prefix, tmp_path / "l.png").stat().st_size > 1000


# --- va chạm ---

def items_rows():
    metas = [{"title": "Son A", "brand": "X", "categories": [["Beauty", "Makeup", "Lips", "Lipstick"]], "description": "đỏ"},
             {"title": "Son A", "brand": "X", "categories": [["Beauty", "Makeup", "Lips", "Lipstick"]], "description": ""},
             {"title": "Kem B", "brand": "Y", "categories": [["Beauty", "Skin Care"]], "description": "dưỡng ẩm"},
             None]
    return [{"item_id": i + 1, "asin": f"A{i + 1}", "metadata_missing": m is None, "metadata": m or {}}
            for i, m in enumerate(metas)]


def test_collision_groups_lists_members_with_causes():
    ids = np.array([[1, 2], [1, 2], [3, 4], [5, 6]])
    emb = np.array([[0.0, 1.0], [0.0, 1.0], [1.0, 0.0], [2.0, 2.0]])
    rows = R.collision_groups({"opq": ids, "random": np.arange(8).reshape(4, 2)}, items_rows(), emb)
    assert [(r["variant"], r["item_id"]) for r in rows] == [("opq", 1), ("opq", 2)]
    assert rows[0]["same_title"] and rows[0]["same_embedding"] and rows[0]["leaf"] == "Lipstick"
    assert rows[1]["description_chars"] == 0 and rows[0]["description_chars"] == 2


def test_collision_groups_checks_row_count():
    with pytest.raises(ValueError):
        R.collision_groups({"opq": np.zeros((3, 2))}, items_rows())


def test_load_items_requires_contiguous_ids(tmp_path):
    p = tmp_path / "items.jsonl"
    p.write_text("\n".join(json.dumps({"item_id": i}) for i in (1, 3)), encoding="utf-8")
    with pytest.raises(ValueError):
        R.load_items(p)


# --- 2D ---

def test_top_categories_order_and_none():
    assert R.top_categories(["b", "a", "b", None, None, None, "c", "a"], 2) == ["a", "b"]


def test_project_2d_pca_is_first_two_components():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(300, 5)) * np.array([5.0, 3.0, 1.0, 0.5, 0.1])
    z = R.project_2d(x, "pca")
    assert z.shape == (300, 2)
    assert z[:, 0].var() > z[:, 1].var() > 0.5
    with pytest.raises(ValueError):
        R.project_2d(x, "tsne")


def test_project_2d_umap_when_installed():
    pytest.importorskip("umap")
    x = np.random.default_rng(0).normal(size=(120, 6))
    assert R.project_2d(x, "umap").shape == (120, 2)


def test_plot_projection_writes_png(tmp_path):
    rng = np.random.default_rng(0)
    coords = rng.normal(size=(100, 2))
    leaf = list(rng.choice(["a", "b", "c", None], 100))
    out = R.plot_projection(coords, leaf, {"pq": rng.integers(0, 20, 100)}, tmp_path / "p.png", "thử", n_top=3)
    assert out.stat().st_size > 1000
