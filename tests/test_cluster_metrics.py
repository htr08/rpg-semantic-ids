"""Unit test cho src/cluster_metrics.py, phần lớn là ví dụ nhỏ tính được bằng tay."""
import json
import math

import numpy as np
import pytest

from src import cluster_metrics as M


# --- Nhãn ---

def test_primary_path_prefers_beauty_and_cleans_html():
    paths = [["Health & Personal Care", "Vitamins"], ["Beauty", "Bath &amp; Body", "Bath", " Bath Salts "]]
    assert M.primary_category_path(paths) == ("Beauty", "Bath & Body", "Bath", "Bath Salts")
    assert M.category_labels(paths) == ("Bath Salts", "Bath")


def test_primary_path_falls_back_to_first_valid_path():
    assert M.primary_category_path([[], ["Health", "Vitamins"], ["Toys", "Games"]]) == ("Health", "Vitamins")


def test_level3_uses_last_label_when_path_is_short():
    assert M.category_labels([["Beauty", "Fragrance"]]) == ("Fragrance", "Fragrance")
    assert M.category_labels([["Beauty", "Makeup", "Nails", "Nail Polish"]]) == ("Nail Polish", "Nails")


@pytest.mark.parametrize("paths", [None, [], [[]], [["", "  "]], "Beauty"])
def test_missing_categories_give_no_label(paths):
    assert M.category_labels(paths) == (None, None)


@pytest.mark.parametrize("raw, expected", [("Essie", "Essie"), (" L&#39;Oreal ", "L'Oreal"),
                                           ("", None), ("Unknown", None), ("unknown ", None), (None, None)])
def test_brand_label_matches_patterns_rule(raw, expected):
    assert M.brand_label(raw) == expected


def write_items(tmp_path, metas, ids=None):
    path = tmp_path / "items.jsonl"
    ids = ids or list(range(1, len(metas) + 1))
    with open(path, "w", encoding="utf-8") as f:
        for i, meta in zip(ids, metas):
            f.write(json.dumps({"item_id": i, "asin": f"A{i}", "metadata_missing": meta is None,
                                "metadata": meta or {}}) + "\n")
    return path


def test_load_item_labels_aligns_with_item_id_and_reports_stats(tmp_path):
    path = write_items(tmp_path, [
        {"categories": [["Beauty", "Makeup", "Nails", "Nail Polish"]], "brand": "Essie"},
        {"categories": [["Health", "Vitamins"], ["Beauty", "Skin Care", "Face"]], "brand": "Unknown"},
        {"categories": [["Health", "Vitamins"]]},
        None,
    ])
    labels, stats = M.load_item_labels(path)
    assert labels["leaf"] == ["Nail Polish", "Face", "Vitamins", None]
    assert labels["level3"] == ["Nails", "Face", "Vitamins", None]
    assert labels["brand"] == ["Essie", None, None, None]
    assert stats["n_items"] == 4 and stats["n_multi_path"] == 1 and stats["n_primary_not_preferred"] == 1
    assert stats["leaf_coverage"] == 0.75 and stats["brand_coverage"] == 0.25 and stats["leaf_n_classes"] == 3


def test_load_item_labels_rejects_gaps_in_item_id(tmp_path):
    path = write_items(tmp_path, [{}, {}], ids=[1, 3])
    with pytest.raises(ValueError):
        M.load_item_labels(path)


# --- Purity, NMI (tính tay) ---

@pytest.mark.parametrize("labels, clusters, expected", [
    (["a", "a", "b", "b"], [0, 0, 1, 1], 1.0),          # mỗi cụm thuần một nhãn
    (["a", "a", "b", "b"], [0, 1, 0, 1], 0.5),          # mỗi cụm nửa a nửa b: (1 + 1) / 4
    (["a", "a", "a", "b"], [0, 0, 1, 1], 0.75),         # cụm 0: 2 a; cụm 1: 1 a 1 b -> (2 + 1) / 4
    (["a", "a", "a", "a"], [0, 1, 2, 3], 1.0),          # mỗi item một cụm: luôn thuần
    (["a", None, "b"], [0, 0, 1], 1.0),                 # item thiếu nhãn bị bỏ qua
])
def test_purity_hand_computed(labels, clusters, expected):
    assert M.purity(labels, clusters) == pytest.approx(expected)


def test_purity_and_nmi_all_missing_is_nan():
    assert math.isnan(M.purity([None, None], [0, 1]))
    assert math.isnan(M.nmi([None, None], [0, 1]))


def test_nmi_perfect_and_independent():
    assert M.nmi(["a", "a", "b", "b"], [5, 5, 9, 9]) == pytest.approx(1.0)   # chỉ khác cách đánh số cụm
    assert M.nmi(["a", "a", "b", "b"], [0, 1, 0, 1]) == pytest.approx(0.0)


def test_nmi_hand_computed_partial():
    # labels a,a,b,b; clusters 0,0,0,1 -> H(L)=1, H(C)=0.8113, I = 0.3113 (bit) -> NMI = 2I/(H(L)+H(C))
    h_c = -(0.75 * math.log2(0.75) + 0.25 * math.log2(0.25))
    mi = 1.0 - 0.75 * (-(2 / 3) * math.log2(2 / 3) - (1 / 3) * math.log2(1 / 3))
    assert M.nmi(["a", "a", "b", "b"], [0, 0, 0, 1]) == pytest.approx(2 * mi / (1 + h_c))


def test_ami_perfect_and_near_zero_for_random_assignment():
    assert M.ami(["a", "a", "b", "b"], [5, 5, 9, 9]) == pytest.approx(1.0)
    rng = np.random.default_rng(0)
    labels = [f"l{x}" for x in rng.integers(0, 50, 3000)]
    clusters = rng.integers(0, 64, 3000)
    assert M.nmi(labels, clusters) > 0.1              # NMI bị thổi phồng khi nhiều cụm, nhiều nhãn (≈ 0,15)
    assert abs(M.ami(labels, clusters)) < 0.01        # AMI đã trừ phần ngẫu nhiên
    assert math.isnan(M.ami([None], [0]))


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        M.purity(["a", "b"], [0])


# --- Entropy, va chạm ---

def test_codebook_entropy_hand_computed():
    ids = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 2], [1, 1, 3]])
    np.testing.assert_allclose(M.codebook_entropy(ids, codebook_size=4), [0.5, 0.5, 1.0])   # log2(2)/log2(4)
    np.testing.assert_allclose(M.codebook_entropy(np.zeros((5, 2), int), 4), [0.0, 0.0])
    assert M.codebook_entropy(np.array([[0], [1]]), 256)[0] == pytest.approx(1 / 8)          # 1 bit / 8 bit


def test_collision_rate_and_full_id_clusters():
    ids = np.array([[1, 2], [1, 2], [3, 4], [5, 6], [5, 6], [5, 6]])
    assert M.collision_rate(ids) == pytest.approx(5 / 6)
    clusters = M.full_id_clusters(ids)
    assert clusters[0] == clusters[1] and clusters[3] == clusters[4] == clusters[5]
    assert len(set(clusters.tolist())) == 3
    assert M.collision_rate(np.eye(3, dtype=int)) == 0.0


# --- Láng giềng ---

PAIRS = np.array([[0, 0, 0], [0, 0, 1], [5, 5, 5], [5, 5, 6]])   # item 0-1 trùng 2 mã, item 2-3 trùng 2 mã


def test_nearest_by_shared_codes_finds_pairs_and_excludes_self():
    nb = M.nearest_by_shared_codes(PAIRS, k=1)
    np.testing.assert_array_equal(nb.ravel(), [1, 0, 3, 2])
    nb2 = M.nearest_by_shared_codes(PAIRS, k=3)
    assert all(i not in row for i, row in enumerate(nb2))
    assert nb2[0, 0] == 1                                             # láng giềng gần nhất xếp đầu


def test_nearest_ties_are_random_but_reproducible():
    ids = np.zeros((50, 2), dtype=int)                                # mọi cặp đều trùng 2 mã: toàn hòa
    a = M.nearest_by_shared_codes(ids, k=5, seed=1)
    np.testing.assert_array_equal(a, M.nearest_by_shared_codes(ids, k=5, seed=1))
    assert not np.array_equal(a, M.nearest_by_shared_codes(ids, k=5, seed=2))
    assert len(set(a[:, 0].tolist())) > 10                            # không dồn về item có chỉ số nhỏ


def test_nearest_chunking_gives_same_result():
    ids = np.random.default_rng(0).integers(0, 4, size=(37, 6))
    np.testing.assert_array_equal(M.nearest_by_shared_codes(ids, k=4, seed=3, chunk=5),
                                  M.nearest_by_shared_codes(ids, k=4, seed=3, chunk=1000))


def test_nearest_rejects_bad_k():
    with pytest.raises(ValueError):
        M.nearest_by_shared_codes(PAIRS, k=4)


def test_neighbor_purity_hand_computed():
    nb = M.nearest_by_shared_codes(PAIRS, k=1)
    assert M.neighbor_purity(nb, ["x", "x", "y", "y"]) == 1.0
    assert M.neighbor_purity(nb, ["x", "y", "x", "y"]) == 0.0
    # item 1 thiếu nhãn: item 0 không có láng giềng có nhãn -> bỏ; còn item 2, 3 -> 1.0
    assert M.neighbor_purity(nb, ["x", None, "y", "y"]) == 1.0


def test_neighbor_purity_counts_only_labeled_neighbors():
    nb = np.array([[1, 2], [0, 2], [0, 1]])
    # item 0: láng giềng 1 (x), 2 (None) -> 1/1; item 1: 0 (x), 2 (None) -> 1/1; item 2 thiếu nhãn -> bỏ
    assert M.neighbor_purity(nb, ["x", "x", None]) == 1.0


def test_chance_neighbor_purity_hand_computed():
    assert M.chance_neighbor_purity(["x", "x", "y", "y"]) == pytest.approx(4 / 12)      # (2·1 + 2·1) / (4·3)
    assert M.chance_neighbor_purity(["x", "x", "x", None]) == pytest.approx(1.0)
    assert math.isnan(M.chance_neighbor_purity(["x", None]))


# --- Tổng hợp ---

def structured_and_random(n=120, m=8, seed=0):
    """ID có cấu trúc: 4 danh mục, item cùng danh mục chia sẻ một nửa số mã. ID ngẫu nhiên: xáo trộn từng cột."""
    rng = np.random.default_rng(seed)
    cat = np.repeat(np.arange(4), n // 4)
    proto = rng.integers(0, 16, size=(4, m))
    ids = rng.integers(0, 16, size=(n, m))
    ids[:, : m // 2] = proto[cat, : m // 2]
    rand = np.stack([rng.permutation(ids[:, j]) for j in range(m)], axis=1)
    return ids, rand, [f"c{c}" for c in cat]


def test_evaluate_separates_structured_from_random_ids():
    ids, rand, cat = structured_and_random()
    labels = {"leaf": cat, "brand": [c if i % 3 else None for i, c in enumerate(cat)]}
    s_struct, per_pos = M.evaluate(ids, labels, k=5, codebook_size=16)
    s_rand, _ = M.evaluate(rand, labels, k=5, codebook_size=16)
    assert len(per_pos) == 8 and {"position", "entropy", "leaf_purity", "leaf_nmi", "leaf_ami", "brand_ami"} <= set(per_pos[0])
    assert s_struct["leaf_neighbor_purity@5"] == 1.0
    assert s_rand["leaf_neighbor_purity@5"] < 0.6
    assert s_struct["leaf_nmi_pos_mean"] > s_rand["leaf_nmi_pos_mean"]
    assert s_struct["leaf_ami_pos_mean"] > 0.1 > abs(s_rand["leaf_ami_pos_mean"])
    assert s_struct["leaf_neighbor_chance"] == pytest.approx(s_rand["leaf_neighbor_chance"])
    # Toàn ID: gần như mỗi item một cụm nên purity toàn ID cao cho cả hai, không phân biệt được
    assert s_struct["leaf_purity_full_id"] == 1.0 and s_rand["leaf_purity_full_id"] > 0.95


def test_evaluate_rejects_label_length_mismatch():
    ids, _, cat = structured_and_random()
    with pytest.raises(ValueError):
        M.evaluate(ids, {"leaf": cat[:-1]}, k=5, codebook_size=16)


def test_cli_writes_csv_files(tmp_path):
    ids, rand, cat = structured_and_random()
    path = write_items(tmp_path, [{"categories": [["Beauty", "X", c]], "brand": "B" + c} for c in cat])
    np.save(tmp_path / "s.npy", ids)
    np.save(tmp_path / "r.npy", rand)
    out = tmp_path / "out"
    assert M.main(["--items", str(path), "--ids", f"struct={tmp_path / 's.npy'}", f"rand={tmp_path / 'r.npy'}",
                   "--out-dir", str(out), "--k", "5", "--codebook-size", "16"]) == 0
    rows = (out / "rq1_cluster_metrics.csv").read_text(encoding="utf-8").splitlines()
    assert rows[0].startswith("variant,") and len(rows) == 3
    assert len((out / "rq1_cluster_metrics_per_position.csv").read_text(encoding="utf-8").splitlines()) == 1 + 2 * 8
    assert json.loads((out / "rq1_label_stats.json").read_text(encoding="utf-8"))["leaf_n_classes"] == 4


def test_cli_rejects_bad_ids_spec(tmp_path):
    path = write_items(tmp_path, [{"categories": [["Beauty", "X"]]}] * 3)
    with pytest.raises(SystemExit):
        M.main(["--items", str(path), "--ids", "no_equal_sign", "--out-dir", str(tmp_path)])
