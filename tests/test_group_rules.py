"""Kiểm tra luật cấp nhóm trên ví dụ nhỏ tính tay."""

import json

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from src.group_rules import (evaluate_rules, group_index, item_groups_by_brand, item_groups_by_category,
                             item_groups_identity, load_category_paths, main, popularity_hits, predicted_groups,
                             user_group_matrix)
from src.patterns import RULE_COLUMNS

# 4 item: 1, 2 cùng brand A và cùng danh mục "Beauty > Makeup"; 3 brand B, danh mục "Beauty > Skin"; 4 thiếu brand,
# hai đường dẫn (Makeup và Skin).
PATHS = {
    1: [["Beauty", "Makeup", "Nails"]],
    2: [["Beauty", "Makeup", "Lips"]],
    3: [["Beauty", "Skin"]],
    4: [["Beauty", "Makeup", "Eyes"], ["Beauty", "Skin", "Face"]],
}
METADATA = {1: {"brand": "Acme"}, 2: {"brand": "ACME"}, 3: {"brand": "Zed"}, 4: {"brand": None}}


def make_rules(rows):
    """rows: (antecedent, consequent) là các tập chỉ số nhóm đếm từ 1."""
    frame = pd.DataFrame({"antecedents": [frozenset(a) for a, _ in rows], "consequents": [frozenset(c) for _, c in rows]})
    for column in RULE_COLUMNS[2:]:
        frame[column] = 1.0
    return frame[RULE_COLUMNS]


def test_category_depth_truncates_and_keeps_multiple_paths():
    groups = item_groups_by_category(PATHS, 4, 2)
    assert groups[0] == groups[1] == frozenset({"Beauty > Makeup"})
    assert groups[2] == frozenset({"Beauty > Skin"})
    assert groups[3] == frozenset({"Beauty > Makeup", "Beauty > Skin"})
    # Đường ngắn hơn độ sâu được giữ nguyên.
    assert item_groups_by_category(PATHS, 4, 3)[2] == frozenset({"Beauty > Skin"})
    with pytest.raises(ValueError):
        item_groups_by_category(PATHS, 4, 0)


def test_brand_ignores_case_and_leaves_missing_empty():
    groups = item_groups_by_brand(METADATA, 4)
    assert groups[0] == groups[1] == frozenset({"Acme"})
    assert groups[2] == frozenset({"Zed"})
    assert groups[3] == frozenset()


def test_user_group_matrix_unions_items_of_the_same_group():
    names, item_group = group_index(item_groups_by_category(PATHS, 4, 2))
    assert names == ["Beauty > Makeup", "Beauty > Skin"]
    matrix = user_group_matrix([np.array([1, 2]), np.array([3]), np.array([4]), np.array([], dtype=np.int64)], item_group)
    assert matrix.toarray().tolist() == [[True, False], [False, True], [True, True], [False, False]]
    with pytest.raises(ValueError):
        user_group_matrix([np.array([5])], item_group)


def test_predicted_groups_requires_whole_antecedent():
    # Ba nhóm; luật 1: {1} => {3}; luật 2: {1, 2} => {3}; luật 3: {2} => {1}.
    rules = make_rules([({1}, {3}), ({1, 2}, {3}), ({2}, {1})])
    history = sparse.csr_matrix(np.array([[1, 0, 0], [1, 1, 0], [0, 0, 1], [0, 1, 0]], dtype=bool))
    predicted = predicted_groups(history, rules, 3).toarray()
    assert predicted.tolist() == [[False, False, True], [True, False, True], [False, False, False], [True, False, False]]
    empty = predicted_groups(history, make_rules([]), 3)
    assert empty.shape == (4, 3) and empty.nnz == 0


def test_popularity_hits_uses_same_set_size_and_tie_break_by_index():
    counts = np.array([5, 9, 9, 1])  # thứ tự phổ biến: nhóm 1, 2, 0, 3
    target = sparse.csr_matrix(np.array([[0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0]], dtype=bool))
    assert popularity_hits(target, np.array([2, 2, 3, 0]), counts).tolist() == [True, False, False, False]
    assert popularity_hits(target, np.array([2, 3, 4, 0]), counts).tolist() == [True, True, True, False]


def test_evaluate_rules_coverage_over_all_users_and_hit_rate_over_scored_users():
    rules = make_rules([({1}, {2})])  # nhóm 1 => nhóm 2
    history = sparse.csr_matrix(np.array([[1, 0, 0], [1, 0, 0], [0, 1, 0], [1, 0, 0]], dtype=bool))
    target = sparse.csr_matrix(np.array([[0, 1, 0], [0, 0, 1], [0, 1, 0], [0, 0, 0]], dtype=bool))
    result = evaluate_rules(rules, history, target, np.array([4, 1, 2]))
    # Phủ: user 1, 2, 4 (tiền đề nhóm 1) trên 4 user. User 4 không có nhóm đích nên không vào hit rate.
    assert result["n_users"] == 4 and result["n_covered_users"] == 3
    assert result["coverage"] == pytest.approx(3 / 4)
    assert result["n_scored_users"] == 2
    assert result["mean_predicted_groups"] == 1.0
    # Luật dự đoán nhóm 2 (chỉ số 1): trúng user 1, trượt user 2.
    assert result["hit_rate_rules"] == 0.5
    # Popularity chọn 1 nhóm phổ biến nhất (chỉ số 0, số đếm 4): không trúng user nào (đích là chỉ số 1 và 2).
    assert result["hit_rate_popularity_same_size"] == 0.0


def write_dataset(root):
    """Hai user có train {1, 3} và {1, 2}; validation 2 và 3; test (4) không được đọc."""
    interactions = [(0, 1, 1), (1, 1, 3), (2, 1, 2), (3, 1, 4), (4, 2, 1), (5, 2, 2), (6, 2, 3), (7, 2, 4)]
    lines = ["interaction_id,user_id,item_id,timestamp,source_line"] + [f"{i},{u},{item},0,{i}" for i, u, item in interactions]
    (root / "interactions.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (root / "splits.csv").write_text("user_id,train_start,train_stop,validation_index,test_index\n1,0,2,2,3\n2,4,6,6,7\n",
                                     encoding="utf-8")
    # Item 4 có brand riêng để việc đọc nhầm item test làm thay đổi số liệu cấp brand.
    brands = {**{i: METADATA[i]["brand"] for i in range(1, 4)}, 4: "Solo"}
    items = [{"item_id": i, "asin": f"a{i}", "metadata": {"brand": brands[i], "title": f"t{i}",
                                                           "categories": PATHS[i]}} for i in range(1, 5)]
    (root / "items.jsonl").write_text("\n".join(json.dumps(r) for r in items) + "\n", encoding="utf-8")
    validation = [{"user_id": 1, "history": [1, 3], "target": 2}, {"user_id": 2, "history": [1, 2], "target": 3}]
    (root / "validation.jsonl").write_text("\n".join(json.dumps(r) for r in validation) + "\n", encoding="utf-8")


def test_main_writes_reports_and_does_not_use_test_items(tmp_path):
    write_dataset(tmp_path)
    args = ["--interactions", str(tmp_path / "interactions.csv"), "--splits", str(tmp_path / "splits.csv"),
            "--items", str(tmp_path / "items.jsonl"), "--validation", str(tmp_path / "validation.jsonl"),
            "--group-supports", "0.75", "--item-supports", "0.75", "--depths", "2", "--show-support", "0.75",
            "--output", str(tmp_path / "out.json"), "--markdown", str(tmp_path / "out.md")]
    assert main(args) == 0
    payload = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
    assert payload["n_transactions"] == 2
    by_level = {level["level"]: level for level in payload["levels"]}
    assert set(by_level) == {"item", "brand", "category depth 2"}
    # Item 4 (brand Solo) chỉ xuất hiện ở test của cả hai user. Số đếm train: Acme 2 user, Zed 1 user; ngưỡng 0,75 × 2 giao
    # dịch cần ≥ 2 user nên chỉ còn Acme. Nếu item test lọt vào train thì Solo cũng có 2 user và số nhóm phổ biến thành 2.
    brand = by_level["brand"]["supports"][0]
    assert by_level["brand"]["n_groups"] == 3 and brand["n_frequent_groups"] == 1
    assert by_level["brand"]["n_items_without_group"] == 0
    # Cấp item: chỉ item 1 (2 user) đạt ngưỡng; item 4 sẽ thành phổ biến nếu item test lọt vào train.
    assert by_level["item"]["supports"][0]["n_frequent_groups"] == 1
    assert by_level["category depth 2"]["supports"][0]["n_frequent_groups"] == 1
    assert by_level["brand"]["supports"][0]["n_users"] == 2
    assert by_level["brand"]["supports"][0]["n_scored_users"] == 0  # không có luật nào ở ngưỡng này


def test_main_reports_mismatched_user_counts(tmp_path, capsys):
    write_dataset(tmp_path)
    (tmp_path / "validation.jsonl").write_text(json.dumps({"user_id": 1, "history": [1], "target": 2}) + "\n", encoding="utf-8")
    args = ["--interactions", str(tmp_path / "interactions.csv"), "--splits", str(tmp_path / "splits.csv"),
            "--items", str(tmp_path / "items.jsonl"), "--validation", str(tmp_path / "validation.jsonl"),
            "--output", str(tmp_path / "out.json"), "--markdown", str(tmp_path / "out.md")]
    assert main(args) == 1
    assert "số user validation" in capsys.readouterr().err
    assert not (tmp_path / "out.json").exists()


def test_load_category_paths_unescapes_html(tmp_path):
    (tmp_path / "items.jsonl").write_text(
        json.dumps({"item_id": 1, "metadata": {"categories": [["Beauty", "Hair &amp; Scalp"]]}}) + "\n", encoding="utf-8")
    assert load_category_paths(tmp_path / "items.jsonl") == {1: [["Beauty", "Hair & Scalp"]]}
    # Tên đệm số 0 để thứ tự tên trùng thứ tự item_id ("02" < "10").
    assert item_groups_identity(10)[1] == frozenset({"02"}) and sorted(set().union(*item_groups_identity(10)))[-1] == "10"
