"""Unit test cho src/evaluate.py; giá trị kỳ vọng được tính tay trong comment."""
import json
import math

import numpy as np
import pytest

from src.evaluate import (
    evaluate,
    filter_seen,
    load_split,
    ndcg_at_k,
    per_user_metrics,
    recall_at_k,
)

# Ví dụ tính tay dùng danh sách ngắn hơn K; cảnh báo này được kiểm tra riêng trong
# test_short_ranking_after_filter_warns.
pytestmark = pytest.mark.filterwarnings("ignore:.*ranked items:UserWarning")


# --- Một nhãn: NDCG@K = 1 / log2(r + 1) với r là vị trí nhãn (đếm từ 1) ---

@pytest.mark.parametrize(
    "rank, k, expected_recall, expected_ndcg",
    [
        (1, 5, 1.0, 1.0),                    # 1 / log2(2) = 1
        (2, 5, 1.0, 0.6309297535714575),     # 1 / log2(3)
        (3, 5, 1.0, 0.5),                    # 1 / log2(4) = 1/2
        (5, 5, 1.0, 0.38685280723454163),    # 1 / log2(6)
        (6, 5, 0.0, 0.0),                    # ngoài top-5
        (6, 10, 1.0, 0.35620718710802218),   # 1 / log2(7)
    ],
)
def test_single_target_by_rank(rank, k, expected_recall, expected_ndcg):
    ranked = [100 + i for i in range(1, 11)]
    target = ranked[rank - 1]
    assert recall_at_k(ranked, [target], k) == expected_recall
    assert ndcg_at_k(ranked, [target], k) == pytest.approx(expected_ndcg)


def test_target_absent():
    assert recall_at_k([1, 2, 3], [9], 3) == 0.0
    assert ndcg_at_k([1, 2, 3], [9], 3) == 0.0


def test_ranking_shorter_than_k():
    # Danh sách 2 item, K = 5: nhãn ở vị trí 2 vẫn được tính.
    assert ndcg_at_k([4, 7], [7], 5) == pytest.approx(1 / math.log2(3))


def test_multiple_relevant_items():
    # Tập liên quan {1, 2}, top-3 = [1, 9, 2].
    # Recall = 2/2 = 1. DCG = 1/log2(2) + 1/log2(4) = 1.5. IDCG = 1 + 1/log2(3) ≈ 1.6309.
    assert recall_at_k([1, 9, 2], [1, 2], 3) == 1.0
    assert ndcg_at_k([1, 9, 2], [1, 2], 3) == pytest.approx(1.5 / (1 + 1 / math.log2(3)))
    # K = 1: chỉ 1 trong 2 item liên quan → Recall = 1/2; IDCG chỉ tính 1 vị trí → NDCG = 1.
    assert recall_at_k([1, 9, 2], [1, 2], 1) == 0.5
    assert ndcg_at_k([1, 9, 2], [1, 2], 1) == 1.0


@pytest.mark.parametrize("k", [0, -1, 2.0])
def test_invalid_k(k):
    with pytest.raises(ValueError):
        recall_at_k([1], [1], k)
    with pytest.raises(ValueError):
        ndcg_at_k([1], [1], k)


def test_empty_relevant_set():
    with pytest.raises(ValueError):
        recall_at_k([1], [], 1)


# --- Lọc item đã thấy ---

def test_filter_seen_keeps_order():
    assert filter_seen([10, 3, 7, 5], seen_items=[10, 5], target=7) == [3, 7]


def test_filter_seen_keeps_repurchased_target():
    # Nhãn 7 cũng nằm trong lịch sử (mua lại): không bị loại.
    assert filter_seen([7, 10, 3], seen_items=[7, 10], target=7) == [7, 3]


def test_seen_filter_changes_rank():
    # Trước lọc nhãn 7 ở vị trí 3 (NDCG = 0.5); sau khi loại 10, nhãn lên vị trí 2.
    split = {1: {"target": 7, "seen_item_ids": [10]}}
    rankings = {1: [10, 3, 7]}
    filtered = evaluate(rankings, split, ks=(5,))
    raw = evaluate(rankings, split, ks=(5,), exclude_seen=False)
    assert filtered["ndcg@5"] == pytest.approx(1 / math.log2(3))
    assert raw["ndcg@5"] == pytest.approx(0.5)


def test_seen_filter_pulls_target_into_top_k():
    # Top-2 trước lọc = [1, 2] (trượt); loại 1, 2 → nhãn 3 ở vị trí 1.
    split = {1: {"target": 3, "seen_item_ids": [1, 2]}}
    assert evaluate({1: [1, 2, 3]}, split, ks=(2,))["recall@2"] == 1.0


# --- Trung bình trên nhiều user ---

def test_mean_over_users():
    # User 1: nhãn ở vị trí 1 → NDCG 1. User 2: vị trí 3 → 0.5. User 3: trượt → 0.
    # Recall@5 = 2/3; NDCG@5 = (1 + 0.5 + 0) / 3 = 0.5.
    # K = 1: chỉ user 1 trúng → Recall@1 = NDCG@1 = 1/3.
    split = {
        1: {"target": 11, "seen_item_ids": []},
        2: {"target": 23, "seen_item_ids": []},
        3: {"target": 99, "seen_item_ids": []},
    }
    rankings = {1: [11, 12, 13], 2: [21, 22, 23], 3: [31, 32, 33]}
    result = evaluate(rankings, split, ks=(1, 5))
    assert result["recall@5"] == pytest.approx(2 / 3)
    assert result["ndcg@5"] == pytest.approx(0.5)
    assert result["recall@1"] == pytest.approx(1 / 3)
    assert result["ndcg@1"] == pytest.approx(1 / 3)
    assert set(result) == {"recall@1", "ndcg@1", "recall@5", "ndcg@5"}


def test_per_user_order_and_values():
    split = {2: {"target": 5, "seen_item_ids": []}, 1: {"target": 4, "seen_item_ids": []}}
    rankings = {1: [4], 2: [9, 5]}
    users, scores = per_user_metrics(rankings, split, ks=(10,))
    assert users.tolist() == [1, 2]
    assert scores["ndcg@10"] == pytest.approx([1.0, 1 / math.log2(3)])


def test_extra_users_in_rankings_are_ignored():
    split = {1: {"target": 4, "seen_item_ids": []}}
    assert evaluate({1: [4], 2: [8]}, split, ks=(5,))["recall@5"] == 1.0


def test_missing_user_raises():
    split = {1: {"target": 4, "seen_item_ids": []}, 2: {"target": 5, "seen_item_ids": []}}
    with pytest.raises(ValueError, match="1 user"):
        evaluate({1: [4]}, split)


@pytest.mark.parametrize("ranking", [["9", "5"], [9.0, 5.0], [True, 5]])
def test_non_integer_item_ids_raise(ranking):
    # '5' != 5: nếu không chặn, user này luôn bị tính trượt.
    split = {1: {"target": 5, "seen_item_ids": []}}
    with pytest.raises(ValueError, match="integers"):
        evaluate({1: ranking}, split)


def test_numpy_integer_ids_accepted():
    split = {1: {"target": 5, "seen_item_ids": []}}
    rankings = {1: np.array([9, 5] + list(range(100, 108)))}
    assert evaluate(rankings, split, ks=(5,))["ndcg@5"] == pytest.approx(1 / math.log2(3))


def test_short_ranking_after_filter_warns():
    # Top-5 = [1..5], đã thấy {1, 2, 3} → còn [4, 5], ít hơn K = 5.
    split = {1: {"target": 5, "seen_item_ids": [1, 2, 3]}}
    with pytest.warns(UserWarning, match="1 user"):
        evaluate({1: [1, 2, 3, 4, 5]}, split, ks=(5,))


def test_full_ranking_is_truncated_before_checks():
    # Toàn bộ catalog 12.101 item, nhãn ở vị trí 3, đã thấy item ở vị trí 1.
    # Sau lọc nhãn ở vị trí 2 → NDCG@10 = 1 / log2(3).
    ranking = np.arange(1, 12102)
    split = {1: {"target": 3, "seen_item_ids": [1]}}
    assert evaluate({1: ranking}, split, ks=(10,))["ndcg@10"] == pytest.approx(1 / math.log2(3))


def test_short_ranking_warns_without_filter():
    split = {1: {"target": 5, "seen_item_ids": []}}
    with pytest.warns(UserWarning, match="exclude_seen=False"):
        evaluate({1: [5]}, split, ks=(5,), exclude_seen=False)


def test_duplicate_items_raise():
    split = {1: {"target": 4, "seen_item_ids": []}}
    with pytest.raises(ValueError, match="duplicate"):
        evaluate({1: [4, 4]}, split)


# --- Đọc file theo data contract ---

def test_load_split(tmp_path):
    rows = [
        {"user_id": 1, "history": [3, 5], "target": 8, "seen_item_ids": [3, 5], "target_in_history": False},
        {"user_id": 2, "history": [7], "target": 7, "seen_item_ids": [7], "target_in_history": True},
    ]
    path = tmp_path / "test.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    split = load_split(path)
    assert split == {
        1: {"target": 8, "seen_item_ids": [3, 5]},
        2: {"target": 7, "seen_item_ids": [7]},
    }
    # User 2 mua lại item 7: nhãn được giữ khi lọc nên vẫn tính là trúng ở vị trí 1.
    assert evaluate({1: [3, 8], 2: [7, 1]}, split, ks=(5,))["ndcg@5"] == pytest.approx(1.0)
