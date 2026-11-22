"""Hàm đánh giá dùng chung cho mọi mô hình: Recall@K, NDCG@K (bản đầu).

Bản đầu: C (Tuần 1). Mở rộng (paired t-test, coverage, tỉ lệ long-tail, cold-start theo nhóm tần suất): B.

Mọi mô hình (RecBole, RPG code gốc, AR-Rec) xuất danh sách xếp hạng cho từng user rồi chấm tại đây,
để tránh khác biệt giữa các cách tính metric của từng thư viện.

Quy ước:

- Mỗi user có đúng một nhãn (leave-last-out): ``target`` trong ``validation.jsonl`` hoặc ``test.jsonl``
  (định dạng theo docs/data_contract.md). Item ID là ``item_id`` theo ``id_mapping.json``;
  ID nội bộ của từng thư viện (ví dụ RecBole) phải đổi về ``item_id`` trước khi chấm.
- Mặc định không loại item đã thấy (``seen_item_ids``) khỏi danh sách xếp hạng; ứng viên là toàn catalog
  (kế hoạch, mục 3.4 và 3.5a). ``exclude_seen=True`` chỉ dùng để đối chiếu với kết quả có lọc.
- Khi ``exclude_seen=True`` có một ngoại lệ do module này chọn (data contract chỉ yêu cầu xử lý tường minh,
  chưa quy định cách xử lý): nếu chính nhãn nằm trong ``seen_item_ids`` (``target_in_history``, mua lại), nhãn không bị loại,
  để mô hình không bị phạt vì một item mà quy tắc lọc tự xóa. Trên Beauty 2014 không có trường hợp này.
- Với một nhãn ở vị trí ``r`` (đếm từ 1) trong top-K đã lọc:
  Recall@K = 1 và NDCG@K = 1 / log2(r + 1); nhãn nằm ngoài top-K cho cả hai bằng 0.
  Với một nhãn duy nhất, IDCG = 1 nên NDCG bằng DCG.
- Giá trị cuối là trung bình cộng trên mọi user có trong tập nhãn.
"""
import json
import math
import warnings
from pathlib import Path

import numpy as np

DEFAULT_KS = (5, 10)


def recall_at_k(ranked_items, relevant_items, k):
    """Recall@K = |top-K ∩ tập liên quan| / |tập liên quan|."""
    _check_k(k)
    relevant = set(relevant_items)
    if not relevant:
        raise ValueError("relevant_items must not be empty")
    hits = sum(1 for item in ranked_items[:k] if item in relevant)
    return hits / len(relevant)


def ndcg_at_k(ranked_items, relevant_items, k):
    """NDCG@K với độ liên quan nhị phân.

    DCG@K = tổng 1 / log2(r + 1) trên các vị trí r (đếm từ 1) chứa item liên quan trong top-K;
    IDCG@K là DCG khi mọi item liên quan đứng đầu danh sách.
    """
    _check_k(k)
    relevant = set(relevant_items)
    if not relevant:
        raise ValueError("relevant_items must not be empty")
    dcg = sum(1.0 / math.log2(rank + 1)
              for rank, item in enumerate(ranked_items[:k], start=1) if item in relevant)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(relevant), k) + 1))
    return dcg / idcg


def filter_seen(ranked_items, seen_items, target):
    """Loại item đã thấy khỏi danh sách xếp hạng, giữ nguyên thứ tự; không loại chính nhãn."""
    seen = set(seen_items)
    seen.discard(target)
    return [item for item in ranked_items if item not in seen]


def per_user_metrics(rankings, split, ks=DEFAULT_KS, exclude_seen=False):
    """Tính Recall@K và NDCG@K cho từng user.

    Args:
        rankings: dict ``user_id -> list item_id`` đã xếp hạng giảm dần theo điểm.
            Cần ít nhất max(ks) + số item đã thấy phần tử để còn đủ K item sau khi lọc.
            Có thể truyền toàn bộ xếp hạng; chỉ phần đầu đủ dùng được đọc, nên item lặp
            nằm sau phần này không được phát hiện.
        split: dict ``user_id -> {"target": int, "seen_item_ids": list}`` (xem ``load_split``).
        ks: các giá trị K.
        exclude_seen: loại item đã thấy trước khi cắt top-K (mặc định False, theo quyết định 19/11).

    Returns:
        ``(user_ids, scores)``: ``user_ids`` là mảng user theo thứ tự tăng dần,
        ``scores`` là dict ``"recall@K"/"ndcg@K" -> mảng numpy`` cùng thứ tự,
        dùng được trực tiếp cho paired t-test theo user.

    Raises:
        ValueError: user thiếu danh sách xếp hạng, danh sách có item lặp,
            hoặc item ID không phải số nguyên (ví dụ token chuỗi của RecBole chưa đổi về item_id).

    Warns:
        UserWarning: có user mà danh sách sau khi lọc ngắn hơn max(ks); điểm của các user này bị tính thấp đi.
    """
    ks = tuple(ks)
    for k in ks:
        _check_k(k)
    missing = [user for user in split if user not in rankings]
    if missing:
        raise ValueError(f"{len(missing)} user(s) have no ranking, e.g. {missing[:5]}")

    user_ids = sorted(split)
    scores = {f"{name}@{k}": np.zeros(len(user_ids)) for k in ks for name in ("recall", "ndcg")}
    max_k = max(ks)
    short_users = 0
    for index, user in enumerate(user_ids):
        target = split[user]["target"]
        seen = split[user]["seen_item_ids"] if exclude_seen else []
        # Lọc chỉ bỏ tối đa len(seen) item, nên max_k + len(seen) phần tử đầu là đủ;
        # cắt trước để toàn bộ xếp hạng 12.101 item không làm chậm vòng lặp.
        ranked = list(rankings[user][:max_k + len(seen)])
        _check_item_ids(user, ranked, target)
        if len(set(ranked)) != len(ranked):
            raise ValueError(f"Ranking of user {user} contains duplicate items")
        if exclude_seen:
            ranked = filter_seen(ranked, seen, target)
        if len(ranked) < max_k:
            short_users += 1
        for k in ks:
            scores[f"recall@{k}"][index] = recall_at_k(ranked, [target], k)
            scores[f"ndcg@{k}"][index] = ndcg_at_k(ranked, [target], k)
    if short_users:
        warnings.warn(
            f"{short_users} user(s) have fewer than {max_k} ranked items "
            f"(exclude_seen={exclude_seen}); export a longer ranking from the model",
            stacklevel=2,
        )
    return np.array(user_ids), scores


def evaluate(rankings, split, ks=DEFAULT_KS, exclude_seen=False):
    """Trung bình Recall@K, NDCG@K trên mọi user của tập nhãn; trả về dict ``"recall@5" -> float``."""
    _, scores = per_user_metrics(rankings, split, ks=ks, exclude_seen=exclude_seen)
    return {name: float(values.mean()) for name, values in scores.items()}


def load_split(path):
    """Đọc ``validation.jsonl`` hoặc ``test.jsonl`` thành dict ``user_id -> {target, seen_item_ids}``."""
    split = {}
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            split[row["user_id"]] = {"target": row["target"], "seen_item_ids": row["seen_item_ids"]}
    return split


def _check_item_ids(user, ranked, target):
    """Item ID phải là số nguyên; so khớp chuỗi '5' với số 5 sẽ luôn trượt mà không báo lỗi."""
    for item in (target, *ranked):
        if isinstance(item, bool) or not isinstance(item, (int, np.integer)):
            raise ValueError(f"User {user}: item IDs must be integers, got {item!r} ({type(item).__name__})")


def _check_k(k):
    if type(k) is not int or k < 1:
        raise ValueError("k must be a positive integer")
