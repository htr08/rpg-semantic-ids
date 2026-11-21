"""Baseline RecBole trên bộ chia cố định của B, không tạo lại split hay mapping.

Mô hình/tương đồng từ RecBole 1.2.0; adapter dữ liệu và
xuất ranking do B viết để dùng đúng hợp đồng dữ liệu và bộ chấm chung.
"""

import sys
from types import ModuleType

import numpy as np
from scipy import sparse


def prepare_recbole_runtime():
    """Dùng chế độ no-TensorFlow có sẵn của TensorBoard trước khi import RecBole.

    RecBole import SummaryWriter dù baseline không dùng TensorBoard. Nếu không
    có marker notf, PyTorch/TensorBoard có thể import TensorFlow rồi JAX của
    Kaggle và vấp xung đột NumPy không liên quan tới các mô hình ở đây.
    Không sửa TensorFlow/JAX hoặc NumPy, không bắt/bỏ qua lỗi import mô hình.
    """
    sys.modules.setdefault("tensorboard.compat.notf", ModuleType("tensorboard.compat.notf"))


def top_items(scores, item_ids, limit):
    """Top theo điểm giảm dần, phá hòa ID tăng; xử lý cả hòa tại ngưỡng cắt."""
    scores, item_ids = np.asarray(scores), np.asarray(item_ids)
    if scores.shape != item_ids.shape or scores.ndim != 1 or not np.isfinite(scores).all():
        raise ValueError("Scores and item IDs must be finite one-dimensional arrays of equal size")
    if type(limit) is not int or limit < 1:
        raise ValueError("limit must be positive")
    limit = min(limit, len(scores))
    if not limit:
        return []
    threshold = np.partition(scores, len(scores) - limit)[len(scores) - limit]
    above = np.flatnonzero(scores > threshold)
    tied = np.flatnonzero(scores == threshold)
    tied = tied[np.argsort(item_ids[tied], kind="stable")][:limit - len(above)]
    selected = np.concatenate((above, tied))
    order = np.lexsort((item_ids[selected], -scores[selected]))
    return item_ids[selected[order]].astype(int).tolist()


def interaction_matrix(train, n_users, n_items):
    """Ma trận user-item nhị phân từ train, hàng/cột 0 dành cho padding."""
    rows, cols = [], []
    for row in train:
        user = row["user_id"]
        if not 1 <= user <= n_users:
            raise ValueError("Train user is outside mapping")
        for item in set(row["item_ids"]):
            if not 1 <= item <= n_items:
                raise ValueError("Train item is outside mapping")
            rows.append(user)
            cols.append(item)
    matrix = sparse.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                               shape=(n_users + 1, n_items + 1))
    matrix.data[:] = 1
    return matrix


class FixedCatalog:
    """Chỉ cung cấp kích thước vocab cho RecBole; ID số không bị remap."""

    def __init__(self, n_users, n_items):
        self.sizes = {"user_id": n_users + 1, "item_id": n_items + 1}

    def num(self, field):
        return self.sizes[field]


def fit_pop(context, n_users):
    """Khởi tạo Pop bằng đúng số lần xuất hiện trong train, không đếm lặp qua epoch."""
    prepare_recbole_runtime()
    import torch
    from recbole.model.general_recommender.pop import Pop

    model = Pop({"USER_ID_FIELD": "user_id", "ITEM_ID_FIELD": "item_id", "NEG_PREFIX": "neg_",
                 "device": "cpu"}, FixedCatalog(n_users, len(context.catalog)))
    counts = [0] + [context.train_counts[item] for item in range(1, len(context.catalog) + 1)]
    model.item_cnt = torch.tensor(counts, dtype=torch.long).reshape(-1, 1)
    model.max_cnt = model.item_cnt.max(dim=0).values.clamp(min=1)
    model.eval()
    return model


def fit_itemknn(train, n_users, n_items, *, k, shrink):
    """Dùng cosine/top-k của RecBole, giữ W thưa, không dựng pred_mat user×item dày.

    Khi chấm test, có thể nhân W cố định với lịch sử gồm validation. Validation
    không tham gia fit W. Không làm tròn kết quả về nhị phân sau khi tính cosine.
    """
    prepare_recbole_runtime()
    from recbole.model.general_recommender.itemknn import ComputeSimilarity

    if type(k) is not int or not 1 <= k <= n_items or not np.isfinite(shrink) or shrink < 0:
        raise ValueError("Invalid ItemKNN k or shrink")
    matrix = interaction_matrix(train, n_users, n_items)
    _, weights = ComputeSimilarity(matrix, topk=k, shrink=shrink).compute_similarity("item")
    return weights.tocsr()


def rank_itemknn(weights, split, catalog, max_k, batch_size=256):
    """Score = lịch sử đầu vào nhị phân × W; toàn catalog là ứng viên."""
    item_ids = np.asarray(sorted(catalog))
    users, rankings = sorted(split), {}
    for start in range(0, len(users), batch_size):
        selected = users[start:start + batch_size]
        rows, cols = [], []
        for index, user in enumerate(selected):
            history_items = set(split[user]["history"])
            rows.extend([index] * len(history_items))
            cols.extend(history_items)
        history = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)),
                                    shape=(len(selected), weights.shape[0]))
        scores = (history @ weights).toarray()
        for index, user in enumerate(selected):
            rankings[user] = top_items(scores[index, item_ids], item_ids,
                                      max_k + len(split[user]["seen_item_ids"]))
    return rankings


def rank_pop(model, split, catalog, max_k, batch_size=256):
    """Score toàn catalog bằng RecBole rồi xuất prefix đủ dài trước lọc seen."""
    prepare_recbole_runtime()
    import torch
    from recbole.data.interaction import Interaction

    users, rankings = sorted(split), {}
    item_ids = np.asarray(sorted(catalog))
    model.eval()
    with torch.no_grad():
        for start in range(0, len(users), batch_size):
            selected = users[start:start + batch_size]
            features = {"user_id": torch.tensor(selected, device=model.device)}
            scores = model.full_sort_predict(Interaction(features)).reshape(len(selected), -1).cpu().numpy()
            for index, user in enumerate(selected):
                rankings[user] = top_items(scores[index, item_ids], item_ids,
                                          max_k + len(split[user]["seen_item_ids"]))
    return rankings
