"""Rút cạnh đồ thị RPG và lấy mẫu ứng viên cho ablation (thành viên B)."""

import numpy as np
from scipy.sparse import csr_matrix


def edges(adjacency):
    """(Hàng nguồn, cột trong adjacency, đích) của cạnh thật. Đích là ID item."""
    items = np.arange(1, len(adjacency) + 1, dtype=np.int32)
    sources, slots = np.where((adjacency > 0) & (adjacency != items[:, None]))
    return sources, slots, adjacency[sources, slots]


def reduce_adjacency(adjacency, method, *, indegree_cap):
    """Giữ cạnh hai chiều hoặc giữ các cạnh vào có hạng từ nguồn cao nhất."""
    sources, slots, targets = edges(adjacency)
    n_items = len(adjacency)
    if method == "mutual":
        arcs = csr_matrix((np.ones(len(sources), dtype=bool), (sources, targets - 1)), shape=(n_items, n_items))
        keep = np.asarray(arcs[targets - 1, sources]).ravel()
    elif method == "indegree_cap":
        if indegree_cap < 1:
            raise ValueError("Giới hạn bậc vào phải dương")
        order = np.lexsort((sources, slots, targets))
        counts = np.bincount(targets, minlength=n_items + 1)
        incoming_rank = np.arange(len(order)) - np.repeat(counts.cumsum() - counts, counts)
        keep = np.zeros(len(order), dtype=bool)
        keep[order[incoming_rank < indegree_cap]] = True
    else:
        raise ValueError(f"Phương pháp giảm hub không hợp lệ: {method}")
    # RPG dùng adjacency hình chữ nhật và torch.unique trên hợp láng giềng.
    # Lặp ID nguồn ở ô rỗng để giữ ứng viên hiện tại mà không thêm cạnh giữa hai item.
    reduced = adjacency.copy()
    reduced[sources[~keep], slots[~keep]] = sources[~keep] + 1
    return reduced


def sample_candidates(budgets, n_items, rng):
    """Lấy mẫu đều không hoàn lại. Mask loại ô đệm trước khi chọn top-K."""
    budgets = np.asarray(budgets, dtype=int)
    if budgets.ndim != 1 or not len(budgets) or budgets.min() < 1 or budgets.max() > n_items:
        raise ValueError("Số ứng viên lấy mẫu phải nằm trong catalog")
    candidates = np.ones((len(budgets), budgets.max()), dtype=np.int64)
    for row, count in enumerate(budgets):
        candidates[row, :count] = np.sort(rng.choice(n_items, size=count, replace=False)) + 1
    return candidates, np.arange(candidates.shape[1]) < budgets[:, None]
