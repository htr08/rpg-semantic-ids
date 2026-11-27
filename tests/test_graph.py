"""Kiểm tra đồ thị giải mã và phép đo bậc vào của RQ3."""

import numpy as np
import pytest

from src.evaluate import build_evaluation_context
from src.graph import analyze_graph, build_adjacency


def test_adjacency_matches_dense_token_cosines_with_rpg_item_ids():
    torch = pytest.importorskip("torch")
    codes = np.array([[0, 1], [1, 0], [0, 1], [1, 2], [2, 0]])
    weights = torch.rand(8, 4, generator=torch.Generator().manual_seed(42))
    selected = torch.stack([weights[1 + 3 * pos + codes[:, pos]] for pos in range(2)])
    real_scores = sum(
        (torch.nn.functional.cosine_similarity(vectors[:, None], vectors[None, :], dim=-1) + 1) / 2
        for vectors in selected
    ) / len(selected)
    padded_scores = torch.nn.functional.pad(real_scores, (1, 0))
    expected = padded_scores.topk(3, dim=1).indices.numpy()

    actual = build_adjacency(codes, weights, codebook_size=3, k=3, block_size=2)

    np.testing.assert_array_equal(actual, expected)
    assert np.all(np.any(actual == np.arange(1, len(codes) + 1)[:, None], axis=1))


def test_graph_counts_directed_neighbors_and_louvain_uses_undirected_union():
    items = [{"item_id": item, "asin": str(item)} for item in range(1, 5)]
    context = build_evaluation_context(
        [{"user_id": 1, "item_ids": [2, 2, 3]}, {"user_id": 2, "item_ids": [3, 2]}], items,
    )
    labels = {"leaf": ["A", "A", "B", None], "level3": ["C"] * 4}
    adjacency = np.array([[1, 2, 3], [2, 1, 3], [3, 1, 2], [4, 1, 2]])

    report, degree, train_counts = analyze_graph(adjacency, items, context, labels, louvain_seed=42)

    np.testing.assert_array_equal(degree, [3, 3, 2, 0])
    np.testing.assert_array_equal(train_counts, [0, 3, 2, 0])
    assert report["directed_edges"] == 8
    assert report["self_edges_excluded"] == 4
    assert report["louvain"]["undirected_edges"] == 5
    assert report["louvain"]["category_agreement"]["leaf"]["labeled_items"] == 3
    assert [hub["item_id"] for hub in report["top_hubs"]] == [1, 2, 3, 4]


def test_graph_handles_padding_and_edgeless_catalog():
    items = [{"item_id": 1, "asin": "A"}, {"item_id": 2, "asin": "B"}]
    context = build_evaluation_context([], items)
    labels = {"leaf": [None, None], "level3": [None, None]}

    report, degree, _ = analyze_graph(
        np.array([[1, 0], [0, 2]]), items, context, labels, louvain_seed=42,
    )

    np.testing.assert_array_equal(degree, [0, 0])
    assert report["padding_edges_excluded"] == 2
    assert report["louvain"]["community_sizes"] == [1, 1]
    assert report["louvain"]["modularity"] is None
    assert report["popularity"]["spearman"] is None
