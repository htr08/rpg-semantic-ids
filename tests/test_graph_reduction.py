"""Kiểm tra rút cạnh và kiểm soát số ứng viên ở ablation RQ3."""

from types import SimpleNamespace

import numpy as np
import pytest

from scripts.run_graph_ablation import matched_random_propagation
from src.graph_reduction import edges, reduce_adjacency, sample_candidates


@pytest.fixture
def adjacency():
    return np.array([[1, 2, 3], [2, 1, 4], [3, 2, 4], [4, 3, 2]])


def test_mutual_keeps_exactly_reciprocal_arcs(adjacency):
    reduced = reduce_adjacency(adjacency, "mutual", indegree_cap=1)

    np.testing.assert_array_equal(reduced, [[1, 2, 1], [2, 1, 4], [3, 3, 4], [4, 3, 2]])
    np.testing.assert_array_equal(adjacency[0], [1, 2, 3])


def test_indegree_cap_preserves_best_source_rank_and_breaks_ties_by_source_id(adjacency):
    reduced = reduce_adjacency(adjacency, "indegree_cap", indegree_cap=1)

    np.testing.assert_array_equal(reduced, [[1, 2, 1], [2, 1, 4], [3, 3, 3], [4, 3, 4]])
    _, _, targets = edges(reduced)
    np.testing.assert_array_equal(np.bincount(targets, minlength=5), [0, 1, 1, 1, 1])


def test_random_candidates_have_exact_budgets_and_no_duplicates():
    budgets = [3, 8, 5]
    candidates, valid = sample_candidates(budgets, 8, np.random.default_rng(42))

    np.testing.assert_array_equal(valid.sum(axis=1), budgets)
    for row, mask, size in zip(candidates, valid, budgets, strict=True):
        assert len(set(row[mask])) == size
        assert all(1 <= item <= 8 for item in row[mask])


def test_random_ablation_uses_rpg_token_scores_and_masks_padding():
    torch = pytest.importorskip("torch")
    model = SimpleNamespace(
        dataset=SimpleNamespace(n_items=4),
        item_id2tokens=torch.tensor([[0, 0], [1, 4], [2, 5], [3, 6]]),
    )
    budgets = [2, 3]
    logits = torch.tensor([[10.0, 4, 7, 1, 4, 7], [1, 8, 7, 1, 8, 7]])
    rank = matched_random_propagation(model, budgets, seed=42)
    candidates, valid = sample_candidates(budgets, 3, np.random.default_rng(42))

    predictions, counts = rank(logits, n_return_sequences=2)

    scores = [[5.5, 4, 7], [1, 8, 7]]
    expected = [sorted(row[mask], key=lambda item: scores[i][item - 1], reverse=True)[:2]
                for i, (row, mask) in enumerate(zip(candidates, valid, strict=True))]
    assert predictions.squeeze(-1).tolist() == expected
    assert counts.squeeze(-1).tolist() == budgets
