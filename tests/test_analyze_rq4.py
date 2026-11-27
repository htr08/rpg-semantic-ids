"""Kiểm tra gộp điểm theo user, phân khúc train và phép đo hub trong top-10."""

import json

import numpy as np
import pytest

from scripts.analyze_rq4 import METRICS, exposure, graph_indegree, load_model, segment_users
from src.evaluate import build_evaluation_context
from src.preprocessing import sha256_file


def test_load_model_pairs_by_user_and_rejects_replaced_rankings(tmp_path):
    inputs = {"split": "test-sha", "train": "train-sha", "items": "items-sha", "manifest": "manifest-sha"}
    dirs = {42: str(tmp_path / "first"), 43: str(tmp_path / "second")}
    for path, rows in zip(map(tmp_path.joinpath, dirs.values()), [[(1, 1.0), (2, 0.0)], [(2, 0.5), (1, 0.5)]],
                          strict=True):
        path.mkdir()
        ranking_path = path / "rankings.jsonl"
        ranking_path.write_text("ranking", encoding="utf-8")
        (path / "per_user.csv").write_text("user_id,ndcg@10\n" + "".join(f"{uid},{score}\n" for uid, score in rows),
                                           encoding="utf-8")
        metrics = {
            "split": "test", "protocol": {"exclude_seen": False, "ks": [5, 10]}, "counts": {"users": len(rows)},
            "metrics": dict.fromkeys(METRICS, 0.5),
            "inputs": {**{key: {"sha256": value} for key, value in inputs.items()},
                       "rankings": {"sha256": sha256_file(ranking_path)}},
        }
        (path / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")

    aggregated, scores = load_model(dirs, [1, 2], inputs)

    assert scores == {1: 0.75, 2: 0.25}
    assert aggregated["metrics"]["ndcg@10"] == {"mean": 0.5, "sample_std": 0.0}
    (tmp_path / "second" / "rankings.jsonl").write_text("replacement", encoding="utf-8")
    with pytest.raises(ValueError, match="ranking không khớp"):
        load_model(dirs, [1, 2], inputs)


def test_segments_use_train_features_and_keep_each_user_once():
    histories = [[1, 1], [1, 1, 1], [2, 3, 2, 3], [2, 3, 2, 3, 2]]
    train = [{"user_id": i, "item_ids": history} for i, history in enumerate(histories, 1)]
    catalog = [
        {"item_id": i, "metadata": {"categories": [[category]]}}
        for i, category in enumerate(("A", "B", "C"), 1)
    ]
    context = build_evaluation_context(train, catalog)

    masks, profiles, method = segment_users(histories, context, n_clusters=2, seed=42)

    np.testing.assert_array_equal(np.sum(masks, axis=0), [1, 1, 1, 1])
    assert [profile["mean_history_length"] for profile in profiles] == [2.5, 4.5]
    assert [profile["mean_head_share"] for profile in profiles] == [1, 0]
    assert method["users_with_imputed_diversity"] == 0


def test_hub_exposure_retains_zero_count_items_and_excludes_self_edges():
    adjacency = np.array([[1, 2, 0], [2, 1, 0], [3, 1, 2], [4, 1, 2]])
    degree = graph_indegree(adjacency, 4)
    result, counts = exposure(degree, {1: [1, 2], 2: [2, 1]})

    np.testing.assert_array_equal(degree, [3, 3, 0, 0])
    np.testing.assert_array_equal(counts, [2, 2, 0, 0])
    assert result["top10_slots"] == 4
    assert result["item_coverage"] == 2
    assert result["spearman"] == pytest.approx(1)
