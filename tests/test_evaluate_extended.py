"""Đối chiếu công thức tính tay, rò rỉ holdout và đường chạy CLI."""

import csv
import json
import math

import numpy as np
import pytest

from src.evaluate import (
    _cold_start_scores, build_evaluation_context, evaluate, load_rankings, load_split,
    main, per_user_metrics,
)
from src.preprocessing import sha256_file, write_json, write_jsonl


@pytest.fixture
def context():
    # Head 20% của 5 item = {1}; item 4 có hai nhãn, item 5 không có nhãn.
    labels = [[['Beauty', 'A']], [['Beauty', 'A']], [['Beauty', 'B']],
              [['Beauty', 'A'], ['Beauty', 'B']], []]
    items = [{"item_id": i, "metadata": {"categories": paths}}
             for i, paths in enumerate(labels, 1)]
    train = [{"user_id": 1, "item_ids": [1, 1, 1, 2, 3]}]
    return build_evaluation_context(train, items)


def test_extended_metrics_share_filtered_top_k(context):
    split = {1: {"target": 2, "seen_item_ids": [1]},
             2: {"target": 4, "seen_item_ids": []}}
    rankings = {1: [1, 2, 3, 4, 5], 2: [1, 4, 2, 3, 5], 99: [5]}
    result = evaluate(rankings, split, ks=(1, 2), context=context)
    # Sau lọc: [2,3] và [1,4]. Hợp = 4/5 catalog; tail = (1 + 1/2)/2.
    # Jaccard(A,B) = 1; Jaccard(A,{A,B}) = 1/2 → trung bình 3/4.
    assert result["coverage@2"] == pytest.approx(4 / 5)
    assert result["long_tail_ratio@2"] == pytest.approx(3 / 4)
    assert result["category_diversity@2"] == pytest.approx(3 / 4)
    assert result["category_label_coverage@2"] == 1
    assert result["recommendation_fill@2"] == 1
    assert result["recall@2"] == 1
    assert result["ndcg@2"] == pytest.approx((1 + 1 / math.log2(3)) / 2)
    assert result["category_diversity@1"] is None
    assert result["category_diversity_users@1"] == 0
    # API cũ vẫn cho đúng cùng độ chính xác và không thêm khóa.
    original = evaluate(rankings, split, ks=(1, 2))
    assert len(original) == 4
    assert all(result[key] == value for key, value in original.items())


def test_context_uses_train_counts_and_catalog_including_unseen_items(context):
    assert context.train_counts == {1: 3, 2: 1, 3: 1, 4: 0, 5: 0}
    assert context.head_items == {1}
    split = {1: {"target": 4, "seen_item_ids": []}}
    result = evaluate({1: [4, 5]}, split, ks=(2,), context=context)
    assert result["long_tail_ratio@2"] == 1  # Item chỉ có trong holdout vẫn thuộc tail.
    assert context.head_items == {1}  # Nhãn không đổi phân nhóm độ phổ biến.


def test_tied_popularity_is_deterministic_and_head_uses_ceiling():
    items = [{"item_id": i, "metadata": {}} for i in [6, 5, 4, 3, 2, 1]]
    ctx = build_evaluation_context([], items)
    assert ctx.head_items == {1, 2}  # ceil(6 * 0.2) = 2, phá hòa theo ID tăng.


def test_cold_start_includes_boundary_frequencies_and_misses():
    users = np.arange(1, 10)
    split = {user: {"target": user + 100} for user in users}
    frequencies = [0, 5, 6, 10, 11, 15, 16, 20, 21]
    train_counts = {split[user]["target"]: freq for user, freq in zip(users, frequencies)}
    scores = {"ndcg@10": np.array([1, 0, .5, 0, 0, 0, .5, 1, 1]), "recall@10": np.ones(9)}
    result = _cold_start_scores(users, scores, split, train_counts)
    assert [(group["min_train_interactions"], group["max_train_interactions"])
            for group in result["groups"]] == [(0, 5), (6, 10), (11, 15), (16, 20)]
    assert [group["users"] for group in result["groups"]] == [2, 2, 2, 2]
    assert [group["ndcg@10"] for group in result["groups"]] == [.5, .25, 0, .75]
    assert result["users_above_20"] == 1


def test_diversity_reports_missing_labels_and_eligible_users(context):
    split = {2: {"target": 3, "seen_item_ids": []},
             1: {"target": 5, "seen_item_ids": []}}
    rankings = {1: [1, 5], 2: [1, 3]}
    users, scores = per_user_metrics(rankings, split, ks=(2,), context=context)
    assert users.tolist() == [1, 2]
    assert np.isnan(scores["category_diversity@2"][0])
    assert scores["category_diversity@2"][1] == 1
    result = evaluate(rankings, split, ks=(2,), context=context)
    assert result["category_label_coverage@2"] == .75
    assert result["category_diversity@2"] == 1
    assert result["category_diversity_users@2"] == 1


def test_full_category_paths_distinguish_same_leaf_name():
    items = [{"item_id": 1, "metadata": {"categories": [["Hair", "Brushes"]]}},
             {"item_id": 2, "metadata": {"categories": [["Makeup", "Brushes"]]}}]
    ctx = build_evaluation_context([], items)
    result = evaluate({1: [1, 2]}, {1: {"target": 1, "seen_item_ids": []}}, ks=(2,), context=ctx)
    assert result["category_diversity@2"] == 1


@pytest.mark.parametrize("ranked,tail,fill", [([1], 0, .5), ([3], 1, .5), ([], 0, 0)])
def test_short_and_empty_lists_are_visible(context, ranked, tail, fill):
    with pytest.warns(UserWarning, match="fewer than 2"):
        result = evaluate({1: ranked}, {1: {"target": 3, "seen_item_ids": []}},
                          ks=(2,), context=context)
    assert result["long_tail_ratio@2"] == tail
    assert result["recommendation_fill@2"] == fill
    assert result["category_diversity@2"] is None


def test_repurchased_target_is_kept_for_all_metrics(context):
    result = evaluate({1: [1, 3]}, {1: {"target": 1, "seen_item_ids": [1]}},
                      ks=(2,), context=context)
    assert result["recall@2"] == 1
    assert result["long_tail_ratio@2"] == .5
    assert result["coverage@2"] == .4


@pytest.mark.parametrize("field,value", [("target", 0), ("target", 8),
                                         ("seen_item_ids", [9]), ("seen_item_ids", [True])])
def test_invalid_split_catalog_ids_raise(context, field, value):
    row = {"target": 2, "seen_item_ids": []}
    row[field] = value
    with pytest.raises(ValueError):
        evaluate({1: [1, 2]}, {1: row}, ks=(2,), context=context)


@pytest.mark.parametrize("ranked", [[0, 1], [1, 99], [1, 1]])
def test_invalid_recommendations_raise(context, ranked):
    with pytest.raises(ValueError):
        evaluate({1: ranked}, {1: {"target": 2, "seen_item_ids": []}}, ks=(2,), context=context)


@pytest.mark.parametrize("fraction", [0, -1, 1.1, float("nan")])
def test_invalid_head_fraction(fraction):
    with pytest.raises(ValueError):
        build_evaluation_context([], [{"item_id": 1}], head_fraction=fraction)


def test_invalid_context_sources_fail():
    with pytest.raises(ValueError, match="empty"):
        build_evaluation_context([], [])
    with pytest.raises(ValueError, match="unique"):
        build_evaluation_context([], [{"item_id": 1}, {"item_id": 1}])
    with pytest.raises(ValueError, match="not in catalog"):
        build_evaluation_context([{"user_id": 1, "item_ids": [9]}], [{"item_id": 1}])
    with pytest.raises(ValueError, match="unique"):
        build_evaluation_context([{"user_id": 1, "item_ids": []}] * 2, [{"item_id": 1}])


@pytest.mark.parametrize("ks,split", [((), {1: {}}), ((2,), {})])
def test_empty_evaluation_inputs_fail(ks, split):
    with pytest.raises(ValueError, match="empty"):
        evaluate({}, split, ks=ks)


@pytest.mark.parametrize("rows", [
    [{"user_id": 1, "ranked_items": [1]}] * 2,
    [{"user_id": 0, "ranked_items": [1]}],
    [{"user_id": 1, "ranked_items": "1"}],
    [{"user_id": 1, "ranked_items": ["1"]}],
    [{"user_id": 1, "ranked_items": [0]}],
])
def test_invalid_ranking_files_fail(tmp_path, rows):
    path = tmp_path / "rankings.jsonl"
    write_jsonl(path, rows)
    with pytest.raises(ValueError):
        load_rankings(path)


def test_duplicate_split_user_is_not_overwritten(tmp_path):
    path = tmp_path / "split.jsonl"
    write_jsonl(path, [{"user_id": 1, "target": 2, "seen_item_ids": []}] * 2)
    with pytest.raises(ValueError, match="unique"):
        load_split(path)


@pytest.fixture
def cli_case(tmp_path):
    processed, splits, output = [tmp_path / name for name in ("processed", "splits", "output")]
    processed.mkdir()
    splits.mkdir()
    write_jsonl(processed / "items.jsonl", [
        {"item_id": i, "metadata": {"categories": [["Beauty", str(i)]]}} for i in range(1, 6)
    ])
    write_jsonl(processed / "train_sequences.jsonl", [{"user_id": 1, "item_ids": [1, 1, 2]}])
    write_jsonl(processed / "validation.jsonl", [{"user_id": 1, "target": 4, "seen_item_ids": [1, 2]}])
    artifacts = {path.name: {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
                 for path in processed.iterdir()}
    write_json(splits / "manifest.json", {"artifacts": artifacts})
    rankings = tmp_path / "rankings.jsonl"
    write_jsonl(rankings, [{"user_id": 1, "ranked_items": [1, 2, 4, 3, 5]}])
    args = ["--rankings", str(rankings), "--model-name", "fixture", "--ks", "2",
            "--processed-dir", str(processed), "--splits-dir", str(splits), "--output-dir", str(output)]
    return args, processed, output, rankings


def test_cli_end_to_end_and_provenance(cli_case):
    args, processed, output, rankings = cli_case
    assert main(args) == 0
    report = json.loads((output / "metrics.json").read_text())
    assert report["protocol"]["exclude_seen"] is True
    assert report["metrics"]["coverage@2"] == .4
    assert report["metrics"]["ndcg@2"] == 1
    groups = report["cold_start"]["groups"]
    assert [group["users"] for group in groups] == [1, 0, 0, 0]
    assert [group["ndcg@2"] for group in groups] == [1, None, None, None]
    assert report["metrics"]["long_tail_ratio@2"] == 1
    assert report["inputs"]["rankings"]["sha256"] == sha256_file(rankings)
    assert report["inputs"]["train"]["sha256"] == sha256_file(processed / "train_sequences.jsonl")
    assert report["counts"]["short_ranking_users"] == {"2": 0}
    with (output / "per_user.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1 and rows[0]["user_id"] == "1"
    assert float(rows[0]["category_diversity@2"]) == 1


def test_cli_keeps_seen_items_when_requested(cli_case):
    args, _, output, _ = cli_case
    assert main([*args, "--no-exclude-seen"]) == 0
    report = json.loads((output / "metrics.json").read_text())
    assert report["protocol"]["exclude_seen"] is False
    assert report["metrics"]["ndcg@2"] == 0
    assert report["cold_start"]["groups"][0]["ndcg@2"] == 0
    with (output / "per_user.csv").open() as stream:
        assert float(next(csv.DictReader(stream))["ndcg@2"]) == 0


def test_cli_rejects_modified_data_before_writing_outputs(cli_case, capsys):
    args, processed, output, _ = cli_case
    with (processed / "train_sequences.jsonl").open("a") as stream:
        stream.write("\n")
    assert main(args) == 1
    assert "manifest" in capsys.readouterr().err
    assert not output.exists()


def test_cli_rejects_user_mismatch(cli_case, capsys):
    args, _, output, rankings = cli_case
    write_jsonl(rankings, [{"user_id": 2, "ranked_items": [1, 2]}])
    assert main(args) == 1
    assert "match" in capsys.readouterr().err
    assert not output.exists()


def test_cli_serializes_unmeasurable_diversity_as_null(cli_case):
    args, _, output, _ = cli_case
    args[args.index("--ks") + 1] = "1"
    assert main(args) == 0
    text = (output / "metrics.json").read_text()
    assert "NaN" not in text
    assert json.loads(text)["metrics"]["category_diversity@1"] is None
    with (output / "per_user.csv").open() as stream:
        row = next(csv.DictReader(stream))
    assert row["category_diversity@1"] == ""
