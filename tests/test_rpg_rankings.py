"""Kiểm tra bàn giao ranking từ lượt chấm RPG mà không cần checkpoint hay GPU."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from adapters.rpg_rankings import USER_COLUMN, run_with_rankings
from scripts.run_rpg import parse_args
from src.evaluate import load_rankings


class FakeTestDataset:
    def add_column(self, name, values):
        assert name == USER_COLUMN
        return SimpleNamespace(with_format=lambda _: values)


@pytest.fixture
def evaluation(tmp_path):
    mapping = {"id2user": [None, "u1", "u2"], "id2item": [None, "a", "b", "c"]}
    (tmp_path / "id_mapping.json").write_text(json.dumps(mapping), encoding="utf-8")
    rows = [
        {"user_id": 1, "history": [3], "target": 1, "seen_item_ids": [3]},
        {"user_id": 2, "history": [1], "target": 2, "seen_item_ids": [1]},
    ]
    (tmp_path / "test.jsonl").write_text("\n".join(map(json.dumps, rows)), encoding="utf-8")

    def generate(batch, n_return_sequences):
        assert USER_COLUMN not in batch and n_return_sequences == 2
        return np.array([[[2], [1]], [[3], [2]]]), np.array([3, 3])

    pipeline = SimpleNamespace(
        accelerator=SimpleNamespace(num_processes=1),
        config={"use_ddp": False, "topk": [1, 2], "max_item_seq_len": 1},
        raw_dataset=SimpleNamespace(id_mapping={
            "id2user": ["[PAD]", "u1", "u2"], "id2item": ["[PAD]", "a", "b", "c"],
            "user2id": {"[PAD]": 0, "u1": 1, "u2": 2}, "item2id": {"[PAD]": 0, "a": 1, "b": 2, "c": 3},
        }),
        split_datasets={"test": [{"user": "u2", "item_seq": ["c", "a", "b"]},
                                 {"user": "u1", "item_seq": ["c", "a"]}]},
        tokenized_datasets={"test": FakeTestDataset()},
        trainer=SimpleNamespace(model=SimpleNamespace(generate=generate)),
    )

    def run():
        user_ids = pipeline.tokenized_datasets["test"]
        result = pipeline.trainer.model.generate(
            {USER_COLUMN: np.array(user_ids), "labels": np.array([[2], [1]])}, n_return_sequences=2,
        )
        np.testing.assert_array_equal(result[0], [[[2], [1]], [[3], [2]]])
        return {"test_results": {"recall@1": .5, "recall@2": .5, "ndcg@1": .5, "ndcg@2": .5}}

    pipeline.run = run
    return pipeline


def test_export_preserves_native_order_and_seen_items(evaluation, tmp_path):
    output = tmp_path / "rankings.jsonl"
    generate = evaluation.trainer.model.generate
    dataset = evaluation.tokenized_datasets["test"]
    _, users, top_k = run_with_rankings(evaluation, tmp_path, output)

    assert (users, top_k) == (2, 2)
    assert load_rankings(output) == {1: [3, 2], 2: [2, 1]}
    assert evaluation.trainer.model.generate is generate
    assert evaluation.tokenized_datasets["test"] is dataset


def test_export_rejects_mismatched_split_before_running(evaluation, tmp_path):
    evaluation.split_datasets["test"][0]["item_seq"] = ["c", "b", "b"]
    output = tmp_path / "rankings.jsonl"

    with pytest.raises(ValueError, match="lệch split chung"):
        run_with_rankings(evaluation, tmp_path, output)
    assert not output.exists()


def test_metric_mismatch_does_not_publish_rankings(evaluation, tmp_path):
    run = evaluation.run

    def mismatched_run():
        result = run()
        result["test_results"]["ndcg@2"] = 0
        return result

    evaluation.run = mismatched_run
    output = tmp_path / "rankings.jsonl"
    with pytest.raises(ValueError, match="ndcg@2 lệch"):
        run_with_rankings(evaluation, tmp_path, output)
    assert not output.exists()


def test_export_requires_a_checkpoint():
    with pytest.raises(SystemExit):
        parse_args(["--rankings-output", "rankings.jsonl"])
