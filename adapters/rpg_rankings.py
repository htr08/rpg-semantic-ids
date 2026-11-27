"""Ghi top-K từ lượt chấm checkpoint của RPG theo ID và split của nhóm."""

import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory

from src.evaluate import evaluate
from src.preprocessing import read_records

USER_COLUMN = "ranking_user_id"


def match_test_users(pipeline, processed_dir):
    """Khớp mapping, lịch sử và nhãn trước khi gắn user ID vào batch RPG."""
    mapping = json.loads((processed_dir / "id_mapping.json").read_bytes())
    raw_mapping = pipeline.raw_dataset.id_mapping
    for kind in ("user", "item"):
        originals = mapping[f"id2{kind}"][1:]
        if (raw_mapping[f"id2{kind}"][1:] != originals
                or any(raw_mapping[f"{kind}2id"].get(key) != i for i, key in enumerate(originals, 1))):
            raise ValueError(f"Mapping {kind} của RPG không khớp mapping chung")
    expected = {row["user_id"]: row for _, row in read_records(processed_dir / "test.jsonl")}
    raw_test = pipeline.split_datasets["test"]
    user_ids = []
    max_history = pipeline.config["max_item_seq_len"]
    for row in raw_test:
        user = raw_mapping["user2id"][row["user"]]
        seq = [raw_mapping["item2id"][item] for item in row["item_seq"][-(max_history + 1):]]
        sample = expected.get(user)
        if not sample or not seq or seq[:-1] != sample["history"] or seq[-1] != sample["target"]:
            raise ValueError(f"Lịch sử hoặc nhãn test của RPG lệch split chung ở user {user}")
        user_ids.append(user)
    if len(user_ids) != len(expected) or set(user_ids) != expected.keys():
        raise ValueError("Tập user test của RPG không khớp split chung")
    return user_ids, expected, len(mapping["id2item"]) - 1


def write_rankings(path, rankings, *, overwrite):
    """Chỉ công bố JSONL hoàn chỉnh sau khi chấm và kiểm tra xong."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as folder:
        temporary = Path(folder) / path.name
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            for user, items in sorted(rankings.items()):
                stream.write(json.dumps({"user_id": user, "ranked_items": items}) + "\n")
        if path.exists() and not overwrite:
            raise FileExistsError(path)
        temporary.replace(path)


def run_with_rankings(pipeline, processed_dir, output, *, overwrite=False, visited=None):
    """Thu ranking từ chính generate() dùng để tính metric ở lượt test."""
    if pipeline.accelerator.num_processes != 1 or pipeline.config["use_ddp"]:
        raise ValueError("Xuất ranking chỉ hỗ trợ một tiến trình")
    users, expected, n_items = match_test_users(pipeline, processed_dir)
    test_data = pipeline.tokenized_datasets["test"]
    model = pipeline.trainer.model
    generate = model.generate
    k = max(pipeline.config["topk"])
    rankings = {}

    def capture(batch, *args, **kwargs):
        output = generate({key: value for key, value in batch.items() if key != USER_COLUMN}, *args, **kwargs)
        predictions = output[0] if isinstance(output, tuple) else output
        batch_users = batch[USER_COLUMN].tolist()
        if visited is not None:
            counts = output[1].squeeze(-1).tolist()
            for user, count in zip(batch_users, counts, strict=True):
                if not k <= count <= n_items or int(count) != count:
                    raise ValueError(f"Số item đã thăm của user {user} ngoài phạm vi")
                visited[user] = int(count)
        # RPG trả tensor (batch, k, 1) vì mỗi ứng viên có một item ID.
        for user, items, labels in zip(batch_users, predictions.squeeze(-1).tolist(),
                                       batch["labels"].tolist(), strict=True):
            if user in rankings or labels != [expected[user]["target"]]:
                raise ValueError(f"User {user} bị lặp hoặc nhãn của batch test không khớp")
            if len(items) != k or len(set(items)) != k or any(not 1 <= item <= n_items for item in items):
                raise ValueError(f"Ranking của user {user} thiếu item, có item trùng hoặc ID ngoài catalog")
            rankings[user] = items
        return output

    pipeline.tokenized_datasets["test"] = test_data.add_column(USER_COLUMN, users).with_format("torch")
    model.generate = capture
    try:
        result = pipeline.run()
    finally:
        model.generate = generate
        pipeline.tokenized_datasets["test"] = test_data

    if rankings.keys() != expected.keys():
        raise ValueError("Ranking không đủ user test")
    scored = evaluate(rankings, expected, ks=pipeline.config["topk"], exclude_seen=False)
    for metric, value in scored.items():
        if not math.isclose(value, result["test_results"][metric], rel_tol=0, abs_tol=1e-6):
            raise ValueError(f"{metric} lệch giữa bộ chấm chung ({value}) và RPG ({result['test_results'][metric]})")
    write_rankings(output, rankings, overwrite=overwrite)
    return result, len(rankings), k
