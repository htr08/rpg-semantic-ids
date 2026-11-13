"""Tự viết (B): lọc k-core, ánh xạ ID và chia dữ liệu theo thời gian.

Chạy bằng ``python -m src.preprocessing``. Chỉ dùng thư viện chuẩn Python.
Train đầy đủ được giữ cho khai phá mẫu; chỉ lịch sử đầu vào bị giới hạn độ dài.
"""

import argparse
import ast
import csv
import gzip
import hashlib
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REVIEWS_FILE = "reviews_Beauty_5.json.gz"
METADATA_FILE = "meta_Beauty.json.gz"
INTERACTION_FIELDS = ["interaction_id", "user_id", "item_id", "timestamp", "source_line"]
SPLIT_FIELDS = ["user_id", "train_start", "train_stop", "validation_index", "test_index"]


@dataclass(frozen=True)
class Review:
    """Một tương tác gốc; source_line bắt đầu từ 1 để truy ngược dữ liệu."""

    user: str
    item: str
    timestamp: int
    source_line: int


def sha256_file(path):
    """Tính checksum dạng luồng, không nạp cả file vào bộ nhớ."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_records(path):
    """Đọc JSON hoặc literal Python của Amazon 2014 mà không thực thi mã."""
    path = Path(path)
    open_file = gzip.open if path.suffix == ".gz" else open
    with open_file(path, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    record = ast.literal_eval(line)
                if not isinstance(record, dict):
                    raise ValueError("record must be an object")
            except (ValueError, SyntaxError) as error:
                raise ValueError(f"{path.name}:{line_number}: invalid record: {error}") from error
            yield line_number, record


def load_reviews(path):
    """Nạp trường cần thiết; giữ nguyên tương tác lặp, không dùng rating."""
    reviews = []
    for line_number, record in read_records(path):
        user = record.get("reviewerID")
        item = record.get("asin")
        timestamp = record.get("unixReviewTime")
        if not isinstance(user, str) or not user.strip():
            raise ValueError(f"{path}:{line_number}: invalid reviewerID")
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{path}:{line_number}: invalid asin")
        if type(timestamp) is not int:
            raise ValueError(f"{path}:{line_number}: unixReviewTime must be an integer")
        reviews.append(Review(user, item, timestamp, line_number))
    if not reviews:
        raise ValueError("The reviews file contains no interactions")
    return reviews


def filter_k_core(reviews, min_core=5):
    """Lọc đồng thời user/item lặp đến hội tụ; trả nhật ký từng vòng loại."""
    if type(min_core) is not int or min_core < 1:
        raise ValueError("min_core must be a positive integer")
    retained = list(reviews)
    rounds = []
    while retained:
        user_counts = Counter(row.user for row in retained)
        item_counts = Counter(row.item for row in retained)
        filtered = [
            row for row in retained
            if user_counts[row.user] >= min_core and item_counts[row.item] >= min_core
        ]
        if len(filtered) == len(retained):
            break
        rounds.append({"before": len(retained), "after": len(filtered)})
        retained = filtered
    return retained, rounds


def iter_training_examples(item_ids, max_history=50):
    """Sinh các tiền tố train; mỗi nhãn chỉ dùng lịch sử trước nó."""
    if type(max_history) is not int or max_history < 1:
        raise ValueError("max_history must be a positive integer")
    for position in range(1, len(item_ids)):
        yield {
            "history": item_ids[max(0, position - max_history):position],
            "target": item_ids[position],
        }


def build_dataset(reviews, max_history=50):
    """Tạo ID từ 1, chuỗi train đầy đủ, ví dụ đánh giá và chỉ số chia từ 0."""
    if type(max_history) is not int or max_history < 1:
        raise ValueError("max_history must be a positive integer")
    grouped = defaultdict(list)
    for review in reviews:
        grouped[review.user].append(review)
    if not grouped:
        raise ValueError("No interactions remain after k-core filtering")

    mapping = {"user2id": {}, "item2id": {}, "id2user": [None], "id2item": [None]}
    result = {
        "id_mapping": mapping, "interactions": [], "train_sequences": [],
        "validation": [], "test": [], "splits": [],
    }
    for user, sequence in grouped.items():
        sequence.sort(key=lambda row: (row.timestamp, row.source_line))
        if len(sequence) < 3:
            raise ValueError(f"User {user!r} needs at least 3 interactions for train/validation/test")
        user_id = len(mapping["id2user"])
        mapping["user2id"][user] = user_id
        mapping["id2user"].append(user)
        start = len(result["interactions"])
        item_ids = []
        for review in sequence:
            if review.item not in mapping["item2id"]:
                mapping["item2id"][review.item] = len(mapping["id2item"])
                mapping["id2item"].append(review.item)
            item_id = mapping["item2id"][review.item]
            item_ids.append(item_id)
            result["interactions"].append({
                "interaction_id": len(result["interactions"]),
                "user_id": user_id, "item_id": item_id,
                "timestamp": review.timestamp, "source_line": review.source_line,
            })
        validation_index = len(result["interactions"]) - 2
        result["splits"].append({
            "user_id": user_id, "train_start": start, "train_stop": validation_index,
            "validation_index": validation_index, "test_index": validation_index + 1,
        })
        result["train_sequences"].append({"user_id": user_id, "item_ids": item_ids[:-2]})
        for key, offset in (("validation", -2), ("test", -1)):
            full_history = item_ids[:offset]
            result[key].append({
                "user_id": user_id, "history": full_history[-max_history:],
                "target": item_ids[offset], "seen_item_ids": sorted(set(full_history)),
                "target_in_history": item_ids[offset] in full_history,
            })
    return result


def load_aligned_metadata(path, id_mapping):
    """Giữ metadata các item đã ánh xạ, theo đúng thứ tự hàng embedding tương lai."""
    selected = {}
    for line_number, record in read_records(path):
        asin = record.get("asin")
        if not isinstance(asin, str) or not asin:
            raise ValueError(f"{path}:{line_number}: invalid asin")
        if asin not in id_mapping["item2id"]:
            continue
        if asin in selected:
            raise ValueError(f"Duplicate metadata for ASIN {asin!r}")
        # Giữ nguyên các trường văn bản cho encoder của A; bỏ ảnh và liên kết mua kèm.
        selected[asin] = {
            key: record[key]
            for key in ("title", "price", "brand", "feature", "categories", "description")
            if key in record
        }
    aligned = []
    for item_id, asin in enumerate(id_mapping["id2item"][1:], start=1):
        aligned.append({
            "item_id": item_id, "asin": asin,
            "metadata_missing": asin not in selected,
            "metadata": selected.get(asin, {}),
        })
    return aligned


def write_json(path, value):
    """Ghi JSON với định dạng cố định để checksum tái lập giữa các lần chạy."""
    with Path(path).open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def write_jsonl(path, rows):
    """Mỗi dòng là một bản ghi UTF-8; không chứa thời gian chạy hay đường dẫn máy."""
    with Path(path).open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_csv(path, rows, fields):
    """Ghi bảng có tên cột và xuống dòng thống nhất trên Windows/Linux."""
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def verify_preprocessed(output_dir, splits_dir, *, config=None):
    """Kiểm tra đầu ra và, khi được truyền vào, cấu hình so với manifest."""
    output_dir, splits_dir = Path(output_dir), Path(splits_dir)
    manifest = json.loads((splits_dir / "manifest.json").read_text(encoding="utf-8"))
    if config is not None:
        for key in ("min_core", "max_history"):
            if type(config[key]) is not int or config[key] != manifest[key]:
                raise ValueError(
                    f"{key}: config value {config[key]!r} does not match manifest value "
                    f"{manifest[key]!r}; rerun preprocessing with this config"
                )
    for name, reference in manifest["artifacts"].items():
        directory = splits_dir if name == "leave_last_out.csv" else output_dir
        path = directory / name
        if path.stat().st_size != reference["size_bytes"] or sha256_file(path) != reference["sha256"]:
            raise ValueError(f"{name}: generated artifact does not match split manifest")
    return manifest


def preprocess(raw_dir, output_dir, splits_dir, *, min_core=5, max_history=50, source_manifest):
    """Kiểm tra nguồn, tạo toàn bộ dữ liệu và manifest kiểm chứng bộ chia."""
    if type(min_core) is not int or min_core < 3:
        raise ValueError("min_core must be an integer >= 3 for leave-last-out")
    if type(max_history) is not int or max_history < 1:
        raise ValueError("max_history must be a positive integer")
    raw_dir, output_dir, splits_dir = map(Path, (raw_dir, output_dir, splits_dir))
    sources = {}
    specifications = {
        spec["filename"]: spec
        for spec in json.loads(Path(source_manifest).read_text(encoding="utf-8"))["files"]
    }
    for name in (REVIEWS_FILE, METADATA_FILE):
        spec = specifications[name]
        path = raw_dir / name
        checksum = sha256_file(path)
        if not spec.get("sha256") or checksum != spec["sha256"]:
            raise ValueError(f"{name}: SHA-256 does not match source manifest")
        sources[name] = {"sha256": checksum, "size_bytes": path.stat().st_size}

    print("Loading reviews and filtering k-core...", flush=True)
    reviews = load_reviews(raw_dir / REVIEWS_FILE)
    retained, rounds = filter_k_core(reviews, min_core)
    dataset = build_dataset(retained, max_history)
    print("Aligning item metadata...", flush=True)
    items = load_aligned_metadata(raw_dir / METADATA_FILE, dataset["id_mapping"])
    train_count = sum(len(row["item_ids"]) for row in dataset["train_sequences"])
    user_count = len(dataset["train_sequences"])
    stats = {
        "source_interactions": len(reviews), "retained_interactions": len(retained),
        "removed_interactions": len(reviews) - len(retained), "filter_rounds": rounds,
        "users": user_count, "items": len(items),
        "min_user_interactions": min(Counter(row.user for row in retained).values()),
        "min_item_interactions": min(Counter(row.item for row in retained).values()),
        "train_interactions": train_count,
        "validation_targets": user_count, "test_targets": user_count,
        "train_plus_validation_interactions": train_count + user_count,
        "average_full_sequence_length": len(retained) / user_count,
        "max_train_sequence_length": max(len(row["item_ids"]) for row in dataset["train_sequences"]),
        "validation_histories_truncated": sum(
            len(row["item_ids"]) > max_history for row in dataset["train_sequences"]
        ),
        "test_histories_truncated": sum(
            len(row["item_ids"]) + 1 > max_history for row in dataset["train_sequences"]
        ),
        "repeated_validation_targets": sum(row["target_in_history"] for row in dataset["validation"]),
        "repeated_test_targets": sum(row["target_in_history"] for row in dataset["test"]),
        "missing_metadata_items": sum(row["metadata_missing"] for row in items),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "id_mapping.json", dataset["id_mapping"])
    write_csv(output_dir / "interactions.csv", dataset["interactions"], INTERACTION_FIELDS)
    for key in ("train_sequences", "validation", "test"):
        write_jsonl(output_dir / f"{key}.jsonl", dataset[key])
    write_jsonl(output_dir / "items.jsonl", items)
    write_json(output_dir / "preprocessing_stats.json", stats)
    write_csv(splits_dir / "leave_last_out.csv", dataset["splits"], SPLIT_FIELDS)

    filenames = [
        "id_mapping.json", "interactions.csv", "train_sequences.jsonl", "validation.jsonl",
        "test.jsonl", "items.jsonl", "preprocessing_stats.json",
    ]
    artifacts = {}
    for name in filenames + ["leave_last_out.csv"]:
        directory = splits_dir if name == "leave_last_out.csv" else output_dir
        path = directory / name
        artifacts[name] = {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
    manifest = {
        "format_version": 1, "protocol": "leave_last_out",
        "min_core": min_core, "max_history": max_history,
        "id_base": 1, "padding_id": 0, "interaction_index_base": 0,
        "train_interval": "[train_start, train_stop)",
        "index_reference": "interaction_id in interactions.csv (header excluded)",
        "id_order": "users in source order; items in first chronological occurrence per user",
        "sources": sources, "artifacts": artifacts,
        "counts": {key: stats[key] for key in (
            "users", "items", "retained_interactions", "train_interactions",
            "validation_targets", "test_targets", "train_plus_validation_interactions",
        )},
    }
    # Manifest ghi cuối cùng để chỉ đánh dấu hoàn tất sau khi mọi đầu ra được tạo.
    write_json(splits_dir / "manifest.json", manifest)
    print(json.dumps(manifest["counts"], ensure_ascii=False, indent=2), flush=True)
    return stats


def main(argv=None):
    """Chạy với đường dẫn mặc định của repo hoặc thư mục riêng qua CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--splits-dir", type=Path, default=ROOT / "data" / "splits")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "preprocessing.json")
    parser.add_argument("--source-manifest", type=Path, default=ROOT / "configs" / "data_sources.json")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        if args.verify_only:
            manifest = verify_preprocessed(args.output_dir, args.splits_dir, config=config)
            print(f"Verified config and {len(manifest['artifacts'])} generated artifacts against split manifest")
            return 0
        preprocess(
            args.raw_dir, args.output_dir, args.splits_dir,
            min_core=config["min_core"], max_history=config["max_history"],
            source_manifest=args.source_manifest,
        )
    except (OSError, EOFError, ValueError, KeyError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
