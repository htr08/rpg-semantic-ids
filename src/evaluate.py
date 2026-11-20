"""Đánh giá chung: Recall/NDCG, coverage, long-tail, đa dạng danh mục và cold-start.

Bản đầu: C (Tuần 1). Mở rộng (paired t-test, coverage, tỉ lệ long-tail, cold-start theo nhóm tần suất): B.

Mọi mô hình (RecBole, RPG code gốc, AR-Rec) xuất danh sách xếp hạng cho từng user rồi chấm tại đây,
để tránh khác biệt giữa các cách tính metric của từng thư viện.

Quy ước:

- Mỗi user có đúng một nhãn (leave-last-out): ``target`` trong ``validation.jsonl`` hoặc ``test.jsonl``
  (định dạng theo docs/data_contract.md). Item ID là ``item_id`` theo ``id_mapping.json``;
  ID nội bộ của từng thư viện (ví dụ RecBole) phải đổi về ``item_id`` trước khi chấm.
- Tham số ``exclude_seen`` điều khiển việc loại item đã xem trước khi cắt top-K.
- Ngoại lệ do module này chọn (data contract chỉ yêu cầu xử lý tường minh, chưa quy định cách xử lý):
  nếu chính nhãn nằm trong ``seen_item_ids`` (``target_in_history``, mua lại), nhãn không bị loại,
  để mô hình không bị phạt vì một item mà quy tắc lọc tự xóa. Trên Beauty 2014 không có trường hợp này.
- Với một nhãn ở vị trí ``r`` (đếm từ 1) trong top-K đã lọc:
  Recall@K = 1 và NDCG@K = 1 / log2(r + 1); nhãn nằm ngoài top-K cho cả hai bằng 0.
  Với một nhãn duy nhất, IDCG = 1 nên NDCG bằng DCG.
- Giá trị cuối là trung bình cộng trên mọi user có trong tập nhãn.
"""
import argparse
import csv
import json
import math
import sys
import warnings
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np

DEFAULT_KS = (5, 10)
COLD_START_GROUPS = ((0, 5), (6, 10), (11, 15), (16, 20))


@dataclass
class EvaluationContext:
    """Catalog và đặc trưng dùng chung; độ phổ biến chỉ lấy từ train."""

    catalog: frozenset
    train_counts: dict
    head_items: frozenset
    item_categories: dict
    head_fraction: float


def build_evaluation_context(train_sequences, items, head_fraction=0.2):
    """Tạo head theo (-tần suất train, item_id), lấy ceil(20% catalog).

    Mỗi nhãn danh mục là một đường dẫn đầy đủ đến lá; một item có thể có
    nhiều nhãn. Danh mục thiếu/rỗng không được coi là một nhãn chung.
    """
    if not math.isfinite(head_fraction) or not 0 < head_fraction <= 1:
        raise ValueError("head_fraction must be in (0, 1]")
    categories = {}
    for row in items:
        item = row["item_id"]
        _check_item_ids("catalog", [item], item)
        if item <= 0 or item in categories:
            raise ValueError("Catalog item IDs must be positive and unique")
        labels = set()
        for path in row.get("metadata", {}).get("categories", []) or []:
            if (isinstance(path, list) and path
                    and all(isinstance(label, str) and label.strip() for label in path)):
                labels.add(tuple(label.strip() for label in path))
        categories[item] = frozenset(labels)
    if not categories:
        raise ValueError("Catalog must not be empty")
    catalog = frozenset(categories)
    counts = Counter()
    train_users = set()
    for row in train_sequences:
        user = row["user_id"]
        if type(user) is not int or user < 1 or user in train_users:
            raise ValueError("Train user IDs must be positive and unique")
        train_users.add(user)
        for item in row["item_ids"]:
            _check_item_ids("train", [item], item)
            if item not in catalog:
                raise ValueError(f"Train item {item} is not in catalog")
            counts[item] += 1
    counts = {item: counts[item] for item in sorted(catalog)}
    head = sorted(catalog, key=lambda item: (-counts[item], item))[:math.ceil(len(catalog) * head_fraction)]
    return EvaluationContext(catalog, counts, frozenset(head), categories, head_fraction)


def category_diversity(ranked_items, item_categories):
    """Trung bình khoảng cách Jaccard giữa các cặp item có nhãn.

    Không đủ hai item có nhãn: NaN (không đo được), không thay bằng 0.
    """
    labels = [item_categories[item] for item in ranked_items if item_categories[item]]
    if len(labels) < 2:
        return float("nan")
    return sum(1 - len(a & b) / len(a | b) for a, b in combinations(labels, 2)) / math.comb(len(labels), 2)


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


def per_user_metrics(rankings, split, ks=DEFAULT_KS, exclude_seen=True, *, context=None):
    """Tính Recall@K và NDCG@K cho từng user.

    Args:
        rankings: dict ``user_id -> list item_id`` đã xếp hạng giảm dần theo điểm.
            Cần ít nhất max(ks) + số item đã thấy phần tử để còn đủ K item sau khi lọc.
            Có thể truyền toàn bộ xếp hạng; chỉ phần đầu đủ dùng được đọc, nên item lặp
            nằm sau phần này không được phát hiện.
        split: dict ``user_id -> {"target": int, "seen_item_ids": list}`` (xem ``load_split``).
        ks: các giá trị K.
        exclude_seen: loại item đã thấy trước khi cắt top-K.
        context: thêm long_tail_ratio, category_diversity, category_label_coverage,
            recommendation_fill cho từng K; xem docs/evaluation.md.

    Returns:
        ``(user_ids, scores)``: ``user_ids`` là mảng user theo thứ tự tăng dần,
        ``scores`` là dict ``"recall@K"/"ndcg@K" -> mảng numpy`` cùng thứ tự,
        dùng được trực tiếp cho paired t-test theo user.

    Raises:
        ValueError: user thiếu danh sách xếp hạng, danh sách có item lặp,
            hoặc item ID không phải số nguyên (ví dụ token chuỗi của RecBole chưa đổi về item_id).

    Warns:
        UserWarning: có user mà danh sách sau khi lọc ngắn hơn max(ks);
            điểm độ chính xác có thể thấp do thiếu ứng viên, xem recommendation_fill.
    """
    users, scores, _ = _score_rankings(rankings, split, ks, exclude_seen, context)
    return users, scores


def _score_rankings(rankings, split, ks, exclude_seen, context):
    """Dùng cùng top-K cho độ chính xác và mọi metric mở rộng."""
    ks = tuple(dict.fromkeys(ks))
    if not ks or not split:
        raise ValueError("ks and split must not be empty")
    for k in ks:
        _check_k(k)
    missing = [user for user in split if user not in rankings]
    if missing:
        raise ValueError(f"{len(missing)} user(s) have no ranking, e.g. {missing[:5]}")

    user_ids = sorted(split)
    scores = {f"{name}@{k}": np.zeros(len(user_ids)) for k in ks for name in ("recall", "ndcg")}
    recommended = {k: set() for k in ks}
    if context is not None:
        for k in ks:
            for name in ("long_tail_ratio", "category_diversity", "category_label_coverage", "recommendation_fill"):
                scores[f"{name}@{k}"] = np.zeros(len(user_ids))
    max_k = max(ks)
    short_users = 0
    for index, user in enumerate(user_ids):
        target = split[user]["target"]
        seen = split[user]["seen_item_ids"] if exclude_seen else []
        # Lọc chỉ bỏ tối đa len(seen) item, nên max_k + len(seen) phần tử đầu là đủ;
        # cắt trước để toàn bộ xếp hạng 12.101 item không làm chậm vòng lặp.
        ranked = list(rankings[user][:max_k + len(seen)])
        _check_item_ids(user, ranked, target)
        if context is not None:
            _check_item_ids(user, seen, target)
            unknown = set(ranked) | {target} | set(seen)
            unknown -= context.catalog
            if unknown:
                raise ValueError(f"User {user}: item IDs outside catalog: {list(unknown)[:5]}")
        if len(set(ranked)) != len(ranked):
            raise ValueError(f"Ranking of user {user} contains duplicate items")
        if exclude_seen:
            ranked = filter_seen(ranked, seen, target)
        if len(ranked) < max_k:
            short_users += 1
        for k in ks:
            scores[f"recall@{k}"][index] = recall_at_k(ranked, [target], k)
            scores[f"ndcg@{k}"][index] = ndcg_at_k(ranked, [target], k)
            if context is not None:
                top = ranked[:k]
                recommended[k].update(top)
                size = len(top)
                scores[f"long_tail_ratio@{k}"][index] = (
                    sum(item not in context.head_items for item in top) / size if size else 0.0
                )
                scores[f"category_diversity@{k}"][index] = category_diversity(top, context.item_categories)
                scores[f"category_label_coverage@{k}"][index] = (
                    sum(bool(context.item_categories[item]) for item in top) / size if size else 0.0
                )
                scores[f"recommendation_fill@{k}"][index] = size / k
    if short_users:
        warnings.warn(
            f"{short_users} user(s) have fewer than {max_k} ranked items "
            f"(exclude_seen={exclude_seen}); export a longer ranking from the model",
            stacklevel=2,
        )
    coverage = {f"coverage@{k}": len(recommended[k]) / len(context.catalog) for k in ks} if context else {}
    return np.array(user_ids), scores, coverage


def _aggregate_scores(scores, coverage):
    result = {}
    for name, values in scores.items():
        valid = values[np.isfinite(values)]
        result[name] = float(valid.mean()) if len(valid) else None
        if name.startswith("category_diversity@"):
            result[name.replace("category_diversity@", "category_diversity_users@")] = len(valid)
    return {**result, **coverage}


def _cold_start_scores(users, scores, split, train_counts):
    """Tính NDCG trung bình theo tần suất train của item đúng trong từng mẫu."""
    frequencies = np.array([train_counts[split[user]["target"]] for user in users])
    ndcg = {name: values for name, values in scores.items() if name.startswith("ndcg@")}
    groups = []
    for lower, upper in COLD_START_GROUPS:
        mask = (frequencies >= lower) & (frequencies <= upper)
        count = int(mask.sum())
        groups.append({
            "min_train_interactions": lower, "max_train_interactions": upper,
            "users": count,
            **{name: float(values[mask].mean()) if count else None for name, values in ndcg.items()},
        })
    return {
        "frequency_source": "train interaction count of the held-out target item",
        "groups": groups,
        "users_above_20": int((frequencies > COLD_START_GROUPS[-1][1]).sum()),
    }


def evaluate(rankings, split, ks=DEFAULT_KS, exclude_seen=True, *, context=None):
    """Trung bình theo user; khi có context thêm các metric mở rộng.

    Coverage tính trên hợp item được gợi ý cho mọi user / toàn catalog.
    Đa dạng chỉ lấy user có ít nhất hai item có nhãn, trả None nếu không có.
    Không truyền context giữ nguyên các khóa Recall/NDCG của API ban đầu.
    """
    _, scores, coverage = _score_rankings(rankings, split, ks, exclude_seen, context)
    return _aggregate_scores(scores, coverage)


def load_split(path):
    """Đọc ``validation.jsonl`` hoặc ``test.jsonl`` thành dict ``user_id -> {target, seen_item_ids}``."""
    split = {}
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            user = row["user_id"]
            if type(user) is not int or user < 1 or user in split:
                raise ValueError("Split user IDs must be positive and unique")
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


def _read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def load_rankings(path):
    """Đọc JSONL: mỗi dòng {user_id: int, ranked_items: list[int]}.

    ID dương theo mapping chung; danh sách theo điểm giảm dần, chưa lọc seen.
    User trùng hoặc thiếu khóa được báo lỗi, không ghi đè âm thầm.
    """
    rankings = {}
    for row in _read_jsonl(path):
        user, ranked = row["user_id"], row["ranked_items"]
        if type(user) is not int or user < 1 or user in rankings:
            raise ValueError("Ranking user IDs must be positive and unique")
        if not isinstance(ranked, list):
            raise ValueError(f"User {user}: ranked_items must be a list")
        _check_item_ids(user, ranked, 1)
        if any(item < 1 for item in ranked):
            raise ValueError(f"User {user}: item IDs must be positive (padding is not an item)")
        if len(set(ranked)) != len(ranked):
            raise ValueError(f"Ranking of user {user} contains duplicate items")
        rankings[user] = ranked
    return rankings


def main(argv=None):
    """Chấm ranking trên dữ liệu khớp manifest; xuất JSON tổng hợp và CSV theo user."""
    from src.preprocessing import ROOT, sha256_file, verify_preprocessed, write_json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rankings", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--processed-dir", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--splits-dir", type=Path, default=ROOT / "data" / "splits")
    parser.add_argument("--ks", type=int, nargs="+", default=list(DEFAULT_KS))
    parser.add_argument("--exclude-seen", action=argparse.BooleanOptionalAction, default=True,
                        help="Loại item đã xem trước khi cắt top-K (mặc định).")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        verify_preprocessed(args.processed_dir, args.splits_dir)
        context = build_evaluation_context(
            _read_jsonl(args.processed_dir / "train_sequences.jsonl"),
            _read_jsonl(args.processed_dir / "items.jsonl"),
        )
        split_path = args.processed_dir / f"{args.split}.jsonl"
        split = load_split(split_path)
        rankings = load_rankings(args.rankings)
        if set(rankings) != set(split):
            raise ValueError("Ranking users must match the selected split exactly")
        users, scores, coverage = _score_rankings(rankings, split, args.ks, args.exclude_seen, context)

        def source(path):
            resolved = path.resolve()
            name = resolved.relative_to(ROOT).as_posix() if resolved.is_relative_to(ROOT) else resolved.as_posix()
            return {"path": name, "sha256": sha256_file(path)}

        report = {
            "model_name": args.model_name, "split": args.split,
            "metrics": _aggregate_scores(scores, coverage),
            "cold_start": _cold_start_scores(users, scores, split, context.train_counts),
            "protocol": {
                "ks": list(dict.fromkeys(args.ks)), "exclude_seen": args.exclude_seen,
                "keep_repurchased_target": True,
                "head_fraction": context.head_fraction,
                "head_order": "train interaction count descending, item_id ascending",
                "category_labels": "set of full category paths to leaves",
                "category_diversity": "mean pairwise Jaccard distance over labelled items; macro over eligible users",
                "short_list_ratio_denominator": "actual returned items; empty list gives 0",
            },
            "counts": {
                "users": len(users), "catalog_items": len(context.catalog),
                "head_items": len(context.head_items),
                "tail_items": len(context.catalog - context.head_items),
                "train_interactions": sum(context.train_counts.values()),
                "catalog_items_with_categories": sum(bool(labels) for labels in context.item_categories.values()),
                "short_ranking_users": {
                    str(k): int(np.sum(scores[f"recommendation_fill@{k}"] < 1)) for k in dict.fromkeys(args.ks)
                },
            },
            "inputs": {
                "rankings": source(args.rankings), "split": source(split_path),
                "train": source(args.processed_dir / "train_sequences.jsonl"),
                "items": source(args.processed_dir / "items.jsonl"),
                "manifest": source(args.splits_dir / "manifest.json"),
                "evaluator": source(Path(__file__)),
            },
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_json(args.output_dir / "metrics.json", report)
        with (args.output_dir / "per_user.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(["user_id", *scores])
            for index, user in enumerate(users):
                writer.writerow([int(user), *[
                    float(values[index]) if np.isfinite(values[index]) else "" for values in scores.values()
                ]])
        print(json.dumps(report["metrics"], ensure_ascii=False, allow_nan=False, indent=2))
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
