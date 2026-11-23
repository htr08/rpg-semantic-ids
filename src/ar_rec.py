"""AR-Rec: recommender dựa trên luật kết hợp (tiền đề nằm trong lịch sử user, xếp hạng theo confidence x lift, bổ sung Popularity).

Phụ trách: C (RQ2).

Cách xếp hạng cho một user có tập item đã thấy ``H``:

1. Một luật ``X => Y`` được khớp khi ``X ⊆ H``.
2. Điểm của item ``y`` là ``max`` của ``confidence x lift`` trên mọi luật khớp có ``y ∈ Y``
   (một item xuất hiện ở nhiều luật khớp chỉ lấy luật mạnh nhất, không cộng dồn).
3. Item có điểm > 0 xếp giảm dần theo điểm, hòa thì ``item_id`` tăng dần.
4. Danh sách được bổ sung bằng Popularity (tần suất train giảm dần, hòa thì ``item_id`` tăng dần,
   cùng quy ước với baseline Pop của B) cho đến đủ độ dài yêu cầu.

Quy ước dữ liệu:

- Luật chỉ được khai phá từ train (``patterns.load_train_transactions``). Lịch sử khớp luật là ``seen_item_ids``
  của bộ chia (toàn bộ item đã thấy, không cắt 50), vì luật được khai phá trên tập item đầy đủ của chuỗi train.
  Với bộ chia test, ``seen_item_ids`` gồm cả item validation, còn luật và Popularity vẫn chỉ lấy từ train.
- Theo quyết định 19/11 (``reports/decisions.md``) mô hình không tự loại item đã thấy; ứng viên là toàn catalog.
  Mô-đun xuất ``max_k + len(seen_item_ids)`` item cho mỗi user (như baseline của B), để bộ chấm chung
  vẫn dùng được chế độ ``--exclude-seen`` khi cần đối chiếu.
- Luật khai phá từ ``interactions.csv`` theo bộ chia, còn Popularity đếm từ ``train_sequences.jsonl``; hai nguồn phải cùng
  một lần chia, nên ``check_train_sequences`` so từng user trước khi dựng mô hình. ``check_split_file`` so tương tự
  ``validation.jsonl`` hoặc ``test.jsonl`` (nguồn của ``seen_item_ids`` và nhãn) với ``interactions.csv`` và bộ chia.
- Bộ luật rỗng (ngưỡng quá cao) vẫn hợp lệ: mô hình khi đó trả Popularity thuần cho mọi user.
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from src.patterns import (
    DEFAULT_SPLITS,
    build_transaction_matrix,
    generate_rules,
    load_train_transactions,
    run_fpgrowth,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROCESSED = ROOT / "data" / "processed"
DEFAULT_KS = (5, 10)
SCORE_DECIMALS = 12  # làm tròn điểm để sai khác dấu phẩy động không phá thứ tự hòa


def popularity_order(train_counts, n_items):
    """Mảng item_id (1..n_items) theo tần suất train giảm dần, hòa thì item_id tăng dần.

    ``train_counts`` là dict ``item_id -> số lần``; item không có mặt tính 0, vẫn nằm trong catalog.
    """
    counts = np.zeros(n_items + 1, dtype=np.int64)
    for item, count in train_counts.items():
        if not 1 <= item <= n_items:
            raise ValueError(f"item_id {item} nằm ngoài 1..{n_items}")
        counts[item] = count
    items = np.arange(1, n_items + 1)
    return items[np.lexsort((items, -counts[1:]))]


def load_train_counts(train_sequences_path):
    """Số lần xuất hiện của từng item trong ``train_sequences.jsonl``, đếm mọi tương tác kể cả lặp (như EDA và B)."""
    counts = {}
    with Path(train_sequences_path).open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            for item in json.loads(line)["item_ids"]:
                counts[item] = counts.get(item, 0) + 1
    return counts


def check_train_sequences(train_sequences_path, interactions_path, splits_path):
    """Báo ``ValueError`` nếu ``train_sequences.jsonl`` không khớp từng user với ``interactions.csv`` theo bộ chia.

    Luật lấy từ ``interactions.csv`` và ``splits_path`` còn Popularity lấy từ ``train_sequences.jsonl``; nếu hai nguồn
    thuộc hai lần chia khác nhau thì mô hình vẫn cho điểm nhưng trộn dữ liệu (ví dụ item validation vào Popularity).
    """
    items = pd.read_csv(interactions_path, usecols=["item_id"])["item_id"].to_numpy()
    splits = pd.read_csv(splits_path, usecols=["user_id", "train_start", "train_stop"])
    if (splits["train_stop"] > len(items)).any():
        raise ValueError(f"bộ chia vượt số dòng của interactions.csv ({len(items)}); dữ liệu thuộc hai lần chia khác nhau")
    expected = {int(user): items[start:stop].tolist()
                for user, start, stop in zip(splits["user_id"], splits["train_start"], splits["train_stop"])}
    name = Path(train_sequences_path).name
    seen_users = set()
    with Path(train_sequences_path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                user = int(record["user_id"])
                matches = user not in seen_users and user in expected and list(record["item_ids"]) == expected[user]
            except (TypeError, KeyError, ValueError) as error:
                raise ValueError(f"{name} dòng {number} không đọc được: {error!r}") from error
            if not matches:
                raise ValueError(f"{name} không khớp interactions.csv và bộ chia (user {user}); "
                                 "dữ liệu thuộc hai lần chia khác nhau")
            seen_users.add(user)
    if seen_users != expected.keys():
        raise ValueError(f"{name} thiếu {len(expected.keys() - seen_users)} user so với bộ chia")


def check_split_file(split_path, split_name, interactions_path, splits_path):
    """Báo ``ValueError`` nếu ``validation.jsonl`` hoặc ``test.jsonl`` không khớp ``interactions.csv`` theo bộ chia.

    Theo ``preprocessing.build_dataset``: ``seen_item_ids`` là tập item (sắp tăng dần) của mọi tương tác trước nhãn,
    tức train với validation và train cộng nhãn validation với test; ``target`` là tương tác tại ``validation_index``
    hoặc ``test_index``. File lệch bộ chia vẫn cho điểm nhưng khớp luật trên lịch sử và độ dài ranking sai.
    """
    if split_name not in ("validation", "test"):
        raise ValueError(f"bộ chia phải là validation hoặc test, nhận {split_name!r}")
    items = pd.read_csv(interactions_path, usecols=["item_id"])["item_id"].to_numpy()
    splits = pd.read_csv(splits_path)
    label_column = f"{split_name}_index"
    if (splits[label_column] >= len(items)).any():
        raise ValueError(f"bộ chia vượt số dòng của interactions.csv ({len(items)}); dữ liệu thuộc hai lần chia khác nhau")
    expected = {}
    for user, start, label in zip(splits["user_id"], splits["train_start"], splits[label_column]):
        expected[int(user)] = (sorted(set(items[start:label].tolist())), int(items[label]))
    seen_users = set()
    name = Path(split_path).name
    with Path(split_path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                user = int(record["user_id"])
                matches = (user not in seen_users and user in expected
                           and (sorted(record["seen_item_ids"]), record["target"]) == expected[user])
            except (TypeError, KeyError, ValueError) as error:
                raise ValueError(f"{name} dòng {number} không đọc được: {error!r}") from error
            if not matches:
                raise ValueError(f"{name} không khớp interactions.csv và bộ chia (user {user}); "
                                 "dữ liệu thuộc hai lần chia khác nhau")
            seen_users.add(user)
    if seen_users != expected.keys():
        raise ValueError(f"{name} thiếu {len(expected.keys() - seen_users)} user so với bộ chia")


class ARRecommender:
    """Khớp luật bằng chỉ mục đảo: item -> các luật có item đó trong tiền đề.

    Một luật khớp khi số item của tiền đề có trong lịch sử bằng đúng kích thước tiền đề.
    """

    def __init__(self, rules, n_items, pop_order):
        missing = {"antecedents", "consequents", "confidence", "lift"} - set(rules.columns)
        if missing:
            raise ValueError(f"rules thiếu cột {sorted(missing)}")
        pop_order = np.asarray(pop_order, dtype=np.int64)
        if sorted(pop_order.tolist()) != list(range(1, n_items + 1)):
            raise ValueError("pop_order phải là một hoán vị của 1..n_items")
        self.n_items = n_items
        self.pop_order = pop_order
        self.n_rules = len(rules)
        antecedents = list(rules["antecedents"])
        consequents = list(rules["consequents"])
        if any(len(a) == 0 for a in antecedents) or any(len(c) == 0 for c in consequents):
            raise ValueError("luật có tiền đề hoặc hậu quả rỗng")
        self.sizes = np.fromiter((len(a) for a in antecedents), dtype=np.int64, count=self.n_rules)
        weights = rules["confidence"].to_numpy(dtype=float) * rules["lift"].to_numpy(dtype=float)
        if not np.isfinite(weights).all() or (weights <= 0).any():
            raise ValueError("confidence x lift phải hữu hạn và dương")
        self.weights = np.round(weights, SCORE_DECIMALS)
        self._by_item = self._csr(antecedents, n_items).tocsc()  # cột = item, hàng = luật
        self._consequents = self._csr(consequents, n_items)

    @staticmethod
    def _csr(itemsets, n_items):
        lengths = np.fromiter((len(s) for s in itemsets), dtype=np.int64, count=len(itemsets))
        indptr = np.concatenate(([0], np.cumsum(lengths)))
        indices = np.fromiter((item for s in itemsets for item in s), dtype=np.int64, count=int(indptr[-1]))
        if indices.size and (indices.min() < 1 or indices.max() > n_items):
            raise ValueError(f"item trong luật nằm ngoài 1..{n_items}")
        return sparse.csr_matrix((np.ones(len(indices), dtype=bool), indices, indptr),
                                 shape=(len(itemsets), n_items + 1))

    def matched_rules(self, seen_items):
        """Chỉ số các luật có tiền đề ⊆ seen_items."""
        seen = np.unique(np.asarray(list(seen_items), dtype=np.int64))
        if seen.size and (seen[0] < 1 or seen[-1] > self.n_items):
            raise ValueError(f"item đã thấy nằm ngoài 1..{self.n_items}")
        postings = [self._by_item.indices[self._by_item.indptr[i]:self._by_item.indptr[i + 1]] for i in seen]
        if not postings:
            return np.empty(0, dtype=np.int64)
        rule_ids, hits = np.unique(np.concatenate(postings), return_counts=True)
        return rule_ids[hits == self.sizes[rule_ids]]

    def rule_scores(self, seen_items):
        """Mảng độ dài ``n_items + 1``: điểm luật (max confidence x lift) của từng item_id, 0 nếu không luật nào khớp."""
        matched = self.matched_rules(seen_items)
        scores = np.zeros(self.n_items + 1)
        if matched.size:
            sub = self._consequents[matched]
            np.maximum.at(scores, sub.indices, np.repeat(self.weights[matched], np.diff(sub.indptr)))
        return scores

    def recommend(self, seen_items, length):
        """Danh sách item_id xếp giảm dần, không lặp, dài ``min(length, n_items)``.

        Returns:
            ``(ranked, n_from_rules)``: ``n_from_rules`` là số item đầu danh sách có điểm từ luật
            (phần còn lại là Popularity).
        """
        if type(length) is not int or length < 1:
            raise ValueError("length phải là số nguyên dương")
        length = min(length, self.n_items)
        scores = self.rule_scores(seen_items)
        scored = np.flatnonzero(scores > 0)
        scored = scored[np.lexsort((scored, -scores[scored]))][:length]
        if len(scored) < length:
            taken = np.zeros(self.n_items + 1, dtype=bool)
            taken[scored] = True
            padding = self.pop_order[~taken[self.pop_order]][:length - len(scored)]
            return scored.tolist() + padding.tolist(), len(scored)
        return scored.tolist(), len(scored)


def rank_users(recommender, split, max_k):
    """Xếp hạng cho mọi user của bộ chia: ``max_k + len(seen_item_ids)`` item mỗi user (xem docstring mô-đun).

    Returns:
        ``(rankings, stats)``: ``rankings`` là dict ``user_id -> list item_id``; ``stats`` gồm
        ``users_with_rule`` (user có ít nhất một luật khớp cho ra item), ``mean_rule_items_in_top_k`` và
        ``rule_items_already_seen_ratio`` (tỉ lệ item từ luật trong top-K đã nằm trong lịch sử).
        Vì mô hình không tự lọc item đã thấy, các item này chiếm chỗ của Popularity nhưng không thể là nhãn
        khi ``target_in_history`` = 0 (Beauty); ``users_with_unseen_rule_item`` đếm user có ít nhất một item luật chưa thấy.
    """
    rankings, with_rule, with_unseen, rule_items_in_top, seen_rule_items = {}, 0, 0, 0, 0
    for user in sorted(split):
        seen = split[user]["seen_item_ids"]
        ranked, n_rule = recommender.recommend(seen, max_k + len(seen))
        rankings[user] = ranked
        in_top = ranked[:min(n_rule, max_k)]
        seen_set = set(seen)
        already_seen = sum(item in seen_set for item in in_top)
        with_rule += n_rule > 0
        with_unseen += already_seen < len(in_top)
        rule_items_in_top += len(in_top)
        seen_rule_items += already_seen
    return rankings, {
        "users": len(rankings),
        "users_with_rule": int(with_rule),
        "users_with_rule_ratio": with_rule / max(len(rankings), 1),
        "users_with_unseen_rule_item": int(with_unseen),
        "mean_rule_items_in_top_k": rule_items_in_top / max(len(rankings), 1),
        "rule_items_already_seen_ratio": seen_rule_items / rule_items_in_top if rule_items_in_top else 0.0,
    }


def write_rankings(path, rankings):
    """Ghi JSONL ``{"user_id", "ranked_items"}`` theo định dạng bàn giao của bộ chấm chung (docs/evaluation.md của B)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for user in sorted(rankings):
            stream.write(json.dumps({"user_id": int(user), "ranked_items": [int(i) for i in rankings[user]]}) + "\n")


def build_recommender(interactions, splits, train_sequences, min_support, min_confidence, max_len):
    """Khai phá luật từ train rồi dựng ARRecommender; trả về (recommender, thống kê khai phá).

    Không có luật nào thỏa ngưỡng thì recommender chỉ còn Popularity (``n_rules`` = 0), không báo lỗi.
    """
    check_train_sequences(train_sequences, interactions, splits)
    n_items = int(pd.read_csv(interactions, usecols=["item_id"])["item_id"].max())
    _, transactions = load_train_transactions(interactions, splits)
    matrix = build_transaction_matrix(transactions, n_items)
    itemsets, fp_seconds = run_fpgrowth(matrix, min_support, max_len=max_len)
    start = time.perf_counter()
    rules = generate_rules(itemsets, matrix.shape[0], min_confidence=min_confidence, max_rule_len=max_len)
    rules_seconds = time.perf_counter() - start
    order = popularity_order(load_train_counts(train_sequences), n_items)
    recommender = ARRecommender(rules, n_items, order)
    return recommender, {
        "n_transactions": int(matrix.shape[0]),
        "n_items": n_items,
        "n_frequent_itemsets": int(len(itemsets)),
        "n_rules": int(len(rules)),
        "fpgrowth_seconds": round(fp_seconds, 3),
        "rules_seconds": round(rules_seconds, 3),
    }


def main(argv=None):
    from src.evaluate import evaluate, load_split

    parser = argparse.ArgumentParser(description="AR-Rec: khai phá luật từ train, xếp hạng và chấm trên validation hoặc test.")
    parser.add_argument("--split", choices=("validation", "test"), default="validation",
                        help="chọn cấu hình trên validation; test chỉ chạy cho cấu hình đã chọn")
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED)
    parser.add_argument("--interactions", type=Path, default=None, help="mặc định: <processed-dir>/interactions.csv")
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--min-support", type=float, default=0.001)
    parser.add_argument("--min-confidence", type=float, default=0.1)
    parser.add_argument("--max-len", type=int, default=4, help="giới hạn |X ∪ Y| của luật")
    parser.add_argument("--ks", type=int, nargs="+", default=list(DEFAULT_KS))
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="ghi rankings.jsonl (định dạng bàn giao của bộ chấm chung) và ar_rec_metrics.json")
    args = parser.parse_args(argv)
    if not 0 < args.min_support <= 1:
        parser.error("--min-support phải thuộc (0, 1]")
    if not 0 < args.min_confidence <= 1:
        parser.error("--min-confidence phải thuộc (0, 1]")
    interactions = args.interactions or args.processed_dir / "interactions.csv"
    try:
        check_split_file(args.processed_dir / f"{args.split}.jsonl", args.split, interactions, args.splits)
        split = load_split(args.processed_dir / f"{args.split}.jsonl")
        recommender, mining = build_recommender(
            interactions, args.splits, args.processed_dir / "train_sequences.jsonl",
            args.min_support, args.min_confidence, args.max_len,
        )
        rankings, stats = rank_users(recommender, split, max(args.ks))
        metrics = {
            "keep_seen": evaluate(rankings, split, ks=args.ks, exclude_seen=False),
            "exclude_seen": evaluate(rankings, split, ks=args.ks, exclude_seen=True),
        }
    except (OSError, ValueError, KeyError, csv.Error) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except MemoryError:
        print("Error: hết bộ nhớ khi sinh luật; tăng --min-support, giảm --max-len hoặc giải phóng RAM "
              "(mức 0,0005 với --max-len 4 tạo hơn 650.000 luật)", file=sys.stderr)
        return 1
    if mining["n_rules"] == 0:
        print("Cảnh báo: không có luật nào thỏa ngưỡng; ranking chỉ gồm Popularity", file=sys.stderr)
    write_rankings(args.output_dir / "rankings.jsonl", rankings)
    result = {
        "model": "ar_rec", "split": args.split,
        "config": {"min_support": args.min_support, "min_confidence": args.min_confidence, "max_len": args.max_len,
                   "score": "max(confidence * lift)", "padding": "popularity (train)"},
        "mining": mining, "ranking": stats, "metrics": metrics,
    }
    (args.output_dir / "ar_rec_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
