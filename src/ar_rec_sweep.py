"""Quét min_support x min_confidence cho AR-Rec trên validation, chọn cấu hình và đo các đại lượng giải thích độ thưa của luật.

Phụ trách: C (RQ2), T7 14/11.

Quy ước:

- Chỉ dùng bộ chia validation (luật khai phá từ train); test không được chạy trong mô-đun này.
- Lưới gồm ba chiều: ``max_len`` (giới hạn |X ∪ Y|), min_support và min_confidence. Mỗi ``max_len`` có tập min_support riêng
  vì số tập phổ biến và số luật tăng rất nhanh theo ``max_len`` khi support thấp (``max_len`` = 4 ở 0,0005 tạo 656.681 luật ở min_confidence 0,1, đỉnh bộ nhớ
  của cả lưới khoảng 0,67 GB); lưới dừng ở 0,0005 cho ``max_len`` 3 và 4 vì ở các support dùng chung hai mức này kém ``max_len`` = 2,
  trừ ``max_len`` = 3 ở 0,001 (NDCG@10 cao hơn 0,00018).
- Với mỗi cặp (``max_len``, min_support), tập phổ biến và luật được khai phá một lần ở min_confidence thấp nhất, các ngưỡng cao
  hơn lọc lại từ cùng bảng luật (kết quả giống khai phá riêng từng ngưỡng vì confidence của một luật không phụ thuộc ngưỡng).
- Tiêu chí chọn, cố định trước khi chạy: NDCG@10 trung bình theo user ở chế độ giữ item đã xem (``exclude_seen = false``,
  quyết định 19/11); hòa thì chọn cấu hình ít luật hơn, rồi min_support lớn hơn, rồi min_confidence lớn hơn, rồi ``max_len`` nhỏ hơn.
- Mức support mà bước khai phá hoặc sinh luật hết bộ nhớ được ghi ``status = "memory_error"`` và bỏ qua; các mức còn lại vẫn chạy.
"""

import argparse
import csv
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from src.ar_rec import (
    DEFAULT_KS,
    DEFAULT_PROCESSED,
    ARRecommender,
    check_split_file,
    check_train_sequences,
    load_train_counts,
    popularity_order,
    rank_users,
    write_rankings,
)
from src.evaluate import load_split, per_user_metrics
from src.patterns import (
    CONFIDENCE_TOLERANCE,
    DEFAULT_SPLITS,
    build_transaction_matrix,
    generate_rules,
    load_train_transactions,
    run_fpgrowth,
    summarize_itemsets,
)

ROOT = Path(__file__).resolve().parents[1]
# max_len -> các min_support; max_len 3 và 4 dừng ở 0,0005 (số luật tăng nhanh khi hạ thêm; xem docstring mô-đun).
DEFAULT_GRID = {
    2: (0.00005, 0.0001, 0.00015, 0.0002, 0.0003, 0.0004, 0.0005, 0.0007, 0.001, 0.0015, 0.002),
    3: (0.0005, 0.0007, 0.001, 0.0015, 0.002),
    4: (0.0005, 0.0007, 0.001, 0.0015, 0.002),
}
DEFAULT_CONFIDENCES = (0.02, 0.05, 0.1, 0.2, 0.3)
DEFAULT_OUTPUT = ROOT / "reports" / "ar_rec_sweep.json"
DEFAULT_MARKDOWN = ROOT / "reports" / "ar_rec_sweep.md"
SELECT_METRIC = "ndcg@10"
# Cận trên (gồm) của số item khác nhau trong lịch sử; nhóm cuối là mọi giá trị lớn hơn.
HISTORY_BUCKETS = (4, 9, 19)


def filter_by_confidence(rules, min_confidence):
    """Luật có confidence >= min_confidence, cùng dung sai với ``patterns.generate_rules``."""
    return rules[rules["confidence"] >= min_confidence - CONFIDENCE_TOLERANCE].reset_index(drop=True)


def bucket_label(index):
    """Nhãn nhóm độ dài lịch sử: ``<=4``, ``5-9``, ``10-19``, ``>=20``."""
    lows = (1,) + tuple(edge + 1 for edge in HISTORY_BUCKETS)
    if index == 0:
        return f"<={HISTORY_BUCKETS[0]}"
    if index == len(HISTORY_BUCKETS):
        return f">={lows[-1]}"
    return f"{lows[index]}-{HISTORY_BUCKETS[index]}"


def history_bucket(n_distinct):
    for index, edge in enumerate(HISTORY_BUCKETS):
        if n_distinct <= edge:
            return index
    return len(HISTORY_BUCKETS)


def pair_counts(matrix):
    """Số giao dịch train chứa từng cặp item (không thứ tự) có xuất hiện cùng nhau ít nhất một lần.

    ``matrix`` là ma trận thưa nhị phân (giao dịch x item); trả về mảng số đếm, mỗi cặp một phần tử.
    """
    binary = matrix.astype(np.int32).tocsr()
    return sparse.triu(binary.T @ binary, k=1).tocoo().data


def cooccurrence_stats(counts, min_count):
    """Số cặp item cùng xuất hiện và số cặp đạt ``min_count``, từ kết quả của ``pair_counts``."""
    return {
        "pairs_cooccurring": int(len(counts)),
        "pairs_reaching_min_count": int((counts >= min_count).sum()),
        "pair_count_median": float(np.median(counts)) if len(counts) else 0.0,
    }


def diagnose(recommender, split):
    """Đại lượng giải thích độ thưa của một bộ luật trên bộ chia.

    - ``users_with_rule_ratio``: tỉ lệ user có ít nhất một item điểm > 0 từ luật (cùng định nghĩa với ``rank_users``).
    - ``target_covered_ratio``: tỉ lệ user có nhãn nằm trong hậu quả của một luật khớp; là cận trên của recall khi chỉ dùng luật.
    - ``by_history_length``: số user và tỉ lệ user có luật khớp theo số item khác nhau trong lịch sử.
    """
    n_buckets = len(HISTORY_BUCKETS) + 1
    users, with_rule = np.zeros(n_buckets, dtype=np.int64), np.zeros(n_buckets, dtype=np.int64)
    covered = 0
    for user in sorted(split):
        seen, target = split[user]["seen_item_ids"], split[user]["target"]
        scores = recommender.rule_scores(seen)
        bucket = history_bucket(len(set(seen)))
        users[bucket] += 1
        has_rule = bool((scores > 0).any())
        with_rule[bucket] += has_rule
        covered += bool(scores[target] > 0)
    total = int(users.sum())
    return {
        "users_with_rule_ratio": float(with_rule.sum() / total),
        "target_covered_ratio": covered / total,
        "by_history_length": {
            bucket_label(i): {"users": int(users[i]), "users_with_rule_ratio": float(with_rule[i] / users[i]) if users[i] else None}
            for i in range(n_buckets)
        },
    }


def level_context(itemsets, matrix, split, min_support, counts):
    """Số liệu theo mức support, không phụ thuộc min_confidence."""
    n_transactions = matrix.shape[0]
    min_count = int(np.ceil(min_support * n_transactions))
    sizes = itemsets["itemsets"].map(len)
    frequent = np.zeros(matrix.shape[1] + 1, dtype=bool)
    for itemset in itemsets.loc[sizes == 1, "itemsets"]:
        frequent[next(iter(itemset))] = True
    with_frequent = sum(bool(frequent[split[u]["seen_item_ids"]].any()) for u in split)
    target_frequent = sum(bool(frequent[split[u]["target"]]) for u in split)
    return {
        "min_count": min_count,
        "n_frequent_itemsets": int(len(itemsets)),
        "itemsets_by_size": summarize_itemsets(itemsets),
        "n_frequent_items": int(frequent.sum()),
        "frequent_items_ratio": float(frequent.sum() / matrix.shape[1]),
        "users_with_frequent_item_ratio": with_frequent / len(split),
        "target_is_frequent_item_ratio": target_frequent / len(split),
        **cooccurrence_stats(counts, min_count),
    }


def paired_difference(a, b):
    """Trung bình và sai số chuẩn của hiệu ``a - b`` theo user."""
    diff = np.asarray(a) - np.asarray(b)
    return {"mean": float(diff.mean()), "standard_error": float(diff.std(ddof=1) / np.sqrt(len(diff)))}


def selection_key(run):
    """Khóa sắp xếp tăng dần: NDCG@10 cao nhất trước, rồi ít luật, support lớn, confidence lớn."""
    return (-run["metrics"]["keep_seen"][SELECT_METRIC], run["n_rules"], -run["min_support"], -run["min_confidence"], run["max_len"])


def config_key(item):
    return item["max_len"], item["min_support"], item["min_confidence"]


def popularity_reference(split, order, ks):
    """Popularity thuần (cùng thứ tự ``popularity_order``): điểm trung bình và điểm theo user ở hai chế độ."""
    max_k = max(ks)
    rankings = {user: order[:max_k + len(split[user]["seen_item_ids"])].tolist() for user in split}
    result, per_user = {}, {}
    for name, exclude in (("keep_seen", False), ("exclude_seen", True)):
        _, scores = per_user_metrics(rankings, split, ks=ks, exclude_seen=exclude)
        result[name] = {metric: float(values.mean()) for metric, values in scores.items()}
        per_user[name] = scores
    return result, per_user


def run_sweep(interactions, splits_path, processed_dir, grid, confidences, ks=DEFAULT_KS, progress=None):
    """Chạy lưới trên validation; trả về ``(payload, best_rankings)``.

    ``grid`` là dict ``max_len -> các min_support``. ``payload`` có ``reference_popularity``, ``levels`` (theo cặp
    ``max_len``, min_support), ``runs`` (theo bộ ba), ``selected``, ``best_by_max_len`` và ``comparison``.
    """
    if SELECT_METRIC not in {f"ndcg@{k}" for k in ks}:
        raise ValueError(f"ks phải chứa 10 để chọn theo {SELECT_METRIC}")
    log = progress or (lambda message: None)
    confidences = sorted(set(confidences))
    if not confidences or not any(grid.values()):
        raise ValueError("lưới phải có ít nhất một min_support và một min_confidence")
    check_train_sequences(Path(processed_dir) / "train_sequences.jsonl", interactions, splits_path)
    check_split_file(Path(processed_dir) / "validation.jsonl", "validation", interactions, splits_path)
    n_items = int(pd.read_csv(interactions, usecols=["item_id"])["item_id"].max())
    _, transactions = load_train_transactions(interactions, splits_path)
    matrix = build_transaction_matrix(transactions, n_items)
    split = load_split(Path(processed_dir) / "validation.jsonl")
    order = popularity_order(load_train_counts(Path(processed_dir) / "train_sequences.jsonl"), n_items)
    reference, reference_scores = popularity_reference(split, order, ks)

    levels, runs, per_user, best_rankings = [], [], {}, {}
    counts = pair_counts(matrix)
    for max_len, supports in sorted(grid.items()):
        for support in supports:
            level = {"min_support": support, "max_len": max_len}
            try:
                itemsets, fp_seconds = run_fpgrowth(matrix, support, max_len=max_len)
                rules_all = generate_rules(itemsets, matrix.shape[0], min_confidence=confidences[0], max_rule_len=max_len)
            except MemoryError:
                itemsets = rules_all = None
                gc.collect()
                level["status"] = "memory_error"
                levels.append(level)
                log(f"max_len={max_len} support={support}: hết bộ nhớ, bỏ qua")
                continue
            context = level_context(itemsets, matrix, split, support, counts)
            level.update(status="ok", fpgrowth_seconds=round(fp_seconds, 3), **context)
            del itemsets
            levels.append(level)
            for confidence in confidences:
                rules = filter_by_confidence(rules_all, confidence)
                run = {"min_support": support, "min_confidence": confidence, "max_len": max_len, "n_rules": int(len(rules))}
                if rules.empty:
                    run["status"] = "no_rules"
                    runs.append(run)
                    continue
                start = time.perf_counter()
                recommender = ARRecommender(rules, n_items, order)
                rankings, stats = rank_users(recommender, split, max(ks))
                scores = {}
                for name, exclude in (("keep_seen", False), ("exclude_seen", True)):
                    _, scores[name] = per_user_metrics(rankings, split, ks=ks, exclude_seen=exclude)
                run.update(
                    status="ok",
                    n_items_in_rules=int(len(set().union(*rules["antecedents"], *rules["consequents"]))),
                    ranking=stats,
                    diagnostics=diagnose(recommender, split),
                    metrics={name: {m: float(v.mean()) for m, v in per.items()} for name, per in scores.items()},
                    seconds=round(time.perf_counter() - start, 3),
                )
                runs.append(run)
                per_user[config_key(run)] = scores["keep_seen"][SELECT_METRIC]
                if selection_key(run) == min(selection_key(r) for r in runs if r["status"] == "ok"):
                    best_rankings = rankings  # giữ lại danh sách của cấu hình đang dẫn đầu
                log(f"max_len={max_len} support={support} confidence={confidence}: "
                    f"{SELECT_METRIC}={run['metrics']['keep_seen'][SELECT_METRIC]:.6f}")
                del recommender, rankings
            del rules_all, rules
            gc.collect()

    evaluated = sorted((r for r in runs if r["status"] == "ok"), key=selection_key)
    if not evaluated:
        raise ValueError("Không cấu hình nào chạy được: giảm min_support hoặc min_confidence nếu không có luật nào, "
                         "tăng min_support hoặc giảm max_len nếu hết bộ nhớ")
    best = evaluated[0]
    best_scores = per_user[config_key(best)]
    comparison = {"metric": f"{SELECT_METRIC} (keep_seen)", "n_users": len(split),
                  "selected_vs_popularity": paired_difference(best_scores, reference_scores["keep_seen"][SELECT_METRIC])}
    if len(evaluated) > 1:
        second = evaluated[1]
        comparison["runner_up"] = {"max_len": second["max_len"], "min_support": second["min_support"],
                                   "min_confidence": second["min_confidence"]}
        comparison["selected_vs_runner_up"] = paired_difference(best_scores, per_user[config_key(second)])
    best_by_max_len = {}
    for run in evaluated:
        best_by_max_len.setdefault(str(run["max_len"]), {
            "min_support": run["min_support"], "min_confidence": run["min_confidence"], "n_rules": run["n_rules"],
            SELECT_METRIC: run["metrics"]["keep_seen"][SELECT_METRIC]})
    payload = {
        "split": "validation",
        "selection": {"metric": SELECT_METRIC, "protocol": "keep_seen",
                      "ties": "ít luật hơn, rồi min_support lớn hơn, rồi min_confidence lớn hơn, rồi max_len nhỏ hơn"},
        "n_transactions": int(matrix.shape[0]),
        "n_items": n_items,
        "reference_popularity": reference,
        "levels": levels,
        "runs": runs,
        "selected": {"max_len": best["max_len"], "min_support": best["min_support"], "min_confidence": best["min_confidence"]},
        "best_by_max_len": best_by_max_len,
        "comparison": comparison,
    }
    return payload, best_rankings


def _pct(value):
    return "-" if value is None else f"{100 * value:.1f}%"


def _is_selected(item, selected):
    return all(item[k] == selected[k] for k in ("max_len", "min_support", "min_confidence") if k in item)


def format_markdown(payload):
    """Báo cáo Markdown của lưới quét; mọi con số lấy từ ``payload``, không có nhận định ngoài số liệu."""
    pop = payload["reference_popularity"]["keep_seen"]
    selected, comparison = payload["selected"], payload["comparison"]
    ok_runs = [r for r in payload["runs"] if r["status"] == "ok"]
    lines = [
        "# Quét max_len, min_support và min_confidence cho AR-Rec (validation)",
        "",
        f"{payload['n_transactions']:,} giao dịch train, {payload['n_items']:,} item. "
        "Sinh bằng `python -m src.ar_rec_sweep`; toàn bộ lưới (từng cấu hình) nằm trong `ar_rec_sweep.json`. "
        "Chỉ dùng bộ chia validation; test chưa chạy.",
        "",
        f"Tiêu chí chọn: {payload['selection']['metric']} ở chế độ giữ item đã xem (`exclude_seen = false`); "
        f"hòa thì {payload['selection']['ties']}. max_len là giới hạn |X ∪ Y| (2 nghĩa là chỉ luật một item ⇒ một item). "
        "Mỗi max_len có tập min_support riêng (xem `DEFAULT_GRID`); max_len lớn chỉ được chạy ở các mức support cao hơn. "
        "Các giá trị tốt nhất theo max_len vì vậy không so sánh trực tiếp được khi tập min_support khác nhau.",
        "",
        f"Popularity thuần (cùng split): Recall@10 {pop['recall@10']:.5f}, NDCG@10 {pop['ndcg@10']:.5f}.",
        "",
        "## Cấu hình tốt nhất theo (max_len, min_support)",
        "",
        "Với mỗi cặp, min_confidence được chọn theo cùng tiêu chí.",
        "",
        "| max_len | min_support | min_confidence | Luật | User có luật | Nhãn nằm trong hậu quả | Recall@10 | NDCG@10 | NDCG@10 (lọc item đã xem) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for level in payload["levels"]:
        if level["status"] != "ok":
            lines.append(f"| {level['max_len']} | {level['min_support']} | - | - | - | - | - | - | không chạy được ({level['status']}) |")
            continue
        candidates = [r for r in ok_runs if (r["max_len"], r["min_support"]) == (level["max_len"], level["min_support"])]
        if not candidates:
            lines.append(f"| {level['max_len']} | {level['min_support']} | - | 0 | - | - | - | - | không có luật nào thỏa ngưỡng |")
            continue
        run = min(candidates, key=selection_key)
        keep, excl, diag = run["metrics"]["keep_seen"], run["metrics"]["exclude_seen"], run["diagnostics"]
        mark = " **(chọn)**" if _is_selected(run, selected) else ""
        lines.append(
            f"| {run['max_len']} | {run['min_support']}{mark} | {run['min_confidence']} | {run['n_rules']:,} | {_pct(diag['users_with_rule_ratio'])} "
            f"| {_pct(diag['target_covered_ratio'])} | {keep['recall@10']:.5f} | {keep['ndcg@10']:.5f} | {excl['ndcg@10']:.5f} |"
        )
    lines += [
        "",
        f"## Ảnh hưởng của min_confidence (max_len = {selected['max_len']}, min_support = {selected['min_support']})",
        "",
        "| min_confidence | Luật | User có luật | Recall@10 | NDCG@10 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for run in ok_runs:
        if (run["max_len"], run["min_support"]) == (selected["max_len"], selected["min_support"]):
            mark = " **(chọn)**" if _is_selected(run, selected) else ""
            lines.append(f"| {run['min_confidence']}{mark} | {run['n_rules']:,} | {_pct(run['diagnostics']['users_with_rule_ratio'])} "
                         f"| {run['metrics']['keep_seen']['recall@10']:.5f} | {run['metrics']['keep_seen']['ndcg@10']:.5f} |")
    lines += [
        "",
        "## Cấu hình được chọn",
        "",
        f"max_len = {selected['max_len']}, min_support = {selected['min_support']}, min_confidence = {selected['min_confidence']}.",
        "",
        "NDCG@10 tốt nhất theo max_len: " + "; ".join(
            f"max_len {k}: {v['ndcg@10']:.5f} (min_support {v['min_support']}, min_confidence {v['min_confidence']})"
            for k, v in sorted(payload["best_by_max_len"].items())) + ".",
    ]
    vs_pop = comparison["selected_vs_popularity"]
    lines += ["", f"Hiệu NDCG@10 so với Popularity theo user: {vs_pop['mean']:+.5f} (sai số chuẩn {vs_pop['standard_error']:.5f}, "
              f"{comparison['n_users']:,} user). Cấu hình được chọn trên chính tập validation này nên các hiệu và sai số chuẩn "
              "chưa hiệu chỉnh cho việc chọn trong nhiều cấu hình; ước lượng không chệch chỉ có từ bộ test."]
    if "selected_vs_runner_up" in comparison:
        ru, diff = comparison["runner_up"], comparison["selected_vs_runner_up"]
        lines.append(f"Hiệu so với cấu hình đứng thứ hai (max_len {ru['max_len']}, min_support {ru['min_support']}, "
                     f"min_confidence {ru['min_confidence']}): {diff['mean']:+.5f} (sai số chuẩn {diff['standard_error']:.5f}).")
    lines += [
        "",
        "## Độ thưa của luật theo min_support",
        "",
        "Cặp item cùng xuất hiện tính trên toàn bộ giao dịch train, không phụ thuộc max_len và min_support.",
        "",
        "| max_len | min_support | Đếm tối thiểu | Item phổ biến | Cặp item cùng xuất hiện | Cặp đạt đếm tối thiểu | Tập phổ biến (≤ max_len) "
        "| User có ≥ 1 item phổ biến | Nhãn là item phổ biến | Thời gian FP-Growth (s) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for level in payload["levels"]:
        if level["status"] != "ok":
            lines.append(f"| {level['max_len']} | {level['min_support']} | - | - | - | - | - | - | - | không chạy được ({level['status']}) |")
            continue
        lines.append(
            f"| {level['max_len']} | {level['min_support']} | {level['min_count']} | {level['n_frequent_items']:,} ({_pct(level['frequent_items_ratio'])}) "
            f"| {level['pairs_cooccurring']:,} | {level['pairs_reaching_min_count']:,} ({_pct(level['pairs_reaching_min_count'] / level['pairs_cooccurring'] if level['pairs_cooccurring'] else None)}) "
            f"| {level['n_frequent_itemsets']:,} | {_pct(level['users_with_frequent_item_ratio'])} "
            f"| {_pct(level['target_is_frequent_item_ratio'])} | {level['fpgrowth_seconds']:.1f} |"
        )
    chosen = next(r for r in ok_runs if _is_selected(r, selected))
    lines += [
        "",
        "## Tỉ lệ user có luật khớp theo độ dài lịch sử (cấu hình được chọn)",
        "",
        "Độ dài lịch sử là số item khác nhau trong phần train của user.",
        "",
        "| Độ dài lịch sử | User | User có luật khớp |",
        "|---|---:|---:|",
    ]
    for label, row in chosen["diagnostics"]["by_history_length"].items():
        lines.append(f"| {label} | {row['users']:,} | {_pct(row['users_with_rule_ratio'])} |")
    return "\n".join(lines) + "\n"


def parse_grid(items):
    """``["2:0.0001,0.0002", "3:0.0005"]`` -> ``{2: (0.0001, 0.0002), 3: (0.0005,)}``."""
    grid = {}
    for item in items:
        max_len, _, supports = item.partition(":")
        if not max_len.isdigit() or int(max_len) < 2 or not supports:
            raise ValueError(f"--grid '{item}' không đúng dạng MAX_LEN:SUPPORT,SUPPORT,... (MAX_LEN >= 2)")
        if int(max_len) in grid:
            raise ValueError(f"max_len {max_len} xuất hiện hơn một lần trong --grid")
        grid[int(max_len)] = tuple(float(x) for x in supports.split(","))
    return grid


def main(argv=None):
    parser = argparse.ArgumentParser(description="Quét max_len x min_support x min_confidence cho AR-Rec trên validation.")
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED)
    parser.add_argument("--interactions", type=Path, default=None, help="mặc định: <processed-dir>/interactions.csv")
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--grid", nargs="+", default=None, metavar="MAX_LEN:SUPPORTS",
                        help="ví dụ 2:0.0001,0.0002 3:0.0005,0.001; mặc định là DEFAULT_GRID trong mô-đun")
    parser.add_argument("--confidences", type=float, nargs="+", default=list(DEFAULT_CONFIDENCES))
    parser.add_argument("--ks", type=int, nargs="+", default=list(DEFAULT_KS))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--rankings-output", type=Path, default=None,
                        help="ghi rankings.jsonl của cấu hình được chọn (định dạng bàn giao của bộ chấm chung)")
    args = parser.parse_args(argv)
    try:
        grid = parse_grid(args.grid) if args.grid else DEFAULT_GRID
    except ValueError as error:
        parser.error(str(error))
    if any(not 0 < s <= 1 for supports in grid.values() for s in supports) or any(not 0 < c <= 1 for c in args.confidences):
        parser.error("min_support và min_confidence phải thuộc (0, 1]")
    interactions = args.interactions or args.processed_dir / "interactions.csv"
    try:
        payload, rankings = run_sweep(interactions, args.splits, args.processed_dir, grid, args.confidences,
                                      tuple(args.ks), progress=lambda m: print(m, file=sys.stderr, flush=True))
    except (OSError, ValueError, KeyError, csv.Error) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.markdown.write_text(format_markdown(payload), encoding="utf-8")
    if args.rankings_output:
        write_rankings(args.rankings_output, rankings)
    print(json.dumps(payload["selected"] | {"comparison": payload["comparison"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
