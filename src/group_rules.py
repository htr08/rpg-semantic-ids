"""Luật kết hợp ở cấp nhóm (thương hiệu, danh mục) so với cấp item.

Phụ trách: C (RQ2, T6 20/11). Item được gộp lên nhóm: mỗi giao dịch (tập item train của một user) trở thành tập các nhóm
chứa item đó, rồi chạy FP-Growth như ở cấp item (src/patterns.py). Mục đích là kiểm tra luật cấp nhóm có còn dùng được
khi luật cấp item thưa, bằng hai số đo trên lịch sử validation:

- coverage: tỉ lệ user (trên mọi user validation) có ít nhất một luật áp dụng được (tiền đề ⊆ nhóm của lịch sử);
- hit rate: trong các user được phủ và có item validation thuộc ít nhất một nhóm, tỉ lệ nhóm của item validation nằm trong
  hợp các hậu quả của luật áp dụng được, đặt cạnh Popularity chọn cùng số nhóm cho mỗi user (cùng kích thước tập dự đoán).
  Hit rate giữa các cấp không so thẳng được: tập đích và kích thước tập dự đoán khác nhau.

Quy ước:
- Luật và số đếm Popularity chỉ lấy từ phần train; validation chỉ dùng để chấm; tương tác test không được dùng.
- Brand lấy từ patterns.load_item_metadata (so khớp không phân biệt hoa thường; rỗng hoặc "Unknown" là thiếu). Item thiếu brand
  không thuộc nhóm nào ở cấp brand và không được suy đoán.
- Danh mục ở độ sâu d là tiền tố d phần tử đầu của đường dẫn ``categories`` (đường ngắn hơn d giữ nguyên), nối bằng " > ".
  Một item có nhiều đường dẫn thuộc nhiều nhóm. Đây là cách roll-up theo độ sâu, không phụ thuộc quy tắc chọn danh mục lá (G4).
- Hậu quả được tính cả khi nhóm đã có trong lịch sử, vì bộ chấm chung của dự án không loại item đã xem.
"""

import argparse
import html
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import sparse

from src.patterns import (DEFAULT_INTERACTIONS, DEFAULT_ITEMS, DEFAULT_SPLITS, build_transaction_matrix, generate_rules,
                          load_item_metadata, load_train_transactions, run_fpgrowth, top_rules_by_lift)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VALIDATION = ROOT / "data" / "processed" / "validation.jsonl"
DEFAULT_OUTPUT = ROOT / "reports" / "group_rules.json"
DEFAULT_MARKDOWN = ROOT / "reports" / "group_rules.md"
CATEGORY_DEPTHS = (2, 3, 4)
GROUP_SUPPORTS = (0.005, 0.01, 0.02)
ITEM_SUPPORTS = (0.0005, 0.001, 0.002)
MAX_LEN = 3


def load_category_paths(items_path=DEFAULT_ITEMS):
    """Dict item_id -> danh sách đường dẫn category (mỗi đường là danh sách chuỗi đã giải mã HTML)."""
    paths = {}
    with open(items_path, encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                raw = (row.get("metadata") or {}).get("categories") or []
                paths[int(row["item_id"])] = [[html.unescape(str(c)).strip() for c in path] for path in raw if path]
    return paths


def item_groups_by_category(paths, n_items, depth):
    """Danh sách (độ dài n_items) các frozenset nhóm danh mục độ sâu ``depth``; phần tử j ứng với item_id j + 1."""
    if depth < 1:
        raise ValueError("depth phải >= 1")
    return [frozenset(" > ".join(path[:depth]) for path in paths.get(i, [])) for i in range(1, n_items + 1)]


def item_groups_by_brand(metadata, n_items):
    """Nhóm brand (không phân biệt hoa thường); item thiếu brand có tập nhóm rỗng. Tên hiển thị là cách viết gặp đầu tiên."""
    display = {}
    groups = []
    for i in range(1, n_items + 1):
        brand = metadata[i]["brand"]
        if brand is None:
            groups.append(frozenset())
        else:
            display.setdefault(brand.casefold(), brand)
            groups.append(frozenset([brand.casefold()]))
    return [frozenset(display[g] for g in group) for group in groups]


def item_groups_identity(n_items):
    """Cấp item: mỗi item là một nhóm; tên là item_id đệm số 0 để thứ tự tên (dùng khi hòa số đếm) trùng thứ tự item_id."""
    width = len(str(n_items))
    return [frozenset([f"{i:0{width}d}"]) for i in range(1, n_items + 1)]


def group_index(item_groups):
    """(tên nhóm đã sắp, ma trận thưa item × nhóm kiểu bool); nhóm đánh chỉ số 0.. theo tên để kết quả xác định."""
    names = sorted(set().union(*item_groups)) if item_groups else []
    index = {name: j for j, name in enumerate(names)}
    rows, cols = [], []
    for i, groups in enumerate(item_groups):
        for name in groups:
            rows.append(i)
            cols.append(index[name])
    matrix = sparse.csr_matrix((np.ones(len(rows), dtype=bool), (rows, cols)), shape=(len(item_groups), len(names)))
    return names, matrix


def user_group_matrix(item_lists, item_group):
    """Ma trận thưa user × nhóm kiểu bool: nhóm có mặt nếu user có ít nhất một item thuộc nhóm đó.

    item_lists là danh sách mảng item_id (đếm từ 1); item_group là ma trận item × nhóm của group_index.
    """
    n_items = item_group.shape[0]
    lengths = np.fromiter((len(items) for items in item_lists), dtype=np.int64, count=len(item_lists))
    indptr = np.concatenate(([0], np.cumsum(lengths)))
    flat = np.concatenate([np.asarray(items, dtype=np.int64) for items in item_lists]) if indptr[-1] else np.empty(0, np.int64)
    if flat.size and (flat.min() < 1 or flat.max() > n_items):
        raise ValueError(f"item_id phải nằm trong 1..{n_items}")
    user_item = sparse.csr_matrix((np.ones(len(flat), dtype=np.int32), flat - 1, indptr), shape=(len(item_lists), n_items))
    return (user_item @ item_group.astype(np.int32)).astype(bool).tocsr()


def rules_to_matrices(rules, n_groups):
    """(ma trận luật × nhóm của tiền đề, của hậu quả, kích thước tiền đề), kiểu số nguyên."""
    def stack(column):
        rows = [r for r, items in enumerate(rules[column]) for _ in items]
        cols = [j - 1 for items in rules[column] for j in sorted(items)]
        return sparse.csr_matrix((np.ones(len(rows), dtype=np.int32), (rows, cols)), shape=(len(rules), n_groups))
    return stack("antecedents"), stack("consequents"), np.asarray([len(a) for a in rules["antecedents"]], dtype=np.int32)


def predicted_groups(history_groups, rules, n_groups):
    """Hợp các hậu quả của luật áp dụng được cho từng user: ma trận thưa user × nhóm kiểu bool.

    Luật áp dụng được khi mọi nhóm của tiền đề có trong nhóm của lịch sử.
    """
    if rules.empty:
        return sparse.csr_matrix((history_groups.shape[0], n_groups), dtype=bool)
    antecedent, consequent, sizes = rules_to_matrices(rules, n_groups)
    # Phần tử (u, r) là số nhóm của tiền đề r có trong lịch sử u; luật áp dụng được khi bằng kích thước tiền đề.
    matched = (history_groups.astype(np.int32) @ antecedent.T).tocoo()
    keep = matched.data == sizes[matched.col]
    fired = sparse.csr_matrix((np.ones(int(keep.sum()), dtype=np.int32), (matched.row[keep], matched.col[keep])),
                              shape=(history_groups.shape[0], len(rules)))
    return (fired @ consequent).astype(bool).tocsr()


def popularity_hits(target_groups, set_sizes, group_counts):
    """Với từng user: có nhóm đích nào nằm trong ``set_sizes[u]`` nhóm phổ biến nhất (theo group_counts) hay không.

    Hòa số đếm thì nhóm có chỉ số nhỏ hơn đứng trước. Đây là Popularity cùng kích thước tập dự đoán với luật.
    """
    order = np.lexsort((np.arange(len(group_counts)), -np.asarray(group_counts)))
    rank = np.empty(len(order), dtype=np.int64)
    rank[order] = np.arange(len(order))
    target_groups = target_groups.tocsr()
    hits = np.zeros(target_groups.shape[0], dtype=bool)
    for u in range(target_groups.shape[0]):
        cols = target_groups.indices[target_groups.indptr[u]:target_groups.indptr[u + 1]]
        hits[u] = cols.size > 0 and rank[cols].min() < set_sizes[u]
    return hits


def evaluate_rules(rules, history_groups, target_groups, group_counts):
    """Coverage và hit rate của luật (và của Popularity cùng kích thước).

    Coverage tính trên mọi user (mẫu số giống nhau ở mọi cấp). Hit rate tính trên user được phủ (có ít nhất một nhóm
    dự đoán) và có item đích thuộc ít nhất một nhóm; cấp brand loại user có item đích thiếu brand khỏi hit rate.
    """
    n_groups = history_groups.shape[1]
    predicted = predicted_groups(history_groups, rules, n_groups)
    sizes = np.diff(predicted.indptr)
    covered = sizes > 0
    has_target = np.asarray(target_groups.sum(axis=1)).ravel() > 0
    scored = covered & has_target
    rule_hit = np.asarray(predicted.multiply(target_groups).sum(axis=1)).ravel() > 0
    pop_hit = popularity_hits(target_groups, sizes, group_counts)
    n_users, n_covered, n_scored = history_groups.shape[0], int(covered.sum()), int(scored.sum())
    return {
        "n_users": int(n_users),
        "n_covered_users": n_covered,
        "coverage": n_covered / n_users if n_users else None,
        "mean_predicted_groups": float(sizes[covered].mean()) if n_covered else None,
        "n_scored_users": n_scored,
        "hit_rate_rules": float(rule_hit[scored].mean()) if n_scored else None,
        "hit_rate_popularity_same_size": float(pop_hit[scored].mean()) if n_scored else None,
    }


def rule_level_summary(rules):
    """Số luật, số luật 1 ⇒ 1, số nhóm có mặt trong luật, confidence TB, lift trung vị."""
    if rules.empty:
        return {"n_rules": 0, "n_pair_rules": 0, "n_groups_in_rules": 0}
    pair = int((rules["antecedents"].map(len).eq(1) & rules["consequents"].map(len).eq(1)).sum())
    groups = set().union(*rules["antecedents"], *rules["consequents"])
    return {
        "n_rules": int(len(rules)), "n_pair_rules": pair, "n_groups_in_rules": len(groups),
        "confidence_mean": float(rules["confidence"].mean()), "lift_median": float(rules["lift"].median()),
        "lift_max": float(rules["lift"].max()),
    }


def run_level(label, item_groups, train_items, validation, supports, min_confidence, max_len, top_n, show_support):
    """Chạy các mức support cho một cấp nhóm; validation là (danh sách lịch sử, mảng item đích)."""
    names, item_group = group_index(item_groups)
    train_matrix = user_group_matrix(train_items, item_group)
    histories, targets = validation
    history_matrix = user_group_matrix(histories, item_group)
    target_matrix = user_group_matrix([[t] for t in targets], item_group)
    group_counts = np.asarray(train_matrix.sum(axis=0)).ravel()
    no_group = [i + 1 for i, g in enumerate(item_groups) if not g]
    interactions = np.fromiter((len(t) for t in train_items), dtype=np.int64).sum()
    lost = sum(1 for t in train_items for i in t if not item_groups[i - 1])
    result = {
        "level": label, "n_groups": len(names), "n_items_without_group": len(no_group),
        "train_interactions_without_group": int(lost), "train_interactions": int(interactions), "supports": [],
    }
    n_transactions = train_matrix.shape[0]
    for support in supports:
        itemsets, seconds = run_fpgrowth(train_matrix, support, max_len=max_len)
        rules = generate_rules(itemsets, n_transactions, min_confidence=min_confidence, max_rule_len=max_len)
        entry = {
            "min_support": support, "min_count": int(np.ceil(support * n_transactions)),
            "n_frequent_groups": int((itemsets["itemsets"].map(len) == 1).sum()),
            "n_frequent_itemsets": int(len(itemsets)), "fpgrowth_seconds": round(seconds, 3),
            **rule_level_summary(rules), **evaluate_rules(rules, history_matrix, target_matrix, group_counts),
        }
        if support == show_support and not rules.empty:
            entry["top_rules"] = [
                {"antecedents": [names[j - 1] for j in sorted(r.antecedents)],
                 "consequents": [names[j - 1] for j in sorted(r.consequents)],
                 "count": int(r.count), "antecedent_count": int(r.antecedent_count), "consequent_count": int(r.consequent_count),
                 "confidence": float(r.confidence), "lift": float(r.lift), "kulczynski": float(r.kulczynski)}
                for r in top_rules_by_lift(rules, n=top_n).itertuples(index=False)
            ]
        result["supports"].append(entry)
        del itemsets, rules
    return result


def load_validation(path, n_items):
    """(danh sách lịch sử, mảng item đích) từ validation.jsonl; kiểm tra item_id hợp lệ."""
    histories, targets = [], []
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                histories.append(np.asarray(row["history"], dtype=np.int64))
                targets.append(int(row["target"]))
    targets = np.asarray(targets, dtype=np.int64)
    if targets.size == 0 or targets.min() < 1 or targets.max() > n_items:
        raise ValueError(f"item đích phải nằm trong 1..{n_items}")
    return histories, targets


def _fmt(value, digits=3):
    return "—" if value is None else f"{value:.{digits}f}"


def format_markdown(payload):
    lines = [
        "# Luật kết hợp cấp nhóm (thương hiệu, danh mục) và cấp item",
        "",
        f"{payload['n_transactions']:,} giao dịch train; min_confidence = {payload['min_confidence']}; |X ∪ Y| ≤ {payload['max_len']}. "
        f"Chấm trên {payload['n_validation_users']:,} lịch sử validation. Sinh bởi `python -m src.group_rules`.",
        "",
        "- Giao dịch cấp nhóm là tập các nhóm chứa item train của user. Luật và số đếm Popularity chỉ lấy từ train; "
        "validation chỉ dùng để chấm; tương tác test không được dùng.",
        "- Coverage: tỉ lệ user (trên mọi user validation) có ít nhất một luật áp dụng được (tiền đề ⊆ nhóm của lịch sử). "
        "Hit rate: trong user được phủ và có item validation thuộc ít nhất một nhóm (cột \"User chấm hit\"), tỉ lệ nhóm của item "
        "validation nằm trong hợp hậu quả của luật áp dụng được. "
        "Popularity: chọn cùng số nhóm phổ biến nhất (theo số user train) cho mỗi user. "
        "Hậu quả được tính cả khi nhóm đã có trong lịch sử. Hit rate giữa các cấp không so thẳng được vì tập đích "
        "và kích thước tập dự đoán khác nhau; so sánh có ý nghĩa là luật với Popularity trong cùng một dòng.",
        "- Danh mục độ sâu d: tiền tố d phần tử đầu của đường dẫn categories; một item có nhiều đường dẫn thuộc nhiều nhóm. "
        "Item thiếu brand không thuộc nhóm nào ở cấp brand.",
        "",
        "| Cấp | Số nhóm | Item không có nhóm | min_support | Số đếm tối thiểu | Nhóm phổ biến | Luật | Luật 1⇒1 | Nhóm trong luật "
        "| Coverage | Số nhóm dự đoán TB | Hit rate: luật | Hit rate: Popularity cùng kích thước | User chấm hit |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for level in payload["levels"]:
        for e in level["supports"]:
            lines.append(
                f"| {level['level']} | {level['n_groups']:,} | {level['n_items_without_group']:,} | {e['min_support']} | {e['min_count']} "
                f"| {e['n_frequent_groups']:,} | {e['n_rules']:,} | {e['n_pair_rules']:,} | {e['n_groups_in_rules']:,} "
                f"| {_fmt(e['coverage'])} | {_fmt(e['mean_predicted_groups'], 1)} | {_fmt(e['hit_rate_rules'])} "
                f"| {_fmt(e['hit_rate_popularity_same_size'])} | {e['n_scored_users']:,} |"
            )
    weak = [f"{level['level']} (min_support {e['min_support']})" for level in payload["levels"] for e in level["supports"]
            if e["hit_rate_rules"] is not None and e["hit_rate_rules"] <= e["hit_rate_popularity_same_size"]]
    lines += ["", "Cấu hình có hit rate của luật không vượt Popularity cùng kích thước: " + ("; ".join(weak) if weak else "không có") + "."]
    for level in payload["levels"]:
        for e in level["supports"]:
            if "top_rules" not in e:
                continue
            lines += ["", f"## {level['level']}: {len(e['top_rules'])} luật lift cao nhất ở min_support = {e['min_support']}", "",
                      "| # | Tiền đề X | Hậu quả Y | Đếm X∪Y | Đếm X | Đếm Y | Confidence | Lift | Kulczynski |",
                      "|---:|---|---|---:|---:|---:|---:|---:|---:|"]
            for rank, r in enumerate(e["top_rules"], start=1):
                x = " ; ".join(r["antecedents"]).replace("|", "/")
                y = " ; ".join(r["consequents"]).replace("|", "/")
                lines.append(f"| {rank} | {x} | {y} | {r['count']} | {r['antecedent_count']} | {r['consequent_count']} "
                             f"| {r['confidence']:.3f} | {r['lift']:.2f} | {r['kulczynski']:.3f} |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Luật kết hợp cấp nhóm (brand, danh mục) so với cấp item (T6 20/11).")
    parser.add_argument("--interactions", type=Path, default=DEFAULT_INTERACTIONS)
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--items", type=Path, default=DEFAULT_ITEMS)
    parser.add_argument("--validation", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--group-supports", type=float, nargs="+", default=list(GROUP_SUPPORTS))
    parser.add_argument("--item-supports", type=float, nargs="+", default=list(ITEM_SUPPORTS))
    parser.add_argument("--depths", type=int, nargs="+", default=list(CATEGORY_DEPTHS))
    parser.add_argument("--min-confidence", type=float, default=0.1)
    parser.add_argument("--max-len", type=int, default=MAX_LEN)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--show-support", type=float, default=0.01, help="mức support của nhóm in bảng luật lift cao nhất")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args(argv)
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    try:
        metadata = load_item_metadata(args.items)
        n_items = max(metadata)
        if sorted(metadata) != list(range(1, n_items + 1)):
            raise ValueError("item_id trong items.jsonl phải là 1..N liên tục")
        _, train_items = load_train_transactions(args.interactions, args.splits)
        validation = load_validation(args.validation, n_items)
        if len(validation[0]) != len(train_items):
            raise ValueError(f"số user validation {len(validation[0])} khác số user train {len(train_items)}")
        paths = load_category_paths(args.items)
        plans = [("item", item_groups_identity(n_items), args.item_supports, None)]
        plans += [("brand", item_groups_by_brand(metadata, n_items), args.group_supports, args.show_support)]
        plans += [(f"category depth {d}", item_groups_by_category(paths, n_items, d), args.group_supports, args.show_support)
                  for d in args.depths]
        levels = [run_level(label, groups, train_items, validation, supports, args.min_confidence, args.max_len, args.top, show)
                  for label, groups, supports, show in plans]
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"Error: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    payload = {
        "n_transactions": len(train_items), "n_items": n_items, "n_validation_users": len(validation[0]),
        "min_confidence": args.min_confidence, "max_len": args.max_len, "levels": levels,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.markdown.write_text(format_markdown(payload), encoding="utf-8")
    print(f"Wrote {args.output} and {args.markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
