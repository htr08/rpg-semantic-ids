"""Khai phá mẫu mua kèm: FP-Growth (mlxtend), PrefixSpan, đối chiếu luật với semantic ID, embedding đồng xuất hiện (item2vec, SVD).

Phụ trách: C (RQ2).

Tuần 1 (T2): dựng tập giao dịch từ file chỉ số chia dữ liệu và chạy FP-Growth thử.
Tuần 1 (T4): quét min_support, sinh luật kết hợp, lập bảng số luật và luật lift cao nhất kèm tên sản phẩm.
Giao dịch = tập item duy nhất trong phần train của mỗi user (không thứ tự, không có validation/test).
"""

import argparse
import csv
import html
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INTERACTIONS = ROOT / "data" / "processed" / "interactions.csv"
DEFAULT_SPLITS = ROOT / "data" / "splits" / "leave_last_out.csv"
DEFAULT_OUTPUT = ROOT / "reports" / "fpgrowth_trial.json"
DEFAULT_ITEMS = ROOT / "data" / "processed" / "items.jsonl"
DEFAULT_SWEEP_OUTPUT = ROOT / "reports" / "fpgrowth_sweep.json"
DEFAULT_TOP_RULES_OUTPUT = ROOT / "reports" / "fpgrowth_top_rules.md"


def load_train_transactions(interactions_path=DEFAULT_INTERACTIONS, splits_path=DEFAULT_SPLITS):
    """Trả về (user_ids, transactions): mỗi giao dịch là mảng item_id đã sắp xếp, không trùng.

    Train của user là khoảng nửa mở [train_start, train_stop) theo interaction_id
    (chỉ số dòng dữ liệu trong interactions.csv, không tính header). Không đọc phần validation/test.
    """
    interactions = pd.read_csv(interactions_path, usecols=["interaction_id", "item_id"])
    if not np.array_equal(interactions["interaction_id"].to_numpy(), np.arange(len(interactions))):
        raise ValueError("interaction_id phải là 0..n-1 theo thứ tự dòng để dùng làm chỉ số chia")
    items = interactions["item_id"].to_numpy()
    splits = pd.read_csv(splits_path)
    user_ids, transactions = [], []
    for row in splits.itertuples(index=False):
        if not row.train_start <= row.train_stop == row.validation_index:
            raise ValueError(f"user {row.user_id}: chỉ số chia không hợp lệ")
        user_ids.append(int(row.user_id))
        transactions.append(np.unique(items[row.train_start:row.train_stop]))
    return user_ids, transactions


def build_transaction_matrix(transactions, n_items):
    """Ma trận thưa nhị phân (số giao dịch x n_items); cột j ứng với item_id j + 1."""
    lengths = np.fromiter((len(t) for t in transactions), dtype=np.int64, count=len(transactions))
    indptr = np.concatenate(([0], np.cumsum(lengths)))
    if len(transactions) == 0 or indptr[-1] == 0:
        indices = np.empty(0, dtype=np.int64)
    else:
        indices = np.concatenate([np.asarray(t, dtype=np.int64) for t in transactions]) - 1
    if indices.size and (indices.min() < 0 or indices.max() >= n_items):
        raise ValueError("item_id nằm ngoài khoảng 1..n_items")
    data = np.ones(len(indices), dtype=bool)
    return sparse.csr_matrix((data, indices, indptr), shape=(len(transactions), n_items))


def _to_sparse_bool_frame(matrix, names):
    """DataFrame thưa kiểu bool (fill_value False), mỗi cột ứng với một item."""
    dtype = pd.SparseDtype(bool, False)
    columns = matrix.tocsc()
    return pd.DataFrame(
        {name: pd.arrays.SparseArray(columns[:, j].toarray().ravel(), dtype=dtype) for j, name in enumerate(names)}
    )


def run_fpgrowth(matrix, min_support, max_len=None):
    """Chạy mlxtend.fpgrowth trên DataFrame thưa; trả về (tập phổ biến, giây).

    max_len giới hạn số item của tập phổ biến (None = không giới hạn); mọi tập con của tập phổ biến
    đều phổ biến nên kết quả khi giới hạn đúng bằng phần tập có kích thước <= max_len của kết quả đầy đủ.
    itemsets là frozenset các item_id (cột + 1). Chỉ đo thời gian khai phá, không tính dựng DataFrame.
    Pandas yêu cầu tên cột của DataFrame thưa là chuỗi hoặc bắt đầu từ 0, nên dùng tên chuỗi rồi đổi lại thành số.
    Cột dùng SparseDtype(bool, False): mlxtend khuyến nghị kiểu bool, còn from_spmatrix với ma trận bool
    (fill_value 0) bị pandas cảnh báo FutureWarning.
    """
    from mlxtend.frequent_patterns import fpgrowth

    names = [str(j) for j in range(1, matrix.shape[1] + 1)]
    frame = _to_sparse_bool_frame(matrix, names)
    start = time.perf_counter()
    itemsets = fpgrowth(frame, min_support=min_support, use_colnames=True, max_len=max_len)
    elapsed = time.perf_counter() - start
    itemsets["itemsets"] = itemsets["itemsets"].map(lambda s: frozenset(int(name) for name in s))
    return itemsets, elapsed


def summarize_itemsets(itemsets):
    """Số tập phổ biến theo kích thước."""
    sizes = itemsets["itemsets"].map(len)
    return {str(int(size)): int(count) for size, count in sizes.value_counts().sort_index().items()}


RULE_SOURCE_COLUMNS = [
    "antecedents", "consequents", "antecedent support", "consequent support", "support", "confidence", "lift",
]
CONFIDENCE_TOLERANCE = 1e-9
RULE_COLUMNS = [
    "antecedents", "consequents", "support", "confidence", "lift",
    "count", "antecedent_count", "consequent_count", "kulczynski",
]


def generate_rules(itemsets, n_transactions, min_confidence=0.1, max_rule_len=None):
    """Sinh luật X => Y từ tập phổ biến bằng mlxtend.association_rules, giữ luật có confidence >= min_confidence.

    max_rule_len giới hạn |X ∪ Y|; tập con của tập phổ biến đều có mặt trong bảng nên giới hạn này không làm sai
    support của tiền đề và hậu quả. Thêm các cột: count (số giao dịch chứa X ∪ Y), antecedent_count, consequent_count
    và kulczynski = (P(Y|X) + P(X|Y)) / 2, độ đo không phụ thuộc số giao dịch không chứa cả X lẫn Y.
    Trả về DataFrame có các cột RULE_COLUMNS, antecedents và consequents là frozenset item_id.
    """
    from mlxtend.frequent_patterns import association_rules

    sizes = itemsets["itemsets"].map(len)
    if max_rule_len is not None:
        itemsets = itemsets[sizes <= max_rule_len]
        sizes = sizes[sizes <= max_rule_len]
    if sizes.empty or sizes.max() < 2:
        return pd.DataFrame(columns=RULE_COLUMNS)
    # confidence là tỉ số hai số đếm nguyên; với ngưỡng 0,1 = 1/10, mọi confidence khác 0,1 cách nó ít nhất
    # 1 / (10 * n_transactions) (cỡ 4e-6 với 22.363 giao dịch), lớn hơn nhiều CONFIDENCE_TOLERANCE. Trừ dung sai này
    # chỉ để luật có confidence đúng bằng ngưỡng không bị loại do làm tròn số thực (mlxtend chia hai support dạng
    # float nên có thể ra 0,0999...).
    rules = association_rules(itemsets, metric="confidence", min_threshold=min_confidence - CONFIDENCE_TOLERANCE)
    if rules.empty:
        return pd.DataFrame(columns=RULE_COLUMNS)
    # association_rules trả thêm leverage, conviction, ...; bỏ tại chỗ các cột không dùng để giảm bộ nhớ (hàng trăm nghìn luật).
    rules.drop(columns=[c for c in rules.columns if c not in RULE_SOURCE_COLUMNS], inplace=True)
    rules.rename(columns={"antecedent support": "antecedent_support", "consequent support": "consequent_support"}, inplace=True)
    rules["count"] = np.rint(rules["support"] * n_transactions).astype(int)
    rules["antecedent_count"] = np.rint(rules["antecedent_support"] * n_transactions).astype(int)
    rules["consequent_count"] = np.rint(rules["consequent_support"] * n_transactions).astype(int)
    rules["kulczynski"] = 0.5 * (rules["confidence"] + rules["count"] / rules["consequent_count"])
    return rules[RULE_COLUMNS].reset_index(drop=True)


def top_rules_by_lift(rules, n=20, unique_itemsets=True):
    """n luật có lift cao nhất; hòa lift thì ưu tiên số đếm lớn hơn, rồi thứ tự item để kết quả xác định.

    unique_itemsets=True chỉ giữ một luật (lift cao nhất) cho mỗi tập X ∪ Y, vì luật A => B và B => A có cùng lift
    và cùng số đếm nên sẽ chiếm hai dòng cho một cặp item.
    """
    if rules.empty:
        return rules.copy()
    # Làm tròn lift khi sắp xếp để sai khác dấu phẩy động (cùng giá trị toán học) không phá thứ tự hòa.
    ranked = rules.assign(
        _lift=rules["lift"].round(9),
        _union=[tuple(sorted(a | c)) for a, c in zip(rules["antecedents"], rules["consequents"])],
        _antecedent=[tuple(sorted(a)) for a in rules["antecedents"]],
    ).sort_values(["_lift", "count", "_union", "_antecedent"], ascending=[False, False, True, True], kind="mergesort")
    if unique_itemsets:
        ranked = ranked.drop_duplicates("_union")
    return ranked.head(n).drop(columns=["_lift", "_union", "_antecedent"]).reset_index(drop=True)


def pair_rules_only(rules):
    """Giữ luật một item => một item (|X| = |Y| = 1), dạng mua kèm đơn giản nhất và không bị lặp theo tập con."""
    if rules.empty:
        return rules.copy()
    mask = rules["antecedents"].map(len).eq(1) & rules["consequents"].map(len).eq(1)
    return rules[mask].reset_index(drop=True)


def load_item_metadata(items_path=DEFAULT_ITEMS):
    """Dict item_id -> {title, brand, leaf_categories} đọc từ items.jsonl.

    Mã HTML trong văn bản (&amp;, &#9733;) được giải mã. brand rỗng hoặc "Unknown" được coi là thiếu (None).
    leaf_categories là tập phần tử cuối của mọi đường dẫn category.
    """
    metadata = {}
    with open(items_path, encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            meta = row.get("metadata") or {}
            brand = html.unescape(meta.get("brand") or "").strip()
            paths = meta.get("categories") or []
            metadata[int(row["item_id"])] = {
                "title": html.unescape(meta.get("title") or "").strip(),
                "brand": None if brand.lower() in ("", "unknown") else brand,
                "leaf_categories": frozenset(path[-1] for path in paths if path),
            }
    return metadata


def describe_rule(antecedents, consequents, metadata):
    """Tên sản phẩm của luật và hai cờ: cùng thương hiệu, có chung danh mục lá.

    same_brand: mọi item trong X ∪ Y có brand (khác thiếu) và chuỗi brand giống nhau, không phân biệt hoa thường
    (các biến thể như "SHANY" và "Shany Cosmetics" vẫn được tính là khác nhau).
    same_leaf_category: mọi item trong X ∪ Y có chung ít nhất một danh mục lá.
    """
    union = sorted(antecedents | consequents)
    brands = {None if metadata[i]["brand"] is None else metadata[i]["brand"].casefold() for i in union}
    leaves = frozenset.intersection(*(metadata[i]["leaf_categories"] for i in union))
    return {
        "antecedent_titles": [metadata[i]["title"] for i in sorted(antecedents)],
        "consequent_titles": [metadata[i]["title"] for i in sorted(consequents)],
        "same_brand": len(brands) == 1 and None not in brands,
        "same_leaf_category": bool(leaves),
    }


def summarize_rules(rules):
    """Thống kê mô tả của tập luật: số luật theo |X ∪ Y|, confidence và lift trung bình/trung vị, số item có mặt trong luật."""
    if rules.empty:
        return {"n_rules": 0, "rules_by_size": {}, "n_items_in_rules": 0}
    union_sizes = [len(a | c) for a, c in zip(rules["antecedents"], rules["consequents"])]
    items = set().union(*rules["antecedents"], *rules["consequents"])
    return {
        "n_rules": int(len(rules)),
        "rules_by_size": {str(k): int(v) for k, v in pd.Series(union_sizes).value_counts().sort_index().items()},
        "n_items_in_rules": len(items),
        "confidence_mean": float(rules["confidence"].mean()),
        "confidence_median": float(rules["confidence"].median()),
        "lift_mean": float(rules["lift"].mean()),
        "lift_median": float(rules["lift"].median()),
        "lift_max": float(rules["lift"].max()),
        "count_median": float(rules["count"].median()),
    }


def _short(text, limit=70):
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _titles(ids, metadata):
    return "<br>".join(f"[{i}] {_short(metadata[i]['title'])}".replace("|", "/") for i in sorted(ids))


def _rule_rows(top, metadata):
    """Chuyển DataFrame luật đã xếp hạng thành danh sách dict kèm tên sản phẩm và cờ brand/danh mục."""
    rows = []
    for row in top.itertuples(index=False):
        info = describe_rule(row.antecedents, row.consequents, metadata)
        rows.append({
            "antecedents": sorted(row.antecedents),
            "consequents": sorted(row.consequents),
            "antecedent_text": _titles(row.antecedents, metadata),
            "consequent_text": _titles(row.consequents, metadata),
            "antecedent_titles": info["antecedent_titles"],
            "consequent_titles": info["consequent_titles"],
            "count": int(row.count),
            "antecedent_count": int(row.antecedent_count),
            "consequent_count": int(row.consequent_count),
            "support": float(row.support),
            "confidence": float(row.confidence),
            "lift": float(row.lift),
            "kulczynski": float(row.kulczynski),
            "same_brand": info["same_brand"],
            "same_leaf_category": info["same_leaf_category"],
        })
    return rows


def _rules_table(rows):
    """Các dòng Markdown của một bảng luật và câu tóm tắt cờ brand/danh mục."""
    lines = [
        "| # | Tiền đề X | Hậu quả Y | Đếm X∪Y | Đếm X | Đếm Y | Confidence | Lift | Kulczynski | Cùng brand | Chung danh mục lá |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|:---:|:---:|",
    ]
    for rank, row in enumerate(rows, start=1):
        lines.append(
            f"| {rank} | {row['antecedent_text']} | {row['consequent_text']} | {row['count']} "
            f"| {row['antecedent_count']} | {row['consequent_count']} | {row['confidence']:.3f} "
            f"| {row['lift']:.1f} | {row['kulczynski']:.3f} "
            f"| {'có' if row['same_brand'] else 'không'} | {'có' if row['same_leaf_category'] else 'không'} |"
        )
    if rows:
        lines += [
            "",
            f"Trong {len(rows)} luật: {sum(r['same_brand'] for r in rows)} luật cùng brand, "
            f"{sum(r['same_leaf_category'] for r in rows)} luật chung danh mục lá; "
            f"số đếm X∪Y nhỏ nhất {min(r['count'] for r in rows)}, lớn nhất {max(r['count'] for r in rows)}.",
        ]
    else:
        lines += ["", "Không có luật."]
    return lines


def format_top_rules_markdown(level_results, n_transactions):
    """Bảng Markdown: số tập phổ biến, số luật theo từng mức min_support và luật lift cao nhất."""
    first = level_results[0]
    lines = [
        "# Luật kết hợp trên giao dịch train (FP-Growth)",
        "",
        f"{n_transactions:,} giao dịch; min_confidence = {first['min_confidence']}; "
        f"tập phổ biến và luật giới hạn |X ∪ Y| ≤ {first['max_len']}. Sinh bằng `python -m src.patterns rules`.",
        "",
        "Ghi chú:",
        "",
        "- Giao dịch là tập item duy nhất trong phần train của mỗi user; không dùng validation và test.",
        f"- Giới hạn |X ∪ Y| ≤ {first['max_len']} chỉ cắt các tập phổ biến dài hơn; số item phổ biến và số đếm của mọi luật "
        "không đổi. Số tập phổ biến không giới hạn độ dài nằm trong `fpgrowth_0.0005.json`, `fpgrowth_0.001.json` "
        "và `fpgrowth_trial.json` (mức 0,002).",
        "- Xếp hạng theo lift giảm dần, hòa thì theo số đếm X ∪ Y giảm dần, rồi theo thứ tự item. Luật A ⇒ B và B ⇒ A của "
        "cùng một cặp có cùng lift và số đếm, nên bảng thứ nhất chỉ giữ một luật cho mỗi tập X ∪ Y.",
        "- Với support thấp, lift bị chặn trên bởi 1 / max(sup(X), sup(Y)); số đếm X ∪ Y ở các luật đứng đầu chỉ vài chục "
        "giao dịch (cột Đếm X∪Y) nên lift lớn không đồng nghĩa với quan hệ mua kèm bền vững. Cột Kulczynski dùng để đối chiếu.",
        "- Cùng brand: mọi item trong X ∪ Y có brand giống nhau (không phân biệt hoa thường; brand rỗng hoặc \"Unknown\" "
        "tính là thiếu). Các biến thể ghi khác nhau của một thương hiệu (ví dụ \"SHANY\" và \"Shany Cosmetics\") "
        "được tính là khác brand.",
        "- Chung danh mục lá: mọi item trong X ∪ Y có chung ít nhất một danh mục lá (phần tử cuối của một đường dẫn "
        "`categories`). Quy tắc chọn danh mục lá chính thức (G4) chưa được chốt; cờ này chỉ dùng để mô tả.",
        "",
        "| min_support | Số giao dịch tối thiểu | Item phổ biến | Tập phổ biến (≤ max_len) | Luật | Luật 1 item ⇒ 1 item "
        "| Item có trong luật | Confidence TB | Lift trung vị | Thời gian FP-Growth (s) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for level in level_results:
        stats = level["rules_summary"]
        lines.append(
            f"| {level['min_support']} | {level['min_count']} | {level['itemsets_by_size'].get('1', 0):,} "
            f"| {level['n_frequent_itemsets']:,} | {stats['n_rules']:,} "
            f"| {stats.get('rules_by_size', {}).get('2', 0):,} | {stats['n_items_in_rules']:,} "
            f"| {stats.get('confidence_mean', float('nan')):.3f} | {stats.get('lift_median', float('nan')):.1f} "
            f"| {level['fpgrowth_seconds']:.1f} |"
        )
    for level in level_results:
        lines += [
            "",
            f"## min_support = {level['min_support']} (số đếm ≥ {level['min_count']})",
            "",
            f"### {len(level['top_rules'])} luật lift cao nhất (mỗi tập X ∪ Y chỉ giữ một luật)",
            "",
            *_rules_table(level["top_rules"]),
            "",
            f"### {len(level['top_pair_rules'])} luật lift cao nhất dạng một item ⇒ một item",
            "",
            *_rules_table(level["top_pair_rules"]),
        ]
    return "\n".join(lines) + "\n"


def run_rule_sweep(matrix, supports, min_confidence, max_len, top_n, metadata):
    """Với từng min_support: FP-Growth (giới hạn max_len), sinh luật, thống kê và top luật lift cao nhất."""
    n_transactions = matrix.shape[0]
    results = []
    for min_support in supports:
        itemsets, fp_seconds = run_fpgrowth(matrix, min_support, max_len=max_len)
        start = time.perf_counter()
        rules = generate_rules(itemsets, n_transactions, min_confidence=min_confidence, max_rule_len=max_len)
        rules_seconds = time.perf_counter() - start
        results.append({
            "min_support": min_support,
            "min_count": int(np.ceil(min_support * n_transactions)),
            "max_len": max_len,
            "min_confidence": min_confidence,
            "n_frequent_itemsets": int(len(itemsets)),
            "itemsets_by_size": summarize_itemsets(itemsets),
            "fpgrowth_seconds": round(fp_seconds, 3),
            "rules_seconds": round(rules_seconds, 3),
            "rules_summary": summarize_rules(rules),
            "top_rules": _rule_rows(top_rules_by_lift(rules, n=top_n), metadata),
            "top_pair_rules": _rule_rows(top_rules_by_lift(pair_rules_only(rules), n=top_n), metadata),
        })
        del itemsets, rules
    return results


def rules_main(argv=None):
    parser = argparse.ArgumentParser(description="Quét min_support, sinh luật kết hợp và lập bảng luật lift cao nhất.")
    parser.add_argument("--interactions", type=Path, default=DEFAULT_INTERACTIONS)
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--items", type=Path, default=DEFAULT_ITEMS)
    parser.add_argument("--supports", type=float, nargs="+", default=[0.0005, 0.001, 0.002])
    parser.add_argument("--min-confidence", type=float, default=0.1)
    parser.add_argument("--max-len", type=int, default=4, help="giới hạn số item của tập phổ biến và của X ∪ Y")
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--output", type=Path, default=DEFAULT_SWEEP_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_TOP_RULES_OUTPUT)
    args = parser.parse_args(argv)
    try:
        n_items = int(pd.read_csv(args.interactions, usecols=["item_id"])["item_id"].max())
        _, transactions = load_train_transactions(args.interactions, args.splits)
        matrix = build_transaction_matrix(transactions, n_items)
        metadata = load_item_metadata(args.items)
        results = run_rule_sweep(matrix, args.supports, args.min_confidence, args.max_len, args.top, metadata)
    except (OSError, ValueError, KeyError, csv.Error) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    payload = {"n_transactions": int(matrix.shape[0]), "n_items": int(matrix.shape[1]), "levels": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.markdown.write_text(format_top_rules_markdown(results, matrix.shape[0]), encoding="utf-8")
    for level in results:
        print(json.dumps({k: v for k, v in level.items() if not k.startswith("top_")}, ensure_ascii=False))
    return 0


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "rules":
        return rules_main(argv[1:])
    parser = argparse.ArgumentParser(description="FP-Growth thử trên tập giao dịch train (T2 tuần 1).")
    parser.add_argument("--interactions", type=Path, default=DEFAULT_INTERACTIONS)
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--min-support", type=float, default=0.002)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    try:
        n_items = int(pd.read_csv(args.interactions, usecols=["item_id"])["item_id"].max())
        _, transactions = load_train_transactions(args.interactions, args.splits)
        matrix = build_transaction_matrix(transactions, n_items)
        itemsets, elapsed = run_fpgrowth(matrix, args.min_support)
    except (OSError, ValueError, KeyError, csv.Error) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    sizes = matrix.sum(axis=1).A1
    result = {
        "min_support": args.min_support,
        "min_count": int(np.ceil(args.min_support * matrix.shape[0])),
        "n_transactions": int(matrix.shape[0]),
        "n_items": int(matrix.shape[1]),
        "n_nonzero": int(matrix.nnz),
        "mean_transaction_size": float(sizes.mean()),
        "n_frequent_itemsets": int(len(itemsets)),
        "itemsets_by_size": summarize_itemsets(itemsets),
        "fpgrowth_seconds": round(elapsed, 3),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
