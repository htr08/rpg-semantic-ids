"""Khai phá mẫu mua kèm: FP-Growth (mlxtend), PrefixSpan, đối chiếu luật với semantic ID, embedding đồng xuất hiện (item2vec, SVD).

Phụ trách: C (RQ2).

Tuần 1 (T2): dựng tập giao dịch từ file chỉ số chia dữ liệu và chạy FP-Growth thử.
Tuần 1 (T4): quét min_support, sinh luật kết hợp, lập bảng số luật và luật lift cao nhất kèm tên sản phẩm.
Tuần 2 (T2 16/11): PrefixSpan trên chuỗi train có thứ tự, so với tập phổ biến và luật không thứ tự.
Giao dịch = tập item duy nhất trong phần train của mỗi user (không thứ tự, không có validation/test).
Chuỗi = item_id của phần train theo thứ tự timestamp, trùng timestamp thì theo số dòng gốc.
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


def load_train_sequences(interactions_path=DEFAULT_INTERACTIONS, splits_path=DEFAULT_SPLITS):
    """Trả về (user_ids, sequences, timestamps): chuỗi item_id train theo thứ tự và timestamp tương ứng.

    Train của user là khoảng nửa mở [train_start, train_stop) theo interaction_id
    (chỉ số dòng dữ liệu trong interactions.csv, không tính header). Không đọc phần validation/test.
    Thứ tự là thứ tự trong interactions.csv: timestamp tăng dần, trùng timestamp thì theo số dòng gốc.
    """
    interactions = pd.read_csv(interactions_path, usecols=["interaction_id", "item_id", "timestamp"])
    if not np.array_equal(interactions["interaction_id"].to_numpy(), np.arange(len(interactions))):
        raise ValueError("interaction_id phải là 0..n-1 theo thứ tự dòng để dùng làm chỉ số chia")
    items = interactions["item_id"].to_numpy()
    times = interactions["timestamp"].to_numpy()
    splits = pd.read_csv(splits_path)
    user_ids, sequences, timestamps = [], [], []
    for row in splits.itertuples(index=False):
        if not row.train_start <= row.train_stop == row.validation_index:
            raise ValueError(f"user {row.user_id}: chỉ số chia không hợp lệ")
        user_ids.append(int(row.user_id))
        sequences.append(items[row.train_start:row.train_stop])
        timestamps.append(times[row.train_start:row.train_stop])
    return user_ids, sequences, timestamps


def load_train_transactions(interactions_path=DEFAULT_INTERACTIONS, splits_path=DEFAULT_SPLITS):
    """Trả về (user_ids, transactions): mỗi giao dịch là mảng item_id đã sắp xếp, không trùng (tập item của chuỗi train)."""
    user_ids, sequences, _ = load_train_sequences(interactions_path, splits_path)
    return user_ids, [np.unique(sequence) for sequence in sequences]


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


# --- PrefixSpan: mẫu tuần tự trên chuỗi train có thứ tự ---

DIRECTION_SHARE = 0.8
DEFAULT_PREFIXSPAN_OUTPUT = ROOT / "reports" / "prefixspan_sweep.json"
DEFAULT_PREFIXSPAN_MARKDOWN = ROOT / "reports" / "prefixspan_patterns.md"


def mine_sequential_patterns(sequences, min_count, max_len=3):
    """PrefixSpan (thư viện prefixspan) trên chuỗi item_id; trả về [(số chuỗi chứa mẫu, mẫu)] đã sắp xếp.

    Mẫu là dãy con (không cần liền kề) có độ dài 1..max_len; số đếm là số user có ít nhất một lần
    xuất hiện đúng thứ tự, mỗi user tính một lần. min_count là số đếm tuyệt đối (số user).
    Sắp xếp theo số đếm giảm dần, hòa thì theo mẫu để kết quả xác định.
    """
    from prefixspan import PrefixSpan

    miner = PrefixSpan([[int(item) for item in sequence] for sequence in sequences])
    miner.maxlen = max_len
    found = miner.frequent(int(min_count))
    return sorted(
        ((int(count), tuple(int(i) for i in pattern)) for count, pattern in found), key=lambda x: (-x[0], x[1])
    )


def count_ordered_pairs(sequences, timestamps, n_items):
    """Đếm mọi cặp có thứ tự (a, b): số user có a xuất hiện trước b.

    Trả về DataFrame (first, second, count, count_strict) sắp theo (first, second). count dùng thứ tự trong chuỗi
    (trùng timestamp theo số dòng gốc), cùng định nghĩa số đếm của PrefixSpan độ dài 2; count_strict chỉ tính khi
    timestamp của a nhỏ hơn timestamp của b (thứ tự do dữ liệu xác định). Mỗi user tính một lần cho mỗi cặp.
    """
    base = n_items + 1
    keys_all, keys_strict = [], []
    for sequence, times in zip(sequences, timestamps):
        if len(sequence) < 2:
            continue
        i, j = np.triu_indices(len(sequence), k=1)
        key = sequence[i].astype(np.int64) * base + sequence[j]
        keys_all.append(np.unique(key))
        keys_strict.append(np.unique(key[times[i] < times[j]]))
    if not keys_all:
        return pd.DataFrame({name: np.empty(0, dtype=np.int64) for name in ("first", "second", "count", "count_strict")})
    all_keys, all_counts = np.unique(np.concatenate(keys_all), return_counts=True)
    strict_keys, strict_counts = np.unique(np.concatenate(keys_strict), return_counts=True)
    strict_full = np.zeros(len(all_keys), dtype=np.int64)
    strict_full[np.searchsorted(all_keys, strict_keys)] = strict_counts
    return pd.DataFrame({
        "first": all_keys // base, "second": all_keys % base,
        "count": all_counts.astype(np.int64), "count_strict": strict_full,
    })


def _lookup_ordered(ordered, first, second, n_items):
    """(count, count_strict) của các cặp có thứ tự (first[k], second[k]); 0 nếu cặp không xuất hiện."""
    wanted = np.asarray(first, dtype=np.int64) * (n_items + 1) + np.asarray(second, dtype=np.int64)
    if ordered.empty:
        return np.zeros(len(wanted), dtype=np.int64), np.zeros(len(wanted), dtype=np.int64)
    keys = ordered["first"].to_numpy() * (n_items + 1) + ordered["second"].to_numpy()
    position = np.minimum(np.searchsorted(keys, wanted), len(keys) - 1)
    found = keys[position] == wanted
    counts = np.where(found, ordered["count"].to_numpy()[position], 0)
    strict = np.where(found, ordered["count_strict"].to_numpy()[position], 0)
    return counts, strict


def direction_null_expectation(totals, share=DIRECTION_SHARE):
    """Số cặp kỳ vọng có một hướng chiếm ít nhất `share` tổng, nếu mỗi user chọn thứ tự độc lập với xác suất 1/2.

    totals là tổng số user của từng cặp (mảng số nguyên; phần tử < 1 bị bỏ). Với n user, số user theo hướng
    a -> b ~ Binomial(n, 1/2); lấy xác suất một trong hai hướng chiếm >= share rồi cộng theo cặp (điều kiện theo n
    quan sát). Đây chỉ là mốc so sánh: giả định user độc lập và thứ tự ngẫu nhiên, trong khi một user đóng góp cho
    nhiều cặp và thứ tự phá hòa không ngẫu nhiên.
    """
    from scipy.stats import binom

    totals = np.asarray(totals, dtype=np.int64)
    totals = totals[totals >= 1]
    k_high = np.ceil(share * totals - 1e-9).astype(np.int64)
    probability = binom.sf(k_high - 1, totals, 0.5) + binom.cdf(totals - k_high, totals, 0.5)
    return float(np.sum(np.minimum(probability, 1.0)))


def compare_sequential_unordered(sequences, timestamps, itemsets, n_items, min_count, min_confidence, max_len=3):
    """So mẫu tuần tự (PrefixSpan) với tập phổ biến không thứ tự (FP-Growth) cùng ngưỡng đếm tối thiểu.

    itemsets: kết quả run_fpgrowth với cùng max_len và min_support tương ứng min_count. Trả về
    (tóm tắt, bảng cặp không thứ tự kèm số đếm hai hướng, danh sách mẫu tuần tự). `checks` trong tóm tắt kiểm tra
    tính nhất quán giữa các thuật toán: số đếm item độ dài 1 khớp FP-Growth; mẫu độ dài 2 khớp bộ đếm cặp độc lập;
    tập item của mọi mẫu tuần tự là tập phổ biến với số đếm không nhỏ hơn; count(a->b) + count(b->a) bằng số đếm
    của {a, b} (đúng khi không user nào lặp item, như trong train Beauty).
    min_count phải bằng ceil(min_support × N) của itemsets; mlxtend lọc item đơn bằng so sánh số thực nên có thể lệch
    ở các cặp (N, min_support) hiếm mà min_support × N là số nguyên, điều không xảy ra với N = 22.363.
    """
    n_sequences = len(sequences)
    start = time.perf_counter()
    patterns = mine_sequential_patterns(sequences, min_count, max_len)
    prefixspan_seconds = time.perf_counter() - start
    by_length = {length: [(c, p) for c, p in patterns if len(p) == length] for length in range(1, max_len + 1)}

    itemset_count = {
        s: int(np.rint(sup * n_sequences)) for s, sup in zip(itemsets["itemsets"], itemsets["support"])
    }
    ordered = count_ordered_pairs(sequences, timestamps, n_items)
    item_count = np.bincount(
        np.concatenate([np.empty(0, dtype=np.int64)] + [np.unique(s) for s in sequences]), minlength=n_items + 1
    )

    items_match = {p[0]: c for c, p in by_length[1]} == {
        next(iter(s)): c for s, c in itemset_count.items() if len(s) == 1
    }
    frequent_ordered = ordered[ordered["count"] >= min_count]
    pairs_counter = {
        (int(a), int(b)): int(c)
        for a, b, c in zip(frequent_ordered["first"], frequent_ordered["second"], frequent_ordered["count"])
    }
    pairs_match = max_len < 2 or pairs_counter == {p: c for c, p in by_length[2]}
    subset_violations = sum(1 for c, p in patterns if itemset_count.get(frozenset(p), -1) < c)

    pair_sets = [s for s in itemset_count if len(s) == 2]
    low = np.array([min(s) for s in pair_sets], dtype=np.int64)
    high = np.array([max(s) for s in pair_sets], dtype=np.int64)
    low_high, low_high_strict = _lookup_ordered(ordered, low, high, n_items)
    high_low, high_low_strict = _lookup_ordered(ordered, high, low, n_items)
    table = pd.DataFrame({
        "low": low, "high": high, "unordered": [itemset_count[s] for s in pair_sets],
        "low_to_high": low_high, "high_to_low": high_low,
        "low_to_high_strict": low_high_strict, "high_to_low_strict": high_low_strict,
    }).sort_values(["low", "high"]).reset_index(drop=True)

    identity_violations = int(np.sum(table["low_to_high"] + table["high_to_low"] != table["unordered"]))
    strict_total = table["low_to_high_strict"] + table["high_to_low_strict"]
    tied_share = float(1.0 - strict_total.sum() / table["unordered"].sum()) if len(table) else None
    forward_frequent = table["low_to_high"] >= min_count
    backward_frequent = table["high_to_low"] >= min_count

    def direction_stats(forward, backward):
        total = forward + backward
        valid = total >= 1
        share = (np.maximum(forward, backward)[valid] / total[valid]).to_numpy()
        return {
            "n_pairs": int(valid.sum()),
            "share_median": float(np.median(share)) if share.size else None,
            "pairs_share_ge_threshold": int(np.sum(share >= DIRECTION_SHARE - 1e-9)),
            "pairs_share_ge_threshold_expected_symmetric": direction_null_expectation(total[valid].to_numpy()),
        }

    tolerance = CONFIDENCE_TOLERANCE
    low_count, high_count = item_count[table["low"]], item_count[table["high"]]
    unordered_rules = int(
        np.sum(table["unordered"] / low_count >= min_confidence - tolerance)
        + np.sum(table["unordered"] / high_count >= min_confidence - tolerance)
    )
    sequential_rules = int(
        np.sum(forward_frequent & (table["low_to_high"] / low_count >= min_confidence - tolerance))
        + np.sum(backward_frequent & (table["high_to_low"] / high_count >= min_confidence - tolerance))
    )

    triples = [s for s in itemset_count if len(s) == 3]
    ordered_triples = {frozenset(p) for _, p in by_length.get(3, [])}
    summary = {
        "min_count": int(min_count),
        "n_sequences": n_sequences,
        "max_len": max_len,
        "prefixspan_seconds": round(prefixspan_seconds, 3),
        "sequential_by_length": {str(k): len(v) for k, v in by_length.items()},
        "unordered_by_length": {str(k): sum(1 for s in itemset_count if len(s) == k) for k in range(1, max_len + 1)},
        "pairs": {
            "unordered_frequent": int(len(table)),
            "both_directions_frequent": int((forward_frequent & backward_frequent).sum()),
            "one_direction_frequent": int((forward_frequent ^ backward_frequent).sum()),
            "no_direction_frequent": int((~forward_frequent & ~backward_frequent).sum()),
            "tied_support_share": tied_share,
            "direction_all_orders": direction_stats(table["low_to_high"], table["high_to_low"]),
            "direction_strict_orders": direction_stats(table["low_to_high_strict"], table["high_to_low_strict"]),
        },
        "triples": {
            "unordered_frequent": len(triples),
            "with_frequent_ordering": sum(1 for s in triples if s in ordered_triples),
            "sequential_patterns": len(by_length.get(3, [])),
        },
        "rules_1_to_1": {"min_confidence": min_confidence, "unordered": unordered_rules, "sequential": sequential_rules},
        "checks": {
            "item_counts_match_fpgrowth": bool(items_match),
            "pair_patterns_match_pair_counter": bool(pairs_match),
            "sequential_itemsets_frequent_violations": int(subset_violations),
            "pair_order_identity_violations": identity_violations,
        },
    }
    return summary, table, patterns


def _top_sequential_pairs(patterns, table, metadata, n):
    """n mẫu tuần tự a -> b (a khác b) có số đếm lớn nhất kèm số đếm hướng ngược, số đếm không thứ tự và thứ tự chặt.

    Mẫu lặp item (a -> a) không có cặp không thứ tự tương ứng nên bị bỏ; train Beauty không có item lặp.
    """
    by_pair = {(int(r.low), int(r.high)): r for r in table.itertuples(index=False)}
    rows = []
    for count, pattern in patterns:
        if len(pattern) != 2 or pattern[0] == pattern[1]:
            continue
        a, b = pattern
        row = by_pair[(min(a, b), max(a, b))]
        forward_is_low = a < b
        rows.append({
            "first": a, "second": b,
            "first_title": metadata[a]["title"], "second_title": metadata[b]["title"],
            "count_forward": int(count),
            "count_backward": int(row.high_to_low if forward_is_low else row.low_to_high),
            "count_unordered": int(row.unordered),
            "count_forward_strict": int(row.low_to_high_strict if forward_is_low else row.high_to_low_strict),
            "count_backward_strict": int(row.high_to_low_strict if forward_is_low else row.low_to_high_strict),
        })
        if len(rows) == n:
            break
    return rows


def run_sequence_sweep(sequences, timestamps, matrix, supports, min_confidence, max_len, top_n, metadata):
    """Với từng min_support: FP-Growth và PrefixSpan cùng ngưỡng đếm, so sánh và lấy top mẫu tuần tự độ dài 2."""
    n_items = matrix.shape[1]
    results = []
    for min_support in supports:
        min_count = int(np.ceil(min_support * len(sequences)))
        itemsets, fp_seconds = run_fpgrowth(matrix, min_support, max_len=max_len)
        summary, table, patterns = compare_sequential_unordered(
            sequences, timestamps, itemsets, n_items, min_count, min_confidence, max_len
        )
        summary.update({
            "min_support": min_support,
            "fpgrowth_seconds": round(fp_seconds, 3),
            "top_sequential_pairs": _top_sequential_pairs(patterns, table, metadata, top_n),
        })
        results.append(summary)
        del itemsets, table, patterns
    return results


def format_sequence_markdown(levels, n_sequences, tie_stats):
    """Báo cáo Markdown: số mẫu theo min_support, so sánh cặp có/không thứ tự, kiểm tra nhất quán và top mẫu."""
    first = levels[0]
    max_len = first["max_len"]
    lines = [
        "# Mẫu tuần tự (PrefixSpan) so với tập phổ biến không thứ tự (FP-Growth)",
        "",
        f"{n_sequences:,} chuỗi train có thứ tự; mẫu và tập phổ biến có độ dài ≤ {max_len}; min_confidence cho luật 1 item ⇒ 1 item "
        f"= {first['rules_1_to_1']['min_confidence']}. Sinh bằng `python -m src.patterns prefixspan`.",
        "",
        "Ghi chú:",
        "",
        "- Số đếm của mẫu là số user có mẫu là dãy con (không cần liền kề) của chuỗi train; mỗi user tính một lần. "
        "Ngưỡng đếm tối thiểu giống FP-Growth.",
        "- Khi không user nào lặp item (đúng với train Beauty), count(a → b) + count(b → a) bằng số đếm của {a, b}: "
        "thứ tự chia số đếm của mỗi cặp thành hai phần. Do đó số mẫu tuần tự không vượt số tập phổ biến không thứ tự, "
        "và một cặp có số đếm dưới 2 × ngưỡng không thể có cả hai hướng đạt ngưỡng; các số này một phần là hệ quả của phép chia, "
        "chưa tự nó cho biết thứ tự có thêm thông tin hay không.",
        f"- Thứ tự theo `interactions.csv`: timestamp tăng dần, trùng timestamp theo số dòng gốc. "
        f"Trong phần train, {tie_stats['rows_in_tie_groups']:,} / {tie_stats['rows']:,} tương tác "
        f"({tie_stats['rows_in_tie_groups'] / tie_stats['rows']:.1%}) thuộc nhóm trùng (user, timestamp), "
        f"và {tie_stats['adjacent_ties']:,} / {tie_stats['adjacent_pairs']:,} cặp liền kề trong chuỗi có cùng timestamp. "
        f"Mọi timestamp là mốc ngày: {'có' if tie_stats['timestamps_whole_days'] else 'không'}. "
        "Cột \"thứ tự chặt\" chỉ tính user có timestamp của hai item khác nhau.",
        f"- Hướng chiếm ≥ {DIRECTION_SHARE:.0%} của một cặp: số cặp quan sát được so với số cặp kỳ vọng nếu mỗi user chọn thứ tự "
        "độc lập với xác suất 1/2 (phân phối nhị thức theo số user của từng cặp). Mốc chỉ cho biết kết quả khác thứ tự ngẫu nhiên "
        "hoàn toàn; không cho biết nguyên nhân (xem mục Hạn chế).",
        "",
        "| min_support | Đếm tối thiểu | Item phổ biến | Cặp phổ biến (không thứ tự) | Mẫu tuần tự độ dài 2 "
        "| Tập 3 item phổ biến | Mẫu tuần tự độ dài 3 | FP-Growth (s) | PrefixSpan (s) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for level in levels:
        lines.append(
            f"| {level['min_support']} | {level['min_count']} | {level['sequential_by_length']['1']:,} "
            f"| {level['unordered_by_length'].get('2', 0):,} | {level['sequential_by_length'].get('2', 0):,} "
            f"| {level['unordered_by_length'].get('3', 0):,} | {level['sequential_by_length'].get('3', 0):,} "
            f"| {level['fpgrowth_seconds']:.1f} | {level['prefixspan_seconds']:.1f} |"
        )
    lines += [
        "",
        "## Cặp phổ biến theo hướng",
        "",
        "| min_support | Cặp phổ biến | Cả hai hướng đạt ngưỡng | Một hướng đạt | Không hướng nào đạt "
        "| Phần số đếm cặp có timestamp trùng | Luật 1 ⇒ 1 không thứ tự | Luật 1 → 1 tuần tự |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for level in levels:
        pairs = level["pairs"]
        lines.append(
            f"| {level['min_support']} | {pairs['unordered_frequent']:,} | {pairs['both_directions_frequent']:,} "
            f"| {pairs['one_direction_frequent']:,} | {pairs['no_direction_frequent']:,} "
            f"| {'—' if pairs['tied_support_share'] is None else format(pairs['tied_support_share'], '.1%')} "
            f"| {level['rules_1_to_1']['unordered']:,} "
            f"| {level['rules_1_to_1']['sequential']:,} |"
        )
    lines += [
        "",
        f"## Mức nghiêng về một hướng (hướng chiếm ≥ {DIRECTION_SHARE:.0%} số user của cặp)",
        "",
        "| min_support | Cách xác định thứ tự | Cặp xét | Tỉ lệ hướng chiếm ưu thế (trung vị) | Cặp có hướng ≥ ngưỡng | Kỳ vọng nếu thứ tự ngẫu nhiên |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for level in levels:
        for label, key in (("theo chuỗi (trùng timestamp theo số dòng)", "direction_all_orders"),
                           ("chặt (chỉ timestamp khác nhau)", "direction_strict_orders")):
            stats = level["pairs"][key]
            median = "—" if stats["share_median"] is None else f"{stats['share_median']:.2f}"
            lines.append(
                f"| {level['min_support']} | {label} | {stats['n_pairs']:,} | {median} "
                f"| {stats['pairs_share_ge_threshold']:,} | {stats['pairs_share_ge_threshold_expected_symmetric']:.1f} |"
            )
    if max_len >= 3:
        lines += [
            "",
            "## Tập 3 item phổ biến và thứ tự",
            "",
            "| min_support | Tập 3 item phổ biến | Có ít nhất một thứ tự phổ biến | Mẫu tuần tự độ dài 3 |",
            "|---:|---:|---:|---:|",
        ]
        for level in levels:
            triples = level["triples"]
            lines.append(
                f"| {level['min_support']} | {triples['unordered_frequent']:,} "
                f"| {triples['with_frequent_ordering']:,} | {triples['sequential_patterns']:,} |"
            )
    lines += [
        "",
        "## Kiểm tra nhất quán giữa PrefixSpan, FP-Growth và bộ đếm cặp độc lập",
        "",
        "| min_support | Số đếm item khớp | Mẫu độ dài 2 khớp bộ đếm cặp | Mẫu có tập item không phổ biến | Cặp có count(a→b) + count(b→a) ≠ số đếm {a, b} |",
        "|---:|:---:|:---:|---:|---:|",
    ]
    for level in levels:
        checks = level["checks"]
        lines.append(
            f"| {level['min_support']} | {'có' if checks['item_counts_match_fpgrowth'] else 'KHÔNG'} "
            f"| {'có' if checks['pair_patterns_match_pair_counter'] else 'KHÔNG'} "
            f"| {checks['sequential_itemsets_frequent_violations']} | {checks['pair_order_identity_violations']} |"
        )
    for level in levels[:1]:
        lines += [
            "",
            f"## min_support = {level['min_support']}: {len(level['top_sequential_pairs'])} mẫu a → b có số đếm lớn nhất",
            "",
            "Bảng chỉ lấy mức min_support thấp nhất; mẫu của các mức cao hơn là tập con của bảng đầy đủ trong JSON.",
            "",
            "| # | a | b | a → b | b → a | {a, b} | a → b (chặt) | b → a (chặt) |",
            "|---:|---|---|---:|---:|---:|---:|---:|",
        ]
        for rank, row in enumerate(level["top_sequential_pairs"], start=1):
            lines.append(
                f"| {rank} | [{row['first']}] {_short(row['first_title'], 55)} | [{row['second']}] {_short(row['second_title'], 55)} "
                f"| {row['count_forward']} | {row['count_backward']} | {row['count_unordered']} "
                f"| {row['count_forward_strict']} | {row['count_backward_strict']} |"
            )
    lines += [
        "",
        "## Hạn chế",
        "",
        "- Thứ tự phá hòa theo số dòng gốc của file raw, và file raw sắp theo ASIN: trong phần train, "
        "57.021 / 57.021 cặp liền kề trùng timestamp có ASIN tăng dần (kiểm tra riêng, không nằm trong pipeline). "
        "Hướng của một cặp tính theo cột \"theo chuỗi\" vì vậy có thể phản ánh quy tắc phá hòa; cột \"chặt\" loại ảnh hưởng này.",
        "- Ở min_support 0,0005, 5.282 cặp có hướng chiếm ≥ 80%; 1% user đóng góp nhiều nhất chiếm 75,4% số lượt (user, cặp) "
        "của các cặp này (độ dài train trung vị của nhóm đó là 41, của toàn tập là 4; kiểm tra riêng). "
        "Số đếm ≥ 12 user vì vậy không tương đương 12 user độc lập về hành vi.",
        "- Các số trên chỉ mô tả; nguyên nhân của việc nghiêng về một hướng (thứ tự mua, thời điểm item được review, "
        "user review nhiều item, quy tắc phá hòa) chưa được tách riêng.",
        "- Train Beauty không có item lặp trong cùng một user; với dữ liệu có lặp, mẫu a → a xuất hiện và đẳng thức "
        "count(a → b) + count(b → a) = số đếm {a, b} không còn đúng.",
        "- Thời gian chạy là một lần đo, hai thư viện (mlxtend, prefixspan) không cùng điều kiện nên không dùng để so sánh tốc độ.",
    ]
    return "\n".join(lines) + "\n"


def tie_statistics(sequences, timestamps):
    """Thống kê trùng timestamp trong chuỗi train: số dòng thuộc nhóm trùng (user, timestamp), số cặp liền kề cùng
    timestamp và cờ mọi timestamp là mốc ngày (chia hết cho 86.400 giây)."""
    rows = int(sum(len(s) for s in sequences))
    in_ties = 0
    adjacent_ties = 0
    whole_days = True
    for times in timestamps:
        whole_days = whole_days and bool(np.all(times % 86400 == 0))
        _, inverse, counts = np.unique(times, return_inverse=True, return_counts=True)
        in_ties += int(np.sum(counts[inverse] > 1))
        adjacent_ties += int(np.sum(np.diff(times) == 0))
    return {
        "rows": rows,
        "rows_in_tie_groups": in_ties,
        "adjacent_pairs": rows - len(sequences),
        "adjacent_ties": adjacent_ties,
        "timestamps_whole_days": whole_days,
    }


def prefixspan_main(argv=None):
    parser = argparse.ArgumentParser(description="PrefixSpan trên chuỗi train có thứ tự, so với FP-Growth cùng ngưỡng.")
    parser.add_argument("--interactions", type=Path, default=DEFAULT_INTERACTIONS)
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--items", type=Path, default=DEFAULT_ITEMS)
    parser.add_argument("--supports", type=float, nargs="+", default=[0.0005, 0.001, 0.002])
    parser.add_argument("--min-confidence", type=float, default=0.1)
    parser.add_argument("--max-len", type=int, default=3, help="độ dài tối đa của mẫu tuần tự và tập phổ biến")
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--output", type=Path, default=DEFAULT_PREFIXSPAN_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_PREFIXSPAN_MARKDOWN)
    args = parser.parse_args(argv)
    if args.max_len < 2:
        print("Error: --max-len phải >= 2 để so sánh mẫu tuần tự với cặp không thứ tự", file=sys.stderr)
        return 1
    try:
        n_items = int(pd.read_csv(args.interactions, usecols=["item_id"])["item_id"].max())
        _, sequences, timestamps = load_train_sequences(args.interactions, args.splits)
        matrix = build_transaction_matrix([np.unique(s) for s in sequences], n_items)
        metadata = load_item_metadata(args.items)
        levels = run_sequence_sweep(
            sequences, timestamps, matrix, args.supports, args.min_confidence, args.max_len, args.top, metadata
        )
    except (OSError, ValueError, KeyError, csv.Error, ImportError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    ties = tie_statistics(sequences, timestamps)
    payload = {"n_sequences": len(sequences), "n_items": n_items, "timestamp_ties": ties, "levels": levels}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.markdown.write_text(format_sequence_markdown(levels, len(sequences), ties), encoding="utf-8")
    for level in levels:
        print(json.dumps({k: v for k, v in level.items() if k != "top_sequential_pairs"}, ensure_ascii=False))
    return 0


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
    if argv and argv[0] == "prefixspan":
        return prefixspan_main(argv[1:])
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
