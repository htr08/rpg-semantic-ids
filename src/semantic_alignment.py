"""Đối chiếu luật kết hợp và phân cụm k-means với semantic ID (OPQ).

Phụ trách: C (RQ1, RQ2). Hai phép đo:

1. Luật ↔ token: với các cặp item (A, B) trong n luật mua kèm có lift cao nhất, đếm số vị trí token trùng nhau
   (hai item có cùng mã ở cùng vị trí j; mỗi vị trí có codebook riêng nên chỉ so cùng vị trí) rồi so với cặp ngẫu nhiên.
2. k-means ↔ token: ARI giữa nhãn k-means (PCA 50 chiều, k = 256) và token ở từng vị trí, nêu riêng token đầu.

Quy ước đầu vào: hàng j của ma trận semantic ID và của embedding ứng với item_id j + 1 (cùng quy ước với src/patterns.py).
Luật chỉ sinh từ phần train (src/patterns.py), không dùng validation/test.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score

from src.clustering import fit_pca, l2_normalize, load_embeddings, run_kmeans

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IDS = ROOT / "data" / "processed" / "semantic_ids.npy"
DEFAULT_EMBEDDINGS = ROOT / "data" / "processed" / "item_embeddings.npy"
DEFAULT_ID_MAPPING = ROOT / "data" / "processed" / "id_mapping.json"
DEFAULT_OUTPUT = ROOT / "reports" / "semantic_alignment.json"
DEFAULT_MARKDOWN = ROOT / "reports" / "semantic_alignment.md"
MAX_SHOWN_THRESHOLD = 6
FREQUENCY_BINS = 3


def load_semantic_ids(path, n_items=None, codebook_size=256):
    """Đọc ma trận semantic ID (n_items, số đoạn) và kiểm tra kiểu số nguyên, giá trị trong 0..codebook_size - 1."""
    ids = np.load(path)
    if ids.ndim != 2:
        raise ValueError(f"semantic ID phải là ma trận 2 chiều, nhận được {ids.ndim} chiều")
    if not np.issubdtype(ids.dtype, np.integer):
        raise ValueError(f"semantic ID phải là số nguyên, nhận được {ids.dtype}")
    if ids.size == 0:
        raise ValueError("semantic ID rỗng")
    if ids.min() < 0 or ids.max() >= codebook_size:
        raise ValueError(f"giá trị token phải nằm trong 0..{codebook_size - 1}, nhận được {ids.min()}..{ids.max()}")
    if n_items is not None and ids.shape[0] != n_items:
        raise ValueError(f"số hàng semantic ID {ids.shape[0]} khác số item {n_items}")
    return ids


def load_item_order(id_mapping_path, items_path):
    """Danh sách asin theo item_id 1..N từ id_mapping.json, sau khi kiểm items.jsonl có cùng thứ tự asin.

    File .npy của semantic ID và embedding không chứa asin nên không kiểm được thứ tự hàng của chúng từ nội dung;
    hàng j ↔ item_id j + 1 là quy ước (docs/data_contract.md), vì vậy báo cáo ghi checksum của các file đầu vào.
    items.jsonl (nguồn brand dùng để tách cặp) thì kiểm được: item_id, asin phải khớp từng dòng với id_mapping.
    """
    name = Path(items_path).name
    mapping = json.loads(Path(id_mapping_path).read_text(encoding="utf-8"))
    id2item = mapping.get("id2item") if isinstance(mapping, dict) else None
    if not isinstance(id2item, list) or not id2item or id2item[0] not in ("[PAD]", None):
        raise ValueError("id_mapping.json phải có id2item là danh sách với phần tử 0 là padding")
    asins = id2item[1:]
    if not all(isinstance(asin, str) and asin for asin in asins):
        raise ValueError("id2item chỉ được chứa asin dạng chuỗi không rỗng")
    if len(set(asins)) != len(asins):
        raise ValueError("id2item có asin trùng nhau")
    with open(items_path, encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if not all(isinstance(row, dict) and type(row.get("item_id")) is int and "asin" in row for row in rows):
        raise ValueError(f"{name}: mỗi dòng phải là đối tượng có item_id nguyên và asin")
    if [row["item_id"] for row in rows] != list(range(1, len(asins) + 1)):
        raise ValueError(f"{name}: item_id phải là 1..{len(asins)} theo thứ tự")
    for row in rows:
        if row["asin"] != asins[row["item_id"] - 1]:
            raise ValueError(f"{name}: asin của item_id {row['item_id']} khác id_mapping.json")
    return asins


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def shared_token_counts(ids, pairs):
    """Số vị trí token trùng nhau của từng cặp; pairs là mảng (n, 2) các item_id (đếm từ 1)."""
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    if pairs.size and (pairs.min() < 1 or pairs.max() > len(ids)):
        raise ValueError(f"item_id phải nằm trong 1..{len(ids)}")
    return (ids[pairs[:, 0] - 1] == ids[pairs[:, 1] - 1]).sum(axis=1)


def random_item_pairs(item_ids, n_pairs, seed=42):
    """n_pairs cặp (A, B) có thứ tự, A khác B, chọn đều trong item_ids; trả về mảng (n_pairs, 2)."""
    pool = np.unique(np.asarray(item_ids, dtype=np.int64))
    if len(pool) < 2:
        raise ValueError("cần ít nhất 2 item khác nhau để lấy cặp ngẫu nhiên")
    rng = np.random.default_rng(seed)
    first = rng.integers(0, len(pool), size=n_pairs)
    # Dịch một đoạn 1..len - 1 rồi lấy dư: B luôn khác A và đều trên các item còn lại.
    second = (first + 1 + rng.integers(0, len(pool) - 1, size=n_pairs)) % len(pool)
    return np.column_stack([pool[first], pool[second]])


def null_threshold(null_counts, alpha=0.05):
    """Số token chung nhỏ nhất t ≥ 1 mà tỉ lệ cặp ngẫu nhiên có ≥ t token chung không vượt alpha."""
    null_counts = np.asarray(null_counts)
    for t in range(1, int(null_counts.max()) + 2):
        if (null_counts >= t).mean() <= alpha:
            return t
    raise AssertionError("không đạt được trong vòng lặp vì tỉ lệ ≥ max + 1 luôn bằng 0")


def overlap_summary(counts, null_counts, threshold):
    """So phân bố số token chung của cặp luật với cặp ngẫu nhiên (mô tả, không kiểm định).

    Trả về trung bình, trung vị, tỉ lệ cặp không có token chung, tỉ lệ cặp có ≥ t token chung (t = 1..MAX_SHOWN_THRESHOLD
    và t = threshold) và tỉ số tỉ lệ luật / ngẫu nhiên tại threshold. Kiểm định nằm ở permutation_test vì các cặp luật
    dùng chung item nên không độc lập.
    """
    counts, null_counts = np.asarray(counts), np.asarray(null_counts)
    thresholds = sorted(set(range(1, MAX_SHOWN_THRESHOLD + 1)) | {threshold})
    rule_rate = float((counts >= threshold).mean())
    null_rate = float((null_counts >= threshold).mean())
    return {
        "n_pairs": int(len(counts)),
        "mean_shared": float(counts.mean()),
        "median_shared": float(np.median(counts)),
        "zero_shared_rate": float((counts == 0).mean()),
        "null_mean_shared": float(null_counts.mean()),
        "null_median_shared": float(np.median(null_counts)),
        "null_zero_shared_rate": float((null_counts == 0).mean()),
        "threshold": int(threshold),
        "rate_at_threshold": rule_rate,
        "null_rate_at_threshold": null_rate,
        "rate_ratio": rule_rate / null_rate if null_rate > 0 else None,
        "rate_by_threshold": {str(t): float((counts >= t).mean()) for t in thresholds},
        "null_rate_by_threshold": {str(t): float((null_counts >= t).mean()) for t in thresholds},
    }


def _permutation_summary(observed, means, rates, threshold):
    """Tóm tắt kiểm định hoán vị: giá trị quan sát, trung bình dưới hoán vị và p-value một phía (lớn hơn hoặc bằng)."""
    n_perm = len(means)
    observed_mean, observed_rate = float(observed.mean()), float((observed >= threshold).mean())
    return {
        "n_perm": int(n_perm),
        "threshold": int(threshold),
        "observed_mean_shared": observed_mean,
        "permuted_mean_shared": float(means.mean()),
        "mean_ratio": observed_mean / float(means.mean()) if means.mean() > 0 else None,
        "p_value_mean": float((1 + (means >= observed_mean - 1e-12).sum()) / (1 + n_perm)),
        "observed_rate_at_threshold": observed_rate,
        "permuted_rate_at_threshold": float(rates.mean()),
        "p_value_rate": float((1 + (rates >= observed_rate - 1e-12).sum()) / (1 + n_perm)),
    }


def permutation_test(ids, pairs, threshold, n_perm=5000, seed=42):
    """Kiểm định hoán vị giữ nguyên đồ thị cặp: hoán vị ngẫu nhiên hàng semantic ID giữa các item có mặt trong cặp.

    Mỗi item giữ nguyên số cặp chứa nó (bậc), nên p-value không bị thổi phồng bởi việc 200 cặp chỉ gồm vài trăm item.
    Thống kê: trung bình số token chung và tỉ lệ cặp có ≥ threshold token chung. p = (1 + số lần hoán vị đạt hoặc vượt
    giá trị quan sát) / (1 + n_perm), nên nhỏ nhất là 1 / (1 + n_perm).
    """
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    if len(pairs) == 0 or n_perm < 1:
        raise ValueError("cần ít nhất một cặp và n_perm >= 1")
    items, inverse = np.unique(pairs, return_inverse=True)
    inverse = inverse.reshape(-1, 2)
    observed = shared_token_counts(ids, pairs)
    rows = ids[items - 1]
    rng = np.random.default_rng(seed)
    means, rates = np.empty(n_perm), np.empty(n_perm)
    for i in range(n_perm):
        shuffled = rows[rng.permutation(len(items))]
        counts = (shuffled[inverse[:, 0]] == shuffled[inverse[:, 1]]).sum(axis=1)
        means[i], rates[i] = counts.mean(), (counts >= threshold).mean()
    result = _permutation_summary(observed, means, rates, threshold)
    result["n_items"] = int(len(items))
    return result


def _brand_key(metadata, item_id):
    brand = metadata[item_id]["brand"]
    return None if brand is None else brand.casefold()


def frequency_bin_edges(item_counts, min_count, n_bins=FREQUENCY_BINS):
    """Biên các nhóm tần suất: phân vị đều của số đếm train của các item đạt min_count (tính trên toàn catalog, không theo brand)."""
    eligible = np.asarray(item_counts)[np.asarray(item_counts) >= min_count]
    if n_bins < 2 or eligible.size == 0:
        return np.empty(0)
    return np.unique(np.quantile(eligible, np.arange(1, n_bins) / n_bins))


def frequency_bins(item_counts, min_count, n_bins=FREQUENCY_BINS):
    """Nhóm tần suất (0..) của từng item; số đếm đúng bằng biên thuộc nhóm trên. Item dưới min_count rơi vào nhóm 0."""
    edges = frequency_bin_edges(item_counts, min_count, n_bins)
    return np.searchsorted(edges, np.asarray(item_counts), side="right"), len(edges) + 1


def within_brand_permutation_test(ids, pairs, metadata, threshold, item_counts, min_count, n_perm=5000, seed=42,
                                  n_bins=FREQUENCY_BINS):
    """Kiểm định hoán vị trong từng brand, ghép theo tần suất train, cho các cặp cùng thương hiệu.

    Chỉ item có số đếm train ≥ min_count (item có thể xuất hiện trong luật ở mức support này) mới vào đối chứng:
    item hiếm bị loại khỏi mọi luật nên không được dùng làm mẫu ngẫu nhiên. Các item đủ điều kiện được chia theo
    số đếm train thành n_bins nhóm (biên là phân vị số đếm của các item đủ điều kiện trong toàn catalog); hoán vị hàng ID chỉ diễn ra trong từng
    (brand, nhóm tần suất). Mỗi item trong cặp nhận ID của một item cùng brand và cùng nhóm tần suất (có thể chính
    nó khi nhóm chỉ có một item, khi đó đối chứng không đổi). Nhờ vậy kiểm định không nhầm khác biệt về tần suất
    hay về khả năng đủ điều kiện với tín hiệu mua kèm. Mọi cặp phải gồm hai item cùng brand và đủ min_count.
    """
    ids, item_counts = np.asarray(ids), np.asarray(item_counts)
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    if len(pairs) == 0 or n_perm < 1:
        raise ValueError("cần ít nhất một cặp và n_perm >= 1")
    if item_counts.shape != (len(ids),):
        raise ValueError(f"item_counts phải có {len(ids)} phần tử, nhận {item_counts.shape}")
    pair_items = np.unique(pairs)
    if pair_items.min() < 1 or pair_items.max() > len(ids) or (item_counts[pair_items - 1] < min_count).any():
        raise ValueError(f"mọi item trong cặp phải nằm trong 1..{len(ids)} và có số đếm train ≥ {min_count}")
    pair_brands = [_brand_key(metadata, int(a)) for a, _ in pairs]
    if any(b is None or b != _brand_key(metadata, int(c)) for b, (_, c) in zip(pair_brands, pairs)):
        raise ValueError("mọi cặp phải gồm hai item cùng brand")
    bins, _ = frequency_bins(item_counts, min_count, n_bins)

    def stratum(item_id):
        return _brand_key(metadata, int(item_id)), int(bins[int(item_id) - 1])

    needed = {stratum(item) for item in pair_items}
    members = {}
    for item_id in metadata:
        if 1 <= item_id <= len(ids) and item_counts[item_id - 1] >= min_count and stratum(item_id) in needed:
            members.setdefault(stratum(item_id), []).append(int(item_id))
    universe = np.array(sorted(item for group in members.values() for item in group), dtype=np.int64)
    position = {int(item): i for i, item in enumerate(universe)}
    groups = [np.array([position[item] for item in group]) for group in members.values() if len(group) > 1]
    first = np.array([position[int(a)] for a in pairs[:, 0]])
    second = np.array([position[int(b)] for b in pairs[:, 1]])
    rows = ids[universe - 1]
    observed = shared_token_counts(ids, pairs)
    rng = np.random.default_rng(seed)
    means, rates = np.empty(n_perm), np.empty(n_perm)
    for i in range(n_perm):
        shuffled = rows.copy()
        for group in groups:
            shuffled[group] = rows[group[rng.permutation(len(group))]]
        counts = (shuffled[first] == shuffled[second]).sum(axis=1)
        means[i], rates[i] = counts.mean(), (counts >= threshold).mean()
    result = _permutation_summary(observed, means, rates, threshold)
    size = {key: len(group) for key, group in members.items()}
    fixed = sum(
        (size[stratum(x)] == 1 and size[stratum(y)] == 1) or (stratum(x) == stratum(y) and size[stratum(x)] == 2)
        for x, y in pairs.tolist()
    )
    result.update({
        "n_brands": len({brand for brand, _ in needed}),
        "n_strata": len(needed),
        "n_freq_bins": int(len(np.unique(bins[item_counts >= min_count]))),
        "n_pair_items": int(len(pair_items)),
        # Cặp có đối chứng bất biến: cả hai item không hoán vị được, hoặc hai item là toàn bộ một nhóm (hoán vị chỉ
        # đổi chỗ, số token chung đối xứng nên không đổi). Các cặp này luôn bảo thủ (null = quan sát).
        "n_pairs_null_fixed": int(fixed),
    })
    return result


def select_top_pairs(matrix, min_support, min_confidence, top_n):
    """Các cặp item của top_n luật 1 item ⇒ 1 item có lift cao nhất ở mức min_support.

    Mỗi cặp chỉ xuất hiện một lần (A ⇒ B và B ⇒ A có cùng lift). Một cặp được giữ nếu ít nhất một chiều
    có confidence ≥ min_confidence. Trả về (DataFrame luật đã chọn, số luật pair thỏa ngưỡng trước khi lấy top).
    """
    from src.patterns import generate_rules, pair_rules_only, run_fpgrowth, top_rules_by_lift

    itemsets, _ = run_fpgrowth(matrix, min_support, max_len=2)
    rules = pair_rules_only(generate_rules(itemsets, matrix.shape[0], min_confidence, max_rule_len=2))
    unique_pairs = top_rules_by_lift(rules, n=len(rules))
    return unique_pairs.head(top_n), int(len(unique_pairs))


def rule_pairs(rules):
    """Mảng (n, 2) item_id của các luật pair, theo thứ tự (tiền đề, hậu quả)."""
    return np.array([[next(iter(a)), next(iter(c))] for a, c in zip(rules["antecedents"], rules["consequents"])],
                    dtype=np.int64).reshape(-1, 2)


def _brand_group(pair, metadata):
    """Nhóm thương hiệu của cặp item.

    "same_brand": mọi item đều có brand và brand giống nhau (không phân biệt hoa thường, cùng quy ước
    patterns.describe_rule); "missing_brand": có ít nhất một item thiếu brand; "different_brand": đủ brand nhưng khác nhau.
    """
    brands = {None if metadata[i]["brand"] is None else metadata[i]["brand"].casefold() for i in pair}
    if None in brands:
        return "missing_brand"
    return "same_brand" if len(brands) == 1 else "different_brand"


BRAND_GROUPS = ("same_brand", "different_brand", "missing_brand")


def rule_alignment(ids, matrix, metadata, min_support, min_confidence=0.1, top_n=200, n_null=100_000, n_perm=5000,
                   seed=42):
    """Đối chiếu top_n luật lift cao nhất ở một mức min_support với semantic ID.

    Đối chứng: (a) n_null cặp ngẫu nhiên chọn đều trong toàn bộ catalog, dùng để cố định ngưỡng "nhiều token chung"
    (t nhỏ nhất để không quá 5% cặp ngẫu nhiên có ≥ t token chung; không phụ thuộc kết quả của luật);
    (b) kiểm định hoán vị hàng ID giữa các item có mặt trong cặp, giữ nguyên đồ thị cặp và bậc từng item, nên p-value
    không giả định các cặp độc lập (permutation_test). metadata (từ patterns.load_item_metadata) dùng để tách
    cặp theo thương hiệu; None thì bỏ qua phần tách này.
    """
    if top_n < 1 or n_null < 1 or n_perm < 1:
        raise ValueError("top_n, n_null và n_perm phải >= 1")
    if ids.shape[0] != matrix.shape[1]:
        raise ValueError(f"số hàng semantic ID {ids.shape[0]} khác số item của ma trận giao dịch {matrix.shape[1]}")
    rules, n_pair_rules = select_top_pairs(matrix, min_support, min_confidence, top_n)
    result = {
        "min_support": min_support,
        "min_count": int(np.ceil(min_support * matrix.shape[0])),
        "min_confidence": min_confidence,
        "top_n": top_n,
        "n_unique_pairs_available": n_pair_rules,
        "n_pairs_used": int(len(rules)),
    }
    if rules.empty:
        return result
    pairs = rule_pairs(rules)
    counts = shared_token_counts(ids, pairs)
    null_all = shared_token_counts(ids, random_item_pairs(np.arange(1, len(ids) + 1), n_null, seed))
    threshold = null_threshold(null_all)
    result.update({
        "count_min": int(rules["count"].min()),
        "count_median": float(rules["count"].median()),
        "lift_min": float(rules["lift"].min()),
        "lift_max": float(rules["lift"].max()),
        "n_items_in_pairs": int(len(np.unique(pairs))),
        "n_identical_id_pairs": int((counts == ids.shape[1]).sum()),
        "vs_random_catalog": overlap_summary(counts, null_all, threshold),
        "permutation": permutation_test(ids, pairs, threshold, n_perm, seed),
    })
    if metadata is not None:
        groups = np.array([_brand_group(pair, metadata) for pair in pairs.tolist()])
        result["by_brand"] = {}
        for name in BRAND_GROUPS:
            mask = groups == name
            result["by_brand"][name] = {
                "n_pairs": int(mask.sum()),
                "mean_shared": float(counts[mask].mean()) if mask.any() else None,
                "rate_at_threshold": float((counts[mask] >= threshold).mean()) if mask.any() else None,
            }
        same = groups == "same_brand"
        if same.any():
            item_counts = np.asarray(matrix.sum(axis=0)).ravel()
            result["by_brand"]["same_brand"]["permutation_within_brand"] = within_brand_permutation_test(
                ids, pairs[same], metadata, threshold, item_counts, result["min_count"], n_perm, seed
            )
    return result


def kmeans_token_alignment(embeddings, ids, k=256, n_components=50, n_init=5, seeds=(42, 43, 44), pca_seed=42,
                           n_shuffle=20):
    """ARI giữa nhãn k-means trên PCA và token ở từng vị trí; mỗi seed là một lần k-means với cùng PCA.

    Trả về ARI với token đầu (vị trí 0) theo seed, ARI theo vị trí (trung bình qua seed), thống kê các vị trí còn lại,
    ARI giữa token đầu và các token khác, mốc xáo trộn (ARI của token đầu đã xáo trộn với nhãn k-means của seed đầu)
    và ARI giữa hai lần k-means khác seed làm mốc ổn định của chính k-means.
    """
    if embeddings.shape[0] != ids.shape[0]:
        raise ValueError(f"số item của embedding {embeddings.shape[0]} khác semantic ID {ids.shape[0]}")
    if not len(seeds):
        raise ValueError("cần ít nhất một seed k-means")
    reduced, variance_kept = fit_pca(l2_normalize(embeddings), n_components, pca_seed)
    all_labels = [run_kmeans(reduced, k, n_init, seed)[0] for seed in seeds]
    by_seed = np.array([[adjusted_rand_score(ids[:, j], labels) for j in range(ids.shape[1])] for labels in all_labels])
    mean_by_position = by_seed.mean(axis=0)
    others = mean_by_position[1:]
    rng = np.random.default_rng(pca_seed)
    shuffled = np.array([adjusted_rand_score(rng.permutation(ids[:, 0]), all_labels[0]) for _ in range(n_shuffle)])
    token_vs_token = np.array([adjusted_rand_score(ids[:, 0], ids[:, j]) for j in range(1, ids.shape[1])])
    result = {
        "k": int(k),
        "pca_dim": int(n_components),
        "pca_explained_variance_ratio": round(float(variance_kept), 6),
        "n_init": int(n_init),
        "seeds": [int(s) for s in seeds],
        "ari_first_token_by_seed": [float(v) for v in by_seed[:, 0]],
        "ari_first_token_mean": float(by_seed[:, 0].mean()),
        "ari_by_position_mean": [float(v) for v in mean_by_position],
        "ari_other_positions": {
            "min": float(others.min()), "median": float(np.median(others)), "max": float(others.max()),
            "argmax_position": int(1 + others.argmax()),
        } if len(others) else None,
        "ari_first_token_shuffled": {
            "n_shuffle": int(n_shuffle), "mean": float(shuffled.mean()), "std": float(shuffled.std()),
        },
        "ari_first_token_vs_other_tokens": {
            "mean": float(token_vs_token.mean()), "max": float(token_vs_token.max()),
        } if len(token_vs_token) else None,
    }
    if len(all_labels) >= 2:
        result["ari_between_kmeans_seeds"] = float(adjusted_rand_score(all_labels[0], all_labels[1]))
    return result


def _fmt(value, digits=3):
    return "—" if value is None else f"{value:.{digits}f}"


def _p(value):
    return f"{value:.2g}"


def format_markdown(result):
    """Báo cáo markdown của hai phép đo."""
    lines = [
        "# Đối chiếu luật kết hợp và k-means với semantic ID (OPQ)",
        "",
        f"{result['n_items']} item, {result['n_positions']} vị trí token, codebook {result['codebook_size']}. "
        "Luật sinh từ phần train. Số token chung của cặp (A, B) là số vị trí có cùng mã ở cùng vị trí. "
        "Sinh bởi `python -m src.semantic_alignment`.",
        "",
        "## Cặp item trong luật lift cao nhất so với cặp ngẫu nhiên",
        "",
        "Cặp là luật một item ⇒ một item, mỗi cặp chỉ tính một lần, xếp theo lift giảm dần. "
        "Cặp ngẫu nhiên: hai item khác nhau chọn đều trong catalog. "
        "Ngưỡng t (\"nhiều token chung\") là số nhỏ nhất để không quá 5% cặp ngẫu nhiên có ≥ t token chung. "
        "Các cặp dùng chung item (cột \"Item\") nên không độc lập; p-value lấy từ kiểm định hoán vị hàng ID giữa các item "
        "có mặt trong cặp, giữ nguyên bậc từng item, và giá trị nhỏ nhất đo được là 1/(1 + số lần hoán vị).",
        "",
        "| min_support | Cặp dùng / có sẵn | Item | Count nhỏ nhất | Trung bình token chung: luật / ngẫu nhiên / hoán vị | "
        "Cặp không trùng token: luật / ngẫu nhiên | Tỉ lệ ≥ 1: luật / ngẫu nhiên | t | "
        "Tỉ lệ ≥ t: luật / ngẫu nhiên / hoán vị | p trung bình | p tỉ lệ ≥ t | Cặp trùng hoàn toàn ID |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for level in result["rules"]:
        if "vs_random_catalog" not in level:
            lines.append(f"| {level['min_support']} | 0 / 0 |" + " — |" * 10)
            continue
        s, perm = level["vs_random_catalog"], level["permutation"]
        lines.append(
            f"| {level['min_support']} | {level['n_pairs_used']} / {level['n_unique_pairs_available']} | "
            f"{level['n_items_in_pairs']} | {level['count_min']} | "
            f"{_fmt(s['mean_shared'])} / {_fmt(s['null_mean_shared'])} / {_fmt(perm['permuted_mean_shared'])} | "
            f"{_fmt(s['zero_shared_rate'])} / {_fmt(s['null_zero_shared_rate'])} | "
            f"{_fmt(s['rate_by_threshold']['1'])} / {_fmt(s['null_rate_by_threshold']['1'])} | {s['threshold']} | "
            f"{_fmt(s['rate_at_threshold'])} / {_fmt(s['null_rate_at_threshold'])} / "
            f"{_fmt(perm['permuted_rate_at_threshold'])} | "
            f"{_p(perm['p_value_mean'])} | {_p(perm['p_value_rate'])} | {level['n_identical_id_pairs']} |"
        )
    if any("by_brand" in level for level in result["rules"]):
        lines += [
            "",
            "Tách theo thương hiệu. Cùng thương hiệu: cả hai item có brand và brand giống nhau (không phân biệt hoa thường); "
            "thiếu thương hiệu: ít nhất một item không có brand; khác thương hiệu: đủ brand nhưng khác nhau.",
            "",
            "Với nhóm cùng thương hiệu, đối chứng là hoán vị hàng ID trong từng (brand, nhóm tần suất train): chỉ các item "
            "có số đếm train ≥ min_count của brand đó (item hiếm không thể vào luật nên không dùng làm mẫu), chia thành "
            f"{FREQUENCY_BINS} nhóm theo phân vị số đếm train của các item đủ điều kiện trong toàn catalog. "
            "Cặp ngẫu nhiên vẫn cùng brand và mỗi item giữ nhóm tần suất của nó, nên khác biệt về tần suất không bị tính là "
            "tín hiệu mua kèm, nhưng 3 nhóm là ghép thô nên có thể còn dư tác động của tần suất; p-value mang tính mô tả. "
            "Cặp có đối chứng bất biến (cột \"Cặp đối chứng bất biến\") là cặp mà cả hai item là phần tử "
            "duy nhất của nhóm, hoặc hai item là toàn bộ nhóm: hoán vị không đổi số token chung nên kiểm định bảo thủ hơn.",
            "",
            "| min_support | Nhóm | Số cặp | Trung bình token chung | Tỉ lệ ≥ t | Trung bình khi hoán vị | "
            "Tỉ lệ ≥ t khi hoán vị | p trung bình | p tỉ lệ ≥ t | Cặp đối chứng bất biến |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for level in result["rules"]:
            for name, group in level.get("by_brand", {}).items():
                within = group.get("permutation_within_brand")
                extra = (
                    f"{_fmt(within['permuted_mean_shared'])} | {_fmt(within['permuted_rate_at_threshold'])} | "
                    f"{_p(within['p_value_mean'])} | {_p(within['p_value_rate'])} | "
                    f"{within['n_pairs_null_fixed']} / {group['n_pairs']}"
                    if within else "— | — | — | — | —"
                )
                lines.append(
                    f"| {level['min_support']} | {name} | {group['n_pairs']} | {_fmt(group['mean_shared'])} | "
                    f"{_fmt(group['rate_at_threshold'])} | {extra} |"
                )
    km = result["kmeans"]
    others = km["ari_other_positions"]
    lines += [
        "",
        "## ARI giữa k-means và token OPQ",
        "",
        f"k-means k = {km['k']} trên PCA {km['pca_dim']} chiều (giữ {km['pca_explained_variance_ratio']} phương sai; "
        f"PCA fit một lần), n_init = {km['n_init']}, seed {km['seeds']}. Mỗi token OPQ mã hóa một không gian con riêng "
        "của vector, nên ARI giữa các token khác nhau cũng được nêu làm mốc.",
        "",
        "| Chỉ số | Giá trị |",
        "|---|---:|",
        f"| ARI với token đầu, từng seed | {', '.join(_fmt(v, 4) for v in km['ari_first_token_by_seed'])} |",
        f"| ARI với token đầu, trung bình | {_fmt(km['ari_first_token_mean'], 4)} |",
    ]
    if others:
        lines.append(
            f"| ARI với các vị trí 1..{result['n_positions'] - 1}, trung bình qua seed "
            f"(nhỏ nhất / trung vị / lớn nhất theo vị trí) | {_fmt(others['min'], 4)} / {_fmt(others['median'], 4)} / "
            f"{_fmt(others['max'], 4)} (lớn nhất tại vị trí {others['argmax_position']}) |"
        )
    shuffled = km["ari_first_token_shuffled"]
    lines.append(
        f"| Mốc xáo trộn: ARI của token đầu đã xáo trộn với k-means seed đầu ({shuffled['n_shuffle']} lần, "
        f"trung bình ± độ lệch chuẩn) | {shuffled['mean']:.1e} ± {shuffled['std']:.1e} |"
    )
    if km.get("ari_first_token_vs_other_tokens"):
        t = km["ari_first_token_vs_other_tokens"]
        lines.append(
            f"| ARI giữa token đầu và token ở vị trí khác (trung bình / lớn nhất) | "
            f"{_fmt(t['mean'], 4)} / {_fmt(t['max'], 4)} |"
        )
    if "ari_between_kmeans_seeds" in km:
        lines.append(
            f"| ARI giữa hai lần k-means khác seed (seed đầu và seed thứ hai) | {_fmt(km['ari_between_kmeans_seeds'], 4)} |"
        )
    if result.get("inputs"):
        lines += [
            "",
            "## Đầu vào",
            "",
            "Thứ tự hàng của semantic ID và embedding (hàng j ↔ `item_id` j + 1) không kiểm được từ nội dung file vì `.npy` "
            "không chứa asin; chỉ kiểm được số hàng bằng số item của `id_mapping.json`, và `items.jsonl` khớp mapping từng dòng. "
            "SHA-256 dưới đây để đối chiếu đúng phiên bản file đầu vào khi tái lập.",
            "",
            "| Đầu vào | File | SHA-256 |",
            "|---|---|---|",
        ]
        lines += [f"| {name} | `{info['file']}` | `{info['sha256']}` |" for name, info in result["inputs"].items()]
    return "\n".join(lines) + "\n"


def run_alignment(ids, embeddings, matrix, metadata, supports, min_confidence, top_n, n_null, n_perm, k, n_components,
                  n_init, seeds, seed):
    """Chạy cả hai phép đo và gom kết quả."""
    rules = [rule_alignment(ids, matrix, metadata, s, min_confidence, top_n, n_null, n_perm, seed) for s in supports]
    return {
        "n_items": int(ids.shape[0]),
        "n_positions": int(ids.shape[1]),
        "codebook_size": 256,
        "n_transactions": int(matrix.shape[0]),
        "n_null_pairs": int(n_null),
        "seed": int(seed),
        "rules": rules,
        "kmeans": kmeans_token_alignment(embeddings, ids, k, n_components, n_init, seeds),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Đối chiếu luật kết hợp và k-means với semantic ID (T4 tuần 2).")
    parser.add_argument("--semantic-ids", type=Path, default=DEFAULT_IDS)
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--supports", type=float, nargs="+", default=[0.0005, 0.001, 0.002])
    parser.add_argument("--min-confidence", type=float, default=0.1)
    parser.add_argument("--top", type=int, default=200)
    parser.add_argument("--null-pairs", type=int, default=100_000)
    parser.add_argument("--permutations", type=int, default=5000)
    parser.add_argument("--k", type=int, default=256)
    parser.add_argument("--components", type=int, default=50)
    parser.add_argument("--n-init", type=int, default=5)
    parser.add_argument("--kmeans-seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args(argv)
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # thông báo lỗi tiếng Việt trên console cp1252
    except (AttributeError, ValueError):
        pass
    try:
        from src.patterns import (DEFAULT_INTERACTIONS, DEFAULT_ITEMS, DEFAULT_SPLITS, build_transaction_matrix,
                                  load_item_metadata, load_train_transactions)

        n_items = len(load_item_order(DEFAULT_ID_MAPPING, DEFAULT_ITEMS))
        ids = load_semantic_ids(args.semantic_ids, n_items)
        embeddings = load_embeddings(args.embeddings)
        if embeddings.shape[0] != n_items:
            raise ValueError(f"số hàng embedding {embeddings.shape[0]} khác số item {n_items}")
        _, transactions = load_train_transactions(DEFAULT_INTERACTIONS, DEFAULT_SPLITS)
        matrix = build_transaction_matrix(transactions, n_items)
        metadata = load_item_metadata(DEFAULT_ITEMS)
        result = run_alignment(ids, embeddings, matrix, metadata, args.supports, args.min_confidence, args.top,
                               args.null_pairs, args.permutations, args.k, args.components, args.n_init,
                               args.kmeans_seeds, args.seed)
        result["inputs"] = {
            name: {"file": path.name, "sha256": file_sha256(path)}
            for name, path in (("semantic_ids", args.semantic_ids), ("embeddings", args.embeddings),
                               ("id_mapping", DEFAULT_ID_MAPPING), ("items", DEFAULT_ITEMS),
                               ("interactions", DEFAULT_INTERACTIONS), ("splits", DEFAULT_SPLITS))
        }
        for path in (args.output, args.markdown):
            path.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        args.markdown.write_text(format_markdown(result), encoding="utf-8")
    except (OSError, ValueError, KeyError) as error:
        print(f"Error: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    # Dòng này chỉ gồm đường dẫn, không in nội dung báo cáo.
    print(f"Wrote {args.output} and {args.markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
