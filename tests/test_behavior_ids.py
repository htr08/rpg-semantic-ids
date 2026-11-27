"""Unit test cho src/behavior_ids.py. CI không có faiss: phần OPQ thật tự bỏ qua; các test khác dùng quantizer giả."""
import json
import math

import numpy as np
import pytest

from adapters import load_ids as A
from src import behavior_ids as B
from src import tokenizer as T


def make_cache(tmp_path, seqs, n_items):
    proc = A.processed_dir(tmp_path / "cache")
    proc.mkdir(parents=True)
    asins = [f"A{i}" for i in range(1, n_items + 1)]
    (proc / "id_mapping.json").write_text(json.dumps(
        {"id2item": ["[PAD]"] + asins, "item2id": {a: i + 1 for i, a in enumerate(asins)}}))
    (proc / "all_item_seqs.json").write_text(json.dumps({u: [f"A{i}" for i in s] for u, s in seqs.items()}))
    return proc


# --- ghép ---

def test_combine_inner_product_is_weighted_cosine():
    rng = np.random.default_rng(0)
    text, cooc = rng.normal(size=(5, 6)), rng.normal(size=(5, 3))
    z = B.combine(text, cooc, 0.25)
    assert z.shape == (5, 9) and z.dtype == np.float32
    cos = lambda m: (m / np.linalg.norm(m, axis=1, keepdims=True)) @ (m / np.linalg.norm(m, axis=1, keepdims=True)).T
    np.testing.assert_allclose(z @ z.T, 0.75 * cos(text) + 0.25 * cos(cooc), atol=1e-5)


def test_combine_alpha_zero_and_zero_rows():
    text = np.array([[3.0, 4.0], [1.0, 0.0]])
    np.testing.assert_allclose(B.combine(text, np.ones((2, 2)), 0.0), [[0.6, 0.8], [1.0, 0.0]], atol=1e-6)
    z = B.combine(text, np.array([[0.0, 0.0], [2.0, 0.0]]), 0.5)
    np.testing.assert_allclose(z[0], [0.6 * math.sqrt(0.5), 0.8 * math.sqrt(0.5), 0, 0], atol=1e-6)  # item ngoài train
    with pytest.raises(ValueError):
        B.combine(text, np.ones((2, 2)), 1.5)
    with pytest.raises(ValueError):
        B.combine(text, np.ones((3, 2)), 0.5)


def test_load_cooc_checks_shape_and_finite(tmp_path):
    np.save(tmp_path / "ok.npy", np.ones((4, 3), dtype=np.float32))
    assert B.load_cooc(tmp_path / "ok.npy", 4).shape == (4, 3)
    with pytest.raises(ValueError):
        B.load_cooc(tmp_path / "ok.npy", 5)
    np.save(tmp_path / "bad.npy", np.array([[np.nan, 1.0]]))
    with pytest.raises(ValueError):
        B.load_cooc(tmp_path / "bad.npy", 1)


# --- chuỗi, cặp, độ phổ biến ---

def test_train_pairs_and_popularity_use_only_train_part(tmp_path):
    proc = make_cache(tmp_path, {"u1": [1, 2, 3, 4, 5], "u2": [2, 2, 6, 1], "u3": [3, 4]}, n_items=6)
    seqs, n = B.train_sequences(proc)
    assert n == 6 and seqs[0] == [0, 1, 2, 3, 4]
    pairs = B.consecutive_train_pairs(seqs)
    # u1 train = [1,2,3] -> (1,2),(2,3); u2 train = [2,2] -> cặp trùng item bị bỏ; u3 không đủ dài
    np.testing.assert_array_equal(pairs, [[0, 1], [1, 2]])
    np.testing.assert_array_equal(B.train_popularity(seqs, n), [1, 3, 1, 0, 0, 0])


# --- kiểm tra thứ tự ---

def test_order_check_passes_aligned_and_fails_shuffled():
    rng = np.random.default_rng(0)
    n = 200
    groups = np.arange(n) % 10
    cooc = rng.normal(size=(10, 8))[groups] + 0.1 * rng.normal(size=(n, 8))
    mask = np.ones(n, bool)
    mask[[5, 50]] = False
    cooc[~mask] = 0
    pairs = np.array([(i, j) for i in range(n) for j in range(i + 1, n) if groups[i] == groups[j]][:300])
    ok = B.order_check(cooc, mask, pairs)
    assert ok["passed"] and ok["zero_rows_match_outside_train"] and ok["cos_consecutive_pairs"] > 0.9
    shuffled = cooc[rng.permutation(n)]
    assert not B.order_check(shuffled, mask, pairs)["passed"]


# --- độ đo hành vi ---

def test_pair_shared_codes_against_random():
    ids = np.zeros((100, 4), dtype=int)
    ids[:, 1:] = np.arange(100)[:, None] * np.array([1, 7, 13])         # 3 vị trí riêng cho từng item
    ids[:, 0] = np.arange(100) // 2                                     # cặp (2k, 2k+1) chung vị trí 0
    pairs = np.array([[2 * k, 2 * k + 1] for k in range(50)])
    out = B.pair_shared_codes(ids, pairs, n_random=5000)
    assert out["pair_shared_mean"] == 1.0 and out["pair_any_shared"] == 1.0
    assert out["random_shared_mean"] < 0.05 and out["pair_lift"] > 20


def test_proxy_finds_target_sharing_codes_with_last_item():
    # 200 item; item 2k và 2k+1 có ID giống hệt nhau trừ vị trí cuối; user mua 2k rồi 2k+1 (đích validation)
    n = 200
    ids = np.stack([np.arange(n) // 2, np.arange(n) // 2, np.arange(n)], axis=1)
    seqs = [[(2 * k + 10) % n, 2 * k, 2 * k + 1, (2 * k + 20) % n] for k in range(n // 2)]
    pop = np.ones(n)
    perfect = B.last_item_overlap_proxy(ids, seqs, pop, n_users=100, k=10)
    assert perfect["proxy_ndcg@10"] == pytest.approx(1.0) and perfect["proxy_users"] == 100
    shuffled = B.last_item_overlap_proxy(ids[np.random.default_rng(0).permutation(n)], seqs, pop, n_users=100)
    assert shuffled["proxy_ndcg@10"] < 0.2


def test_proxy_breaks_ties_by_popularity():
    ids = np.zeros((5, 2), dtype=int)                                   # mọi item cùng ID: chỉ còn độ phổ biến
    seqs = [[0, 0, 3, 4]]
    assert B.last_item_overlap_proxy(ids, seqs, np.array([0, 0, 0, 9, 0]), k=1)["proxy_recall@1"] == 1.0
    assert B.last_item_overlap_proxy(ids, seqs, np.array([0, 9, 0, 1, 0]), k=1)["proxy_recall@1"] == 0.0


# --- thử nhanh đầu cuối với quantizer giả ---

def fake_quantize(z, mask):
    return T.rq_ids(z, n_codebook=4, codebook_size=4, train_mask=mask, backend="sklearn")


def items_jsonl(tmp_path, n):
    path = tmp_path / "items.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for i in range(n):
            meta = {"categories": [["Beauty", "X", f"L{i % 4}"]], "brand": f"B{i % 3}"}
            f.write(json.dumps({"item_id": i + 1, "asin": f"A{i + 1}", "metadata_missing": False, "metadata": meta}) + "\n")
    return path


def trial_setup(tmp_path, n=80):
    rng = np.random.default_rng(0)
    seqs = {f"u{u}": [int(x) for x in rng.choice(np.arange(1, n + 1), size=6, replace=False)] for u in range(60)}
    proc = make_cache(tmp_path, seqs, n)
    text = (rng.normal(size=(4, 32))[np.arange(n) % 4] + rng.normal(size=(n, 32))).astype(np.float32)
    cooc = rng.normal(size=(n, 8)).astype(np.float32)
    cooc[~T.train_item_mask(proc)] = 0
    from src import cluster_metrics as M
    labels, _ = M.load_item_labels(items_jsonl(tmp_path, n))
    return proc, text, cooc, labels


def test_run_trial_rows_files_and_recommendation(tmp_path):
    proc, text, cooc, labels = trial_setup(tmp_path)
    out = tmp_path / "out"
    rows = B.run_trial(text, {"svd": cooc}, [0, 0.5], proc, labels, out, quantize=fake_quantize,
                       extra_ids={"ref": np.zeros((80, 4), dtype=int)}, n_proxy_users=50, pca_dim=8)
    assert [r["config"] for r in rows] == ["ref", "text", "svd_a50"]
    assert (out / "trial_ids" / "text.npy").exists() and (out / "trial_ids" / "svd_a50.npy").exists()
    assert not (out / "trial_ids" / "ref.npy").exists()
    assert rows[0]["collision_rate"] == 1.0 and rows[0]["cooc"] == "có sẵn"
    for key in ("proxy_ndcg@10", "pair_shared_mean", "leaf_neighbor_purity@10", "entropy_min"):
        assert key in rows[1]
    rec = B.recommend(rows, max_collision=1.0, min_entropy=0.0, min_purity_ratio=0.0)
    assert rec["config"] in (None, "svd_a50")


def test_recommend_rules():
    base = {"config": "text", "cooc": "", "alpha": 0.0, "collision_rate": 0.0, "entropy_min": 0.98,
            "leaf_neighbor_purity@10": 0.20, "proxy_ndcg@10": 0.010}
    rows = [base,
            dict(base, config="svd_a25", cooc="svd", alpha=0.25, **{"proxy_ndcg@10": 0.013}),
            dict(base, config="svd_a50", cooc="svd", alpha=0.5, **{"proxy_ndcg@10": 0.020, "leaf_neighbor_purity@10": 0.10}),
            dict(base, config="opq_goc", cooc="có sẵn", alpha="", **{"proxy_ndcg@10": 0.05})]
    rec = B.recommend(rows)
    assert rec["config"] == "svd_a25" and rec["proxy_gain"] == pytest.approx(0.3)   # svd_a50 mất ngữ nghĩa, bị loại
    worse = [base, dict(base, config="svd_a25", cooc="svd", alpha=0.25, **{"proxy_ndcg@10": 0.009})]
    assert B.recommend(worse)["config"] is None
    with pytest.raises(ValueError):
        B.recommend(rows[1:])


def test_behavior_ids_with_real_opq():
    pytest.importorskip("faiss")
    rng = np.random.default_rng(0)
    text = rng.normal(size=(600, 48)).astype(np.float32)
    cooc = rng.normal(size=(600, 8)).astype(np.float32)
    ids = B.behavior_ids(text, cooc, 0.25, np.ones(600, bool), pca_dim=16,
                         quantize=lambda z, m: T.opq_ids(z, n_codebook=4, codebook_size=16, train_mask=m, threads=1))
    assert ids.shape == (600, 4) and ids.max() < 16


def test_cli_stops_when_order_check_fails(tmp_path, capsys):
    proc, text, cooc, _ = trial_setup(tmp_path)
    text.tofile(proc / "sentence-t5-base.sent_emb")
    np.save(tmp_path / "bad.npy", np.roll(cooc, 7, axis=0))                         # lệch thứ tự hàng
    code = B.main(["trial", "--processed-dir", str(proc), "--items", str(tmp_path / "items.jsonl"),
                   "--cooc", f"svd={tmp_path / 'bad.npy'}", "--out-dir", str(tmp_path / "out"), "--emb-dim", "32", "--pca-dim", "8"])
    assert code == 2 and "DỪNG" in capsys.readouterr().out
    assert json.loads((tmp_path / "out" / "order_check.json").read_text(encoding="utf-8"))["svd"]["passed"] is False


# --- chia codebook (mode split) ---

def test_split_features_puts_behaviour_in_last_dims():
    rng = np.random.default_rng(0)
    text = rng.normal(size=(300, 40))
    cooc = rng.normal(size=(300, 12))
    z = B.split_features(text, cooc, 2, n_codebook=8, pca_dim=16)          # 16/8 = 2 chiều mỗi mã; 2 mã -> 4 chiều
    assert z.shape == (300, 16)
    np.testing.assert_allclose(z[:, :12], T.pca_whiten(text, 12), atol=1e-4)
    cooc_l2 = cooc / np.linalg.norm(cooc, axis=1, keepdims=True)
    np.testing.assert_allclose(z[:, 12:], T.pca_whiten(cooc_l2, 4), atol=1e-4)
    with pytest.raises(ValueError):
        B.split_features(text, cooc, 0, n_codebook=8, pca_dim=16)
    with pytest.raises(ValueError):
        B.split_features(text, cooc, 2, n_codebook=6, pca_dim=16)


def test_split_ids_with_real_pq_keeps_text_codes_independent_of_behaviour():
    pytest.importorskip("faiss")
    rng = np.random.default_rng(0)
    text = rng.normal(size=(600, 48)).astype(np.float32)
    c1, c2 = rng.normal(size=(600, 12)), rng.normal(size=(600, 12))
    kw = dict(n_codebook=8, codebook_size=16, pca_dim=16, threads=1)
    a = B.split_ids(text, c1, 2, np.ones(600, bool), **kw)
    b = B.split_ids(text, c2, 2, np.ones(600, bool), **kw)
    np.testing.assert_array_equal(a[:, :6], b[:, :6])                       # 6 mã văn bản không đổi theo hành vi
    assert not np.array_equal(a[:, 6:], b[:, 6:])


def test_paired_difference():
    d = B.paired_difference([1.0, 2.0, 3.0, 4.0], [0.0, 2.0, 2.0, 4.0])
    assert d["diff"] == pytest.approx(0.5) and d["se"] == pytest.approx(np.std([1, 0, 1, 0], ddof=1) / 2)


def test_proxy_per_user_matches_mean():
    ids = np.stack([np.arange(40) // 2, np.arange(40)], axis=1)
    seqs = [[(2 * k + 10) % 40, 2 * k, 2 * k + 1, 0] for k in range(20)]
    out = B.last_item_overlap_proxy(ids, seqs, np.ones(40), n_users=20, per_user=True)
    assert out["per_user_ndcg"].shape == (20,) and out["proxy_ndcg@10"] == pytest.approx(out["per_user_ndcg"].mean())


def test_run_trial_split_mode_with_baseline(tmp_path):
    proc, text, cooc, labels = trial_setup(tmp_path)
    fake = lambda z, m: T.rq_ids(z, n_codebook=4, codebook_size=4, train_mask=m, backend="sklearn")
    rows = B.run_trial(text, {"svd": cooc}, [], proc, labels, tmp_path / "o", mode="split", behavior_codes=[1, 2],
                       baseline="pq_ref", quantize=fake, extra_ids={"pq_ref": np.zeros((80, 4), dtype=int)},
                       n_proxy_users=50, pca_dim=8, n_codebook=4)
    assert [r["config"] for r in rows] == ["pq_ref", "svd_b1", "svd_b2"]
    assert rows[1]["behavior_codes"] == 1 and rows[1]["alpha"] == ""
    assert rows[0]["proxy_diff_vs_baseline"] == 0.0 and rows[0]["baseline"] == "pq_ref"
    assert {"proxy_diff_se", "proxy_diff_z"} <= set(rows[2])
    assert (tmp_path / "o" / "trial_ids" / "svd_b2.npy").exists()
    with pytest.raises(ValueError):
        B.run_trial(text, {"svd": cooc}, [], proc, labels, tmp_path / "o", mode="mix")


def test_recommend_custom_baseline_and_min_z():
    base = {"config": "pq_rq1", "cooc": "có sẵn", "collision_rate": 0.0, "entropy_min": 0.98,
            "leaf_neighbor_purity@10": 0.20, "proxy_ndcg@10": 0.010}
    cand = dict(base, config="svd_b8", cooc="svd", behavior_codes=8, **{"proxy_ndcg@10": 0.012, "proxy_diff_z": 1.5})
    assert B.recommend([base, cand], baseline="pq_rq1")["config"] == "svd_b8"
    assert B.recommend([base, cand], baseline="pq_rq1", min_z=2.0)["config"] is None


def test_cli_split_requires_reference(tmp_path):
    proc, text, cooc, _ = trial_setup(tmp_path)
    text.tofile(proc / "sentence-t5-base.sent_emb")
    np.save(tmp_path / "svd.npy", cooc)
    with pytest.raises(SystemExit):
        B.main(["trial", "--mode", "split", "--processed-dir", str(proc), "--items", str(tmp_path / "items.jsonl"),
                "--cooc", f"svd={tmp_path / 'svd.npy'}", "--out-dir", str(tmp_path / "o"), "--emb-dim", "32",
                "--pca-dim", "8"])
