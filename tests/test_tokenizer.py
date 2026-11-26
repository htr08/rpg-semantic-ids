"""Unit test cho src/tokenizer.py (PQ, OPQ kiểm tra, RQ, token ngẫu nhiên).

CI không cài faiss: test PQ/OPQ bằng FAISS tự bỏ qua, RQ dùng backend sklearn. Trên Kaggle/Colab có faiss
nên mọi test đều chạy; notebook còn kiểm tra ``opq_ids`` tái tạo đúng semantic ID của code gốc trên dữ liệu thật.
"""
import json

import numpy as np
import pytest

from adapters import load_ids as A
from src import tokenizer as T


def make_cache(tmp_path, seqs, n_items):
    proc = A.processed_dir(tmp_path / "cache")
    proc.mkdir(parents=True)
    asins = [f"A{i}" for i in range(1, n_items + 1)]
    (proc / "id_mapping.json").write_text(json.dumps(
        {"id2item": ["[PAD]"] + asins, "item2id": {a: i + 1 for i, a in enumerate(asins)}}))
    (proc / "all_item_seqs.json").write_text(json.dumps(
        {u: [f"A{i}" for i in s] for u, s in seqs.items()}))
    return proc


# --- Mặt nạ item train ---

def test_train_mask_drops_last_two_items_like_code_goc(tmp_path):
    # train = chuỗi bỏ 2 item cuối (genrec/dataset.py dòng 124)
    proc = make_cache(tmp_path, {"u1": [1, 2, 3, 4], "u2": [2, 5, 6]}, n_items=6)
    np.testing.assert_array_equal(T.train_item_mask(proc), [True, True, False, False, False, False])


def test_train_mask_skips_users_with_two_items_or_less(tmp_path):
    proc = make_cache(tmp_path, {"u1": [1, 2], "u2": [3, 4, 5]}, n_items=5)
    np.testing.assert_array_equal(T.train_item_mask(proc), [False, False, True, False, False])


def test_load_embeddings_checks_size(tmp_path):
    proc = make_cache(tmp_path, {"u": [1, 2, 3]}, n_items=3)
    np.arange(3 * 4, dtype=np.float32).tofile(proc / "e.sent_emb")
    assert T.load_embeddings(proc, "e", 4).shape == (3, 4)
    with pytest.raises(ValueError):
        T.load_embeddings(proc, "e", 5)


# --- PCA ---

def test_pca_whiten_shape_and_unit_variance():
    x = np.random.default_rng(0).normal(size=(200, 20)) @ np.random.default_rng(1).normal(size=(20, 20))
    z = T.pca_whiten(x, n_components=8)
    assert z.shape == (200, 8) and z.dtype == np.float32
    np.testing.assert_allclose(z.mean(0), 0, atol=1e-4)
    np.testing.assert_allclose(z.std(0, ddof=1), 1, atol=1e-3)


def test_pca_is_reproducible_with_seed_even_with_randomized_solver():
    # 540 x 96 -> 8: scikit-learn chọn solver randomized; không có random_state thì mỗi lần ra khác nhau
    x = np.random.default_rng(0).normal(size=(540, 96))
    np.testing.assert_array_equal(T.pca_whiten(x, 8, seed=42), T.pca_whiten(x, 8, seed=42))


def test_pca_whiten_rejects_too_few_columns():
    with pytest.raises(ValueError):
        T.pca_whiten(np.ones((10, 4)), n_components=8)


# --- PQ / OPQ bằng FAISS ---

def test_faiss_factory_strings():
    assert T.faiss_factory() == "IVF1,PQ32x8"                          # PQ: bỏ phần OPQ32,
    assert T.faiss_factory(rotation=True) == "OPQ32,IVF1,PQ32x8"       # đúng chuỗi của code gốc
    assert T.faiss_factory(4, 16) == "IVF1,PQ4x4"
    with pytest.raises(ValueError):
        T.faiss_factory(4, 100)


def blobs(n=600, d=16, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.normal(size=(8, d))[rng.integers(0, 8, n)] * 3 + rng.normal(size=(n, d))).astype(np.float32)


@pytest.mark.parametrize("fn", ["pq_ids", "opq_ids"])
def test_faiss_quantizers_shape_range_and_determinism(fn):
    pytest.importorskip("faiss")
    x = blobs()
    quant = getattr(T, fn)
    a = quant(x, n_codebook=4, codebook_size=16, threads=1)
    b = quant(x, n_codebook=4, codebook_size=16, threads=1)
    assert a.shape == (600, 4) and a.dtype == np.int16 and a.min() >= 0 and a.max() < 16
    np.testing.assert_array_equal(a, b)


def test_pq_and_opq_differ_only_by_rotation_and_both_cover_codes():
    pytest.importorskip("faiss")
    x = blobs()
    pq = T.pq_ids(x, n_codebook=4, codebook_size=16, threads=1)
    opq = T.opq_ids(x, n_codebook=4, codebook_size=16, threads=1)
    assert not np.array_equal(pq, opq)                                  # phép xoay làm đổi mã
    assert min(len(np.unique(pq[:, j])) for j in range(4)) > 8


def test_faiss_quantize_assigns_codes_to_rows_outside_train_mask():
    pytest.importorskip("faiss")
    x = blobs()
    mask = np.r_[np.ones(500, bool), np.zeros(100, bool)]
    ids = T.pq_ids(x, n_codebook=4, codebook_size=16, train_mask=mask, threads=1)
    assert ids.shape == (600, 4)


# --- RQ ---

def coarse_fine(n_per=40, seed=0):
    """2 nhóm thô cách nhau 20 đơn vị, mỗi nhóm 2 nhóm con cách nhau 2 đơn vị (3 chiều)."""
    rng = np.random.default_rng(seed)
    coarse = np.repeat([0, 1, 0, 1], n_per)
    fine = np.repeat([0, 0, 1, 1], n_per)
    x = coarse[:, None] * 20.0 + fine[:, None] * 2.0 + rng.normal(0, 0.05, (4 * n_per, 3))
    return x.astype(np.float32), coarse, fine


def test_rq_defaults_follow_faiss_defaults():
    assert (T.FAISS_NITER, T.FAISS_SEED) == (25, 1234)
    import inspect
    sig = inspect.signature(T.rq_ids)
    assert sig.parameters["niter"].default == 25 and sig.parameters["seed"].default == 1234
    assert sig.parameters["backend"].default == "faiss"


def test_rq_shape_dtype_range_and_any_dim():
    x, _, _ = coarse_fine()
    ids = T.rq_ids(x, n_codebook=4, codebook_size=2, backend="sklearn")   # 3 chiều, không cần chia hết cho 4
    assert ids.shape == (160, 4) and ids.dtype == np.int16
    assert ids.min() >= 0 and ids.max() < 2


def test_rq_first_level_is_coarse_and_second_level_is_fine():
    x, coarse, fine = coarse_fine()
    ids = T.rq_ids(x, n_codebook=2, codebook_size=2, backend="sklearn")
    assert len(set(zip(coarse.tolist(), ids[:, 0].tolist()))) == 2      # tầng 1 tách đúng 2 nhóm thô
    assert len(set(zip(fine.tolist(), ids[:, 1].tolist()))) == 2        # tầng 2 tách nhóm con trên phần dư


def test_rq_residual_energy_decreases():
    x, _, _ = coarse_fine()
    ids, books = T.rq_ids(x, n_codebook=3, codebook_size=2, backend="sklearn", return_codebooks=True)
    energy = T.rq_residual_energy(x, ids, books)
    assert energy.shape == (4,) and energy[0] == 1.0
    assert energy[1] < 0.05 and energy[2] < energy[1] and energy[3] <= energy[2] + 1e-6


def test_rq_learns_first_level_only_from_train_rows():
    rng = np.random.default_rng(0)
    train = np.concatenate([rng.normal(0, 0.1, (20, 1)), rng.normal(10, 0.1, (20, 1))])
    other = rng.normal(5, 0.1, (40, 1))
    x = np.concatenate([train, other]).astype(np.float32)
    mask = np.r_[np.ones(40, bool), np.zeros(40, bool)]
    ids, books = T.rq_ids(x, n_codebook=1, codebook_size=2, train_mask=mask,
                          backend="sklearn", return_codebooks=True)
    np.testing.assert_allclose(np.sort(books[0].ravel()), [0, 10], atol=0.2)
    assert ids.shape == (80, 1)


def test_rq_is_reproducible_with_same_seed():
    x, _, _ = coarse_fine(seed=2)
    a = T.rq_ids(x, n_codebook=3, codebook_size=2, seed=7, backend="sklearn")
    b = T.rq_ids(x, n_codebook=3, codebook_size=2, seed=7, backend="sklearn")
    np.testing.assert_array_equal(a, b)


def test_rq_with_faiss_backend():
    pytest.importorskip("faiss")
    x, _, _ = coarse_fine(n_per=100)
    a = T.rq_ids(x, n_codebook=3, codebook_size=2)
    np.testing.assert_array_equal(a, T.rq_ids(x, n_codebook=3, codebook_size=2))
    assert a.shape == (400, 3) and a.max() < 2


@pytest.mark.parametrize("kwargs", [
    dict(x=np.ones(90), n_codebook=1, codebook_size=4),                                  # mảng 1 chiều
    dict(x=np.ones((10, 2)), n_codebook=2, codebook_size=4, train_mask=np.r_[np.ones(3, bool), np.zeros(7, bool)]),
    dict(x=np.ones((90, 4)), n_codebook=4, codebook_size=3, train_mask=np.ones(10, bool)),  # mặt nạ sai độ dài
    dict(x=np.ones((90, 4)), n_codebook=4, codebook_size=3, backend="gpu"),
])
def test_rq_rejects_invalid_input(kwargs):
    x = kwargs.pop("x")
    kwargs.setdefault("backend", "sklearn")
    with pytest.raises(ValueError):
        T.rq_ids(x, **kwargs)


def test_rq_output_loads_through_adapter(tmp_path):
    proc = make_cache(tmp_path, {"u": list(range(1, 161))}, n_items=160)
    x, _, _ = coarse_fine()
    ids = T.rq_ids(x, n_codebook=4, codebook_size=2, backend="sklearn")
    A.write_semantic_ids(ids, proc, "rq", n_codebook=4, codebook_size=2)
    np.testing.assert_array_equal(A.read_semantic_ids(proc, "rq", n_codebook=4, codebook_size=2), ids)


# --- Token ngẫu nhiên ---

def test_permuted_ids_keep_code_usage_of_each_codebook():
    ids = np.random.default_rng(0).integers(0, 256, size=(500, 32))
    out = T.permuted_ids(ids, seed=1)
    assert out.shape == ids.shape and out.dtype == np.int16
    for j in range(32):
        np.testing.assert_array_equal(np.sort(out[:, j]), np.sort(ids[:, j]))


def test_permuted_ids_shuffle_each_column_independently():
    # Nếu chỉ xáo nguyên hàng, mọi ID mới vẫn là một ID cũ (100%); hoán vị độc lập theo cột thì gần như không.
    ids = np.random.default_rng(0).integers(0, 256, size=(500, 32))
    out = T.permuted_ids(ids, seed=1)
    old_rows = {tuple(r) for r in ids.tolist()}
    assert np.mean([tuple(r) in old_rows for r in out.tolist()]) < 0.01


def test_permuted_ids_remove_collisions():
    ids = np.zeros((6, 3), dtype=int)
    ids[:, 0] = np.arange(6) % 3                                   # đầu vào có ID trùng
    ids[:, 1] = np.arange(6) % 2
    ids[:, 2] = [0, 0, 1, 1, 2, 2]
    out = T.permuted_ids(ids, seed=0)
    assert len(np.unique(out, axis=0)) == 6


def test_permuted_ids_reproducible_and_seed_dependent():
    ids = np.random.default_rng(0).integers(0, 256, size=(100, 32))
    np.testing.assert_array_equal(T.permuted_ids(ids, seed=5), T.permuted_ids(ids, seed=5))
    assert not np.array_equal(T.permuted_ids(ids, seed=5), T.permuted_ids(ids, seed=6))


def test_permuted_ids_fails_when_uniqueness_impossible():
    with pytest.raises(ValueError):
        T.permuted_ids(np.zeros((4, 3), dtype=int), max_tries=5)


def test_summary():
    s = T.summary(np.array([[0, 1], [0, 1], [2, 3]]), codebook_size=4)
    assert s == {"shape": [3, 2], "codes_used_min": 2, "codes_used_max": 2,
                 "codebook_size": 4, "n_colliding_items": 2}


# --- CLI ---

def cli_cache(tmp_path, n=60, n_out=6):
    """Cache giả: n item, trong đó n_out item chỉ ở 2 vị trí cuối của chuỗi (không thuộc phần train)."""
    rng = np.random.default_rng(0)
    train_part = list(range(1, n - n_out + 1))
    seqs = {"all": train_part + [n - n_out + 1, n - n_out + 2]}
    for u in range(n_out // 2):
        seqs[f"tail{u}"] = [1, 2, n - n_out + 1 + 2 * u, n - n_out + 2 + 2 * u]
    proc = make_cache(tmp_path, seqs, n_items=n)
    rng.normal(size=(n, 16)).astype(np.float32).tofile(proc / "sentence-t5-base.sent_emb")
    return proc


def test_cli_rq_uses_train_mask_and_is_reproducible(tmp_path, capsys):
    n, n_out = 60, 6
    proc = cli_cache(tmp_path, n, n_out)
    small = ["--n-codebook", "4", "--codebook-size", "4", "--emb-dim", "16", "--pca-dim", "8", "--backend", "sklearn"]
    out1, out2 = tmp_path / "rq1.npy", tmp_path / "rq2.npy"
    assert T.main(["rq", "--processed-dir", str(proc), "--out", str(out1), *small]) == 0
    printed = capsys.readouterr().out
    assert f"item dùng để học tâm cụm: {n - n_out} / {n}" in printed      # mặt nạ train được dùng thật
    assert T.main(["rq", "--processed-dir", str(proc), "--out", str(out2), *small]) == 0
    np.testing.assert_array_equal(np.load(out1), np.load(out2))           # cùng seed -> cùng ID


def test_cli_random_preserves_source_code_usage(tmp_path):
    n = 60
    proc = cli_cache(tmp_path, n)
    src = np.random.default_rng(1).integers(0, 16, size=(n, 4))       # 16^4 tổ hợp: đủ chỗ cho 60 ID không trùng
    A.write_semantic_ids(src, proc, "src", n_codebook=4, codebook_size=16)
    out = tmp_path / "random.npy"
    assert T.main(["random", "--processed-dir", str(proc), "--out", str(out), "--source-name", "src",
                   "--n-codebook", "4", "--codebook-size", "16"]) == 0
    rd = np.load(out)
    for j in range(4):
        np.testing.assert_array_equal(np.sort(rd[:, j]), np.sort(src[:, j]))


def test_cli_pq_with_faiss(tmp_path, capsys):
    pytest.importorskip("faiss")
    n = 300
    proc = cli_cache(tmp_path, n, n_out=10)
    out = tmp_path / "pq.npy"
    assert T.main(["pq", "--processed-dir", str(proc), "--out", str(out), "--emb-dim", "16", "--pca-dim", "8",
                   "--n-codebook", "4", "--codebook-size", "16", "--threads", "1"]) == 0
    assert np.load(out).shape == (n, 4)
    assert f"item dùng để học tâm cụm: {n - 10} / {n}" in capsys.readouterr().out


def test_cli_prints_vietnamese_on_non_utf8_console(tmp_path, monkeypatch):
    # Console Windows mặc định cp1252: trước đây "Đã ghi …" gây UnicodeEncodeError và mã thoát 1.
    import io
    import sys

    n = 60
    proc = cli_cache(tmp_path, n)
    src = np.random.default_rng(1).integers(0, 16, size=(n, 4))
    A.write_semantic_ids(src, proc, "src", n_codebook=4, codebook_size=16)
    console = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
    monkeypatch.setattr(sys, "stdout", console)
    assert T.main(["random", "--processed-dir", str(proc), "--out", str(tmp_path / "r.npy"), "--source-name", "src",
                   "--n-codebook", "4", "--codebook-size", "16"]) == 0
    console.flush()
    assert "Đã ghi".encode("utf-8") in console.buffer.getvalue()
