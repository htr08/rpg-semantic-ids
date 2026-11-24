"""Unit test cho src/tokenizer.py (PQ và token ngẫu nhiên).

CI không cài faiss nên các test dùng backend sklearn; test riêng cho faiss tự bỏ qua khi thiếu.
"""
import json

import numpy as np
import pytest

from adapters import load_ids as A
from src import tokenizer as T


def separable(n_per=30, m=4, k=3, dsub=2, seed=0):
    """Dữ liệu có k cụm tách biệt rõ trong từng đoạn; trả về x (N, m*dsub) và nhãn thật (N, m)."""
    rng = np.random.default_rng(seed)
    n = n_per * k
    truth = np.stack([rng.permutation(np.repeat(np.arange(k), n_per)) for _ in range(m)], axis=1)
    centers = np.arange(k)[:, None] * 10.0
    x = np.concatenate([centers[truth[:, j]] + rng.normal(0, 0.1, (n, dsub)) for j in range(m)], axis=1)
    return x.astype(np.float32), truth


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


def test_pca_whiten_rejects_too_few_columns():
    with pytest.raises(ValueError):
        T.pca_whiten(np.ones((10, 4)), n_components=8)


# --- PQ ---

def test_pq_shape_dtype_range():
    x, _ = separable()
    ids = T.pq_ids(x, n_codebook=4, codebook_size=3, backend="sklearn")
    assert ids.shape == (90, 4) and ids.dtype == np.int16
    assert ids.min() >= 0 and ids.max() < 3


def test_pq_recovers_clusters_in_each_subspace():
    x, truth = separable()
    ids = T.pq_ids(x, n_codebook=4, codebook_size=3, backend="sklearn")
    for j in range(4):
        # mỗi cụm thật ứng với đúng một mã và ngược lại (chỉ khác cách đánh số)
        pairs = set(zip(truth[:, j].tolist(), ids[:, j].tolist()))
        assert len(pairs) == 3


def test_pq_uses_contiguous_subvectors_without_rotation():
    # Đoạn j chỉ phụ thuộc cột [j*dsub, (j+1)*dsub): đổi cột của đoạn 0 không làm đổi mã đoạn 1
    x, _ = separable(m=2)
    ids = T.pq_ids(x, n_codebook=2, codebook_size=3, backend="sklearn")
    x2 = x.copy()
    x2[:, :2] = np.random.default_rng(5).normal(size=(len(x), 2))
    ids2 = T.pq_ids(x2, n_codebook=2, codebook_size=3, backend="sklearn")
    np.testing.assert_array_equal(ids[:, 1], ids2[:, 1])


def test_pq_learns_centroids_only_from_train_rows():
    # Hàng train ở quanh 0 và 10; hàng không train ở quanh 5 không được kéo tâm cụm về 5
    rng = np.random.default_rng(0)
    train = np.concatenate([rng.normal(0, 0.1, (20, 1)), rng.normal(10, 0.1, (20, 1))])
    other = rng.normal(5, 0.1, (40, 1))
    x = np.concatenate([train, other]).astype(np.float32)
    mask = np.r_[np.ones(40, bool), np.zeros(40, bool)]
    ids, books = T.pq_ids(x, n_codebook=1, codebook_size=2, train_mask=mask,
                          backend="sklearn", return_codebooks=True)
    np.testing.assert_allclose(np.sort(books[0].ravel()), [0, 10], atol=0.2)
    assert ids.shape == (80, 1)                                   # hàng không train vẫn được gán mã


def test_pq_is_reproducible_with_same_seed():
    x, _ = separable(seed=3)
    a = T.pq_ids(x, n_codebook=4, codebook_size=3, seed=7, backend="sklearn")
    b = T.pq_ids(x, n_codebook=4, codebook_size=3, seed=7, backend="sklearn")
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("kwargs, err", [
    (dict(x=np.ones((90, 7)), n_codebook=4, codebook_size=3), ValueError),       # 7 không chia hết cho 4
    (dict(x=np.ones(90), n_codebook=1, codebook_size=3), ValueError),            # mảng 1 chiều
    (dict(x=np.ones((90, 4)), n_codebook=4, codebook_size=3,
          train_mask=np.zeros(90, bool)), ValueError),                           # không có hàng train
    (dict(x=np.ones((90, 4)), n_codebook=4, codebook_size=3,
          train_mask=np.ones(10, bool)), ValueError),                            # mặt nạ sai độ dài
    (dict(x=np.ones((90, 4)), n_codebook=4, codebook_size=3, backend="gpu"), ValueError),
])
def test_pq_rejects_invalid_input(kwargs, err):
    x = kwargs.pop("x")
    kwargs.setdefault("backend", "sklearn")
    with pytest.raises(err):
        T.pq_ids(x, **kwargs)


def test_pq_with_faiss_backend():
    # Chỉ kiểm tra tính chất không phụ thuộc điểm khởi tạo: k-means của faiss chọn tâm đầu ngẫu nhiên
    # nên có thể rơi vào cực trị địa phương trên dữ liệu nhỏ; chất lượng cụm đo trên dữ liệu thật.
    pytest.importorskip("faiss")
    x, _ = separable(n_per=60)
    a = T.pq_ids(x, n_codebook=4, codebook_size=3, seed=1, backend="faiss")
    b = T.pq_ids(x, n_codebook=4, codebook_size=3, seed=1, backend="faiss")
    assert a.shape == (180, 4) and a.dtype == np.int16 and a.min() >= 0 and a.max() < 3
    np.testing.assert_array_equal(a, b)


def test_pq_output_loads_through_adapter(tmp_path):
    proc = make_cache(tmp_path, {"u": list(range(1, 91))}, n_items=90)
    x, _ = separable()
    ids = T.pq_ids(x, n_codebook=4, codebook_size=3, backend="sklearn")
    A.write_semantic_ids(ids, proc, "pq", n_codebook=4, codebook_size=4)
    np.testing.assert_array_equal(A.read_semantic_ids(proc, "pq", n_codebook=4, codebook_size=4), ids)


# --- Token ngẫu nhiên ---

def test_permuted_ids_keep_code_usage_of_each_codebook():
    ids = np.random.default_rng(0).integers(0, 256, size=(500, 32))
    out = T.permuted_ids(ids, seed=1)
    assert out.shape == ids.shape and out.dtype == np.int16
    for j in range(32):
        np.testing.assert_array_equal(np.sort(out[:, j]), np.sort(ids[:, j]))
    assert (out == ids).all(1).mean() < 0.01                        # gần như không item nào giữ ID cũ


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

def test_cli_pq_and_random(tmp_path):
    n = 40
    rng = np.random.default_rng(0)
    seqs = {"all": list(range(1, n + 1)) + [1, 2]}                  # mọi item đều có trong phần train
    proc = make_cache(tmp_path, seqs, n_items=n)
    rng.normal(size=(n, 16)).astype(np.float32).tofile(proc / "sentence-t5-base.sent_emb")
    small = ["--n-codebook", "4", "--codebook-size", "4"]

    out_pq = tmp_path / "pq.npy"
    assert T.main(["pq", "--processed-dir", str(proc), "--out", str(out_pq), "--emb-dim", "16",
                   "--pca-dim", "8", "--backend", "sklearn", *small]) == 0
    pq = np.load(out_pq)
    assert pq.shape == (n, 4) and pq.max() < 4

    # Token ngẫu nhiên hoán vị từ một file .sem_ids nguồn (ở đây dùng chính kết quả PQ)
    A.write_semantic_ids(pq, proc, "src", n_codebook=4, codebook_size=4)
    out_rd = tmp_path / "random.npy"
    assert T.main(["random", "--processed-dir", str(proc), "--out", str(out_rd),
                   "--source-name", "src", *small]) == 0
    rd = np.load(out_rd)
    for j in range(4):
        np.testing.assert_array_equal(np.sort(rd[:, j]), np.sort(pq[:, j]))
