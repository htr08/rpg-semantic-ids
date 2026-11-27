"""Kiểm tra embedding đồng xuất hiện trên ví dụ nhỏ tính tay."""

import json

import numpy as np
import pytest
from scipy import sparse

from adapters.load_ids import validate_embeddings
from src.cooccurrence import (centroid_rankings, cooccurrence_matrix, finalize, item2vec_embeddings, main,
                              popularity_check, popularity_ranking, ppmi_matrix, quality_check, svd_embeddings)


def basket_matrix(baskets, n_items):
    """Ma trận user × item nhị phân; item_id đếm từ 1."""
    rows = [u for u, b in enumerate(baskets) for _ in b]
    cols = [i - 1 for b in baskets for i in b]
    return sparse.csr_matrix((np.ones(len(rows), dtype=bool), (rows, cols)), shape=(len(baskets), n_items))


def test_cooccurrence_counts_users_buying_both_and_drops_diagonal():
    matrix = cooccurrence_matrix(basket_matrix([[1, 2], [1, 2], [1, 3]], 4))
    assert matrix.toarray().tolist() == [[0, 2, 1, 0], [2, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]]


def test_ppmi_matches_hand_computation_and_drops_non_positive_values():
    # r = [10, 10, 18], tổng = 38; PMI(1,2) = log(1·38/100) < 0 bị loại; PMI(1,3) = log(9·38/180) = log 1,9.
    c = sparse.csr_matrix(np.array([[0, 1, 9], [1, 0, 9], [9, 9, 0]]))
    ppmi = ppmi_matrix(c).toarray()
    assert ppmi[0, 1] == ppmi[1, 0] == 0
    assert ppmi[0, 2] == pytest.approx(np.log(1.9)) and ppmi[2, 0] == pytest.approx(np.log(1.9))
    assert ppmi_matrix(sparse.csr_matrix((3, 3))).nnz == 0


def test_item2vec_is_deterministic_and_leaves_missing_items_zero():
    baskets = [np.array([1, 2, 3]), np.array([1, 2]), np.array([2, 3]), np.array([1, 3])]
    first = item2vec_embeddings(baskets, 5, dim=8, epochs=5)
    second = item2vec_embeddings(baskets, 5, dim=8, epochs=5)
    assert first.shape == (5, 8) and first.dtype == np.float32
    assert np.array_equal(first, second)
    assert first[:3].any(axis=1).all() and not first[3:].any()  # item 4, 5 không có trong giao dịch
    with pytest.raises(ValueError):
        item2vec_embeddings([np.array([9])], 5, dim=8, epochs=1)
    with pytest.raises(ValueError):
        item2vec_embeddings([], 5, dim=8, epochs=1)


def test_svd_embeddings_shape_determinism_and_dim_check():
    baskets = [[1, 2, 3], [1, 2], [2, 3, 4], [3, 4, 5], [1, 5], [4, 5, 6], [1, 6], [2, 6]]
    x = basket_matrix(baskets, 6)
    first, variance = svd_embeddings(x, 3, seed=7)
    second, _ = svd_embeddings(x, 3, seed=7)
    assert first.shape == (6, 3) and first.dtype == np.float32 and 0 < variance <= 1
    assert np.allclose(first, second)
    with pytest.raises(ValueError):
        svd_embeddings(x, 6)


def test_svd_embedding_gram_matrix_equals_rank_k_approximation_of_ppmi():
    # U·√Σ có Gram E·Eᵀ = U Σ Uᵀ; U·Σ hoặc U/Σ cho giá trị khác. So với SVD dày chính xác của PPMI.
    baskets = [[1, 2, 3], [1, 2], [2, 3, 4], [3, 4, 5], [1, 5], [4, 5, 6], [1, 6], [2, 6], [1, 2, 6], [3, 5, 6]]
    x = basket_matrix(baskets, 6)
    ppmi = ppmi_matrix(cooccurrence_matrix(x)).toarray()
    u, sigma, _ = np.linalg.svd(ppmi)
    k = 3
    expected = (u[:, :k] * sigma[:k]) @ u[:, :k].T
    embedding, _ = svd_embeddings(x, k)
    assert np.allclose(embedding @ embedding.T, expected, atol=1e-4)


def test_svd_rejects_empty_ppmi():
    # Mỗi user chỉ có một item: không có cặp đồng xuất hiện nên không có PPMI.
    with pytest.raises(ValueError, match="PPMI rỗng"):
        svd_embeddings(basket_matrix([[1], [2], [3]], 4), 2)


def test_finalize_normalizes_rows_and_keeps_zero_rows():
    matrix, zeros = finalize(np.array([[3.0, 4.0], [0.0, 0.0], [0.0, 2.0]]))
    assert zeros == 1
    assert np.allclose(np.linalg.norm(matrix, axis=1), [1.0, 0.0, 1.0])
    # Đầu ra qua được kiểm tra của adapter nạp embedding vào code gốc.
    assert validate_embeddings(matrix, 3).dtype == np.float32


def test_centroid_rankings_excludes_seen_and_returns_empty_for_zero_centroid():
    emb = np.array([[1, 0], [1, 0], [0.8, 0.6], [0, 1], [0, 0]], dtype=np.float32)
    emb[2] /= np.linalg.norm(emb[2])
    rankings = centroid_rankings(emb, [[1], [4], [5]], [{1}, {4}, {5}], k=2)
    assert rankings[0] == [2, 3]  # item 1 đã xem; cosine với item 2 là 1, với item 3 là 0,8
    assert rankings[1][0] == 3 and 4 not in rankings[1]
    assert rankings[2] == []  # lịch sử chỉ có item không có vector
    # k lớn hơn số item chưa xem: không báo lỗi và không có giá trị bị loại lọt vào danh sách.
    assert centroid_rankings(emb, [[1]], [{1, 2, 3, 4}], k=10) == [[5]]


def reference_quality(emb, rows, k):
    """Recall@k và NDCG@k tính trực tiếp bằng vòng lặp: xếp toàn catalog theo cosine với trung bình vector lịch sử."""
    recalls, ndcgs = [], []
    for history, seen, target in rows:
        score = emb @ emb[np.asarray(history) - 1].mean(axis=0)
        score[np.asarray(sorted(seen)) - 1] = -np.inf
        ranked = [int(i) + 1 for i in np.argsort(-score, kind="stable")[:k]]
        hit = target in ranked
        recalls.append(float(hit))
        ndcgs.append(1 / np.log2(ranked.index(target) + 2) if hit else 0.0)
    return float(np.mean(recalls)), float(np.mean(ndcgs))


def test_quality_check_matches_direct_computation_with_k_and_full_history(tmp_path):
    rng = np.random.default_rng(0)
    n_items = 40
    emb = rng.normal(size=(n_items, 6)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    rows = []
    for _ in range(60):
        history = rng.choice(np.arange(1, n_items + 1), size=int(rng.integers(5, 12)), replace=False).tolist()
        target = int(rng.choice([i for i in range(1, n_items + 1) if i not in history]))
        rows.append((history, set(history), target))
    path = tmp_path / "validation.jsonl"
    lines = [json.dumps({"user_id": u + 1, "history": h, "seen_item_ids": sorted(s), "target": t})
             for u, (h, s, t) in enumerate(rows)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for k in (3, 7):
        recall, ndcg = reference_quality(emb, rows, k)
        result = quality_check(emb, path, k=k)
        assert result[f"recall@{k}"] == pytest.approx(recall) and result[f"ndcg@{k}"] == pytest.approx(ndcg)
    counts = np.arange(n_items, 0, -1)  # item 1 phổ biến nhất
    popular = popularity_check(path, counts, k=3)
    expected = np.mean([t in [i for i in range(1, n_items + 1) if i not in s][:3] for _, s, t in rows])
    assert popular["recall@3"] == pytest.approx(expected)


def test_popularity_ranking_breaks_ties_by_item_id_and_skips_seen():
    counts = np.array([5, 9, 9, 1])
    assert popularity_ranking(counts, set(), 3) == [2, 3, 1]
    assert popularity_ranking(counts, {2}, 3) == [3, 1, 4]


def test_main_writes_embeddings_without_reading_test_items(tmp_path):
    interactions = [(0, 1, 1), (1, 1, 2), (2, 1, 3), (3, 1, 4), (4, 2, 1), (5, 2, 2), (6, 2, 4), (7, 2, 5),
                    (8, 3, 2), (9, 3, 3), (10, 3, 4), (11, 3, 1)]
    lines = ["interaction_id,user_id,item_id,timestamp,source_line"] + [f"{i},{u},{item},0,{i}" for i, u, item in interactions]
    (tmp_path / "interactions.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    # Mỗi user: 2 item train, 1 validation, 1 test. Train chỉ gồm item 1, 2, 3.
    (tmp_path / "splits.csv").write_text(
        "user_id,train_start,train_stop,validation_index,test_index\n1,0,2,2,3\n2,4,6,6,7\n3,8,10,10,11\n", encoding="utf-8")
    validation = [{"user_id": 1, "history": [1, 2], "target": 3, "seen_item_ids": [1, 2]},
                  {"user_id": 2, "history": [1, 2], "target": 4, "seen_item_ids": [1, 2]},
                  {"user_id": 3, "history": [2, 3], "target": 4, "seen_item_ids": [2, 3]}]
    (tmp_path / "validation.jsonl").write_text("\n".join(json.dumps(r) for r in validation) + "\n", encoding="utf-8")
    out = {"v2v": tmp_path / "v2v.npy", "svd": tmp_path / "svd.npy", "json": tmp_path / "r.json", "md": tmp_path / "r.md"}
    args = ["--interactions", str(tmp_path / "interactions.csv"), "--splits", str(tmp_path / "splits.csv"),
            "--validation", str(tmp_path / "validation.jsonl"), "--dim", "2", "--epochs", "3",
            "--item2vec-out", str(out["v2v"]), "--svd-out", str(out["svd"]), "--output", str(out["json"]),
            "--markdown", str(out["md"])]
    assert main(args) == 0
    for name in ("v2v", "svd"):
        matrix = np.load(out[name])
        assert matrix.shape == (5, 2) and matrix.dtype == np.float32
    payload = json.loads(out["json"].read_text(encoding="utf-8"))
    assert payload["n_transactions"] == 3 and payload["n_items"] == 5
    for name in ("popularity", "item2vec", "svd"):
        assert set(payload["quality"][name]) == {"recall@10", "ndcg@10"}
        assert all(0.0 <= v <= 1.0 for v in payload["quality"][name].values())
    # Item 4 chỉ có ở validation/test và item 5 chỉ có ở test nên cả hai không có vector ở cả hai embedding.
    assert payload["n_items_without_train"] == 2
    for name in ("v2v", "svd"):
        assert not np.load(out[name])[3:].any()
    assert "Embedding đồng xuất hiện" in out["md"].read_text(encoding="utf-8")


def test_main_rejects_invalid_dim_before_training(tmp_path, capsys, monkeypatch):
    import src.cooccurrence as cooc

    called = []
    monkeypatch.setattr(cooc, "item2vec_embeddings", lambda *a, **k: called.append(1))
    rows = ["interaction_id,user_id,item_id,timestamp,source_line", "0,1,1,0,0", "1,1,2,0,1", "2,1,3,0,2", "3,1,4,0,3"]
    (tmp_path / "interactions.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    splits = ["user_id,train_start,train_stop,validation_index,test_index", "1,0,2,2,3"]
    (tmp_path / "splits.csv").write_text("\n".join(splits) + "\n", encoding="utf-8")
    assert main(["--interactions", str(tmp_path / "interactions.csv"), "--splits", str(tmp_path / "splits.csv"),
                 "--dim", "4", "--output", str(tmp_path / "r.json"), "--markdown", str(tmp_path / "r.md")]) == 1
    assert "dim phải trong" in capsys.readouterr().err and not called


def test_main_reports_missing_input(tmp_path, capsys):
    assert main(["--interactions", str(tmp_path / "none.csv"), "--output", str(tmp_path / "r.json"),
                 "--markdown", str(tmp_path / "r.md")]) == 1
    assert "Error" in capsys.readouterr().err
