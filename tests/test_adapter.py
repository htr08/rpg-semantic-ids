"""Unit test cho adapters/load_ids.py.

Dùng thư mục cache giả (tmp_path) có cùng cấu trúc với code gốc RPG_KDD2025 nên không cần
GPU, torch hay faiss. Kiểm tra trên dữ liệu thật (12.101 item) nằm trong notebook Colab.
"""
import json
import os

import numpy as np
import pytest

from adapters import load_ids as A

ASINS = ["B001", "B002", "B003", "B004", "B005"]


def make_cache(tmp_path, first="[PAD]", asins=ASINS):
    proc = A.processed_dir(tmp_path / "cache")
    proc.mkdir(parents=True)
    id2item = [first] + list(asins)
    mapping = {
        "item2id": {a: i for i, a in enumerate(id2item) if i},
        "id2item": id2item,
    }
    (proc / "id_mapping.json").write_text(json.dumps(mapping), encoding="utf-8")
    return proc


def code_goc_sem_ids_to_tokens(item2sem_ids, n_digit, codebook_size):
    """Chép nguyên logic tokenizer.py dòng 241–247 (code gốc @ 7dcf95c) để so sánh."""
    for item in item2sem_ids:
        tokens = list(item2sem_ids[item])
        for digit in range(n_digit):
            # "+ 1" as 0 is reserved for padding
            tokens[digit] += codebook_size * digit + 1
        item2sem_ids[item] = tuple(tokens)
    return item2sem_ids


# --- Tên file theo quy ước code gốc ---

def test_index_factory_default_matches_code_goc():
    assert A.index_factory() == "OPQ32,IVF1,PQ32x8"
    assert A.index_factory(4, 16) == "OPQ4,IVF1,PQ4x4"


@pytest.mark.parametrize("bad", [0, 1, 100, 255])
def test_index_factory_rejects_non_power_of_two(bad):
    with pytest.raises(ValueError):
        A.index_factory(32, bad)


def test_file_names_follow_basename_of_sent_emb_model(tmp_path):
    # tokenizer.py dòng 260–263 và 267–270: tên file = basename(sent_emb_model)
    proc = make_cache(tmp_path)
    model = A.run_args("pq")["sent_emb_model"]
    expected_ids = os.path.join(str(proc), f"{os.path.basename(model)}_OPQ32,IVF1,PQ32x8.sem_ids")
    expected_emb = os.path.join(str(proc), f"{os.path.basename(model)}.sent_emb")
    assert str(A.sem_ids_path(proc, "pq")) == expected_ids
    assert str(A.sent_emb_path(proc, "pq")) == expected_emb


def test_run_args():
    assert A.run_args("pq") == {"sent_emb_model": "x/pq"}
    assert A.run_args("behav_a05", 1024) == {"sent_emb_model": "x/behav_a05", "sent_emb_dim": 1024}


# --- Thứ tự item ---

@pytest.mark.parametrize("first", ["[PAD]", None])
def test_load_id2item_accepts_both_padding_styles(tmp_path, first):
    proc = make_cache(tmp_path, first=first)
    assert A.load_id2item(proc) == ASINS


def test_load_id2item_rejects_missing_padding(tmp_path):
    proc = make_cache(tmp_path, first="B000")
    with pytest.raises(ValueError):
        A.load_id2item(proc)


# --- Chế độ (b): semantic ID ---

def random_ids(n=len(ASINS), m=4, k=16, seed=0):
    return np.random.default_rng(seed).integers(0, k, size=(n, m))


def test_write_ids_rows_follow_item_id_order(tmp_path):
    proc = make_cache(tmp_path)
    ids = np.arange(len(ASINS) * 4).reshape(len(ASINS), 4) % 16   # mỗi hàng khác nhau
    path = A.write_semantic_ids(ids, proc, "pq", n_codebook=4, codebook_size=16)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert list(data) == ASINS                                   # thứ tự item_id 1..N
    for i, asin in enumerate(ASINS):                              # hàng i <-> item_id i + 1
        assert data[asin] == ids[i].tolist()
    assert all(type(v) is int for row in data.values() for v in row)   # JSON số nguyên thuần


def test_write_then_read_roundtrip(tmp_path):
    proc = make_cache(tmp_path)
    ids = random_ids()
    A.write_semantic_ids(ids, proc, "rq", n_codebook=4, codebook_size=16)
    back = A.read_semantic_ids(proc, "rq", n_codebook=4, codebook_size=16)
    np.testing.assert_array_equal(back, ids)


def test_raw_codes_are_stored_without_offset(tmp_path):
    # Code gốc tự cộng offset khi nạp; nếu adapter cộng trước thì offset bị cộng hai lần.
    proc = make_cache(tmp_path)
    ids = np.full((len(ASINS), 4), 15)
    path = A.write_semantic_ids(ids, proc, "pq", n_codebook=4, codebook_size=16)
    values = [v for row in json.loads(path.read_text()).values() for v in row]
    assert max(values) == 15


def test_tokens_match_code_goc_offset_formula(tmp_path):
    proc = make_cache(tmp_path)
    ids = random_ids(k=16)
    path = A.write_semantic_ids(ids, proc, "pq", n_codebook=4, codebook_size=16)
    reference = code_goc_sem_ids_to_tokens(json.loads(path.read_text()), n_digit=4, codebook_size=16)
    np.testing.assert_array_equal(np.array([reference[a] for a in ASINS]), A.to_tokens(ids, 16))


def test_token_ranges_do_not_overlap_between_codebooks():
    ids = np.array([[0] * 32, [255] * 32])
    tokens = A.to_tokens(ids)
    for digit in range(32):
        assert tokens[0, digit] == 256 * digit + 1
        assert tokens[1, digit] == 256 * (digit + 1)
    assert A.to_tokens(np.array([[5] * 32]))[0, 31] == 7942       # ví dụ trong ghi chú đọc code
    assert tokens.min() == 1 and tokens.max() == 8192             # 0 = padding, 8193 = eos


def test_default_shape_is_32_codes_of_256(tmp_path):
    proc = make_cache(tmp_path)
    ids = np.random.default_rng(1).integers(0, 256, size=(len(ASINS), 32), dtype=np.int16)
    path = A.write_semantic_ids(ids, proc, "random")
    assert path.name == "random_OPQ32,IVF1,PQ32x8.sem_ids"
    np.testing.assert_array_equal(A.read_semantic_ids(proc, "random"), ids)


@pytest.mark.parametrize("ids, err", [
    (np.zeros((len(ASINS) - 1, 4), dtype=int), ValueError),    # thiếu một item
    (np.zeros((len(ASINS) + 1, 4), dtype=int), ValueError),    # thừa một item
    (np.zeros((len(ASINS), 3), dtype=int), ValueError),        # sai số codebook
    (np.full((len(ASINS), 4), 16), ValueError),                # mã = codebook_size
    (np.full((len(ASINS), 4), -1), ValueError),                # mã âm
    (np.zeros((len(ASINS), 4), dtype=float), TypeError),       # không phải số nguyên
    (np.zeros(len(ASINS) * 4, dtype=int), ValueError),         # mảng 1 chiều
])
def test_write_ids_rejects_invalid_input(tmp_path, ids, err):
    proc = make_cache(tmp_path)
    with pytest.raises(err):
        A.write_semantic_ids(ids, proc, "bad", n_codebook=4, codebook_size=16)
    assert not A.sem_ids_path(proc, "bad", 4, 16).exists()


@pytest.mark.parametrize("name", ["sentence-t5-base", "x/pq", "", "../pq", "p q"])
def test_rejects_reserved_or_unsafe_names(tmp_path, name):
    proc = make_cache(tmp_path)
    with pytest.raises(ValueError):
        A.write_semantic_ids(random_ids(), proc, name, n_codebook=4, codebook_size=16)


def test_refuses_to_overwrite_without_flag(tmp_path):
    proc = make_cache(tmp_path)
    A.write_semantic_ids(random_ids(seed=1), proc, "pq", n_codebook=4, codebook_size=16)
    with pytest.raises(FileExistsError):
        A.write_semantic_ids(random_ids(seed=2), proc, "pq", n_codebook=4, codebook_size=16)
    A.write_semantic_ids(random_ids(seed=2), proc, "pq", n_codebook=4, codebook_size=16, overwrite=True)
    np.testing.assert_array_equal(A.read_semantic_ids(proc, "pq", n_codebook=4, codebook_size=16),
                                  random_ids(seed=2))


def test_read_rejects_file_missing_items(tmp_path):
    proc = make_cache(tmp_path)
    path = A.sem_ids_path(proc, "broken", 4, 16)
    path.write_text(json.dumps({a: [0, 0, 0, 0] for a in ASINS[:-1]}))
    with pytest.raises(ValueError, match="thiếu 1 item"):
        A.read_semantic_ids(proc, "broken", n_codebook=4, codebook_size=16)


def test_sidecar_records_source_and_collisions(tmp_path):
    proc = make_cache(tmp_path)
    ids = np.zeros((len(ASINS), 4), dtype=int)
    ids[0] = 1                                                   # 4 item còn lại trùng nhau
    A.write_semantic_ids(ids, proc, "pq", n_codebook=4, codebook_size=16, source="pq.npy")
    info = json.loads(A.sidecar_path(proc, "pq").read_text())
    assert info["mode"] == "semantic_ids" and info["source"] == "pq.npy"
    assert info["n_colliding_items"] == 4 and info["n_collision_groups"] == 1


def test_collision_stats():
    ids = np.array([[1, 2], [1, 2], [3, 4], [5, 6], [5, 6], [5, 6]])
    s = A.collision_stats(ids)
    assert s == {"n_items": 6, "n_unique_ids": 3, "n_colliding_items": 5,
                 "n_collision_groups": 2, "collision_rate": 5 / 6}


# --- Chế độ (a): embedding ---

def test_write_embeddings_matches_code_goc_reader(tmp_path):
    proc = make_cache(tmp_path)
    emb = np.random.default_rng(0).normal(size=(len(ASINS), 7))      # float64 đầu vào
    path = A.write_embeddings(emb, proc, "behav")
    assert path.name == "behav.sent_emb"
    # Cách code gốc đọc (tokenizer.py dòng 273)
    back = np.fromfile(path, dtype=np.float32).reshape(-1, 7)
    np.testing.assert_array_equal(back, emb.astype(np.float32))      # hàng i <-> item_id i + 1
    np.testing.assert_array_equal(A.read_embeddings(proc, "behav", 7), back)
    assert path.stat().st_size == len(ASINS) * 7 * 4                 # không có header


def test_read_embeddings_detects_wrong_dim(tmp_path):
    proc = make_cache(tmp_path)
    A.write_embeddings(np.ones((len(ASINS), 6)), proc, "behav")
    with pytest.raises(ValueError):
        A.read_embeddings(proc, "behav", 4)                         # 30 số, không chia hết cho 4
    with pytest.raises(ValueError):
        A.read_embeddings(proc, "behav", 3)                         # ra 10 hàng thay vì 5


@pytest.mark.parametrize("emb", [
    np.ones((len(ASINS) - 1, 4)),
    np.ones(len(ASINS)),
    np.array([[np.nan] * 4] * len(ASINS)),
    np.array([[np.inf] * 4] * len(ASINS)),
])
def test_write_embeddings_rejects_invalid_input(tmp_path, emb):
    proc = make_cache(tmp_path)
    with pytest.raises(ValueError):
        A.write_embeddings(emb, proc, "behav")
    assert not A.sent_emb_path(proc, "behav").exists()


def test_write_embeddings_handles_stale_sem_ids(tmp_path):
    # Có sẵn .sem_ids cùng tên thì code gốc sẽ bỏ qua PCA + OPQ (dòng 265), nên phải xử lý.
    proc = make_cache(tmp_path)
    A.write_semantic_ids(np.zeros((len(ASINS), 32), dtype=int), proc, "behav")
    with pytest.raises(FileExistsError):
        A.write_embeddings(np.ones((len(ASINS), 4)), proc, "behav")
    A.write_embeddings(np.ones((len(ASINS), 4)), proc, "behav", overwrite=True)
    assert not A.sem_ids_path(proc, "behav").exists()
    assert A.sent_emb_path(proc, "behav").exists()


def test_write_embeddings_rejects_reserved_name(tmp_path):
    proc = make_cache(tmp_path)
    with pytest.raises(ValueError):
        A.write_embeddings(np.ones((len(ASINS), 4)), proc, "sentence-t5-base")


# --- CLI ---

def test_cli_ids_and_export_roundtrip(tmp_path, capsys):
    proc = make_cache(tmp_path)
    ids = np.random.default_rng(3).integers(0, 256, size=(len(ASINS), 32))
    np.save(tmp_path / "pq.npy", ids)
    assert A.main(["ids", "--processed-dir", str(proc), "--npy", str(tmp_path / "pq.npy"), "--name", "pq"]) == 0
    assert "x/pq" in capsys.readouterr().out
    assert A.main(["export", "--processed-dir", str(proc), "--name", "pq", "--out", str(tmp_path / "back.npy")]) == 0
    np.testing.assert_array_equal(np.load(tmp_path / "back.npy"), ids)


def test_cli_emb(tmp_path, capsys):
    proc = make_cache(tmp_path)
    np.save(tmp_path / "e.npy", np.ones((len(ASINS), 9)))
    assert A.main(["emb", "--processed-dir", str(proc), "--npy", str(tmp_path / "e.npy"), "--name", "e"]) == 0
    assert "'sent_emb_dim': 9" in capsys.readouterr().out
