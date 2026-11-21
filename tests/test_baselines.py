"""Bộ chia cố định, scoring tính tay và tích hợp mô hình RecBole trên CPU."""

import os
import subprocess
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import pytest
from scipy import sparse

from src.baselines import (
    fit_itemknn, fit_pop, interaction_matrix, rank_itemknn, rank_pop,
    top_items,
)
from src.evaluate import build_evaluation_context, evaluate


def test_top_items_breaks_cutoff_ties_by_id():
    ids = np.array([5, 2, 4, 1, 3])
    assert top_items([1, 1, 4, 1, 1], ids, 3) == [4, 1, 2]
    assert top_items([1, 1, 4, 1, 1], ids, 10) == [4, 1, 2, 3, 5]


def test_top_items_matches_full_sort():
    rng = np.random.default_rng(42)
    for size in (1, 5, 101):
        ids = rng.permutation(np.arange(1, size + 1))
        scores = rng.integers(-3, 4, size=size)
        expected = ids[np.lexsort((ids, -scores))].tolist()
        for limit in (1, 3, size, size + 1):
            assert top_items(scores, ids, limit) == expected[:limit]


def test_train_matrix_is_binary_and_has_no_holdout():
    train = [{"user_id": 1, "item_ids": [1, 2, 2]}, {"user_id": 2, "item_ids": [1, 3]}]
    matrix = interaction_matrix(train, 2, 4)
    assert matrix.shape == (3, 5)
    assert matrix[1, 2] == 1
    assert matrix[:, 4].sum() == 0
    assert matrix[0].sum() == 0


def test_itemknn_can_use_validation_history_without_refitting():
    # Item 5 có cạnh mạnh nhưng nằm ngoài cửa sổ history của cả hai split.
    weights = sparse.csr_matrix(([1., 2., 3.], ([1, 2, 5], [3, 4, 2])), shape=(6, 6))
    original = weights.copy()
    val = {1: {"history": [1], "target": 2, "seen_item_ids": [5, 1]}}
    test = {1: {"history": [1, 2], "target": 3, "seen_item_ids": [5, 1, 2]}}
    assert rank_itemknn(weights, val, range(1, 6), 1)[1][0] == 3
    assert rank_itemknn(weights, test, range(1, 6), 1)[1][0] == 4
    assert (original != weights).nnz == 0


@pytest.fixture
def recbole_ready():
    pytest.importorskip("torch")
    pytest.importorskip("recbole")
    pytest.importorskip("kmeans_pytorch")


def test_recbole_pop_counts_events_exactly(recbole_ready):
    train = [{"user_id": 1, "item_ids": [1, 1, 1, 2]}]
    context = build_evaluation_context(train, [{"item_id": i} for i in range(1, 5)])
    model = fit_pop(context, 1)
    assert model.item_cnt[:, 0].tolist() == [0, 3, 1, 0, 0]
    split = {1: {"target": 3, "seen_item_ids": [1, 2]}}
    rankings = rank_pop(model, split, context.catalog, 2)
    assert rankings[1] == [1, 2, 3, 4]
    assert evaluate(rankings, split, ks=(2,), context=context)["recall@2"] == 0
    assert evaluate(rankings, split, ks=(2,), context=context, exclude_seen=True)["recall@2"] == 1


def test_recbole_itemknn_cosine_matches_hand_calculation(recbole_ready):
    # Deux user: item 1 chez les deux, item 2 chez u1, item 3 chez u2.
    train = [{"user_id": 1, "item_ids": [1, 2, 2]}, {"user_id": 2, "item_ids": [1, 3]}]
    w = fit_itemknn(train, 2, 3, k=3, shrink=0)
    assert w[1, 2] == pytest.approx(1 / (np.sqrt(2) + 1e-6), rel=1e-6)
    assert w[1, 3] == pytest.approx(w[1, 2])
    assert w[2, 3] == 0
    assert np.all(w.diagonal() == 0)


@pytest.mark.parametrize("prepared", [False, True])
def test_tensorboard_does_not_load_broken_tensorflow(recbole_ready, tmp_path, prepared):
    # Tái hiện import gián tiếp TensorFlow/JAX bị hỏng vì thiếu StringDType.
    # Đối chứng không chuẩn bị runtime phải lỗi; baseline thực tế không import TF.
    package = tmp_path / "tensorflow"
    package.mkdir()
    (package / "__init__.py").write_text(
        "raise AttributeError(\"module 'numpy.dtypes' has no attribute 'StringDType'\")\n",
        encoding="utf-8",
    )
    if prepared:
        program = (
            "from src.baselines import fit_pop; from src.evaluate import build_evaluation_context; "
            "context=build_evaluation_context([{'user_id':1,'item_ids':[1]}],[{'item_id':1}]); "
            "model=fit_pop(context,1); assert model.item_cnt[1].item()==1; "
            "import sys; assert 'tensorflow' not in sys.modules and 'jax' not in sys.modules"
        )
    else:
        program = "from torch.utils.tensorboard import SummaryWriter"
    env = {**os.environ, "PYTHONPATH": str(tmp_path) + os.pathsep + os.getcwd(),
           "CUDA_VISIBLE_DEVICES": ""}
    result = subprocess.run([sys.executable, "-c", program], env=env, capture_output=True,
                            text=True, timeout=60)
    if prepared:
        assert result.returncode == 0, result.stderr
    else:
        assert result.returncode != 0
        assert "StringDType" in result.stderr
