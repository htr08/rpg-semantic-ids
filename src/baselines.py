"""Baseline RecBole trên bộ chia cố định của B, không tạo lại split hay mapping.

Mô hình/tương đồng từ RecBole 1.2.0; adapter dữ liệu, vòng chọn checkpoint và
xuất ranking do B viết để dùng đúng hợp đồng dữ liệu và bộ chấm chung.
"""

import os
import random
import sys
import time
from pathlib import Path
from types import ModuleType

import numpy as np
from scipy import sparse

from src.evaluate import evaluate
from src.preprocessing import iter_training_examples


def prepare_recbole_runtime():
    """Dùng chế độ no-TensorFlow có sẵn của TensorBoard trước khi import RecBole.

    RecBole import SummaryWriter dù baseline không dùng TensorBoard. Nếu không
    có marker notf, PyTorch/TensorBoard có thể import TensorFlow rồi JAX của
    Kaggle và vấp xung đột NumPy không liên quan tới các mô hình ở đây.
    Không sửa TensorFlow/JAX hoặc NumPy, không bắt/bỏ qua lỗi import mô hình.
    """
    sys.modules.setdefault("tensorboard.compat.notf", ModuleType("tensorboard.compat.notf"))


def top_items(scores, item_ids, limit):
    """Top theo điểm giảm dần, phá hòa ID tăng; xử lý cả hòa tại ngưỡng cắt."""
    scores, item_ids = np.asarray(scores), np.asarray(item_ids)
    if scores.shape != item_ids.shape or scores.ndim != 1 or not np.isfinite(scores).all():
        raise ValueError("Scores and item IDs must be finite one-dimensional arrays of equal size")
    if type(limit) is not int or limit < 1:
        raise ValueError("limit must be positive")
    limit = min(limit, len(scores))
    if not limit:
        return []
    threshold = np.partition(scores, len(scores) - limit)[len(scores) - limit]
    above = np.flatnonzero(scores > threshold)
    tied = np.flatnonzero(scores == threshold)
    tied = tied[np.argsort(item_ids[tied], kind="stable")][:limit - len(above)]
    selected = np.concatenate((above, tied))
    order = np.lexsort((item_ids[selected], -scores[selected]))
    return item_ids[selected[order]].astype(int).tolist()


def interaction_matrix(train, n_users, n_items):
    """Ma trận user-item nhị phân từ train, hàng/cột 0 dành cho padding."""
    rows, cols = [], []
    for row in train:
        user = row["user_id"]
        if not 1 <= user <= n_users:
            raise ValueError("Train user is outside mapping")
        for item in set(row["item_ids"]):
            if not 1 <= item <= n_items:
                raise ValueError("Train item is outside mapping")
            rows.append(user)
            cols.append(item)
    matrix = sparse.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                               shape=(n_users + 1, n_items + 1))
    matrix.data[:] = 1
    return matrix


def sequence_examples(train, max_history):
    """Sinh tiền tố chỉ từ train; thứ tự item giữ nguyên, cắt lịch sử từ bên trái."""
    histories, lengths, targets = [], [], []
    for row in train:
        for example in iter_training_examples(row["item_ids"], max_history):
            history = example["history"]
            histories.append(history + [0] * (max_history - len(history)))
            lengths.append(len(history))
            targets.append(example["target"])
    if not histories:
        raise ValueError("Train has no sequential training examples")
    return (np.asarray(histories, dtype=np.int64), np.asarray(lengths, dtype=np.int64),
            np.asarray(targets, dtype=np.int64))


def select_device(requested):
    """Mặc định CPU; GPU chỉ được phép trong runtime Kaggle theo yêu cầu người dùng."""
    if requested not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    if requested == "cuda" and not (Path("/kaggle/working").is_dir() and Path("/kaggle/input").is_dir()):
        raise ValueError("GPU runs are restricted to Kaggle; use --device cpu locally")
    if requested == "cuda":
        import torch
        if not torch.cuda.is_available():
            raise ValueError("Enable a GPU accelerator in Kaggle before running SASRec")
    return requested


def _config(device, max_history):
    return {"USER_ID_FIELD": "user_id", "ITEM_ID_FIELD": "item_id", "NEG_PREFIX": "neg_",
            "LIST_SUFFIX": "_list", "ITEM_LIST_LENGTH_FIELD": "item_length",
            "MAX_ITEM_LIST_LENGTH": max_history, "device": device}


class FixedCatalog:
    """Chỉ cung cấp kích thước vocab cho RecBole; ID số không bị remap."""

    def __init__(self, n_users, n_items):
        self.sizes = {"user_id": n_users + 1, "item_id": n_items + 1}

    def num(self, field):
        return self.sizes[field]


def fit_pop(context, n_users):
    """Khởi tạo Pop bằng đúng số lần xuất hiện trong train, không đếm lặp qua epoch."""
    prepare_recbole_runtime()
    import torch
    from recbole.model.general_recommender.pop import Pop

    model = Pop(_config("cpu", 50), FixedCatalog(n_users, len(context.catalog)))
    counts = [0] + [context.train_counts[item] for item in range(1, len(context.catalog) + 1)]
    model.item_cnt = torch.tensor(counts, dtype=torch.long).reshape(-1, 1)
    model.max_cnt = model.item_cnt.max(dim=0).values.clamp(min=1)
    model.eval()
    return model


def fit_itemknn(train, n_users, n_items, *, k, shrink):
    """Dùng cosine/top-k của RecBole, giữ W thưa, không dựng pred_mat user×item dày.

    Khi chấm test, có thể nhân W cố định với lịch sử gồm validation. Validation
    không tham gia fit W. Không làm tròn kết quả về nhị phân sau khi tính cosine.
    """
    prepare_recbole_runtime()
    from recbole.model.general_recommender.itemknn import ComputeSimilarity

    if type(k) is not int or not 1 <= k <= n_items or not np.isfinite(shrink) or shrink < 0:
        raise ValueError("Invalid ItemKNN k or shrink")
    matrix = interaction_matrix(train, n_users, n_items)
    _, weights = ComputeSimilarity(matrix, topk=k, shrink=shrink).compute_similarity("item")
    return weights.tocsr()


def rank_itemknn(weights, split, catalog, max_k, batch_size=256):
    """Score = lịch sử đầu vào nhị phân × W; toàn catalog là ứng viên."""
    item_ids = np.asarray(sorted(catalog))
    users, rankings = sorted(split), {}
    for start in range(0, len(users), batch_size):
        selected = users[start:start + batch_size]
        rows, cols = [], []
        for index, user in enumerate(selected):
            history_items = set(split[user]["history"])
            rows.extend([index] * len(history_items))
            cols.extend(history_items)
        history = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)),
                                    shape=(len(selected), weights.shape[0]))
        scores = (history @ weights).toarray()
        for index, user in enumerate(selected):
            rankings[user] = top_items(scores[index, item_ids], item_ids,
                                      max_k + len(split[user]["seen_item_ids"]))
    return rankings


def rank_torch(model, split, catalog, max_k, max_history=50, batch_size=256):
    """Score toàn catalog bằng RecBole rồi xuất prefix đủ dài trước lọc seen."""
    prepare_recbole_runtime()
    import torch
    from recbole.data.interaction import Interaction

    users, rankings = sorted(split), {}
    item_ids = np.asarray(sorted(catalog))
    model.eval()
    with torch.no_grad():
        for start in range(0, len(users), batch_size):
            selected = users[start:start + batch_size]
            features = {"user_id": torch.tensor(selected, device=model.device)}
            if hasattr(model, "ITEM_SEQ"):
                histories = [split[user]["history"][-max_history:] for user in selected]
                if any(not history for history in histories):
                    raise ValueError("SASRec evaluation histories must not be empty")
                width = max(map(len, histories))
                features["item_id_list"] = torch.tensor(
                    [history + [0] * (width - len(history)) for history in histories], device=model.device)
                features["item_length"] = torch.tensor(list(map(len, histories)), device=model.device)
            scores = model.full_sort_predict(Interaction(features)).reshape(len(selected), -1).cpu().numpy()
            for index, user in enumerate(selected):
                rankings[user] = top_items(scores[index, item_ids], item_ids,
                                          max_k + len(split[user]["seen_item_ids"]))
    return rankings


def train_sasrec(train, validation, context, n_users, settings, *, seed, device, output_dir,
                 exclude_seen=True):
    """Train mô hình/loss SASRec gốc RecBole; chọn checkpoint bằng NDCG@10 chung."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    prepare_recbole_runtime()
    import torch
    from recbole.data.interaction import Interaction
    from recbole.model.sequential_recommender.sasrec import SASRec
    from src.preprocessing import write_json

    select_device(device)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(settings["cpu_threads"])
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
    config = {**_config(device, settings["max_history"]), **settings["model"]}
    model = SASRec(config, FixedCatalog(n_users, len(context.catalog))).to(device)
    arrays = sequence_examples(train, settings["max_history"])
    histories, lengths, targets = [torch.from_numpy(array).to(device) for array in arrays]
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"],
                                 weight_decay=settings["weight_decay"])
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best, best_epoch, stale, history = -1.0, 0, 0, []
    started = time.perf_counter()
    for epoch in range(1, settings["epochs"] + 1):
        model.train()
        permutation = torch.randperm(len(targets), device=device)
        loss_total = 0.0
        for start in range(0, len(targets), settings["batch_size"]):
            indices = permutation[start:start + settings["batch_size"]]
            width = int(lengths[indices].max().item())
            interaction = Interaction({"item_id_list": histories[indices, :width],
                                       "item_length": lengths[indices], "item_id": targets[indices]})
            optimizer.zero_grad(set_to_none=True)
            loss = model.calculate_loss(interaction)
            if not torch.isfinite(loss):
                raise ValueError("SASRec training loss is not finite")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), settings["clip_grad_norm"])
            optimizer.step()
            loss_total += float(loss.detach().item()) * len(indices)
        rankings = rank_torch(model, validation, context.catalog, 10, settings["max_history"],
                              settings["eval_batch_size"])
        ndcg = evaluate(rankings, validation, ks=(10,), exclude_seen=exclude_seen)["ndcg@10"]
        history.append({"epoch": epoch, "loss": loss_total / len(targets), "val_ndcg@10": ndcg})
        print(f"SASRec seed={seed} epoch={epoch} loss={history[-1]['loss']:.5f} val_ndcg@10={ndcg:.6f}", flush=True)
        if ndcg > best:
            best, best_epoch, stale = ndcg, epoch, 0
            torch.save({"state_dict": model.state_dict(), "epoch": epoch, "val_ndcg@10": best,
                        "seed": seed, "config": config, "exclude_seen": exclude_seen},
                       output_dir / "checkpoint.pt")
        else:
            stale += 1
        write_json(output_dir / "training.json", {
            "seed": seed, "device": device, "best_epoch": best_epoch, "best_val_ndcg@10": best,
            "exclude_seen": exclude_seen,
            "training_examples": len(targets), "elapsed_seconds": time.perf_counter() - started,
            "epochs": history,
        })
        if stale >= settings["patience"]:
            break
    checkpoint = torch.load(output_dir / "checkpoint.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    return model, {"best_epoch": best_epoch, "epochs_run": len(history), "train_examples": len(targets),
                   "train_seconds": time.perf_counter() - started, "best_val_ndcg@10": best,
                   "exclude_seen": exclude_seen}
