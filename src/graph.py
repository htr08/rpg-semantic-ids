"""Dựng và phân tích đồ thị giải mã RPG từ trọng số token của checkpoint (B, RQ3)."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import skew, spearmanr

from src.cluster_metrics import ami, nmi, load_item_labels
from src.evaluate import build_evaluation_context
from src.preprocessing import read_records, sha256_file, verify_preprocessed

ROOT = Path(__file__).resolve().parents[1]
RPG_REVISION = "7dcf95ce9e08601810efdd1c850ed6dada5f97c2"


def build_adjacency(codes, token_weights, *, codebook_size, k, block_size=512):
    """Trả về top-k ID (hàng i ứng với item i+1), giữ tự nối như bộ giải mã.

    RPG lấy trung bình cosine đã đưa về [0, 1] qua các vị trí của semantic ID.
    Tính từng khối hàng để tránh giữ ma trận N×N. Cột 0 có điểm 0 như RPG.
    """
    import torch
    from torch.nn.functional import normalize

    codes = np.asarray(codes)
    if codes.ndim != 2 or not codes.size or codes.dtype.kind not in "iu":
        raise ValueError("Semantic ID phải là ma trận mã nguyên khác rỗng")
    n_items, n_positions = codes.shape
    if not 1 <= k <= n_items or block_size < 1 or codebook_size < 1:
        raise ValueError("Cần 1 <= k <= số item, block_size và codebook_size dương")
    if codes.min() < 0 or codes.max() >= codebook_size:
        raise ValueError("Semantic ID ngoài phạm vi codebook")
    weights = torch.as_tensor(token_weights, device="cpu").detach().to(torch.float32)
    if (weights.ndim != 2 or weights.shape[0] != n_positions * codebook_size + 2
            or not weights.shape[1] or not torch.isfinite(weights).all()):
        raise ValueError("Trọng số token phải khớp số vị trí và codebook, kể cả PAD và EOS")

    weights = normalize(weights[1:-1].reshape(n_positions, codebook_size, -1), dim=-1)
    similarities = (torch.bmm(weights, weights.transpose(1, 2)) + 1.0) * 0.5
    code_columns = torch.as_tensor(codes.T.copy(), dtype=torch.long)
    neighbors = np.empty((n_items, k), dtype=np.int32)
    for start in range(0, n_items, block_size):
        stop = min(start + block_size, n_items)
        scores = torch.zeros((stop - start, n_items + 1), dtype=torch.float32)
        for lookup, column in zip(similarities, code_columns):
            scores[:, 1:] += lookup[column[start:stop]][:, column]
        scores[:, 1:] /= n_positions
        neighbors[start:stop] = scores.topk(k, dim=1).indices.numpy()
    return neighbors


def analyze_graph(adjacency, items, context, labels, *, louvain_seed):
    """Đo hubness trên đồ thị có hướng và chạy Louvain trên hợp cạnh vô hướng."""
    import networkx as nx
    from scipy.sparse import csr_matrix

    n_items = len(items)
    item_ids = np.arange(1, n_items + 1)
    self_edges = adjacency == item_ids[:, None]
    sources, slots = np.where((adjacency > 0) & ~self_edges)
    targets = adjacency[sources, slots] - 1
    degree = np.bincount(targets, minlength=n_items)
    train_counts = np.array([context.train_counts.get(int(item), 0) for item in item_ids])
    rank = np.lexsort((item_ids, -degree))[:20]
    rho = spearmanr(degree, train_counts) if np.ptp(degree) and np.ptp(train_counts) else None
    hubs = []
    for index in rank:
        item = items[index]
        hubs.append({
            "item_id": item["item_id"], "asin": item["asin"],
            "title": (item.get("metadata") or {}).get("title"),
            "in_degree": int(degree[index]), "train_interactions": int(train_counts[index]),
            "is_head": item["item_id"] in context.head_items,
        })

    arcs = csr_matrix((np.ones(len(sources), dtype=bool), (sources, targets)), shape=(n_items, n_items))
    graph = nx.from_scipy_sparse_array(arcs.maximum(arcs.T), edge_attribute=None)
    if graph.number_of_edges():
        communities = nx.community.louvain_communities(graph, weight=None, seed=louvain_seed)
        modularity = float(nx.community.modularity(graph, communities, weight=None))
    else:
        communities = [{i} for i in range(n_items)]
        modularity = None
    membership = np.empty(n_items, dtype=int)
    for index, community in enumerate(communities):
        membership[list(community)] = index
    agreement = {}
    for name in ("leaf", "level3"):
        category = labels[name]
        labeled_items = sum(label is not None for label in category)
        agreement[name] = {
            "labeled_items": labeled_items,
            "nmi": nmi(category, membership) if labeled_items else None,
            "ami": ami(category, membership) if labeled_items else None,
        }

    report = {
        "n_items": n_items,
        "neighbors_per_item": adjacency.shape[1],
        "self_edges_excluded": int(self_edges.sum()),
        "padding_edges_excluded": int(np.count_nonzero(adjacency == 0)),
        "directed_edges": len(sources),
        "indegree": {
            "mean": float(degree.mean()), "std": float(degree.std()),
            "skewness": float(skew(degree, bias=False)) if np.ptp(degree) and n_items >= 3 else None,
            **{f"p{p}": float(np.percentile(degree, p)) for p in (0, 50, 90, 95, 99, 100)},
            "zero_items": int(np.count_nonzero(degree == 0)),
        },
        "popularity": {
            "spearman": float(rho.statistic) if rho is not None else None,
            "p_value": float(rho.pvalue) if rho is not None and n_items >= 3 else None,
            "head_fraction": context.head_fraction,
            "head_hubs": sum(hub["is_head"] for hub in hubs),
        },
        "top_hubs": hubs,
        "louvain": {
            "seed": louvain_seed, "undirected_edges": graph.number_of_edges(),
            "n_communities": len(communities), "modularity": modularity,
            "community_sizes": sorted((len(c) for c in communities), reverse=True),
            "category_agreement": agreement,
        },
    }
    return report, degree, train_counts


def plot_hubness(path, degree, train_counts):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    figure, (histogram, scatter) = plt.subplots(1, 2, figsize=(10, 4.1), layout="constrained")
    histogram.hist(degree, bins=40, color="#087e8b")
    histogram.set(xlabel="Bậc vào (bỏ cạnh tự nối)", ylabel="Số sản phẩm", title="Phân bố bậc vào")
    scatter.scatter(np.log10(1 + train_counts), degree, s=7, alpha=0.18, color="#24639f", linewidths=0)
    scatter.set(xlabel="log₁₀(1 + số tương tác train)", ylabel="Bậc vào", title="Bậc vào và độ phổ biến")
    for axis in (histogram, scatter):
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.18)
        axis.set_axisbelow(True)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="State dict .pth của RPG")
    parser.add_argument("--semantic-ids", type=Path, required=True, help="File .sem_ids của RPG, khóa là ASIN")
    parser.add_argument("--processed-dir", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--splits-dir", type=Path, default=ROOT / "data" / "splits")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "graph_hubness.yaml")
    parser.add_argument("--output", type=Path, required=True, help="Báo cáo JSON")
    parser.add_argument("--adjacency-output", type=Path, help="Đồ thị gốc lưu thành .npy")
    parser.add_argument("--figure", type=Path, help="Biểu đồ bậc vào và độ phổ biến")
    args = parser.parse_args(argv)

    try:
        import torch

        verify_preprocessed(args.processed_dir, args.splits_dir)
        items_path = args.processed_dir / "items.jsonl"
        train_path = args.processed_dir / "train_sequences.jsonl"
        items = [row for _, row in read_records(items_path)]
        labels, _ = load_item_labels(items_path)
        context = build_evaluation_context((row for _, row in read_records(train_path)), items)
        raw_codes = json.loads(args.semantic_ids.read_text(encoding="utf-8"))
        if not isinstance(raw_codes, dict) or raw_codes.keys() != {item["asin"] for item in items}:
            raise ValueError("Các ASIN trong .sem_ids phải khớp catalog đã xử lý")
        codes = np.asarray([raw_codes[item["asin"]] for item in items])
        config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        weights = torch.load(args.checkpoint, map_location="cpu", weights_only=True)["gpt2.wte.weight"]
        adjacency = build_adjacency(
            codes, weights, codebook_size=config["codebook_size"],
            k=config["n_edges"], block_size=config["block_size"],
        )
        report, degree, train_counts = analyze_graph(
            adjacency, items, context, labels, louvain_seed=config["louvain_seed"],
        )
        report["rpg_source_revision"] = RPG_REVISION
        report["torch_version"] = str(torch.__version__)
        report["inputs"] = {
            key: {"file": path.name, "sha256": sha256_file(path)}
            for key, path in {
                "checkpoint": args.checkpoint, "semantic_ids": args.semantic_ids,
                "config": args.config, "split_manifest": args.splits_dir / "manifest.json",
            }.items()
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        if args.adjacency_output:
            args.adjacency_output.parent.mkdir(parents=True, exist_ok=True)
            with args.adjacency_output.open("wb") as stream:
                np.save(stream, adjacency, allow_pickle=False)
        if args.figure:
            plot_hubness(args.figure, degree, train_counts)
    except (KeyError, OSError, ValueError) as error:
        parser.exit(1, f"Lỗi: {error}\n")
    print(f"Đã ghi {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
