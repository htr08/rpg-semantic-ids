"""Chấm checkpoint RPG bằng đồ thị gốc, hai đồ thị rút cạnh và lấy mẫu ngẫu nhiên."""

import argparse
import sys
import time
from collections import Counter
from pathlib import Path
from statistics import mean

import numpy as np
import yaml
from scipy.stats import skew, spearmanr

from adapters.load_ids import processed_dir, read_semantic_ids, sem_ids_path
from adapters.rpg_rankings import match_test_users, run_with_rankings
from scripts.paired_t_test import paired_t_test
from scripts.run_rpg import DEFAULT_SENT_EMB_MODEL, build_config, parse_args, patch_torchvision_videoreader
from src.evaluate import build_evaluation_context, evaluate, load_rankings, load_split, per_user_metrics
from src.graph_reduction import edges, reduce_adjacency, sample_candidates
from src.preprocessing import ROOT, read_records, sha256_file, verify_preprocessed, write_json


def matched_random_propagation(model, budgets, seed):
    """Dùng logits RPG gốc. Mỗi user lấy mẫu đúng số item đồ thị gốc đã thăm."""
    import torch

    rng = np.random.default_rng(seed)
    n_items = model.dataset.n_items - 1
    offset = 0

    def rank(token_logits, n_return_sequences):
        nonlocal offset
        batch_size = len(token_logits)
        counts = budgets[offset:offset + batch_size]
        if len(counts) != batch_size or min(counts) < n_return_sequences:
            raise ValueError("Thiếu ngân sách ứng viên của batch")
        offset += batch_size
        candidates, valid = sample_candidates(counts, n_items, rng)
        candidates = torch.as_tensor(candidates, device=token_logits.device)
        valid = torch.as_tensor(valid, device=token_logits.device)
        indices = model.item_id2tokens[candidates] - 1
        item_scores = token_logits[:, None].expand(-1, candidates.shape[1], -1).gather(2, indices).mean(dim=-1)
        indices = item_scores.masked_fill(~valid, -torch.inf).topk(n_return_sequences, dim=1).indices
        chosen = candidates.gather(1, indices).unsqueeze(-1)
        return chosen, torch.tensor(counts, dtype=torch.float32).unsqueeze(-1)

    return rank


def graph_details(adjacency, counts):
    origins, _, targets = edges(adjacency)
    n_items = len(adjacency)
    indegree = np.bincount(targets, minlength=n_items + 1)[1:]
    outdegree = np.bincount(origins, minlength=n_items)
    rho = spearmanr(indegree, counts).statistic if np.ptp(indegree) and np.ptp(counts) else None
    return {
        "edges": len(targets),
        "indegree_mean": float(indegree.mean()),
        "indegree_std": float(indegree.std()),
        "indegree_p99": float(np.percentile(indegree, 99)),
        "indegree_max": int(indegree.max()),
        "indegree_skewness": float(skew(indegree, bias=False)) if np.ptp(indegree) else None,
        "minimum_outdegree": int(outdegree.min()),
        "spearman_indegree_vs_top10_count": float(rho) if rho is not None else None,
    }, indegree


def compare_adjacency(native, reference):
    if native.shape != reference.shape:
        raise ValueError(f"Adjacency đối chiếu có shape {reference.shape}, cần {native.shape}")
    difference = native != reference
    return {
        "rows_with_different_ordered_neighbors": int(np.count_nonzero(difference.any(axis=1))),
        "positions_different": int(difference.sum()),
        "rows_with_different_neighbor_set": int(np.count_nonzero(
            np.any(np.sort(native, axis=1) != np.sort(reference, axis=1), axis=1)
        )),
    }


def run(args):
    import torch

    manifest = verify_preprocessed(args.processed_dir, args.splits_dir)
    spec = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seed, cap = int(spec["seed"]), int(spec["indegree_cap"])
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    modes = ("native", "mutual", "indegree_cap", "random_matched")
    if not args.overwrite and any((output / mode / "rankings.jsonl").exists() for mode in modes):
        raise FileExistsError("Đã có ranking trong thư mục kết quả. Chọn thư mục mới hoặc dùng --overwrite")
    base_args = parse_args([
        "--root", str(args.root), "--checkpoint", str(args.checkpoint), "--seed", str(seed),
        "--run_id", "graph_ablation_opq",
    ])
    config = build_config(base_args)
    cache = processed_dir(config["cache_dir"], config["category"])
    encoder = DEFAULT_SENT_EMB_MODEL.rsplit("/", 1)[-1]
    sem_ids = sem_ids_path(cache, encoder, config["n_codebook"], config["codebook_size"])
    read_semantic_ids(cache, encoder, n_codebook=config["n_codebook"], codebook_size=config["codebook_size"])
    reference = load_rankings(args.reference_rankings) if args.reference_rankings else None
    split = load_split(args.processed_dir / "test.jsonl")
    if reference is not None and reference.keys() != split.keys():
        raise ValueError("Tập user của ranking đối chiếu không khớp test")
    context = build_evaluation_context(
        (row for _, row in read_records(args.processed_dir / "train_sequences.jsonl")),
        (row for _, row in read_records(args.processed_dir / "items.jsonl")),
    )

    sys.path.insert(0, str(args.repo.resolve()))
    sys.argv = [Path(sys.argv[0]).name]
    patch_torchvision_videoreader()
    from genrec.pipeline import Pipeline

    pipeline = Pipeline(
        model_name="RPG", dataset_name="AmazonReviews2014",
        checkpoint_path=str(args.checkpoint), config_dict=config,
    )
    model = pipeline.trainer.model
    device = pipeline.accelerator.device
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    test_users, _, n_items = match_test_users(pipeline, args.processed_dir)
    base_propagation = model.graph_propagation
    base_graph = native_hubs = native_visited = native_scores = None
    parity = {}
    runs = {}
    for mode in modes:
        start = time.perf_counter()
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state(cuda_rng, device)
        adjacency = None
        if mode in ("mutual", "indegree_cap"):
            adjacency = reduce_adjacency(base_graph, mode, indegree_cap=cap)
            origins, _, _ = edges(adjacency)
            if np.bincount(origins, minlength=n_items).min() + 1 < model.num_beams:
                raise ValueError(f"{mode}: có hàng không đủ ứng viên cho num_beams của RPG")
            model.adjacency[1:] = torch.as_tensor(adjacency, device=device)
        elif mode == "random_matched":
            model.graph_propagation = matched_random_propagation(model, [native_visited[user] for user in test_users], seed)

        folder = output / mode
        ranking_path = folder / "rankings.jsonl"
        visited = {}
        print(f"[{mode}] Bắt đầu chấm test", flush=True)
        try:
            native_result, _, _ = run_with_rankings(
                pipeline, args.processed_dir, ranking_path, overwrite=args.overwrite, visited=visited,
            )
        finally:
            model.graph_propagation = base_propagation
        ranking = load_rankings(ranking_path)
        metrics = evaluate(ranking, split, ks=(5, 10), context=context)
        users, scores = per_user_metrics(ranking, split, ks=(10,))
        by_user = dict(zip(users.tolist(), scores["ndcg@10"].tolist(), strict=True))
        counts = Counter(item for row in ranking.values() for item in row[:10])
        count_array = np.array([counts[item] for item in range(1, n_items + 1)])
        if mode == "native":
            base_graph = model.adjacency[1:].cpu().numpy().copy()
            if (base_graph.min() < 1 or base_graph.max() > n_items
                    or not np.all(np.any(base_graph == np.arange(1, n_items + 1)[:, None], axis=1))):
                raise ValueError("Thí nghiệm này cần đồ thị có cạnh tự nối và chỉ dùng item ID thật")
            adjacency = base_graph
            native_visited, native_scores = visited, by_user
            if reference is not None:
                reference_metrics = evaluate(reference, split, ks=(5, 10))
                parity["ranking"] = {
                    "reference_sha256": sha256_file(args.reference_rankings),
                    "same_users_and_ordered_items": sum(row == reference[user] for user, row in ranking.items()),
                    "metric_deltas": {metric: metrics[metric] - value for metric, value in reference_metrics.items()},
                }
            if args.reference_adjacency:
                reference_graph = np.load(args.reference_adjacency, allow_pickle=False)
                parity["adjacency"] = {
                    "reference_sha256": sha256_file(args.reference_adjacency),
                    **compare_adjacency(base_graph, reference_graph),
                }
        if mode == "random_matched" and visited != native_visited:
            raise ValueError("Ngân sách lấy mẫu phải khớp đồ thị gốc theo từng user")
        topology = None
        if adjacency is not None:
            topology, degree = graph_details(adjacency, count_array)
            if mode == "native":
                native_hubs = np.lexsort((np.arange(n_items), -degree))[:20] + 1
        slots = sum(counts.values())
        hub_slots = sum(counts[int(item)] for item in native_hubs)
        runs[mode] = {
            "metrics": {key: metrics[key] for key in (
                "recall@5", "recall@10", "ndcg@5", "ndcg@10", "coverage@10", "long_tail_ratio@10",
                "category_diversity@10",
            )},
            "mean_visited_items": mean(visited.values()),
            "rpg_float32_mean_visited_items": native_result["test_results"]["n_visited_items"],
            "topology": topology,
            "reference_hub_slots": hub_slots,
            "reference_hub_share": hub_slots / slots,
            "paired_t_test_vs_native": paired_t_test(by_user, native_scores) if mode != "native" else None,
            "ranking_sha256": sha256_file(ranking_path),
            "elapsed_seconds": round(time.perf_counter() - start, 2),
        }
        write_json(folder / "visited.json", visited)
        write_json(folder / "summary.json", runs[mode])
        print(f"[{mode}] NDCG@10 {metrics['ndcg@10']:.6f}, {len(visited)} user", flush=True)

    report = {
        "protocol": {"split": "test", "exclude_seen": False, "seed": seed, "users": len(test_users),
                     "catalog_items": n_items, "indegree_cap": cap, "num_beams": model.num_beams,
                     "n_edges": model.n_edges, "propagation_steps": model.propagation_steps},
        "runtime": {"device": str(device), "torch": str(torch.__version__)},
        "inputs_sha256": {
            "checkpoint": sha256_file(args.checkpoint), "semantic_ids": sha256_file(sem_ids),
            "split_manifest": sha256_file(args.splits_dir / "manifest.json"),
            "test": manifest["artifacts"]["test.jsonl"]["sha256"], "config": sha256_file(args.config),
        },
        "native_parity": parity,
        "reference_top_hub_item_ids": native_hubs.tolist(),
        "random_budget_matches_native_per_user": True,
        "runs": runs,
    }
    write_json(output / "summary.json", report)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True, help="Checkout RPG_KDD2025 tại 7dcf95c")
    parser.add_argument("--root", type=Path, required=True, help="Thư mục chứa cache của RPG")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint OPQ đã chọn trên validation")
    parser.add_argument("--processed-dir", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--splits-dir", type=Path, default=ROOT / "data" / "splits")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "rpg_graph_ablation.yaml")
    parser.add_argument("--reference-rankings", type=Path, help="Ranking đồ thị gốc từ PR 2")
    parser.add_argument("--reference-adjacency", type=Path, help="Adjacency dựng trên CPU từ PR 1")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        run(args)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Lỗi: {error}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
