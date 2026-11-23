"""Chạy Popularity và ItemKNN bằng RecBole trên split của dự án."""

import argparse
import importlib.metadata
import json
import os
import sys
import time
from pathlib import Path
from unittest.mock import patch

from src.baselines import fit_itemknn, fit_pop, prepare_recbole_runtime, rank_itemknn, rank_pop
from src.evaluate import build_evaluation_context, filter_seen, main as evaluate_main
from src.preprocessing import ROOT, read_records, sha256_file, verify_preprocessed, write_json, write_jsonl


def recbole_accuracy(rankings, split, ks, exclude_seen):
    """Dùng RecBole tính Recall/NDCG trên đúng thứ tự ứng viên do adapter xuất."""
    prepare_recbole_runtime()
    import numpy as np
    import torch
    from recbole.evaluator import Evaluator
    from recbole.evaluator.collector import DataStruct

    max_k = max(ks)
    hits = []
    for user, case in split.items():
        target = case["target"]
        ranked = rankings[user]
        if exclude_seen:
            ranked = filter_seen(ranked, case["seen_item_ids"], target)
        row = [int(item == target) for item in ranked[:max_k]]
        # Cột cuối của rec.topk là số nhãn đúng. Vị trí thiếu tính là trượt.
        hits.append(row + [0] * (max_k - len(row)) + [1])
    data = DataStruct()
    data.set("rec.topk", torch.tensor(hits))
    evaluator = Evaluator({"metrics": ["Recall", "NDCG"], "topk": ks, "metric_decimal_place": 12})
    # RecBole 1.2.0 uses the former alias np.float in NDCG.
    with patch.object(np, "float", float, create=True):
        return dict(evaluator.evaluate(data))


def run(args):
    import yaml
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    verify_preprocessed(args.processed_dir, args.splits_dir)
    mapping = json.loads((args.processed_dir / "id_mapping.json").read_text(encoding="utf-8"))
    train = [row for _, row in read_records(args.processed_dir / "train_sequences.jsonl")]
    items = [row for _, row in read_records(args.processed_dir / "items.jsonl")]
    context = build_evaluation_context(train, items)
    n_users = len(mapping["id2user"]) - 1
    n_items = len(mapping["id2item"]) - 1
    if context.catalog != frozenset(range(1, n_items + 1)):
        raise ValueError("Catalog IDs must match mapping exactly")
    splits = {}
    for name in args.splits:
        splits[name] = {row["user_id"]: row for _, row in read_records(args.processed_dir / f"{name}.jsonl")}
    ks = config["eval"]["topk"]
    exclude_seen = config["eval"]["exclude_seen"]
    if type(exclude_seen) is not bool:
        raise ValueError("eval.exclude_seen must be a YAML boolean")
    seen_flag = "--exclude-seen" if exclude_seen else "--no-exclude-seen"
    summaries = []
    for name in args.models:
        tag = name.lower()
        run_dir = args.output_dir / tag
        run_dir.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        print(f"Starting {tag} (device=cpu)", flush=True)
        if name == "Pop":
            model = fit_pop(context, n_users)
        else:
            model = fit_itemknn(train, n_users, n_items, **config["itemknn"])
        fit_seconds = time.perf_counter() - started
        report = {"model": name, "config": config, "python_version": sys.version,
                  "fit_seconds": fit_seconds, "device": "cpu",
                  "config_sha256": sha256_file(args.config),
                  "manifest_sha256": sha256_file(args.splits_dir / "manifest.json"),
                  "baseline_code_sha256": sha256_file(ROOT / "src" / "baselines.py"),
                  "runner_code_sha256": sha256_file(Path(__file__)),
                  "versions": {package: importlib.metadata.version(package) for package in
                               ("torch", "recbole", "numpy", "scipy", "pandas")},
                  "results": {}}
        for split_name in args.splits:
            split = splits[split_name]
            if name == "ItemKNN":
                rankings = rank_itemknn(model, split, context.catalog, max(ks), config["eval_batch_size"])
            else:
                rankings = rank_pop(model, split, context.catalog, max(ks), config["eval_batch_size"])
            output = run_dir / split_name
            output.mkdir(parents=True, exist_ok=True)
            ranking_path = output / "rankings.jsonl"
            write_jsonl(ranking_path, ({"user_id": user, "ranked_items": ranking}
                                      for user, ranking in sorted(rankings.items())))
            code = evaluate_main(["--rankings", str(ranking_path), "--model-name", tag,
                                  "--split", split_name, "--processed-dir", str(args.processed_dir),
                                  "--splits-dir", str(args.splits_dir), "--output-dir", str(output),
                                  "--ks", *map(str, ks), seen_flag])
            if code:
                raise ValueError(f"Evaluation failed for {tag}/{split_name}")
            result = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
            reference = recbole_accuracy(rankings, split, ks, exclude_seen)
            max_error = max(abs(value - result["metrics"][metric]) for metric, value in reference.items())
            if max_error >= 1e-4:
                raise ValueError(f"Recall/NDCG differ from RecBole for {tag}/{split_name}: {max_error}")
            result["recbole_accuracy_check"] = {"metrics": reference, "max_absolute_error": max_error}
            report["results"][split_name] = result
        report["elapsed_seconds"] = time.perf_counter() - started
        write_json(run_dir / "run.json", report)
        summaries.append(report)
    write_json(args.output_dir / "summary.json", summaries)
    return summaries


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "baselines_recbole.yaml")
    parser.add_argument("--models", nargs="+", choices=("Pop", "ItemKNN"), default=["Pop", "ItemKNN"])
    parser.add_argument("--splits", nargs="+", choices=("validation", "test"), default=["validation"])
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--processed-dir", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--splits-dir", type=Path, default=ROOT / "data" / "splits")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "experiments" / "baselines")
    args = parser.parse_args(argv)
    try:
        run(args)
        return 0
    except (OSError, ValueError, KeyError, ImportError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
