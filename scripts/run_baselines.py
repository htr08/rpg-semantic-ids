"""Chạy Popularity và ItemKNN bằng RecBole trên split của dự án."""

import argparse
import importlib.metadata
import json
import os
import sys
import time
from pathlib import Path

from src.baselines import fit_itemknn, fit_pop, rank_itemknn, rank_torch, select_device, train_sasrec
from src.evaluate import build_evaluation_context, main as evaluate_main
from src.preprocessing import ROOT, read_records, sha256_file, verify_preprocessed, write_json, write_jsonl


def run(args):
    import yaml
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if "SASRec" in args.models and "sasrec" not in config:
        raise ValueError("See docs/evaluation.md for SASRec-BCE using S³-Rec. "
                         "The legacy runner requires an archived SASRec config.")
    select_device(args.device)
    if args.device == "cpu":
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
    for name in set(args.splits) | {"validation"}:
        splits[name] = {row["user_id"]: row for _, row in read_records(args.processed_dir / f"{name}.jsonl")}
    ks = config["eval"]["topk"]
    exclude_seen = config["eval"]["exclude_seen"]
    if type(exclude_seen) is not bool:
        raise ValueError("eval.exclude_seen must be a YAML boolean")
    seen_flag = "--exclude-seen" if exclude_seen else "--no-exclude-seen"
    summaries = []
    for name in args.models:
        seeds = args.seeds if name == "SASRec" else [None]
        for seed in seeds:
            tag = f"{name.lower()}_s{seed}" if seed is not None else name.lower()
            run_dir = args.output_dir / tag
            run_dir.mkdir(parents=True, exist_ok=True)
            started = time.perf_counter()
            print(f"Starting {tag} (device={args.device if name == 'SASRec' else 'cpu'})", flush=True)
            extra = {}
            if name == "Pop":
                model = fit_pop(context, n_users)
            elif name == "ItemKNN":
                model = fit_itemknn(train, n_users, n_items, **config["itemknn"])
            else:
                settings = {**config["sasrec"], **{key: config[key] for key in
                            ("max_history", "cpu_threads", "eval_batch_size")}}
                if args.epochs is not None:
                    settings["epochs"] = args.epochs
                model, extra = train_sasrec(train, splits["validation"], context, n_users, settings,
                                           seed=seed, device=args.device, output_dir=run_dir,
                                           exclude_seen=exclude_seen)
            fit_seconds = time.perf_counter() - started
            report = {"model": name, "seed": seed, "config": config,
                      "python_version": sys.version,
                      "runtime_policy": {"tensorboard": "no_tensorflow", "gpu": "Kaggle only"},
                      "effective_epochs": settings["epochs"] if name == "SASRec" else None,
                      "fit_seconds": fit_seconds, "selection": extra,
                      "device": args.device if name == "SASRec" else "cpu",
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
                    rankings = rank_torch(model, split, context.catalog, max(ks), config["max_history"],
                                          config["eval_batch_size"])
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
                report["results"][split_name] = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
            report["elapsed_seconds"] = time.perf_counter() - started
            write_json(run_dir / "run.json", report)
            summaries.append(report)
    write_json(args.output_dir / "summary.json", summaries)
    return summaries


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "baselines_recbole.yaml")
    parser.add_argument("--models", nargs="+", choices=("Pop", "ItemKNN", "SASRec"), default=["Pop", "ItemKNN"],
                        help="Cấu hình mặc định chạy Pop và ItemKNN. SASRec ở đây dành cho config lưu trữ.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--splits", nargs="+", choices=("validation", "test"), default=["validation"])
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--epochs", type=int, help="Giới hạn epoch cho kiểm tra; không thay config gốc")
    parser.add_argument("--processed-dir", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--splits-dir", type=Path, default=ROOT / "data" / "splits")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "experiments" / "baselines")
    args = parser.parse_args(argv)
    try:
        if args.epochs is not None and args.epochs < 1:
            raise ValueError("epochs must be positive")
        run(args)
        return 0
    except (OSError, ValueError, KeyError, ImportError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
