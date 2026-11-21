"""Chấm checkpoint SASRec của S³-Rec bằng bộ chia và bộ chấm của dự án."""

import argparse
from collections import Counter, defaultdict
from io import BytesIO
import json
from pathlib import Path
from statistics import mean, stdev
import sys
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
import torch

from scripts.reproduce_sasrec_paper import DATA_HASHES, REFERENCE_DIR, RUN_NAME, SEEDS, UPSTREAM_CODE, UPSTREAM_REVISION
from src.baselines import top_items
from src.evaluate import main as evaluate_main
from src.preprocessing import ROOT, read_records, sha256_file, verify_preprocessed, write_json, write_jsonl


METRICS = ("recall@5", "recall@10", "ndcg@5", "ndcg@10")
MODEL_SOURCE = {name: UPSTREAM_CODE[name] for name in ("models.py", "modules.py")}


def match_item_ids(reference_sequences, local_sequences):
    """Ghép item bằng vị trí xuất hiện, rồi kiểm tra nguyên chuỗi theo thứ tự."""
    def by_signature(sequences):
        occurrences = defaultdict(list)
        for sequence in sequences:
            for position, item in enumerate(sequence):
                occurrences[item].append((len(sequence), position))
        result = {tuple(sorted(positions)): item for item, positions in occurrences.items()}
        if len(result) != len(occurrences):
            raise ValueError("Beauty item signatures are ambiguous")
        return result

    reference = by_signature(reference_sequences)
    local = by_signature(local_sequences)
    if reference.keys() != local.keys():
        raise ValueError("Reference and project Beauty item signatures differ")
    ref_to_local = {item: local[signature] for signature, item in reference.items()}
    mapped = Counter(tuple(ref_to_local[item] for item in sequence) for sequence in reference_sequences)
    if mapped != Counter(map(tuple, local_sequences)):
        raise ValueError("Reference and project Beauty sequences differ after remapping")
    return {local_id: reference_id for reference_id, local_id in ref_to_local.items()}


def rank(model, split, local_to_ref):
    item_ids = np.asarray(sorted(local_to_ref))
    ref_items = torch.tensor([local_to_ref[item] for item in item_ids])
    users = sorted(split)
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(users), 256):
            batch = users[start:start + 256]
            histories = []
            for user in batch:
                history = [local_to_ref[item] for item in split[user]["history"]]
                histories.append([0] * (50 - len(history)) + history)
            states = model.finetune(torch.tensor(histories))[:, -1, :]
            scores = states @ model.item_embeddings.weight[ref_items].T
            for user, row in zip(batch, scores.numpy(), strict=True):
                yield {"user_id": user, "ranked_items": top_items(row, item_ids, 10)}


def run(reference_dir, archive_path, output_dir):
    processed, splits_dir = ROOT / "data/processed", ROOT / "data/splits"
    verify_preprocessed(processed, splits_dir)
    splits = {
        split: {row["user_id"]: row for _, row in read_records(processed / f"{split}.jsonl")}
        for split in ("validation", "test")
    }
    sequences = [
        row["item_ids"] + [splits[split][row["user_id"]]["target"] for split in splits]
        for _, row in read_records(processed / "train_sequences.jsonl")
    ]
    reference_data = reference_dir / "Beauty.txt"
    attribute_path = reference_dir / "Beauty_item2attributes.json"
    input_hashes = {path.name: sha256_file(path) for path in (reference_data, attribute_path)}
    if input_hashes != DATA_HASHES:
        raise ValueError("Beauty inputs are from a different revision")
    reference_sequences = [
        list(map(int, line.split()[1:])) for line in reference_data.read_text().splitlines()
    ]
    local_to_ref = match_item_ids(reference_sequences, sequences)
    attributes = json.loads(attribute_path.read_bytes())
    args = SimpleNamespace(
        max_seq_length=50, hidden_size=64, num_hidden_layers=2, num_attention_heads=2,
        hidden_act="gelu", attention_probs_dropout_prob=0.5, hidden_dropout_prob=0.5,
        initializer_range=0.02, item_size=max(local_to_ref.values()) + 2, cuda_condition=False,
        attribute_size=max(attr for item in attributes.values() for attr in item) + 1,
    )
    source = reference_dir / "upstream"
    if any(sha256_file(source / name) != checksum for name, checksum in MODEL_SOURCE.items()):
        raise ValueError("S³-Rec model source is from a different revision")
    sys.path.insert(0, str(source.resolve()))
    from models import S3RecModel

    torch.set_num_threads(4)
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = []
    with ZipFile(archive_path) as archive:
        paper = json.loads(archive.read("summary.json"))
        if tuple(run["seed"] for run in paper["runs"]) != SEEDS:
            raise ValueError("Expected paper checkpoints for seeds 42, 43, 44")
        for saved in paper["runs"]:
            seed = saved["seed"]
            if saved["input_files"] != input_hashes or saved["revision"] != UPSTREAM_REVISION:
                raise ValueError(f"Checkpoint provenance mismatch for seed {seed}")
            model = S3RecModel(args)
            path = f"seed_{seed}/{RUN_NAME}.pt"
            checkpoint = torch.load(BytesIO(archive.read(path)), map_location="cpu", weights_only=True)
            model.load_state_dict(checkpoint)
            record = {"seed": seed, "paper_test": saved["test"]}
            for split, cases in splits.items():
                output = output_dir / f"seed_{seed}" / split
                output.mkdir(parents=True, exist_ok=True)
                ranking = output / "rankings.jsonl"
                write_jsonl(ranking, rank(model, cases, local_to_ref))
                if evaluate_main(["--rankings", str(ranking), "--split", split, "--no-exclude-seen",
                                  "--model-name", f"sasrec_s3rec_s{seed}", "--output-dir", str(output)]):
                    raise ValueError(f"Unable to evaluate seed {seed} on {split}")
                report = json.loads((output / "metrics.json").read_bytes())
                record[split] = {key: report["metrics"][key] for key in METRICS}
                record[split]["ranking_sha256"] = report["inputs"]["rankings"]["sha256"]
                record[split]["per_user_sha256"] = sha256_file(output / "per_user.csv")
            runs.append(record)
    aggregate = {}
    for key in METRICS:
        values = [record["test"][key] for record in runs]
        aggregate[key] = {"mean": mean(values), "sample_std": stdev(values)}
    write_json(output_dir / "summary.json", {
        "model": "SASRec (S³-Rec, BCE)", "checkpoint_selection": "S³-Rec validation NDCG@20, patience 10",
        "evaluation": "leave-last-out, whole project catalog, keep seen items, K=5 and 10",
        "source_revision": UPSTREAM_REVISION, "source_files": MODEL_SOURCE, "input_files": input_hashes,
        "archive_sha256": sha256_file(archive_path), "split_manifest_sha256": sha256_file(splits_dir / "manifest.json"),
        "users": len(sequences), "catalog_items": len(local_to_ref),
        "paper_reproduction": {
            "evaluation": "S³-Rec full ranking, set scores of seen items to zero before top-K",
            "reference_ndcg@10": paper["paper_beauty_sasrec_ndcg@10"],
            "aggregate_test": paper["aggregate"],
        },
        "aggregate_test": aggregate, "runs": runs,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, default=REFERENCE_DIR)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "experiments/sasrec_paper_no_seen")
    cli = parser.parse_args()
    run(cli.reference_dir, cli.archive or cli.reference_dir / "sasrec_paper_beauty_outputs.zip", cli.output_dir)
