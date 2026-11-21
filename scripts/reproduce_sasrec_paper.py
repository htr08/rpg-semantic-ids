"""Prepare the pinned Beauty inputs or train three SASRec-BCE seeds with S³-Rec."""

import argparse
import ast
import hashlib
import subprocess
import sys
from pathlib import Path
from statistics import mean, stdev
from tempfile import TemporaryDirectory
from urllib.request import urlopen
from zipfile import ZIP_DEFLATED, ZipFile

from src.preprocessing import ROOT, sha256_file, write_json


UPSTREAM_REVISION = "49a890dcb516c9ed3126564495234ea74d5cd566"
UPSTREAM_URL = f"https://raw.githubusercontent.com/aHuiWang/CIKM2020-S3Rec/{UPSTREAM_REVISION}"
REFERENCE_DIR = ROOT / "experiments/sasrec_reference"
DATA_HASHES = {
    "Beauty.txt": "226cce9c3105299ca0db9615d7d3fb32b3175e90da43100ae352599f0f0107b8",
    "Beauty_item2attributes.json": "129e40f67486f6877eff0f5318cfa8530ee8ac23d8b0d1de04e1bb7a2ba0b2d4",
}
UPSTREAM_CODE = {
    "datasets.py": "2de25229e5885ae776efb8943170796d652ef2a65a3323738d61ebaa23c28d55",
    "models.py": "cdb7dd02f870b9bd16e2e6d9fb094be29ef490f5b7d19f85e2b72994761bd27a",
    "modules.py": "348190dcfba4ded63441041deee9a3f8931bd1dedae9fc3cdd300beea89bba7d",
    "run_finetune_full.py": "fd4462c85552c981769ef6614e1850d4b8145e03fdb5ab69dcccf18ee8e46e75",
    "trainers.py": "7610a25c72ecda35091d5e0f2fcad73981753c6d6ac3ccac31add98d8bc035e4",
    "utils.py": "534c191895e979755ac191d289246228c10121bd92a160c36bf8d55e5ff85cd0",
}
SEEDS = (42, 43, 44)
RUN_NAME = "Finetune_full-Beauty-0"


def prepare_reference(directory):
    files = {f"data/{name}": (directory / name, checksum) for name, checksum in DATA_HASHES.items()}
    files.update({name: (directory / "upstream" / name, checksum) for name, checksum in UPSTREAM_CODE.items()})
    for remote_name, (path, checksum) in files.items():
        if path.is_file() and sha256_file(path) == checksum:
            continue
        with urlopen(f"{UPSTREAM_URL}/{remote_name}", timeout=45) as response:
            contents = response.read()
        if hashlib.sha256(contents).hexdigest() != checksum:
            raise ValueError(f"Unexpected SHA-256 for {remote_name}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)


def train(directory, archive):
    archive = archive.resolve()
    if archive.exists():
        raise FileExistsError(f"Choose a new path to retain the existing checkpoint archive: {archive}")
    import torch

    if not torch.cuda.is_available():
        raise ValueError("Enable a GPU in the training runtime")
    archive.parent.mkdir(parents=True, exist_ok=True)
    source = directory / "upstream" / "run_finetune_full.py"
    runs = []
    with TemporaryDirectory(prefix="sasrec-", dir=archive.parent) as temporary:
        work = Path(temporary)
        for seed in SEEDS:
            output = work / f"seed_{seed}"
            subprocess.run([
                sys.executable, "-u", str(source), "--data_dir", directory.as_posix() + "/",
                "--output_dir", str(output), "--data_name", "Beauty", "--ckp", "0",
                "--seed", str(seed), "--epochs", "200", "--batch_size", "256",
                "--lr", "0.001", "--max_seq_length", "50",
            ], cwd=source.parent, check=True)
            log = (output / f"{RUN_NAME}.txt").read_text(encoding="utf-8").splitlines()
            if not log or "cuda_condition=True" not in log[0] or not (output / f"{RUN_NAME}.pt").is_file():
                raise ValueError(f"Seed {seed} has no GPU checkpoint")
            test = ast.literal_eval(log[-1])
            scores = {f"{metric}@{k}": float(test[f"{name}@{k}"])
                      for k in (5, 10) for metric, name in (("recall", "HIT"), ("ndcg", "NDCG"))}
            runs.append({"seed": seed, "revision": UPSTREAM_REVISION, "input_files": DATA_HASHES, "test": scores})

        aggregates = {}
        for metric in runs[0]["test"]:
            values = [run["test"][metric] for run in runs]
            aggregates[metric] = {"mean": mean(values), "sample_std": stdev(values)}
        write_json(work / "summary.json", {
            "paper_beauty_sasrec_ndcg@10": 0.0318, "aggregate": aggregates, "runs": runs,
        })
        bundle = work / "outputs.zip"
        with ZipFile(bundle, "w", ZIP_DEFLATED) as output:
            output.write(work / "summary.json", "summary.json")
            for path in sorted(work.glob("seed_*/*")):
                output.write(path, path.relative_to(work).as_posix())
        bundle.replace(archive)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, default=REFERENCE_DIR)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args(argv)
    directory = args.reference_dir.resolve()
    try:
        prepare_reference(directory)
        if not args.prepare_only:
            train(directory, args.archive or directory / "sasrec_paper_beauty_outputs.zip")
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
