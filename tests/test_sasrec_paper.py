import hashlib
import json
import sys
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import pytest

from scripts import reproduce_sasrec_paper as reproduction
from scripts.evaluate_sasrec_paper import match_item_ids


def test_match_item_ids_accepts_remapping_and_reordered_users():
    reference = [[1, 2, 1], [2, 3], [3, 1, 2]]
    local = [[30, 10, 20], [10, 20, 10], [20, 30]]
    assert match_item_ids(reference, local) == {10: 1, 20: 2, 30: 3}


def test_match_item_ids_rejects_ambiguous_signatures():
    with pytest.raises(ValueError, match="ambiguous"):
        match_item_ids([[1, 2], [2, 1]], [[3, 4], [4, 3]])


def test_match_item_ids_requires_complete_sequence_agreement():
    # Swapping equal-length suffixes preserves signatures but changes the sequences.
    reference = [[1, 2, 5], [3, 4, 6], [1, 5], [2, 5, 1, 4]]
    local = [[1, 2, 6], [3, 4, 5], [1, 5], [2, 5, 1, 4]]
    with pytest.raises(ValueError, match="sequences differ after remapping"):
        match_item_ids(reference, local)


def test_reference_download_rejects_wrong_bytes_before_write(monkeypatch, tmp_path):
    expected = b"reference bytes"
    monkeypatch.setattr(reproduction, "DATA_HASHES", {"Beauty.txt": hashlib.sha256(expected).hexdigest()})
    monkeypatch.setattr(reproduction, "UPSTREAM_CODE", {})

    def download(url, *, timeout):
        assert url == f"{reproduction.UPSTREAM_URL}/data/Beauty.txt"
        return BytesIO(b"unexpected")

    monkeypatch.setattr(reproduction, "urlopen", download)
    with pytest.raises(ValueError, match="Unexpected SHA-256"):
        reproduction.prepare_reference(tmp_path)
    assert not (tmp_path / "Beauty.txt").exists()


def test_training_packages_checkpoints_for_all_seeds(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True)))

    def run(command, *, cwd, check):
        assert check
        assert command[command.index("--ckp") + 1] == "0"
        seed = int(command[command.index("--seed") + 1])
        output = Path(command[command.index("--output_dir") + 1])
        assert output.name == f"seed_{seed}"
        output.mkdir()
        (output / f"{reproduction.RUN_NAME}.pt").write_bytes(bytes([seed]))
        (output / f"{reproduction.RUN_NAME}.txt").write_text(
            "Namespace(cuda_condition=True)\n"
            "{'HIT@5': '0.01', 'NDCG@5': '0.005', 'HIT@10': '0.04', 'NDCG@10': '0.02'}\n",
            encoding="utf-8",
        )

    monkeypatch.setattr(reproduction.subprocess, "run", run)
    archive = tmp_path / "results.zip"
    reproduction.train(tmp_path / "reference", archive)
    with ZipFile(archive) as bundle:
        summary = json.loads(bundle.read("summary.json"))
        assert tuple(row["seed"] for row in summary["runs"]) == reproduction.SEEDS
        assert summary["aggregate"]["ndcg@10"] == {"mean": 0.02, "sample_std": 0.0}
        for seed in reproduction.SEEDS:
            assert bundle.read(f"seed_{seed}/{reproduction.RUN_NAME}.pt") == bytes([seed])
