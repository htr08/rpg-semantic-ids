"""Unit test cho scripts/run_rpg.py, không cần torch hay code gốc (thay bằng module giả)."""
import csv
import json
import sys
import types

import pytest

from scripts import run_rpg as R


def test_default_config_matches_team_setup():
    cfg = R.build_config(R.parse_args(["--root", "/r"]))
    assert cfg == {
        "category": "Beauty", "lr": 0.01, "temperature": 0.03,
        "n_codebook": 32, "codebook_size": 256,
        "num_beams": 20, "n_edges": 200, "propagation_steps": 3,
        "sent_emb_model": "sentence-transformers/sentence-t5-base",
        "sent_emb_dim": 768, "sent_emb_pca": 128,
        "rand_seed": 42, "epochs": 150, "run_id": "main_s42",
        "cache_dir": "/r/cache/", "ckpt_dir": "/r/ckpt/main_s42/",
        "log_dir": "/r/logs/", "tensorboard_log_dir": "/r/tb/",
        "faiss_omp_num_threads": cfg["faiss_omp_num_threads"],
    }


def test_adapter_and_salvage_options():
    cfg = R.build_config(R.parse_args(["--root", "/r", "--seed", "43", "--sent_emb_model", "x/behav",
                                       "--sent_emb_dim", "1024", "--run_id", "behav_s43",
                                       "--checkpoint", "/r/ckpt/x.pth"]))
    assert cfg["sent_emb_model"] == "x/behav" and cfg["sent_emb_dim"] == 1024
    assert cfg["rand_seed"] == 43 and cfg["run_id"] == "behav_s43" and cfg["ckpt_dir"] == "/r/ckpt/behav_s43/"
    assert cfg["steps"] == 0                                          # chỉ chấm test
    assert "steps" not in R.build_config(R.parse_args([]))


def fake_torchvision(monkeypatch, with_reader):
    tv, io = types.ModuleType("torchvision"), types.ModuleType("torchvision.io")
    if with_reader:
        io.VideoReader = object
    tv.io = io
    monkeypatch.setitem(sys.modules, "torchvision", tv)
    monkeypatch.setitem(sys.modules, "torchvision.io", io)
    return io


def test_patch_adds_videoreader_when_missing(monkeypatch):
    io = fake_torchvision(monkeypatch, with_reader=False)
    assert R.patch_torchvision_videoreader() is True
    assert not isinstance([1, 2], io.VideoReader)


def test_patch_keeps_existing_videoreader(monkeypatch):
    io = fake_torchvision(monkeypatch, with_reader=True)
    assert R.patch_torchvision_videoreader() is False
    assert io.VideoReader is object


def test_write_results_appends_one_header(tmp_path):
    out = {"run_id": "a", "seed": 42, "gpu_hours": 1.0, "best_epoch": 3, "best_val_ndcg10": 0.05,
           "test": {"recall@5": 0.1, "recall@10": 0.2, "ndcg@5": 0.3, "ndcg@10": 0.4}, "finished_at": "t"}
    R.write_results(out, tmp_path)
    R.write_results({**out, "run_id": "b"}, tmp_path)
    rows = list(csv.reader(open(tmp_path / "results" / "log.csv", encoding="utf-8")))
    assert rows[0] == R.LOG_COLUMNS and [r[0] for r in rows[1:]] == ["a", "b"]
    assert json.loads((tmp_path / "results" / "a.json").read_text())["test"]["ndcg@10"] == 0.4


def test_main_calls_code_goc_with_short_argv(tmp_path, monkeypatch):
    seen = {}

    class FakePipeline:
        def __init__(self, model_name, dataset_name, checkpoint_path, config_dict):
            seen.update(model=model_name, dataset=dataset_name, cfg=config_dict, argv=list(sys.argv))
            self.trainer = types.SimpleNamespace(saved_model_ckpt="ckpt.pth")

        def run(self):
            return {"best_epoch": 7, "best_val_score": 0.05,
                    "test_results": {"recall@5": 0.1, "recall@10": 0.2, "ndcg@5": 0.03, "ndcg@10": 0.04}}

    genrec, pipeline = types.ModuleType("genrec"), types.ModuleType("genrec.pipeline")
    pipeline.Pipeline = FakePipeline
    monkeypatch.setitem(sys.modules, "genrec", genrec)
    monkeypatch.setitem(sys.modules, "genrec.pipeline", pipeline)
    monkeypatch.setattr(sys, "argv", ["/content/x/run_rpg.py", "--a-very-long-argument=" + "y" * 300])
    monkeypatch.setattr(sys, "path", list(sys.path))

    out = R.main(["--root", str(tmp_path), "--repo", str(tmp_path), "--sent_emb_model", "x/pq",
                  "--run_id", "pq_s42"])
    assert seen["argv"] == ["run_rpg.py"]                              # tên checkpoint ngắn
    assert seen["model"] == "RPG" and seen["dataset"] == "AmazonReviews2014"
    assert seen["cfg"]["sent_emb_model"] == "x/pq"
    assert out["checkpoint"] == "ckpt.pth" and out["best_epoch"] == 7
    assert (tmp_path / "results" / "pq_s42.json").exists()
