"""Unit test cho scripts/run_rpg.py, không cần torch hay code gốc (thay bằng module giả)."""
import csv
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import yaml

from adapters import load_ids as A
from scripts import run_rpg as R

ROOT_DIR = Path(__file__).resolve().parents[1]
ASINS = ["B001", "B002", "B003"]


# --- Cấu hình ---

def test_config_matches_rpg_main_yaml():
    # Đối chiếu với file cấu hình đã ghi trong log cho main_s42-44, không lặp lại hằng số trong test.
    spec = yaml.safe_load((ROOT_DIR / "configs" / "rpg_main.yaml").read_text(encoding="utf-8"))
    cfg = R.build_config(R.parse_args(["--root", "/r"]))
    assert cfg["lr"] == spec["train"]["lr"]
    assert cfg["temperature"] == spec["model"]["temperature"]
    assert cfg["n_codebook"] == spec["semantic_id"]["num_subspaces"]
    assert cfg["codebook_size"] == spec["semantic_id"]["codebook_size"]
    assert cfg["sent_emb_pca"] == spec["semantic_id"]["pca_dim"]
    assert cfg["sent_emb_model"].endswith(spec["encoder"])
    for key in ("num_beams", "n_edges", "propagation_steps"):
        assert cfg[key] == spec["decoding"][key]
    assert cfg["rand_seed"] in spec["train"]["seeds"]
    assert cfg["category"] == "Beauty" and cfg["sent_emb_dim"] == 768 and "steps" not in cfg


def test_outputs_go_under_root():
    cfg = R.build_config(R.parse_args(["--root", "/r"]))
    assert (cfg["cache_dir"], cfg["ckpt_dir"], cfg["log_dir"], cfg["tensorboard_log_dir"]) == \
        ("/r/cache/", "/r/ckpt/main_s42/", "/r/logs/", "/r/tb/")


@pytest.mark.parametrize("argv, expected", [
    ([], "main_s42"),
    (["--seed", "43"], "main_s43"),
    (["--sent_emb_model", "x/pq", "--seed", "43"], "pq_s43"),              # biến thể không đè main_s43
    (["--checkpoint", "/r/ckpt/main_s42/a.pth"], "main_s42_salvage"),
    (["--sent_emb_model", "x/pq", "--run_id", "custom"], "custom"),
])
def test_default_run_id(argv, expected):
    args = R.parse_args(argv)
    assert R.default_run_id(args) == expected
    assert R.build_config(args)["run_id"] == expected


def test_checkpoint_mode_sets_zero_steps():
    cfg = R.build_config(R.parse_args(["--checkpoint", "/r/ckpt/main_s42/a.pth"]))
    assert cfg["steps"] == 0


# --- Vá VideoReader ---

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


# --- Ghi kết quả ---

def result(run_id="a"):
    return {"run_id": run_id, "seed": 42, "gpu_hours": 1.0, "best_epoch": 3, "best_val_ndcg10": 0.05,
            "test": {"recall@5": 0.1, "recall@10": 0.2, "ndcg@5": 0.3, "ndcg@10": 0.4}, "finished_at": "t"}


def test_write_results_appends_one_header(tmp_path):
    R.write_results(result("a"), tmp_path)
    R.write_results(result("b"), tmp_path)
    rows = list(csv.reader(open(tmp_path / "results" / "log.csv", encoding="utf-8")))
    assert rows[0] == R.LOG_COLUMNS and [r[0] for r in rows[1:]] == ["a", "b"]
    assert json.loads((tmp_path / "results" / "a.json").read_text())["test"]["ndcg@10"] == 0.4


def test_write_results_refuses_to_overwrite(tmp_path):
    R.write_results(result("a"), tmp_path)
    with pytest.raises(FileExistsError):
        R.write_results(result("a"), tmp_path)
    rows = list(csv.reader(open(tmp_path / "results" / "log.csv", encoding="utf-8")))
    assert len(rows) == 2                                             # không thêm dòng trùng run_id
    R.write_results(result("a"), tmp_path, overwrite=True)


# --- main() với code gốc giả ---

class FakePipeline:
    calls = []
    best_epoch, best_val = 7, 0.05

    def __init__(self, model_name, dataset_name, checkpoint_path, config_dict):
        FakePipeline.calls.append({"model": model_name, "dataset": dataset_name, "cfg": config_dict,
                                   "argv": list(sys.argv), "checkpoint": checkpoint_path})
        self.trainer = types.SimpleNamespace(saved_model_ckpt="ckpt.pth")

    def run(self):
        return {"best_epoch": self.best_epoch, "best_val_score": self.best_val,
                "test_results": {"recall@5": 0.1, "recall@10": 0.2, "ndcg@5": 0.03, "ndcg@10": 0.04}}


@pytest.fixture
def fake_code_goc(monkeypatch):
    FakePipeline.calls = []
    FakePipeline.best_epoch, FakePipeline.best_val = 7, 0.05
    genrec, pipeline = types.ModuleType("genrec"), types.ModuleType("genrec.pipeline")
    pipeline.Pipeline = FakePipeline
    monkeypatch.setitem(sys.modules, "genrec", genrec)
    monkeypatch.setitem(sys.modules, "genrec.pipeline", pipeline)
    monkeypatch.setattr(sys, "argv", ["/content/x/run_rpg.py", "--a-very-long-argument=" + "y" * 300])
    monkeypatch.setattr(sys, "path", list(sys.path))
    return FakePipeline


def make_cache(root):
    proc = A.processed_dir(Path(root) / "cache")
    proc.mkdir(parents=True)
    (proc / "id_mapping.json").write_text(json.dumps({"id2item": ["[PAD]"] + ASINS}))
    return proc


def run(tmp_path, *extra):
    return R.main(["--root", str(tmp_path), "--repo", str(tmp_path), *extra])


def test_main_calls_code_goc_with_short_argv(tmp_path, fake_code_goc):
    out = run(tmp_path)
    call = fake_code_goc.calls[0]
    assert call["argv"] == ["run_rpg.py"]                              # tên checkpoint ngắn
    assert call["model"] == "RPG" and call["dataset"] == "AmazonReviews2014"
    assert out["run_id"] == "main_s42" and out["checkpoint"] == "ckpt.pth"
    assert (tmp_path / "results" / "main_s42.json").exists()


def test_main_stops_before_training_if_run_id_taken(tmp_path, fake_code_goc):
    run(tmp_path)
    with pytest.raises(FileExistsError):
        run(tmp_path)
    assert len(fake_code_goc.calls) == 1                               # lần hai không train


def test_main_adapter_mode_b(tmp_path, fake_code_goc):
    proc = make_cache(tmp_path)
    A.write_semantic_ids(np.zeros((3, 32), dtype=int), proc, "pq")
    out = run(tmp_path, "--sent_emb_model", "x/pq")
    assert out["run_id"] == "pq_s42"
    assert fake_code_goc.calls[0]["cfg"]["sent_emb_model"] == "x/pq"


def test_main_adapter_mode_a_takes_dim_from_sidecar(tmp_path, fake_code_goc):
    proc = make_cache(tmp_path)
    A.write_embeddings(np.ones((3, 1024)), proc, "behav")
    run(tmp_path, "--sent_emb_model", "x/behav")
    assert fake_code_goc.calls[0]["cfg"]["sent_emb_dim"] == 1024


@pytest.mark.parametrize("extra, err", [
    (["--sent_emb_model", "x/behav", "--sent_emb_dim", "512"], ValueError),     # khác dim trong sidecar
    (["--sent_emb_model", "x/missing"], FileNotFoundError),                       # chưa ghi file adapter
])
def test_main_rejects_bad_adapter_before_training(tmp_path, fake_code_goc, extra, err):
    proc = make_cache(tmp_path)
    A.write_embeddings(np.ones((3, 1024)), proc, "behav")
    with pytest.raises(err):
        run(tmp_path, *extra)
    assert fake_code_goc.calls == []


def test_main_mode_a_detects_wrong_size_without_sidecar(tmp_path, fake_code_goc):
    proc = make_cache(tmp_path)
    np.ones((3, 6), dtype=np.float32).tofile(A.sent_emb_path(proc, "raw"))   # 18 số: reshape(-1, 3) vẫn chạy
    with pytest.raises(ValueError):
        run(tmp_path, "--sent_emb_model", "x/raw", "--sent_emb_dim", "3")
    run(tmp_path, "--sent_emb_model", "x/raw", "--sent_emb_dim", "6")
    assert fake_code_goc.calls[0]["cfg"]["sent_emb_dim"] == 6


def test_main_checkpoint_mode_fills_best_epoch_from_source_run(tmp_path, fake_code_goc):
    run(tmp_path)                                                       # tạo results/main_s42.json
    fake_code_goc.best_epoch, fake_code_goc.best_val = None, None       # code gốc trả rỗng khi steps = 0
    out = run(tmp_path, "--checkpoint", str(tmp_path / "ckpt" / "main_s42" / "a.pth"))
    assert out["run_id"] == "main_s42_salvage" and out["mode"] == "evaluate_only"
    assert out["source_run"] == "main_s42" and out["best_epoch"] == 7 and out["best_val_ndcg10"] == 0.05
    assert fake_code_goc.calls[-1]["cfg"]["steps"] == 0
