"""Chạy code gốc RPG_KDD2025 (không sửa) với cấu hình của nhóm; mọi output ghi ra một thư mục gốc.

Dùng trên Colab (code gốc clone ở ``/content/RPG_KDD2025`` tại commit 7dcf95c, output trên Google Drive)::

    python scripts/run_rpg.py --seed=42                                  # cấu hình chính -> run_id main_s42
    python scripts/run_rpg.py --sent_emb_model=x/pq --seed=42            # adapter chế độ (b) -> run_id pq_s42
    python scripts/run_rpg.py --sent_emb_model=x/behav --seed=42         # chế độ (a); số chiều lấy từ behav.adapter.json
    python scripts/run_rpg.py --checkpoint=<root>/ckpt/main_s42/<file>.pth   # chỉ chấm test -> run_id main_s42_salvage

Khác lệnh trong README của code gốc ở 3 điểm:

1. Encoder: config mặc định của code gốc dùng ``text-embedding-3-large`` (OpenAI, cần API trả phí);
   nhóm dùng ``sentence-t5-base`` 768 chiều với PCA 128 như dòng đang comment trong
   ``genrec/models/RPG/config.yaml``. Mốc so sánh là NDCG@10 = 0,0429 (Table 4 của paper).
   Siêu tham số khớp ``configs/rpg_main.yaml`` (có test kiểm tra).
2. Tên file checkpoint: code gốc ghép toàn bộ tham số dòng lệnh vào tên file; với các tham số trên
   tên dài khoảng 276 ký tự, vượt giới hạn 255. Script truyền config bằng Python và rút gọn
   ``sys.argv`` trước khi gọi code gốc.
3. Output: cache, checkpoint, log và kết quả ghi vào ``--root`` (mặc định Drive) để không mất khi
   runtime Colab ngắt. Mỗi lần chạy ghi ``results/<run_id>.json`` và thêm một dòng ``results/log.csv``.

``run_id`` mặc định theo encoder và seed (``main_s42``, ``pq_s42``...), nên các biến thể không ghi
đè kết quả của nhau. Nếu ``results/<run_id>.json`` đã có, script dừng ngay trước khi train
(thêm ``--overwrite_results`` để chạy lại).

Adapter (``adapters/load_ids.py``) ghi file của nhóm vào cache; ``--sent_emb_model=x/<tên>`` khiến
code gốc đọc file đó thay cho file của nó. Trước khi train, script kiểm tra file adapter có tồn tại,
đủ item, và (chế độ a) kích thước khớp ``--sent_emb_dim``.

Chế độ ``--checkpoint`` chỉ chấm test: code gốc trả ``best_epoch`` và điểm validation rỗng
(``trainer.py`` dòng 75–78), nên script lấy hai giá trị này từ ``results/<run gốc>.json`` nếu có
(run gốc = tên thư mục chứa checkpoint) và ghi ``source_run``.

Chưa làm: xuất top-K theo từng user để chấm bằng ``src/evaluate.py`` và paired t-test (Tuần 4).

Phụ trách: A.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Sequence

DEFAULT_REPO = os.environ.get("RPG_KDD2025_DIR", "/content/RPG_KDD2025")
DEFAULT_ROOT = os.environ.get("RPG_ROOT", "/content/drive/MyDrive/rpg")
DEFAULT_SENT_EMB_MODEL = "sentence-transformers/sentence-t5-base"
DEFAULT_SENT_EMB_DIM = 768
PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_COLUMNS = ["run_id", "seed", "gpu_hours", "best_epoch", "val_ndcg10",
               "test_recall5", "test_recall10", "test_ndcg5", "test_ndcg10", "finished_at"]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python scripts/run_rpg.py", description=__doc__.split("\n")[0])
    p.add_argument("--repo", default=DEFAULT_REPO, help="thư mục code gốc RPG_KDD2025 (commit 7dcf95c)")
    p.add_argument("--root", default=DEFAULT_ROOT, help="thư mục gốc chứa cache, ckpt, logs, results")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--run_id", default=None,
                   help="mặc định main_s<seed>, <tên>_s<seed> khi dùng adapter, <run gốc>_salvage khi --checkpoint")
    p.add_argument("--cache_name", default="cache", help="thư mục cache bên trong --root")
    p.add_argument("--sent_emb_model", default=DEFAULT_SENT_EMB_MODEL,
                   help="adapter: 'x/<tên>' để code gốc đọc file <tên>.sent_emb hoặc <tên>_OPQ32,IVF1,PQ32x8.sem_ids")
    p.add_argument("--sent_emb_dim", type=int, default=None,
                   help="số chiều embedding; mặc định 768, hoặc lấy từ <tên>.adapter.json ở chế độ (a)")
    p.add_argument("--checkpoint", default=None, help="chỉ chấm test từ checkpoint có sẵn, không train")
    p.add_argument("--overwrite_results", action="store_true", help="cho phép ghi đè results/<run_id>.json")
    return p.parse_args(argv)


def adapter_name(args: argparse.Namespace) -> str | None:
    """Tên file adapter nếu đang dùng adapter (``x/<tên>``), ngược lại None."""
    if args.sent_emb_model == DEFAULT_SENT_EMB_MODEL:
        return None
    return os.path.basename(args.sent_emb_model)


def default_run_id(args: argparse.Namespace) -> str:
    if args.run_id:
        return args.run_id
    if args.checkpoint:
        return f"{Path(args.checkpoint).parent.name}_salvage"
    name = adapter_name(args)
    return f"main_s{args.seed}" if name is None else f"{name}_s{args.seed}"


def build_config(args: argparse.Namespace) -> dict:
    """Config truyền cho ``genrec.pipeline.Pipeline`` của code gốc (tên khóa đúng như code gốc dùng)."""
    run_id = default_run_id(args)
    root = args.root
    cfg = {
        # cấu hình Beauty theo README của code gốc (khớp configs/rpg_main.yaml)
        "category": "Beauty", "lr": 0.01, "temperature": 0.03,
        "n_codebook": 32, "codebook_size": 256,
        "num_beams": 20, "n_edges": 200, "propagation_steps": 3,
        # encoder mở thay cho OpenAI (dòng comment trong genrec/models/RPG/config.yaml)
        "sent_emb_model": args.sent_emb_model,
        "sent_emb_dim": args.sent_emb_dim or DEFAULT_SENT_EMB_DIM, "sent_emb_pca": 128,
        "rand_seed": args.seed, "epochs": args.epochs, "run_id": run_id,
        # mọi output vào --root
        "cache_dir": f"{root}/{args.cache_name}/",
        "ckpt_dir": f"{root}/ckpt/{run_id}/",
        "log_dir": f"{root}/logs/",
        "tensorboard_log_dir": f"{root}/tb/",
        "faiss_omp_num_threads": os.cpu_count() or 2,
    }
    if args.checkpoint:
        cfg["steps"] = 0  # code gốc bỏ qua train khi steps = 0, chỉ chấm test
    return cfg


def results_path(root: str | Path, run_id: str) -> Path:
    return Path(root) / "results" / f"{run_id}.json"


def check_results_free(root: str | Path, run_id: str, overwrite: bool = False) -> None:
    """Dừng sớm nếu run_id đã có kết quả, để không train xong mới phát hiện ghi đè."""
    path = results_path(root, run_id)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} đã tồn tại; đặt --run_id khác hoặc thêm --overwrite_results")


def check_adapter(args: argparse.Namespace) -> dict | None:
    """Kiểm tra file adapter trước khi train; điền ``args.sent_emb_dim`` cho chế độ (a).

    Trả về None nếu không dùng adapter.
    """
    name = adapter_name(args)
    if name is None:
        return None
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from adapters.load_ids import processed_dir, resolve_run

    info = resolve_run(processed_dir(Path(args.root) / args.cache_name), name, args.sent_emb_dim)
    if info["mode"] == "embeddings":
        args.sent_emb_dim = info["sent_emb_dim"]
    return info


def source_run_metrics(root: str | Path, checkpoint: str) -> dict:
    """best_epoch và điểm validation của run đã tạo checkpoint (run = thư mục cha của file .pth)."""
    source = Path(checkpoint).parent.name
    path = results_path(root, source)
    if not path.exists():
        return {"source_run": source}
    src = json.loads(path.read_text(encoding="utf-8"))
    return {"source_run": source, "best_epoch": src.get("best_epoch"),
            "best_val_ndcg10": src.get("best_val_ndcg10")}


def patch_torchvision_videoreader() -> bool:
    """Vá tương thích Colab (14/11/2026): ``datasets`` 4.0.0 import ``torchvision.io.VideoReader`` khi
    tensor hóa batch, nhưng torchvision mới đã bỏ class này nên báo ImportError ở batch đầu tiên.
    Thêm một class rỗng: ``isinstance(x, VideoReader)`` luôn False nên dữ liệu vẫn tensor hóa bình thường.
    Trả về True nếu đã vá.
    """
    try:
        import torchvision.io as tvio
    except ImportError:
        return False
    if hasattr(tvio, "VideoReader"):
        return False

    class _NoVideoReader:
        pass

    tvio.VideoReader = _NoVideoReader
    return True


def write_results(out: dict, root: str | Path, overwrite: bool = False) -> Path:
    """Ghi ``results/<run_id>.json`` (từ chối ghi đè nếu thiếu ``overwrite``) và thêm một dòng ``results/log.csv``."""
    check_results_free(root, out["run_id"], overwrite)
    path = results_path(root, out["run_id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")

    log_path = path.parent / "log.csv"
    new = not log_path.exists()
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(LOG_COLUMNS)
        t = out["test"]
        w.writerow([out["run_id"], out["seed"], out["gpu_hours"], out["best_epoch"], out["best_val_ndcg10"],
                    t.get("recall@5"), t.get("recall@10"), t.get("ndcg@5"), t.get("ndcg@10"), out["finished_at"]])
    return path


def main(argv: Sequence[str] | None = None) -> dict:
    args = parse_args(argv)
    run_id = default_run_id(args)
    check_results_free(args.root, run_id, args.overwrite_results)
    info = check_adapter(args)
    if info:
        print(f"[run_rpg] Adapter {adapter_name(args)}: chế độ {info['mode']}, sent_emb_dim={args.sent_emb_dim}")
    cfg = build_config(args)

    sys.path.insert(0, args.repo)
    sys.argv = [os.path.basename(sys.argv[0]) or "run_rpg.py"]  # giữ tên file checkpoint < 255 ký tự
    if patch_torchvision_videoreader():
        print("[run_rpg] Đã vá torchvision.io.VideoReader")
    from genrec.pipeline import Pipeline

    t0 = time.time()
    pipe = Pipeline(model_name="RPG", dataset_name="AmazonReviews2014",
                    checkpoint_path=args.checkpoint, config_dict=cfg)
    res = pipe.run()
    out = {
        "run_id": run_id, "seed": args.seed, "gpu_hours": round((time.time() - t0) / 3600, 2),
        "best_epoch": res["best_epoch"], "best_val_ndcg10": res["best_val_score"],
        "test": dict(res["test_results"]),
        "sent_emb_model": args.sent_emb_model,
        "checkpoint": args.checkpoint or pipe.trainer.saved_model_ckpt,
        "finished_at": time.strftime("%Y-%m-%d %H:%M"),
    }
    if args.checkpoint:
        out["mode"] = "evaluate_only"
        for key, value in source_run_metrics(args.root, args.checkpoint).items():
            if key == "source_run" or out.get(key) is None:
                out[key] = value
    write_results(out, args.root, args.overwrite_results)
    print("DONE", json.dumps(out, default=str))
    return out


if __name__ == "__main__":
    main()
