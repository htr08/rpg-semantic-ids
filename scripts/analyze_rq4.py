"""Tổng hợp đánh giá RQ4 và đo mức xuất hiện của hub trong top-10 (thành viên B)."""

import argparse
import json
import math
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import yaml
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from scripts.paired_t_test import load_scores, paired_t_test
from src.evaluate import (
    COLD_START_GROUPS, build_evaluation_context, category_diversity, load_rankings, load_split,
)
from src.preprocessing import ROOT, read_records, sha256_file, verify_preprocessed, write_json

METRICS = (
    "recall@5", "recall@10", "ndcg@5", "ndcg@10",
    "coverage@10", "long_tail_ratio@10", "category_diversity@10",
)


def load_model(runs, user_ids, inputs):
    """Kiểm tra thí nghiệm chấm cùng dữ liệu và gộp điểm mỗi user qua các seed."""
    directories = [ROOT / path for path in runs.values()]
    signature = ("test", inputs["split"], inputs["manifest"], False)
    by_metric = {metric: [] for metric in METRICS}
    provenance = {}
    for run, directory in zip(runs, directories, strict=True):
        report_path = directory / "metrics.json"
        report = json.loads(report_path.read_bytes())
        if (any(report["inputs"][key]["sha256"] != digest for key, digest in inputs.items())
                or set(report["protocol"]["ks"]) != {5, 10}):
            raise ValueError(f"{directory}: bộ dữ liệu hoặc K lệch protocol")
        rankings_path = directory / "rankings.jsonl"
        rankings_digest = sha256_file(rankings_path)
        if rankings_digest != report["inputs"]["rankings"]["sha256"]:
            raise ValueError(f"{directory}: ranking không khớp báo cáo đã chấm")
        for key in METRICS:
            by_metric[key].append(report["metrics"][key])
        provenance[str(run)] = {
            "directory": str(runs[run]),
            "metrics_sha256": sha256_file(report_path),
            "rankings_sha256": rankings_digest,
            "per_user_sha256": sha256_file(directory / "per_user.csv"),
        }
    scores, _ = load_scores(directories, signature)
    if scores.keys() != set(user_ids):
        raise ValueError("Danh sách user đã chấm không khớp split test")
    summary = {
        key: {"mean": mean(values), "sample_std": stdev(values) if len(values) > 1 else None}
        for key, values in by_metric.items()
    }
    if not math.isclose(mean(scores.values()), summary["ndcg@10"]["mean"], abs_tol=1e-12):
        raise ValueError("NDCG@10 theo user không khớp số tổng hợp")
    return {"metrics": summary, "runs": provenance}, scores


def segment_users(histories, context, *, n_clusters, seed):
    """Phân cụm chỉ từ train, sắp cụm theo độ dài lịch sử trung bình."""
    features = np.array([
        (len(history), category_diversity(history, context.item_categories),
         sum(item in context.head_items for item in history) / len(history))
        for history in histories
    ], dtype=float)
    missing = np.isnan(features[:, 1])
    if missing.all():
        raise ValueError("Cần lịch sử có ít nhất hai item có danh mục để tính đa dạng")
    imputed = float(np.nanmedian(features[:, 1]))
    features[missing, 1] = imputed
    transformed = features.copy()
    transformed[:, 0] = np.log1p(transformed[:, 0])
    standardized = StandardScaler().fit_transform(transformed)
    labels = KMeans(n_clusters=n_clusters, n_init=20, random_state=seed).fit_predict(standardized)
    if len(set(labels)) != n_clusters:
        raise ValueError("Đặc trưng train không tạo đủ số phân khúc yêu cầu")
    groups = sorted((labels == index for index in range(n_clusters)),
                    key=lambda mask: features[mask, 0].mean())
    profiles = []
    for index, mask in enumerate(groups, 1):
        length, diversity, head_share = features[mask].mean(axis=0)
        profiles.append({
            "segment": index,
            "mean_history_length": float(length),
            "mean_category_diversity": float(diversity),
            "mean_head_share": float(head_share),
        })
    method = {"n_clusters": n_clusters, "seed": seed, "n_init": 20,
              "users_with_imputed_diversity": int(missing.sum()), "imputed_diversity": imputed}
    return groups, profiles, method


def group_scores(mask, scores):
    """Trung bình NDCG@10 của các user trong nhóm, với cùng trọng số mỗi user."""
    return {
        "users": int(mask.sum()),
        "ndcg@10": {name: float(values[mask].mean()) if mask.any() else None for name, values in scores.items()},
    }


def graph_indegree(adjacency, n_items):
    """Bậc vào của các item thật sau khi bỏ padding và cạnh tự nối."""
    if (adjacency.ndim != 2 or adjacency.shape[0] != n_items or not adjacency.shape[1]
            or adjacency.dtype.kind not in "iu" or adjacency.min() < 0 or adjacency.max() > n_items):
        raise ValueError("Adjacency phải dùng item ID đúng catalog")
    items = np.arange(1, n_items + 1)
    neighbors = adjacency[(adjacency > 0) & (adjacency != items[:, None])]
    return np.bincount(neighbors.astype(np.int64, copy=False), minlength=n_items + 1)[1:]


def exposure(indegree, rankings):
    """So bậc vào và tần suất top-10 trên toàn catalog, kể cả item chưa được gợi ý."""
    n_items = len(indegree)
    items = np.arange(1, n_items + 1)
    if any(item < 1 or item > n_items for row in rankings.values() for item in row[:10]):
        raise ValueError("Ranking có item ngoài catalog")
    counts = np.bincount([item for row in rankings.values() for item in row[:10]], minlength=n_items + 1)[1:]
    hubs = np.lexsort((items, -indegree))[:20]
    slots = int(counts.sum())
    rho = spearmanr(indegree, counts) if np.ptp(indegree) and np.ptp(counts) else None
    result = {
        "spearman": float(rho.statistic) if rho is not None else None,
        "p_value": float(rho.pvalue) if rho is not None and n_items >= 3 else None,
        "top10_slots": slots,
        "item_coverage": int(np.count_nonzero(counts)),
        "hub_slots": int(counts[hubs].sum()),
        "hub_share": float(counts[hubs].sum() / slots) if slots else None,
    }
    return result, counts


def analyze(config, processed_dir, splits_dir):
    verify_preprocessed(processed_dir, splits_dir)
    train_path = processed_dir / "train_sequences.jsonl"
    items_path = processed_dir / "items.jsonl"
    test_path = processed_dir / "test.jsonl"
    inputs = {
        "split": sha256_file(test_path), "manifest": sha256_file(splits_dir / "manifest.json"),
        "train": sha256_file(train_path), "items": sha256_file(items_path),
    }
    split = load_split(test_path)
    users = sorted(split)
    train = sorted((row for _, row in read_records(train_path)), key=lambda row: row["user_id"])
    context = build_evaluation_context(train, (row for _, row in read_records(items_path)))
    if [row["user_id"] for row in train] != users:
        raise ValueError("Tập user train và test không khớp")
    groups, profiles, method = segment_users([row["item_ids"] for row in train], context, **config["segments"])

    model_reports, user_scores = {}, {}
    for name, spec in config["models"].items():
        model, scores = load_model(spec["runs"], users, inputs)
        model_reports[name] = {"label": spec["label"], **model}
        user_scores[name] = scores
    scores = {name: np.array([values[user] for user in users]) for name, values in user_scores.items()}
    segments = [{**profile, **group_scores(mask, scores)} for profile, mask in zip(profiles, groups, strict=True)]
    frequencies = np.array([context.train_counts[split[user]["target"]] for user in users])
    cold_start = [
        {"min": low, "max": high, **group_scores((low <= frequencies) & (frequencies <= high), scores)}
        for low, high in COLD_START_GROUPS
    ]
    reference = config["reference_model"]
    paired = {
        name: paired_t_test(values, user_scores[reference])
        for name, values in user_scores.items() if name != reference
    }

    graph = config["hub_exposure"]
    adjacency_path = ROOT / graph["adjacency"]
    if sha256_file(adjacency_path) != graph["sha256"]:
        raise ValueError("Adjacency không khớp checksum của đồ thị tham chiếu")
    n_items = len(context.catalog)
    indegree = graph_indegree(np.load(adjacency_path, allow_pickle=False), n_items)
    hubs = np.lexsort((np.arange(1, n_items + 1), -indegree))[:20]
    correlations, exposure_counts = {}, {}
    for name, seed in graph["runs"].items():
        rankings = load_rankings(ROOT / config["models"][name]["runs"][seed] / "rankings.jsonl")
        if rankings.keys() != split.keys():
            raise ValueError(f"Ranking hub của {name} không khớp user test")
        result, counts = exposure(indegree, rankings)
        correlations[name] = {"seed": seed, **result}
        exposure_counts[name] = counts

    report = {
        "protocol": {"split": "test", "ks": [5, 10], "exclude_seen": False, "users": len(users),
                     "catalog_items": n_items, "inputs_sha256": inputs},
        "models": model_reports,
        "cold_start": {"groups": cold_start, "users_above_20": int((frequencies > COLD_START_GROUPS[-1][1]).sum())},
        "segmentation": {"method": method, "groups": segments},
        "paired_t_test": {"metric": "ndcg@10", "reference": reference, "comparisons": paired},
        "hub_exposure": {
            "adjacency": graph["adjacency"], "adjacency_sha256": graph["sha256"], "hubs": len(hubs),
            "hub_catalog_fraction": len(hubs) / n_items,
            "runs": correlations,
            "top_hubs": [{"item_id": int(i + 1), "in_degree": int(indegree[i]),
                          "top10_count": {name: int(values[i]) for name, values in exposure_counts.items()}} for i in hubs],
        },
    }
    return report, indegree, exposure_counts


def plot_results(output, report, indegree, exposure_counts):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.ticker import PercentFormatter

    output.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    cold = report["cold_start"]["groups"]
    segments = report["segmentation"]["groups"]
    models = report["models"]
    axes[0].set(xlabel="Tần suất tương tác train của item đích", ylabel="NDCG@10",
                title="Cold-start theo item")
    axes[1].set(xlabel="Phân khúc theo lịch sử train", title="NDCG theo phân khúc")
    for axis, groups, labels in (
        (axes[0], cold, [f"{g['min']}–{g['max']}\n(n={g['users']:,})" for g in cold]),
        (axes[1], segments, [f"Nhóm {g['segment']}\n(n={g['users']:,})" for g in segments]),
    ):
        for name, model in models.items():
            axis.plot(range(len(groups)), [g["ndcg@10"][name] for g in groups], marker="o", label=model["label"])
        axis.set_xticks(range(len(groups)), labels)
        axis.yaxis.set_major_formatter(PercentFormatter(1))
        axis.set_ylim(bottom=0)
        axis.grid(axis="y", alpha=.22)
    figure.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncols=3, frameon=False)
    figure.savefig(output / "rq4_ndcg.png", dpi=150)
    plt.close(figure)

    figure, axes = plt.subplots(1, len(exposure_counts), figsize=(10.5, 3.9), layout="constrained", squeeze=False)
    for axis, (name, values) in zip(axes[0], exposure_counts.items(), strict=True):
        run = report["hub_exposure"]["runs"][name]
        rho = f"{run['spearman']:.3f}" if run["spearman"] is not None else "—"
        axis.scatter(indegree, np.log10(1 + values), s=7, alpha=.22, color="#24639f", linewidths=0)
        axis.set(xlabel="Bậc vào đồ thị tham chiếu", ylabel="log₁₀(1 + số lần xuất hiện trong top-10)",
                 title=f"{models[name]['label']} seed {run['seed']}  ·  ρ = {rho}")
        axis.grid(axis="y", alpha=.22)
    figure.savefig(output / "rpg_hub_exposure.png", dpi=150)
    plt.close(figure)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "rq4_analysis.yaml")
    parser.add_argument("--processed-dir", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--splits-dir", type=Path, default=ROOT / "data" / "splits")
    parser.add_argument("--output", type=Path, required=True, help="Báo cáo JSON")
    parser.add_argument("--figures-dir", type=Path, help="Thư mục biểu đồ")
    args = parser.parse_args(argv)
    try:
        config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        report, indegree, exposure_counts = analyze(config, args.processed_dir, args.splits_dir)
        report["protocol"]["config_sha256"] = sha256_file(args.config)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, report)
        if args.figures_dir:
            plot_results(args.figures_dir, report, indegree, exposure_counts)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Lỗi: {error}\n")
    print(f"Đã ghi {args.output}")
    return 0


if __name__ == "__main__":
    main()
