"""Tự viết (B): EDA chuỗi, long-tail, độ thưa và tình trạng thiếu metadata."""

import argparse
import csv
import html
import json
import math
import random
import re
import sys
from collections import Counter
from pathlib import Path

from src.preprocessing import ROOT, sha256_file, verify_preprocessed, write_json


def read_jsonl(path):
    """Đọc đầu ra JSONL đã kiểm chứng bằng manifest."""
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def clean_text(value):
    """Chuẩn hóa để đo độ dài/vẽ ví dụ; không thay thế cách encode của RPG."""
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return " ".join(filter(None, (clean_text(part) for part in value)))
    text = html.unescape(str(value))
    return " ".join(re.sub(r"<[^>]*>", " ", text).split())


def describe(values):
    """Thống kê với phân vị nội suy tuyến tính; chấp nhận nhóm rỗng."""
    ordered = sorted(values)
    if not ordered:
        return {"count": 0, "min": None, "median": None, "p90": None, "p95": None, "max": None, "mean": None}

    def percentile(fraction):
        position = (len(ordered) - 1) * fraction
        low = int(position)
        high = min(low + 1, len(ordered) - 1)
        return ordered[low] + (ordered[high] - ordered[low]) * (position - low)

    return {
        "count": len(ordered), "min": ordered[0], "median": percentile(0.5),
        "p90": percentile(0.9), "p95": percentile(0.95),
        "max": ordered[-1], "mean": sum(ordered) / len(ordered),
    }


def analyze_data(processed_dir, splits_dir, *, seed=42, head_fraction=0.2):
    """Thống kê EDA tuần 1; độ phổ biến chỉ tính từ train."""
    if not 0 < head_fraction <= 1:
        raise ValueError("head_fraction must be in (0, 1]")
    processed_dir = Path(processed_dir)
    manifest = verify_preprocessed(processed_dir, splits_dir)
    train = read_jsonl(processed_dir / "train_sequences.jsonl")
    items = read_jsonl(processed_dir / "items.jsonl")
    with (processed_dir / "interactions.csv").open(encoding="utf-8", newline="") as stream:
        events = [{key: int(value) for key, value in row.items()} for row in csv.DictReader(stream)]
    catalog = {row["item_id"] for row in items}
    counts = Counter(item for row in train for item in row["item_ids"])
    head_size = math.ceil(len(catalog) * head_fraction)
    head = set(sorted(catalog, key=lambda item: (-counts.get(item, 0), item))[:head_size])
    user_count = len(train)
    pairs = {(row["user_id"], row["item_id"]) for row in events}
    triples = {(row["user_id"], row["item_id"], row["timestamp"]) for row in events}
    timestamp_pairs = {(row["user_id"], row["timestamp"]) for row in events}
    train_lengths = [len(row["item_ids"]) for row in train]
    full_lengths = [length + 2 for length in train_lengths]
    max_history = manifest["max_history"]
    missing_fields = {
        field: sum(not clean_text(row["metadata"].get(field)) for row in items)
        for field in ("title", "brand", "categories", "description")
    }
    missing_both = sum(
        not clean_text(row["metadata"].get("title"))
        and not clean_text(row["metadata"].get("description")) for row in items
    )
    summary = {
        "users": user_count, "items": len(items), "interactions": len(events),
        "train_interactions": sum(train_lengths),
        "validation_targets": manifest["counts"]["validation_targets"],
        "test_targets": manifest["counts"]["test_targets"],
        "unique_user_item_pairs": len(pairs), "duplicate_user_item_events": len(events) - len(pairs),
        "duplicate_user_item_timestamp_events": len(events) - len(triples),
        "tied_user_timestamp_events": len(events) - len(timestamp_pairs),
        "density": len(pairs) / (user_count * len(items)),
        "sparsity": 1 - len(pairs) / (user_count * len(items)),
        "sequence_length_full": describe(full_lengths), "sequence_length_train": describe(train_lengths),
        "history_window": {
            "max_history": max_history,
            "validation_users_truncated": sum(length > max_history for length in train_lengths),
            "test_users_truncated": sum(length + 1 > max_history for length in train_lengths),
        },
        "train_item_frequency": describe([counts.get(item, 0) for item in sorted(catalog)]),
        "head_fraction": head_fraction, "head_items": len(head),
        "head_train_interaction_share": sum(counts[item] for item in head) / sum(counts.values()),
        "metadata_missing_items": sum(row["metadata_missing"] for row in items),
        "missing_fields": missing_fields,
        "missing_field_ratios": {field: count / len(items) for field, count in missing_fields.items()},
        "missing_title_and_description": missing_both,
        "missing_description_with_title": missing_fields["description"] - missing_both,
    }
    return {
        "summary": summary, "source_manifest": manifest,
        "full_lengths": full_lengths, "train_lengths": train_lengths,
        "train_counts": {item: counts.get(item, 0) for item in sorted(catalog)},
        "examples": random.Random(seed).sample(items, min(4, len(items))),
    }

def plot_eda(analysis, figures_dir, *, dpi=160):
    """Xuất hai hình EDA tuần 1 bằng backend Agg."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    def save(fig, name):
        fig.tight_layout()
        path = figures_dir / name
        fig.savefig(path, dpi=dpi, facecolor="white")
        plt.close(fig)
        paths.append(path)

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, values, title in zip(axes, (analysis["full_lengths"], analysis["train_lengths"]), ("Full sequences", "Train sequences (untruncated)")):
        ax.hist(values, bins=range(min(values), max(values) + 2), color="#2563eb", alpha=0.85)
        ax.set(xlabel="Interactions per user", ylabel="Users (log scale)", title=title, yscale="log")
    save(fig, "eda_sequence_lengths.png")

    frequencies = np.array(sorted(analysis["train_counts"].values(), reverse=True), dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    nonzero = frequencies[frequencies > 0]
    axes[0].loglog(np.arange(1, len(nonzero) + 1), nonzero, color="#0f766e")
    axes[0].set(xlabel="Item popularity rank (log)", ylabel="Train interactions (log)", title="Popularity fitted on train only")
    x = np.arange(len(frequencies) + 1) / len(frequencies)
    y = np.concatenate(([0], np.cumsum(frequencies) / frequencies.sum()))
    axes[1].plot(x, y, color="#0f766e", linewidth=2)
    axes[1].plot([0, 1], [0, 1], "--", color="#94a3b8", label="Uniform reference")
    share = analysis["summary"]["head_train_interaction_share"]
    fraction = analysis["summary"]["head_items"] / len(frequencies)
    axes[1].plot(fraction, share, "o", color="#dc2626", label=f"Top {fraction:.0%} items: {share:.1%} interactions")
    axes[1].set(xlabel="Fraction of items (most popular first)", ylabel="Cumulative train interaction share", title="Popularity concentration", xlim=(0, 1), ylim=(0, 1.03))
    axes[1].legend(loc="lower right", fontsize=8)
    save(fig, "eda_long_tail.png")

    return paths


def write_eda_report(analysis, reports_dir, *, config, config_path):
    """Lưu hai hình và nhận xét EDA đúng phạm vi tuần 1 của B."""
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    paths = plot_eda(analysis, reports_dir / "figures", dpi=config["figure_dpi"])
    stats = analysis["summary"]
    write_json(reports_dir / "eda_summary.json", stats)
    write_json(reports_dir / "eda_provenance.json", {
        "run_name": "eda_beauty_5core", "config": config, "config_sha256": sha256_file(config_path),
        "sources": analysis["source_manifest"]["sources"],
        "processed_artifacts": analysis["source_manifest"]["artifacts"],
        "figures": {path.name: sha256_file(path) for path in paths},
    })

    def escape(value):
        return clean_text(value).replace("|", "\\|")

    train_stats = stats["sequence_length_train"]
    window = stats["history_window"]
    missing_example = next(
        (row for row in analysis["examples"] if not clean_text(row["metadata"].get("description"))), None
    )
    example_observation = (
        f"- Item {missing_example['item_id']} thiếu description → biểu diễn nội dung có thể ít thông tin hơn; cần fallback title + category."
        if missing_example is not None else
        "- Các ví dụ chỉ minh họa nội dung sản phẩm, không đại diện toàn bộ catalog."
    )
    lines = [
        "# EDA Beauty 2014 — Tuần 1 của B", "",
        "Phạm vi: độ dài chuỗi, long-tail, độ thưa, tình trạng thiếu metadata và ví dụ mô tả sản phẩm.", "",
        "Nhận xét nối kết quả quan sát với hệ quả có thể xảy ra; ảnh hưởng đến mô hình cần được kiểm chứng bằng thí nghiệm.", "",
        "Run: eda_beauty_5core; config: configs/eda.json; log: experiments/log.csv.",
        "Số liệu: reports/eda_summary.json; nguồn và checksum: reports/eda_provenance.json.", "",
        "## Quy mô, độ thưa và bộ chia", "",
        "| Đại lượng | Giá trị |", "|---|---:|",
        *[f"| {label} | {stats[key]:,} |" for label, key in (
            ("User", "users"), ("Item", "items"), ("Tương tác sau 5-core", "interactions"),
            ("Train", "train_interactions"), ("Validation", "validation_targets"), ("Test", "test_targets"))],
        "", "**Nhận xét:**", "",
        f"- Độ thưa **{stats['sparsity']:.4%}** → ít bằng chứng đồng xuất hiện; độ tương đồng item và luật mua kèm có thể kém ổn định.",
        f"- Có **{stats['tied_user_timestamp_events']:,}** tương tác dư khi gộp theo (user, timestamp) "
        "→ thứ tự trong các nhóm này chưa xác định; đổi quy tắc xử lý có thể làm thay đổi mẫu PrefixSpan.",
        f"- Train + validation có **{stats['train_interactions'] + stats['validation_targets']:,}** tương tác, "
        f"khác tổng **{stats['interactions']:,}** → dùng lẫn hai phạm vi có thể dẫn đến kết luận sai rằng pipeline không khớp paper.", "",
        "## Độ dài chuỗi", "",
        "![Độ dài chuỗi](figures/eda_sequence_lengths.png)", "",
        "**Nhận xét:**", "",
        f"- Trung bình chuỗi đầy đủ **{stats['sequence_length_full']['mean']:.2f}** lớn hơn trung vị "
        f"**{stats['sequence_length_full']['median']:.0f}** → chỉ nhìn trung bình có thể đánh giá quá cao lượng lịch sử của user điển hình.",
        f"- Train có trung vị **{train_stats['median']:.0f}** item, p95 **{train_stats['p95']:.0f}**: "
        "lịch sử ngắn → mẫu mua kèm dài có thể ít lặp lại và khó đạt ngưỡng support.",
        f"- Cửa sổ **{window['max_history']}** item chỉ cắt "
        f"**{window['validation_users_truncated'] / stats['users']:.2%}** lịch sử validation và "
        f"**{window['test_users_truncated'] / stats['users']:.2%}** lịch sử test "
        "→ phần lớn user giữ đủ ngữ cảnh, nhưng nhóm lịch sử dài có thể mất tín hiệu cũ. Chuỗi train đầy đủ vẫn được giữ cho khai phá mẫu.", "",
        "## Phân bố độ phổ biến và long-tail", "",
        "![Long-tail](figures/eda_long_tail.png)", "",
        "**Nhận xét:**", "",
        f"- Head gồm **{stats['head_items']:,} item** (top {config['head_fraction']:.0%} catalog), "
        f"chiếm **{stats['head_train_interaction_share']:.2%}** tương tác train → mô hình có thể ưu tiên sản phẩm phổ biến và ít gợi ý item thuộc long-tail.",
        f"- Long-tail gồm **{stats['items'] - stats['head_items']:,} item**, chia sẻ "
        f"**{1 - stats['head_train_interaction_share']:.2%}** tương tác train → ít dữ liệu cho từng item, khiến biểu diễn hành vi có thể kém ổn định.",
        "- Độ phổ biến chỉ tính từ train; tính cả validation/test có thể đưa thông tin tương lai vào việc phân nhóm item.", "",
        "## Tình trạng thiếu metadata", "",
        f"Thống kê trên **{stats['items']:,} item**. Trường thiếu/rỗng được xác định sau khi bỏ HTML và khoảng trắng; dữ liệu gốc được giữ nguyên.", "",
        "| Nội dung thiếu | Số item | Tỷ lệ |", "|---|---:|---:|",
        f"| Cả bản ghi metadata | {stats['metadata_missing_items']:,} | {stats['metadata_missing_items'] / stats['items']:.2%} |",
        *[f"| {field} | {count:,} | {stats['missing_field_ratios'][field]:.2%} |" for field, count in stats["missing_fields"].items()],
        f"| Cả title và description | {stats['missing_title_and_description']:,} | {stats['missing_title_and_description'] / stats['items']:.2%} |",
        "", "**Nhận xét:**", "",
        f"- Thiếu cả bản ghi: **{stats['metadata_missing_items']} item** → có metadata cho toàn catalog không đồng nghĩa mọi trường đều đầy đủ. Cờ metadata_missing chỉ đánh dấu thiếu cả bản ghi.",
        f"- **{stats['missing_fields']['description']:,} item thiếu description** → biểu diễn nội dung có thể ít thông tin hơn; "
        f"**{stats['missing_description_with_title']:,} item** còn title có thể dùng title + category, "
        f"**{stats['missing_title_and_description']:,} item** thiếu cả hai cần dùng thông tin còn lại, tối thiểu là category nếu có.",
        f"- **{stats['missing_fields']['brand']:,} item thiếu brand** → thiếu tín hiệu thương hiệu; không tự đoán brand, và khi phân tích theo brand phải báo độ phủ nhãn.",
        f"- **{stats['missing_fields']['categories']:,} item thiếu categories** → category có thể hỗ trợ fallback; cần thống nhất cách chọn từ đường dẫn danh mục.",
        "- Pipeline hiện giữ nguyên metadata và chia tương tác theo user/item/timestamp; **chưa tạo văn bản fallback**. Cần thực hiện fallback trước khi encode, giữ nguyên mapping và bộ chia.", "",
        "## Ví dụ mô tả sản phẩm", "",
        f"Mẫu seed {config['seed']}; description hiển thị tối đa 240 ký tự.", "",
        "| Item ID | ASIN | Title | Brand | Description (trích ngắn) |", "|---:|---|---|---|---|",
        *[f"| {row['item_id']} | {row['asin']} | {escape(row['metadata'].get('title'))} | "
          f"{escape(row['metadata'].get('brand'))} | {escape(row['metadata'].get('description'))[:240]} |" for row in analysis["examples"]],
        "", "**Nhận xét:**", "",
        example_observation,
        "- Title và description chứa dung tích, tên dòng sản phẩm và nội dung quảng bá → embedding có thể phản ánh cách viết bên cạnh công dụng, làm các item khác công dụng vẫn gần nhau.", "",
    ]
    with (reports_dir / "data_quality.md").open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(lines))
    return paths

def main(argv=None):
    """Tái tạo toàn bộ EDA, hình và báo cáo từ dữ liệu processed đã kiểm chứng."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--splits-dir", type=Path, default=ROOT / "data" / "splits")
    parser.add_argument("--reports-dir", type=Path, default=ROOT / "reports")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "eda.json")
    args = parser.parse_args(argv)
    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        analysis = analyze_data(args.processed_dir, args.splits_dir, seed=config["seed"], head_fraction=config["head_fraction"])
        paths = write_eda_report(analysis, args.reports_dir, config=config, config_path=args.config)
        print(f"EDA complete: {len(paths)} figures; report: {args.reports_dir / 'data_quality.md'}")
        print(json.dumps(analysis["summary"], ensure_ascii=False, indent=2))
    except (OSError, ValueError, KeyError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
