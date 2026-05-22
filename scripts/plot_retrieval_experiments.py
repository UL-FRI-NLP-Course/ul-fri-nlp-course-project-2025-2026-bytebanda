#!/usr/bin/env python3
"""Plot retrieval experiment summaries from retrieval_experiments.csv."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


NUMERIC_COLUMNS = {
    "score",
    "source_hit_rate",
    "question_law_hit_rate",
    "article_hit_rate",
    "chunk_hit_rate",
    "all_phrase_hit_rate",
    "phrase_hit_rate",
    "context_relevance_mean",
}


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for column in NUMERIC_COLUMNS:
            if column in row:
                row[column] = float(row[column])
    return rows


def read_many(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(read_rows(path))
    return rows


def short_embedding(name: str) -> str:
    return name.rsplit("/", 1)[-1]


def display_name(key: str, value: str) -> str:
    if key == "embedding":
        mapping = {
            "sentence-transformers/paraphrase-multilingual-mpnet-base-v2": "MPNet",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": "MiniLM",
            "intfloat/multilingual-e5-base": "E5-base",
            "BAAI/bge-m3": "BGE-M3",
        }
        return mapping.get(value, short_embedding(value))
    if key == "chunk_config":
        mapping = {
            "fixed-1200-o200": "Baseline",
            "fixed-800-o150": "Fixed small",
            "legal-1000-o100": "Legal small",
            "legal-1800-o150": "Legal medium",
            "legal-2500-o250": "Legal large",
        }
        return mapping.get(value, value)
    if key == "retrieval_config":
        mapping = {
            "dense-k3": "Dense top-3",
            "hybrid-k3-c200-l040-s025-a060": "Hybrid top-3",
            "hybrid-k5-c200-l040-s025-a060": "Hybrid top-5",
            "hybrid-title-k3-c200-l035-t030-s025-a060": "Hybrid-title top-3",
            "hybrid-title-k5-c250-l040-t035-s025-a060": "Hybrid-title top-5",
            "hybrid-title-k8-c300-l045-t040-s030-a070": "Hybrid-title top-8",
        }
        return mapping.get(value, value)
    if key == "dataset":
        mapping = {
            "citation": "Citation",
            "natural": "Natural",
            "dual_mixed": "Dual mixed",
            "dual_natural": "Dual natural",
            "dual_citation": "Dual citation",
        }
        return mapping.get(value, value)
    return value


def label_for(row: dict[str, Any]) -> str:
    return (
        f"{display_name('dataset', row['dataset'])} | {display_name('chunk_config', row['chunk_config'])} | "
        f"{display_name('embedding', row['embedding'])} | {display_name('retrieval_config', row['retrieval_config'])}"
    )


def ensure_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is not installed. Install it with: pip install matplotlib"
        ) from exc
    return plt


def configure_style(report_style: bool = False) -> None:
    plt = ensure_matplotlib()
    if not report_style:
        return
    plt.rcParams.update(
        {
            "axes.titlesize": 25,
            "axes.labelsize": 20,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "legend.fontsize": 15,
            "figure.titlesize": 25,
            "font.size": 16,
        }
    )


def std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    avg = sum(values) / len(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / (len(values) - 1))


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def save_top_score_plot(rows: list[dict[str, Any]], out_dir: Path, top_n: int) -> None:
    plt = ensure_matplotlib()
    top = sorted(rows, key=lambda row: row["score"], reverse=True)[:top_n]
    labels = [label_for(row) for row in top][::-1]
    scores = [row["score"] for row in top][::-1]

    height = max(5, 0.42 * len(top))
    fig, ax = plt.subplots(figsize=(13, height))
    bars = ax.barh(labels, scores, color="#2f6f7e")
    ax.set_title(f"Top {len(top)} Retrieval Configurations")
    ax.set_xlabel("Combined retrieval score")
    ax.set_xlim(0, max(1.0, max(scores, default=0) * 1.08))
    ax.bar_label(bars, fmt="%.3f", padding=4, fontsize=8)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "top_retrieval_scores.png", dpi=180)
    plt.close(fig)


def grouped_scores_by_dataset(rows: list[dict[str, Any]], key: str) -> list[tuple[str, float, float, int]]:
    """For each key value, average its best score on each dataset, then return mean/std."""
    datasets = sorted({row["dataset"] for row in rows})
    key_values = sorted({row[key] for row in rows})
    results = []
    for value in key_values:
        dataset_best = []
        for dataset in datasets:
            matching = [
                row["score"]
                for row in rows
                if row[key] == value and row["dataset"] == dataset
            ]
            if matching:
                dataset_best.append(max(matching))
        if dataset_best:
            results.append((value, mean(dataset_best), std(dataset_best), len(dataset_best)))
    return sorted(results, key=lambda item: item[1], reverse=True)


def save_mean_std_plot(rows: list[dict[str, Any]], out_dir: Path, key: str, filename: str) -> None:
    plt = ensure_matplotlib()
    items = grouped_scores_by_dataset(rows, key)
    labels = [display_name(key, name) for name, _m, _s, _n in items]
    means = [m for _name, m, _s, _n in items]
    stds = [s for _name, _m, s, _n in items]

    fig, ax = plt.subplots(figsize=(13, max(5, 0.65 * len(items))))
    y = list(range(len(items)))
    bars = ax.barh(y, means, xerr=stds, color="#356f8c", ecolor="#222222", capsize=6)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_title(f"Best Retrieval Score by {key.replace('_', ' ').title()} Across Datasets")
    ax.set_xlabel("Mean of per-dataset best score (+/- std)")
    ax.set_xlim(0, max(1.0, max((m + s for m, s in zip(means, stds)), default=0) * 1.08))
    ax.bar_label(bars, labels=[f"{m:.3f}" for m in means], padding=8)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / filename, dpi=220)
    plt.close(fig)


def save_dataset_component_heatmap(rows: list[dict[str, Any]], out_dir: Path, key: str, filename: str) -> None:
    plt = ensure_matplotlib()
    datasets = sorted({row["dataset"] for row in rows})
    values = sorted({row[key] for row in rows})
    matrix = []
    for dataset in datasets:
        line = []
        for value in values:
            matching = [
                row["score"]
                for row in rows
                if row["dataset"] == dataset and row[key] == value
            ]
            line.append(max(matching) if matching else 0.0)
        matrix.append(line)

    display_values = [display_name(key, value) for value in values]
    fig, ax = plt.subplots(figsize=(max(10, 2.5 * len(values)), max(5, 1.0 * len(datasets))))
    image = ax.imshow(matrix, cmap="YlGnBu", vmin=0, vmax=max(1.0, max(max(row) for row in matrix)))
    ax.set_title(f"Best Score per Dataset by {key.replace('_', ' ').title()}")
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(display_values, rotation=30, ha="right")
    ax.set_yticks(range(len(datasets)))
    ax.set_yticklabels(datasets)
    for row_idx, line in enumerate(matrix):
        for col_idx, value in enumerate(line):
            ax.text(col_idx, row_idx, f"{value:.3f}", ha="center", va="center", fontsize=14)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_dir / filename, dpi=220)
    plt.close(fig)


def save_dataset_grouped_bars(
    rows: list[dict[str, Any]],
    out_dir: Path,
    key: str,
    filename: str,
    legend_title: str,
) -> None:
    """Plot component performance for each dataset with std over other configs."""
    plt = ensure_matplotlib()
    datasets = sorted({row["dataset"] for row in rows})
    groups = sorted({row[key] for row in rows})
    x = list(range(len(datasets)))
    width = min(0.22, 0.75 / max(len(groups), 1))
    colors = ["#2f6f7e", "#b66a3c", "#5f8a45", "#7566a0", "#a64f6b"]

    fig, ax = plt.subplots(figsize=(16, 8))
    for idx, group in enumerate(groups):
        means = []
        stds = []
        for dataset in datasets:
            values = [
                row["score"]
                for row in rows
                if row["dataset"] == dataset and row[key] == group
            ]
            means.append(mean(values))
            stds.append(std(values))
        offsets = [pos + (idx - (len(groups) - 1) / 2) * width for pos in x]
        bars = ax.bar(
            offsets,
            means,
            width=width,
            yerr=stds,
            capsize=5,
            label=display_name(key, group),
            color=colors[idx % len(colors)],
            alpha=0.92,
        )
        ax.bar_label(bars, labels=[f"{value:.2f}" for value in means], padding=4, fontsize=12)

    ax.set_title(f"{legend_title} Across Evaluation Datasets")
    ax.set_ylabel("Mean retrieval score (+/- std)")
    ax.set_xlabel("")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend(
        title=legend_title,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=min(3, len(groups)),
        frameon=False,
    )
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out_dir / filename, dpi=220)
    plt.close(fig)


def save_dataset_embedding_grouped_bars(rows: list[dict[str, Any]], out_dir: Path) -> None:
    """Plot embedding performance for each dataset with std over chunk/retrieval configs."""
    save_dataset_grouped_bars(
        rows,
        out_dir,
        key="embedding",
        filename="report_dataset_embedding_grouped_bars.png",
        legend_title="Embedding",
    )


def save_metric_group_plot(rows: list[dict[str, Any]], out_dir: Path, top_n: int) -> None:
    plt = ensure_matplotlib()
    top = sorted(rows, key=lambda row: row["score"], reverse=True)[:top_n]
    metrics = [
        "article_hit_rate",
        "all_phrase_hit_rate",
        "chunk_hit_rate",
        "source_hit_rate",
    ]
    labels = [label_for(row) for row in top]
    x = list(range(len(top)))
    width = 0.20

    fig, ax = plt.subplots(figsize=(14, 7))
    colors = ["#326b8c", "#c46a42", "#5e8c4a", "#8b6fb3"]
    for idx, metric in enumerate(metrics):
        offsets = [pos + (idx - 1.5) * width for pos in x]
        ax.bar(offsets, [row[metric] for row in top], width=width, label=metric, color=colors[idx])

    ax.set_title(f"Retrieval Metrics for Top {len(top)} Configurations")
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "top_retrieval_metric_breakdown.png", dpi=180)
    plt.close(fig)


def aggregate(rows: list[dict[str, Any]], key: str) -> list[tuple[str, float, int]]:
    groups: dict[str, list[float]] = {}
    for row in rows:
        groups.setdefault(row[key], []).append(row["score"])
    return sorted(
        ((name, sum(values) / len(values), len(values)) for name, values in groups.items()),
        key=lambda item: item[1],
        reverse=True,
    )


def aggregate_best(rows: list[dict[str, Any]], key: str) -> list[tuple[str, float, dict[str, Any]]]:
    best: dict[str, dict[str, Any]] = {}
    for row in rows:
        name = row[key]
        if name not in best or row["score"] > best[name]["score"]:
            best[name] = row
    return sorted(
        ((name, row["score"], row) for name, row in best.items()),
        key=lambda item: item[1],
        reverse=True,
    )


def save_aggregate_plot(rows: list[dict[str, Any]], out_dir: Path, key: str, filename: str) -> None:
    plt = ensure_matplotlib()
    items = aggregate(rows, key)
    labels = [display_name(key, name) for name, _score, _count in items]
    values = [score for _name, score, _count in items]

    fig, ax = plt.subplots(figsize=(11, max(4, 0.45 * len(items))))
    bars = ax.barh(labels[::-1], values[::-1], color="#6b7f3f")
    ax.set_title(f"Mean Retrieval Score by {key.replace('_', ' ').title()}")
    ax.set_xlabel("Mean combined retrieval score")
    ax.set_xlim(0, max(1.0, max(values, default=0) * 1.08))
    ax.bar_label(bars, fmt="%.3f", padding=4, fontsize=8)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / filename, dpi=180)
    plt.close(fig)


def save_best_aggregate_plot(rows: list[dict[str, Any]], out_dir: Path, key: str, filename: str) -> None:
    plt = ensure_matplotlib()
    items = aggregate_best(rows, key)
    labels = [display_name(key, name) for name, _score, _row in items]
    values = [score for _name, score, _row in items]

    fig, ax = plt.subplots(figsize=(11, max(4, 0.48 * len(items))))
    bars = ax.barh(labels[::-1], values[::-1], color="#2f6f7e")
    ax.set_title(f"Best Retrieval Score by {key.replace('_', ' ').title()}")
    ax.set_xlabel("Best combined retrieval score")
    ax.set_xlim(0, max(1.0, max(values, default=0) * 1.08))
    ax.bar_label(bars, fmt="%.3f", padding=4, fontsize=8)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / filename, dpi=180)
    plt.close(fig)


def pivot_best(rows: list[dict[str, Any]], row_key: str, col_key: str) -> tuple[list[str], list[str], list[list[float]]]:
    row_names = sorted({row[row_key] for row in rows})
    col_names = sorted({row[col_key] for row in rows})
    values: list[list[float]] = []
    for row_name in row_names:
        line = []
        for col_name in col_names:
            matching = [
                row["score"]
                for row in rows
                if row[row_key] == row_name and row[col_key] == col_name
            ]
            line.append(max(matching) if matching else 0.0)
        values.append(line)
    return row_names, col_names, values


def save_heatmap(rows: list[dict[str, Any]], out_dir: Path, row_key: str, col_key: str, filename: str) -> None:
    plt = ensure_matplotlib()
    row_names, col_names, values = pivot_best(rows, row_key, col_key)
    display_rows = [display_name(row_key, name) for name in row_names]
    display_cols = [display_name(col_key, name) for name in col_names]

    fig, ax = plt.subplots(figsize=(max(7, 1.7 * len(col_names)), max(4, 0.65 * len(row_names))))
    image = ax.imshow(values, cmap="YlGnBu", vmin=0, vmax=max(1.0, max(max(row) for row in values)))
    ax.set_title(f"Best Score Heatmap: {row_key.replace('_', ' ').title()} x {col_key.replace('_', ' ').title()}")
    ax.set_xticks(range(len(col_names)))
    ax.set_xticklabels(display_cols, rotation=35, ha="right")
    ax.set_yticks(range(len(row_names)))
    ax.set_yticklabels(display_rows)
    for row_idx, line in enumerate(values):
        for col_idx, value in enumerate(line):
            ax.text(col_idx, row_idx, f"{value:.3f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_dir / filename, dpi=180)
    plt.close(fig)


def write_best_table(rows: list[dict[str, Any]], out_dir: Path, top_n: int) -> None:
    top = sorted(rows, key=lambda row: row["score"], reverse=True)[:top_n]
    columns = [
        "rank",
        "dataset",
        "chunk_config",
        "embedding",
        "retrieval_config",
        "score",
        "article_hit_rate",
        "all_phrase_hit_rate",
        "context_relevance_mean",
        "results_jsonl",
    ]
    lines = [
        "# Top Retrieval Configurations",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for rank, row in enumerate(top, start=1):
        values = []
        for column in columns:
            if column == "rank":
                value = rank
            elif column in {"dataset", "chunk_config", "embedding", "retrieval_config"}:
                value = display_name(column, row[column])
            elif isinstance(row.get(column), float):
                value = f"{row[column]:.3f}"
            else:
                value = row.get(column, "")
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    (out_dir / "top_retrieval_configurations.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path, help="Path to retrieval_experiments.csv")
    parser.add_argument(
        "--extra-csv",
        type=Path,
        action="append",
        default=[],
        help="Additional retrieval_experiments.csv files to merge into the plots.",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--top-n", type=int, default=12)
    parser.add_argument("--report-style", action="store_true", help="Use larger report-ready fonts.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    rows = read_many([args.csv_path, *args.extra_csv])
    if not rows:
        raise SystemExit(f"No rows found in {args.csv_path}")

    out_dir = args.out_dir or args.csv_path.parent / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    configure_style(args.report_style)

    save_top_score_plot(rows, out_dir, args.top_n)
    save_metric_group_plot(rows, out_dir, min(args.top_n, 8))
    save_aggregate_plot(rows, out_dir, "embedding", "mean_score_by_embedding.png")
    save_aggregate_plot(rows, out_dir, "chunk_config", "mean_score_by_chunk_config.png")
    save_aggregate_plot(rows, out_dir, "retrieval_config", "mean_score_by_retrieval_config.png")
    save_aggregate_plot(rows, out_dir, "dataset", "mean_score_by_dataset.png")
    save_best_aggregate_plot(rows, out_dir, "embedding", "best_score_by_embedding.png")
    save_best_aggregate_plot(rows, out_dir, "chunk_config", "best_score_by_chunk_config.png")
    save_best_aggregate_plot(rows, out_dir, "retrieval_config", "best_score_by_retrieval_config.png")
    save_best_aggregate_plot(rows, out_dir, "dataset", "best_score_by_dataset.png")
    save_heatmap(rows, out_dir, "embedding", "chunk_config", "heatmap_embedding_by_chunk.png")
    save_heatmap(rows, out_dir, "embedding", "retrieval_config", "heatmap_embedding_by_retrieval.png")
    save_heatmap(rows, out_dir, "dataset", "retrieval_config", "heatmap_dataset_by_retrieval.png")
    save_mean_std_plot(rows, out_dir, "embedding", "report_embedding_mean_std.png")
    save_mean_std_plot(rows, out_dir, "chunk_config", "report_chunking_mean_std.png")
    save_mean_std_plot(rows, out_dir, "retrieval_config", "report_retrieval_mean_std.png")
    save_dataset_component_heatmap(rows, out_dir, "embedding", "report_dataset_by_embedding.png")
    save_dataset_component_heatmap(rows, out_dir, "chunk_config", "report_dataset_by_chunking.png")
    save_dataset_component_heatmap(rows, out_dir, "retrieval_config", "report_dataset_by_retrieval.png")
    save_dataset_embedding_grouped_bars(rows, out_dir)
    save_dataset_grouped_bars(
        rows,
        out_dir,
        key="chunk_config",
        filename="report_dataset_chunking_grouped_bars.png",
        legend_title="Chunking",
    )
    save_dataset_grouped_bars(
        rows,
        out_dir,
        key="retrieval_config",
        filename="report_dataset_retrieval_grouped_bars.png",
        legend_title="Retrieval",
    )
    write_best_table(rows, out_dir, args.top_n)

    print(f"Wrote plots and table to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
