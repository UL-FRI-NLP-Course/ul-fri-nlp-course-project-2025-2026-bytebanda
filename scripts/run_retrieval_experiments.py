#!/usr/bin/env python3
"""Run report-ready retrieval experiments over chunking and embedding variants."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = PROJECT_ROOT / "downloads" / "pisrs"
DEFAULT_NATURAL_QUESTIONS = PROJECT_ROOT / "evaluation" / "generated_tax_eval_questions_natural.jsonl"
DEFAULT_CITATION_QUESTIONS = PROJECT_ROOT / "evaluation" / "generated_tax_eval_questions.jsonl"
DEFAULT_DUAL_NATURAL_QUESTIONS = PROJECT_ROOT / "evaluation" / "generated_tax_eval_questions_dual_natural.jsonl"
DEFAULT_DUAL_CITATION_QUESTIONS = PROJECT_ROOT / "evaluation" / "generated_tax_eval_questions_dual_citation.jsonl"
DEFAULT_DUAL_MIXED_QUESTIONS = PROJECT_ROOT / "evaluation" / "generated_tax_eval_questions_dual_mixed.jsonl"
DEFAULT_OUT_DIR = PROJECT_ROOT / "experiments" / "retrieval"


@dataclass(frozen=True)
class ChunkConfig:
    name: str
    strategy: str
    chunk_size: int
    overlap: int


@dataclass(frozen=True)
class RetrievalConfig:
    name: str
    mode: str
    top_k: int
    candidate_k: int
    lexical_weight: float
    source_boost: float
    article_boost: float
    title_weight: float


def slugify(value: str) -> str:
    value = value.lower().replace("/", "-")
    value = re.sub(r"[^a-z0-9._-]+", "-", value)
    return re.sub(r"-+", "-", value).strip("-")


def run_command(command: list[str], cwd: Path, dry_run: bool = False) -> None:
    print("+ " + " ".join(command), flush=True)
    if dry_run:
        return
    subprocess.run(command, cwd=cwd, check=True)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize_results(path: Path) -> dict[str, Any]:
    records = read_jsonl(path)
    count = len(records)
    hits = [record.get("hits") or {} for record in records]
    scores = [record.get("scores") or {} for record in records]

    return {
        "questions": count,
        "source_hit_rate": mean([float(hit.get("source_hit", False)) for hit in hits]),
        "question_law_hit_rate": mean(
            [float(hit.get("question_law_hit", False)) for hit in hits]
        ),
        "article_hit_rate": mean([float(hit.get("article_hit", False)) for hit in hits]),
        "chunk_hit_rate": mean([float(hit.get("chunk_hit", False)) for hit in hits]),
        "all_phrase_hit_rate": mean(
            [float(hit.get("all_phrases_hit", False)) for hit in hits]
        ),
        "phrase_hit_rate": mean(
            [
                (hit.get("phrase_hit_count", 0) / hit.get("phrase_count", 1))
                if hit.get("phrase_count", 0)
                else 0.0
                for hit in hits
            ]
        ),
        "context_relevance_mean": mean(
            [float(score.get("context_relevance", 0)) for score in scores]
        ),
    }


def score_row(row: dict[str, Any]) -> float:
    """Single ranking score for report sorting; retrieval quality only."""
    return (
        0.35 * float(row["article_hit_rate"])
        + 0.25 * float(row["all_phrase_hit_rate"])
        + 0.20 * (float(row["context_relevance_mean"]) / 2.0)
        + 0.10 * float(row["chunk_hit_rate"])
        + 0.10 * float(row["source_hit_rate"])
    )


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def format_float(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def write_markdown(rows: list[dict[str, Any]], path: Path, top_n: int = 20) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
        "chunk_hit_rate",
        "source_hit_rate",
        "results_jsonl",
    ]
    lines = [
        "# Retrieval Experiment Results",
        "",
        "Ranking score = 0.35 article hit + 0.25 all phrase hit + 0.20 normalized context relevance + 0.10 chunk hit + 0.10 source hit.",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for rank, row in enumerate(rows[:top_n], start=1):
        values = []
        for column in columns:
            value = rank if column == "rank" else row.get(column, "")
            values.append(format_float(value))
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def chunk_configs(mode: str) -> list[ChunkConfig]:
    if mode == "medium":
        return [
            ChunkConfig("legal-1000-o100", "legal", 1000, 100),
            ChunkConfig("legal-1800-o150", "legal", 1800, 150),
            ChunkConfig("legal-2500-o250", "legal", 2500, 250),
        ]
    configs = [
        ChunkConfig("fixed-1200-o200", "fixed", 1200, 200),
        ChunkConfig("legal-1800-o150", "legal", 1800, 150),
    ]
    if mode == "full":
        configs.extend(
            [
                ChunkConfig("legal-1000-o100", "legal", 1000, 100),
                ChunkConfig("legal-2500-o250", "legal", 2500, 250),
                ChunkConfig("fixed-800-o150", "fixed", 800, 150),
            ]
        )
    return configs


def embedding_models(mode: str) -> list[str]:
    if mode == "medium":
        return [
            "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
            "intfloat/multilingual-e5-base",
            "BAAI/bge-m3",
        ]
    models = [
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    ]
    if mode == "full":
        models.extend(
            [
                "intfloat/multilingual-e5-base",
                "BAAI/bge-m3",
            ]
        )
    return models


def retrieval_configs(mode: str, include_dual: bool = False) -> list[RetrievalConfig]:
    if mode == "medium":
        configs = [
            RetrievalConfig("dense-k3", "dense", 3, 30, 0.0, 0.0, 0.0, 0.0),
            RetrievalConfig(
                "hybrid-title-k3-c200-l035-t030-s025-a060",
                "hybrid",
                3,
                200,
                0.35,
                0.25,
                0.60,
                0.30,
            ),
            RetrievalConfig(
                "hybrid-title-k5-c250-l040-t035-s025-a060",
                "hybrid",
                5,
                250,
                0.40,
                0.25,
                0.60,
                0.35,
            ),
        ]
        if include_dual:
            configs.append(
                RetrievalConfig(
                    "hybrid-title-k8-c300-l045-t040-s030-a070",
                    "hybrid",
                    8,
                    300,
                    0.45,
                    0.30,
                    0.70,
                    0.40,
                )
            )
        return configs

    configs = [
        RetrievalConfig("dense-k3", "dense", 3, 30, 0.0, 0.0, 0.0, 0.0),
        RetrievalConfig("hybrid-k3-c200-l040-s025-a060", "hybrid", 3, 200, 0.40, 0.25, 0.60, 0.0),
    ]
    if include_dual:
        configs.append(
            RetrievalConfig("hybrid-k5-c200-l040-s025-a060", "hybrid", 5, 200, 0.40, 0.25, 0.60, 0.0)
        )
    if mode == "full":
        configs.extend(
            [
                RetrievalConfig("hybrid-k3-c100-l025-s020-a035", "hybrid", 3, 100, 0.25, 0.20, 0.35, 0.0),
                RetrievalConfig("hybrid-k3-c300-l050-s030-a080", "hybrid", 3, 300, 0.50, 0.30, 0.80, 0.0),
                RetrievalConfig("hybrid-k5-c300-l050-s030-a080", "hybrid", 5, 300, 0.50, 0.30, 0.80, 0.0),
            ]
        )
    return configs


def dataset_configs(args: argparse.Namespace) -> list[tuple[str, Path]]:
    available = {
        "natural": args.natural_questions,
        "citation": args.citation_questions,
        "dual_natural": args.dual_natural_questions,
        "dual_citation": args.dual_citation_questions,
        "dual_mixed": args.dual_mixed_questions,
    }
    if args.datasets:
        names = [name.strip() for name in args.datasets.split(",") if name.strip()]
        unknown = sorted(set(names) - set(available))
        if unknown:
            raise SystemExit(
                "Unknown dataset name(s): "
                + ", ".join(unknown)
                + ". Available: "
                + ", ".join(sorted(available))
            )
        datasets = [(name, available[name]) for name in names]
    elif args.only_dual:
        datasets = [
            ("dual_natural", args.dual_natural_questions),
            ("dual_citation", args.dual_citation_questions),
            ("dual_mixed", args.dual_mixed_questions),
        ]
    else:
        datasets = [("natural", args.natural_questions), ("citation", args.citation_questions)]
        if args.include_dual:
            datasets.extend(
                [
                    ("dual_natural", args.dual_natural_questions),
                    ("dual_citation", args.dual_citation_questions),
                    ("dual_mixed", args.dual_mixed_questions),
                ]
            )
    return [(name, path) for name, path in datasets if path.exists()]


def selected_datasets_include_dual(args: argparse.Namespace) -> bool:
    """Return whether the configured dataset selection includes any dual dataset."""
    return any(name.startswith("dual_") for name, _path in dataset_configs(args))


def build_index(
    chunk: ChunkConfig,
    embedding: str,
    index_dir: Path,
    args: argparse.Namespace,
) -> None:
    index_path = index_dir / "faiss.index"
    chunks_path = index_dir / "chunks.jsonl"
    processed_path = index_dir / "processed_chunks.jsonl"
    if index_path.exists() and chunks_path.exists() and not args.rebuild:
        print(f"Skipping existing index: {index_dir}", flush=True)
        return

    run_command(
        [
            sys.executable,
            "-m",
            "src.rag_cli",
            "--build-index",
            "--raw-dir",
            str(args.raw_dir),
            "--chunk-strategy",
            chunk.strategy,
            "--chunk-size",
            str(chunk.chunk_size),
            "--overlap",
            str(chunk.overlap),
            "--embedding-model",
            embedding,
            "--batch-size",
            str(args.batch_size),
            "--processed-chunks-path",
            str(processed_path),
            "--index-path",
            str(index_path),
            "--index-chunks-path",
            str(chunks_path),
        ],
        cwd=PROJECT_ROOT,
        dry_run=args.dry_run,
    )


def evaluate(
    dataset_name: str,
    questions: Path,
    chunk: ChunkConfig,
    embedding: str,
    retrieval: RetrievalConfig,
    index_dir: Path,
    results_path: Path,
    args: argparse.Namespace,
) -> None:
    command = [
        sys.executable,
        "-m",
        "src.evaluate_rag",
        "--questions",
        str(questions),
        "--results-jsonl",
        str(results_path),
        "--index-path",
        str(index_dir / "faiss.index"),
        "--chunks-path",
        str(index_dir / "chunks.jsonl"),
        "--embedding-model",
        embedding,
        "--top-k",
        str(retrieval.top_k),
        "--retrieval-mode",
        retrieval.mode,
        "--candidate-k",
        str(retrieval.candidate_k),
        "--lexical-weight",
        str(retrieval.lexical_weight),
        "--source-boost",
        str(retrieval.source_boost),
        "--article-boost",
        str(retrieval.article_boost),
        "--title-weight",
        str(retrieval.title_weight),
        "--run-label",
        f"{dataset_name}-{chunk.name}-{slugify(embedding)}-{retrieval.name}",
        "--prompt-label",
        "retrieval-only",
        "--no-generate",
    ]
    if args.limit:
        command.extend(["--limit", str(args.limit)])
    run_command(command, cwd=PROJECT_ROOT, dry_run=args.dry_run)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["quick", "medium", "full"], default="quick")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--natural-questions", type=Path, default=DEFAULT_NATURAL_QUESTIONS)
    parser.add_argument("--citation-questions", type=Path, default=DEFAULT_CITATION_QUESTIONS)
    parser.add_argument("--dual-natural-questions", type=Path, default=DEFAULT_DUAL_NATURAL_QUESTIONS)
    parser.add_argument("--dual-citation-questions", type=Path, default=DEFAULT_DUAL_CITATION_QUESTIONS)
    parser.add_argument("--dual-mixed-questions", type=Path, default=DEFAULT_DUAL_MIXED_QUESTIONS)
    parser.add_argument(
        "--include-dual",
        action="store_true",
        help="Include dual-source natural/citation/mixed datasets in addition to single-source datasets.",
    )
    parser.add_argument(
        "--only-dual",
        action="store_true",
        help="Run only dual-source natural/citation/mixed datasets.",
    )
    parser.add_argument(
        "--datasets",
        default=None,
        help=(
            "Comma-separated dataset names to run. Available: "
            "natural,citation,dual_natural,dual_citation,dual_mixed. "
            "Overrides --include-dual and --only-dual."
        ),
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--limit", type=int, default=0, help="Limit questions per dataset for smoke tests.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--rebuild", action="store_true", help="Rebuild indexes even if they already exist.")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or uuid.uuid4().hex[:8]
    run_dir = args.out_dir / run_id
    index_root = run_dir / "indexes"
    logs_dir = run_dir / "logs"
    reports_dir = run_dir / "reports"
    rows: list[dict[str, Any]] = []

    print(f"RUN_ID={run_id}")
    print(f"Output directory: {run_dir}")

    start = time.time()
    datasets = dataset_configs(args)
    if not datasets:
        raise SystemExit("No question datasets found. Generate them first with scripts/make_eval_dataset.py.")

    for chunk in chunk_configs(args.mode):
        for embedding in embedding_models(args.mode):
            embedding_slug = slugify(embedding)
            index_dir = index_root / chunk.name / embedding_slug
            build_index(chunk, embedding, index_dir, args)

            for retrieval in retrieval_configs(
                args.mode,
                include_dual=selected_datasets_include_dual(args),
            ):
                for dataset_name, questions in datasets:
                    result_name = (
                        f"{dataset_name}__{chunk.name}__{embedding_slug}__{retrieval.name}.jsonl"
                    )
                    results_path = logs_dir / result_name
                    evaluate(
                        dataset_name=dataset_name,
                        questions=questions,
                        chunk=chunk,
                        embedding=embedding,
                        retrieval=retrieval,
                        index_dir=index_dir,
                        results_path=results_path,
                        args=args,
                    )
                    if args.dry_run:
                        continue
                    summary = summarize_results(results_path)
                    row = {
                        "run_id": run_id,
                        "dataset": dataset_name,
                        "chunk_config": chunk.name,
                        "chunk_strategy": chunk.strategy,
                        "chunk_size": chunk.chunk_size,
                        "overlap": chunk.overlap,
                        "embedding": embedding,
                        "retrieval_config": retrieval.name,
                        "retrieval_mode": retrieval.mode,
                        "top_k": retrieval.top_k,
                        "candidate_k": retrieval.candidate_k,
                        "lexical_weight": retrieval.lexical_weight,
                        "source_boost": retrieval.source_boost,
                        "article_boost": retrieval.article_boost,
                        "title_weight": retrieval.title_weight,
                        **summary,
                        "results_jsonl": str(results_path.relative_to(PROJECT_ROOT)),
                    }
                    row["score"] = score_row(row)
                    rows.append(row)

    if not args.dry_run:
        rows.sort(key=lambda row: row["score"], reverse=True)
        write_csv(rows, reports_dir / "retrieval_experiments.csv")
        write_markdown(rows, reports_dir / "retrieval_experiments.md")
        print()
        print(f"Wrote CSV: {reports_dir / 'retrieval_experiments.csv'}")
        print(f"Wrote Markdown: {reports_dir / 'retrieval_experiments.md'}")
        if rows:
            best = rows[0]
            print(
                "Best: "
                f"dataset={best['dataset']} chunk={best['chunk_config']} "
                f"embedding={best['embedding']} retrieval={best['retrieval_config']} "
                f"score={best['score']:.3f}"
            )

    print(f"Elapsed: {(time.time() - start) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
