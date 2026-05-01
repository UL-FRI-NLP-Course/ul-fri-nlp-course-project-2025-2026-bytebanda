"""Compare multiple RAG evaluation JSONL result files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read JSON Lines records."""
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def summarize(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize one evaluation result file."""
    total = len(records) or 1
    source_hits = sum(int(record.get("hits", {}).get("source_hit", False)) for record in records)
    article_hits = sum(int(record.get("hits", {}).get("article_hit", False)) for record in records)
    chunk_hits = sum(int(record.get("hits", {}).get("chunk_hit", False)) for record in records)
    phrase_hits = sum(int(record.get("hits", {}).get("all_phrases_hit", False)) for record in records)
    generation_failures = sum(1 for record in records if record.get("answer_error"))
    answered = sum(1 for record in records if record.get("answer"))
    context = sum(int(record.get("scores", {}).get("context_relevance", 0)) for record in records)
    faithfulness = sum(int(record.get("scores", {}).get("faithfulness", 0)) for record in records)
    correctness = sum(int(record.get("scores", {}).get("answer_correctness", 0)) for record in records)

    return {
        "n": len(records),
        "source_hit": source_hits / total,
        "article_hit": article_hits / total,
        "chunk_hit": chunk_hits / total,
        "all_phrase_hit": phrase_hits / total,
        "answered": answered,
        "generation_failures": generation_failures,
        "context": context / total,
        "faithfulness": faithfulness / total,
        "correctness": correctness / total,
    }


def parse_run_spec(spec: str) -> tuple[str, Path]:
    """Parse LABEL=PATH or PATH syntax."""
    if "=" in spec:
        label, path = spec.split("=", 1)
        return label, Path(path)
    path = Path(spec)
    return path.stem, path


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""
    parser = argparse.ArgumentParser(description="Compare RAG evaluation JSONL files.")
    parser.add_argument("runs", nargs="+", help="Evaluation runs as LABEL=path.jsonl")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Print a compact comparison table."""
    parser = build_parser()
    args = parser.parse_args(argv)

    rows = []
    for spec in args.runs:
        label, path = parse_run_spec(spec)
        records = read_jsonl(path)
        summary = summarize(records)
        rows.append((label, path, summary))

    print("RAG evaluation comparison")
    print(
        "label\tn\tsource@k\tarticle@k\tchunk@k\tphrases@k\t"
        "context/2\tfaithful/2\tcorrect/2\tanswered\tgen_fail"
    )
    for label, path, summary in rows:
        print(
            f"{label}\t"
            f"{summary['n']}\t"
            f"{summary['source_hit']:.3f}\t"
            f"{summary['article_hit']:.3f}\t"
            f"{summary['chunk_hit']:.3f}\t"
            f"{summary['all_phrase_hit']:.3f}\t"
            f"{summary['context']:.3f}\t"
            f"{summary['faithfulness']:.3f}\t"
            f"{summary['correctness']:.3f}\t"
            f"{summary['answered']}\t"
            f"{summary['generation_failures']}"
        )
        print(f"  file={path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
