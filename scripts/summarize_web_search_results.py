#!/usr/bin/env python3
"""Summarize an existing web-search answer evaluation JSONL file.

This script does not run web search, retrieval, or answer generation. It only
reads saved result rows from scripts/evaluate_web_search_answers.py and
recomputes aggregate metrics.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


DEFAULT_RESULTS = Path("logs/web-search-answer-eval-20-fast.jsonl")
DEFAULT_SUMMARY = Path("logs/web-search-answer-eval-20-fast-offline-summary.json")
DEFAULT_WEAK_CASES = Path("logs/web-search-answer-eval-20-fast-weak-cases.jsonl")


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}: {exc}") from exc
    return rows


def score_value(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    scores = row.get("scores") or {}
    try:
        return float(scores.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def judge_correctness(row: dict[str, Any]) -> float | None:
    judge = row.get("llm_judge")
    if not isinstance(judge, dict) or judge.get("parse_error"):
        return None
    value = judge.get("answer_correctness_vs_ground_truth")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def answer_text(row: dict[str, Any]) -> str:
    answer = row.get("answer")
    if isinstance(answer, dict):
        return str(answer.get("text") or "")
    return str(answer or "")


def build_summary(rows: list[dict[str, Any]], results_path: Path) -> dict[str, Any]:
    correctness = [value for row in rows if (value := judge_correctness(row)) is not None]
    source_hits = [
        1.0 if (row.get("scores") or {}).get("expected_source_domain_hit") else 0.0
        for row in rows
    ]
    unsupported = [
        row
        for row in rows
        if not answer_text(row).strip()
        or "nisem našel" in answer_text(row).lower()
        or "not found" in answer_text(row).lower()
    ]
    return {
        "questions": len(rows),
        "results_jsonl": str(results_path),
        "run_ids": sorted({str(row.get("run_id")) for row in rows if row.get("run_id")}),
        "mean_token_recall_vs_gt": round(mean([score_value(row, "token_recall_vs_gt") for row in rows]), 4),
        "mean_token_f1_vs_gt": round(mean([score_value(row, "token_f1_vs_gt") for row in rows]), 4),
        "expected_source_domain_hit_rate": round(mean(source_hits), 4),
        "mean_official_source_count": round(mean([score_value(row, "official_source_count") for row in rows]), 4),
        "mean_source_relevance": round(mean([score_value(row, "mean_source_relevance") for row in rows]), 4),
        "mean_llm_judge_correctness": round(mean(correctness), 4) if correctness else None,
        "answered_count": len(rows) - len(unsupported),
        "unsupported_or_empty_answer_count": len(unsupported),
    }


def weak_case_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row.get("id"),
        "question": row.get("question"),
        "token_f1_vs_gt": score_value(row, "token_f1_vs_gt"),
        "expected_source_domain_hit": bool((row.get("scores") or {}).get("expected_source_domain_hit")),
        "official_source_count": score_value(row, "official_source_count"),
        "mean_source_relevance": score_value(row, "mean_source_relevance"),
        "expected_answer": row.get("expected_answer"),
        "predicted_answer": answer_text(row),
        "top_sources": [
            {
                "title": item.get("title"),
                "url": item.get("url"),
                "relevance": item.get("relevance"),
                "source_name": item.get("source_name"),
            }
            for item in (row.get("suggestions") or [])[:3]
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize saved web-search answer evaluation results.")
    parser.add_argument("--results-jsonl", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--weak-cases-jsonl", type=Path, default=DEFAULT_WEAK_CASES)
    parser.add_argument("--weak-cases", type=int, default=5)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    rows = load_jsonl(args.results_jsonl)
    summary = build_summary(rows, args.results_jsonl)

    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    weak_rows = sorted(
        rows,
        key=lambda row: (
            score_value(row, "token_f1_vs_gt"),
            1.0 if (row.get("scores") or {}).get("expected_source_domain_hit") else 0.0,
            score_value(row, "mean_source_relevance"),
        ),
    )[: max(args.weak_cases, 0)]
    args.weak_cases_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.weak_cases_jsonl.open("w", encoding="utf-8") as handle:
        for row in weak_rows:
            handle.write(json.dumps(weak_case_row(row), ensure_ascii=False) + "\n")

    print(f"Wrote summary to {args.summary_json}")
    print(f"Wrote weak cases to {args.weak_cases_jsonl}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
