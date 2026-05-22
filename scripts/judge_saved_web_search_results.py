#!/usr/bin/env python3
"""Run LLM judging on an existing web-search answer evaluation JSONL file.

This does not run web search or regenerate answers. It only reads saved rows,
judges the existing predicted answer against the expected answer, and writes a
new JSONL plus summary.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from evaluate_web_search_answers import DEFAULT_LOCAL_MODEL_PATH, llm_judge, source_summary_text  # noqa: E402


DEFAULT_INPUT = PROJECT_ROOT / "logs" / "web-search-answer-eval-20-fast.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "logs" / "web-search-answer-eval-20-judged-existing.jsonl"
DEFAULT_SUMMARY = PROJECT_ROOT / "logs" / "web-search-answer-eval-20-judged-existing-summary.json"


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


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def answer_text(row: dict[str, Any]) -> str:
    answer = row.get("answer")
    if isinstance(answer, dict):
        return str(answer.get("text") or "")
    return str(answer or "")


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
    try:
        return float(judge.get("answer_correctness_vs_ground_truth"))
    except (TypeError, ValueError):
        return None


def build_summary(rows: list[dict[str, Any]], output_path: Path) -> dict[str, Any]:
    correctness = [value for row in rows if (value := judge_correctness(row)) is not None]
    return {
        "questions": len(rows),
        "results_jsonl": str(output_path),
        "run_ids": sorted({str(row.get("run_id")) for row in rows if row.get("run_id")}),
        "mean_token_recall_vs_gt": round(mean([score_value(row, "token_recall_vs_gt") for row in rows]), 4),
        "mean_token_f1_vs_gt": round(mean([score_value(row, "token_f1_vs_gt") for row in rows]), 4),
        "expected_source_domain_hit_rate": round(
            mean([1.0 if (row.get("scores") or {}).get("expected_source_domain_hit") else 0.0 for row in rows]),
            4,
        ),
        "mean_official_source_count": round(mean([score_value(row, "official_source_count") for row in rows]), 4),
        "mean_source_relevance": round(mean([score_value(row, "mean_source_relevance") for row in rows]), 4),
        "mean_llm_judge_correctness": round(mean(correctness), 4) if correctness else None,
        "judged_count": len(correctness),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Judge saved web-search answers without re-running search.")
    parser.add_argument("--input-jsonl", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-jsonl", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_LOCAL_MODEL_PATH)
    parser.add_argument("--judge-max-new-tokens", type=int, default=120)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    rows = load_jsonl(args.input_jsonl)
    if args.limit > 0:
        rows = rows[: args.limit]

    from src.generate_answer import load_llm

    print(f"Loaded {len(rows)} saved result row(s) from {args.input_jsonl}")
    print(f"Loading LLM judge model from {args.model_path}", flush=True)
    llm_bundle = load_llm(args.model_path)

    judged_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        if args.skip_existing and row.get("llm_judge"):
            judged_rows.append(row)
            print(f"[{index}/{len(rows)}] id={row.get('id')} kept existing judge")
            continue

        judge = llm_judge(
            question=str(row.get("question") or ""),
            expected_answer=str(row.get("expected_answer") or ""),
            predicted_answer=answer_text(row),
            source_summaries=source_summary_text(row.get("suggestions") or []),
            llm_bundle=llm_bundle,
            max_new_tokens=args.judge_max_new_tokens,
        )
        row = dict(row)
        row["llm_judge"] = judge
        judged_rows.append(row)
        correctness = judge.get("answer_correctness_vs_ground_truth") if isinstance(judge, dict) else None
        parse_note = " parse_error" if isinstance(judge, dict) and judge.get("parse_error") else ""
        print(f"[{index}/{len(rows)}] id={row.get('id')} judge_correctness={correctness}{parse_note}")

    write_jsonl(judged_rows, args.output_jsonl)
    summary = build_summary(judged_rows, args.output_jsonl)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote judged rows to {args.output_jsonl}")
    print(f"Wrote summary to {args.summary_json}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
