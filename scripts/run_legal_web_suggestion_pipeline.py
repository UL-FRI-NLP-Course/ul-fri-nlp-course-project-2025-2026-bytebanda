#!/usr/bin/env python3
"""Run the standalone legal web-suggestion helper scripts.

This wrapper calls scripts/legal_web_suggestions.py without touching the main RAG
pipeline. It is useful for reproducible experiments from one command.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUGGESTION_SCRIPT = PROJECT_ROOT / "scripts" / "legal_web_suggestions.py"
DEFAULT_QUESTIONS = PROJECT_ROOT / "evaluation" / "web_suggestion_seed_questions.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "logs" / "web_suggestions.jsonl"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run legal source suggestion collection.")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--mode", choices=["local", "web", "both"], default="local")
    parser.add_argument("--provider", choices=["duckduckgo", "brave", "tavily"], default="duckduckgo")
    parser.add_argument("--broad", action="store_true")
    parser.add_argument("--allow-curated-fallback", action="store_true")
    parser.add_argument("--source-policy", choices=["auto", "all"], default="auto")
    parser.add_argument("--min-relevance", type=float, default=0.25)
    parser.add_argument("--best-local-context", action="store_true")
    parser.add_argument("--best-local-limit", type=int, default=4)
    parser.add_argument("--answer", action="store_true")
    parser.add_argument("--llm-answer", action="store_true")
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--llm-context-chars", type=int, default=1800)
    parser.add_argument("--fetch-pages", action="store_true")
    parser.add_argument("--answer-sources", type=int, default=5)
    parser.add_argument("--answer-sentences", type=int, default=4)
    parser.add_argument("--tiers", default="1,2,3")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-suggestions", type=int, default=8)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    command = [
        sys.executable,
        str(SUGGESTION_SCRIPT),
        "--questions",
        str(args.questions),
        "--output",
        str(args.output),
        "--mode",
        args.mode,
        "--provider",
        args.provider,
        "--tiers",
        args.tiers,
        "--max-suggestions",
        str(args.max_suggestions),
        "--source-policy",
        args.source_policy,
        "--min-relevance",
        str(args.min_relevance),
        "--best-local-limit",
        str(args.best_local_limit),
        "--answer-sources",
        str(args.answer_sources),
        "--answer-sentences",
        str(args.answer_sentences),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--llm-context-chars",
        str(args.llm_context_chars),
    ]
    if args.limit > 0:
        command.extend(["--limit", str(args.limit)])
    if args.broad:
        command.append("--broad")
    if args.allow_curated_fallback:
        command.append("--allow-curated-fallback")
    if args.best_local_context:
        command.append("--best-local-context")
    if args.answer:
        command.append("--answer")
    if args.llm_answer:
        command.append("--llm-answer")
    if args.model_path is not None:
        command.extend(["--model-path", str(args.model_path)])
    if args.fetch_pages:
        command.append("--fetch-pages")

    print("Running:", " ".join(command))
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
