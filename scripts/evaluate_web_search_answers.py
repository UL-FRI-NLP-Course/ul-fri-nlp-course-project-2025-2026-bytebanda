#!/usr/bin/env python3
"""Evaluate web-search grounded answers against generated_web_questions."""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from legal_web_suggestions import (  # noqa: E402
    DEFAULT_LOCAL_MODEL_PATH,
    best_local_rag_suggestions,
    make_extractive_answer,
    make_llm_answer,
    source_role,
    web_suggestions,
)
from src.evaluate_web_questions import answer_overlap, source_domain  # noqa: E402
from src.retrieve import tokenize_for_matching  # noqa: E402


DEFAULT_QUESTIONS = PROJECT_ROOT / "evaluation" / "generated_web_questions.json"
DEFAULT_RESULTS = PROJECT_ROOT / "logs" / "web-search-answer-eval.jsonl"
DEFAULT_SUMMARY = PROJECT_ROOT / "logs" / "web-search-answer-eval-summary.json"


def load_loose_json_records(path: Path) -> list[dict[str, Any]]:
    """Load JSON array, JSONL, or consecutive pretty-printed JSON objects."""
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return []
    if raw.startswith("["):
        try:
            return list(json.loads(raw))
        except json.JSONDecodeError:
            raw = raw[1:]
            if raw.rstrip().endswith("]"):
                raw = raw.rstrip()[:-1]

    records: list[dict[str, Any]] = []
    decoder = json.JSONDecoder()
    index = 0
    while index < len(raw):
        while index < len(raw) and raw[index] in " \t\r\n,":
            index += 1
        if index >= len(raw):
            break
        try:
            record, end = decoder.raw_decode(raw, index)
        except json.JSONDecodeError:
            # Last-resort parser for the hand-edited generated_web_questions file.
            pattern = re.compile(
                r'"id"\s*:\s*(?P<id>\d+)\s*,\s*'
                r'"question"\s*:\s*"(?P<question>(?:\\.|[^"\\])*)"\s*,\s*'
                r'"answer"\s*:\s*"(?P<answer>(?:\\.|[^"\\])*)"\s*,\s*'
                r'"source_title"\s*:\s*"(?P<source_title>(?:\\.|[^"\\])*)"\s*,\s*'
                r'"source_url"\s*:\s*"(?P<source_url>(?:\\.|[^"\\])*)"',
                flags=re.DOTALL,
            )
            records = []
            for match in pattern.finditer(raw):
                records.append(
                    {
                        "id": int(match.group("id")),
                        "question": json.loads(f'"{match.group("question")}"'),
                        "answer": json.loads(f'"{match.group("answer")}"'),
                        "source_title": json.loads(f'"{match.group("source_title")}"'),
                        "source_url": json.loads(f'"{match.group("source_url")}"'),
                    }
                )
            break
        if isinstance(record, dict):
            records.append(record)
        index = end
    return [record for record in records if str(record.get("question") or "").strip()]


def write_jsonl(records: Iterable[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def token_f1(expected: str, actual: str) -> float:
    expected_tokens = tokenize_for_matching(expected)
    actual_tokens = tokenize_for_matching(actual)
    if not expected_tokens or not actual_tokens:
        return 0.0
    expected_counts = {token: expected_tokens.count(token) for token in set(expected_tokens)}
    actual_counts = {token: actual_tokens.count(token) for token in set(actual_tokens)}
    overlap = sum(min(expected_counts.get(token, 0), actual_counts.get(token, 0)) for token in expected_counts)
    if overlap == 0:
        return 0.0
    precision = overlap / len(actual_tokens)
    recall = overlap / len(expected_tokens)
    return 2 * precision * recall / (precision + recall)


def contains_expected_source(expected_url: str, suggestions: list[dict[str, Any]]) -> bool:
    domain = source_domain(expected_url)
    if not domain:
        return False
    urls = [str(item.get("url") or "").lower() for item in suggestions]
    domains = [str(item.get("domain") or "").lower() for item in suggestions]
    return any(domain in url for url in urls) or any(domain in item_domain for item_domain in domains)


def official_source_count(suggestions: list[dict[str, Any]]) -> int:
    official_roles = {
        "primary_legal_source",
        "official_guidance",
        "official_register_guidance",
        "official_public_source",
    }
    return sum(1 for item in suggestions if source_role(item) in official_roles)


def parse_judge_json(text: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    parsed: dict[str, Any] = {}
    patterns = {
        "answer_relevance": r"(?:answer[_\s-]*relevance|relevance|relevantnost)\D{0,20}([0-5])",
        "answer_correctness_vs_ground_truth": (
            r"(?:answer[_\s-]*correctness[_\s-]*vs[_\s-]*ground[_\s-]*truth|"
            r"correctness|correct|pravilnost|ocena)\D{0,20}([0-5])"
        ),
        "source_support": r"(?:source[_\s-]*support|support|podpora|viri)\D{0,20}([0-5])",
    }
    for key, pattern in patterns.items():
        value_match = re.search(pattern, text, flags=re.IGNORECASE)
        if value_match:
            parsed[key] = int(value_match.group(1))

    # Last fallback: if the model returned just one or more numbers, use the
    # first as correctness. This is intentionally conservative and keeps raw.
    if "answer_correctness_vs_ground_truth" not in parsed:
        numbers = re.findall(r"\b([0-5])(?:\s*/\s*5)?\b", text)
        if numbers:
            parsed["answer_correctness_vs_ground_truth"] = int(numbers[0])

    if parsed:
        parsed.setdefault("answer_relevance", parsed.get("answer_correctness_vs_ground_truth"))
        parsed.setdefault("source_support", parsed.get("answer_correctness_vs_ground_truth"))
        parsed["parse_warning"] = "Recovered judge scores from non-JSON output"
        return parsed

    return {"parse_error": "No JSON object or recoverable score found", "raw": text}


def llm_judge(
    question: str,
    expected_answer: str,
    predicted_answer: str,
    source_summaries: str,
    llm_bundle: tuple[Any, Any, Any],
    max_new_tokens: int,
) -> dict[str, Any]:
    from src.generate_answer import generate_from_prompt

    tokenizer, model, torch_module = llm_bundle
    prompt = f"""<s>[INST] Evaluate this Slovenian legal/tax QA answer.
Return only one valid JSON object. Use integer scores from 0 to 5.
{{
  "answer_relevance": 0,
  "answer_correctness_vs_ground_truth": 0,
  "source_support": 0,
  "missing_or_wrong_points": "...",
  "short_comment": "..."
}}

Question:
{question}

Ground-truth answer:
{expected_answer}

Predicted answer:
{predicted_answer}

Retrieved source snippets:
{source_summaries[:5000]}
[/INST]"""
    raw = generate_from_prompt(
        prompt,
        tokenizer,
        model,
        torch_module,
        max_new_tokens=max_new_tokens,
    )
    parsed = parse_judge_json(raw)
    parsed["raw"] = raw
    return parsed


def source_summary_text(suggestions: list[dict[str, Any]], limit: int = 5) -> str:
    blocks = []
    for index, item in enumerate(suggestions[:limit], start=1):
        blocks.append(
            f"[{index}] {item.get('title')}\n"
            f"url={item.get('url')}\n"
            f"summary={item.get('summary')}"
        )
    return "\n\n".join(blocks)


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate web-search LLM answers against generated web questions.")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--results-jsonl", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--provider", choices=["duckduckgo", "brave", "tavily"], default="tavily")
    parser.add_argument("--tiers", default="1")
    parser.add_argument("--source-policy", choices=["auto", "all"], default="auto")
    parser.add_argument("--min-relevance", type=float, default=0.25)
    parser.add_argument("--per-domain", type=int, default=1)
    parser.add_argument("--max-suggestions", type=int, default=5)
    parser.add_argument("--broad", action="store_true")
    parser.add_argument("--best-local-context", action="store_true")
    parser.add_argument("--best-local-limit", type=int, default=3)
    parser.add_argument("--extractive-answer", action="store_true")
    parser.add_argument("--llm-answer", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--judge-with-llm", action="store_true")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_LOCAL_MODEL_PATH)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--judge-max-new-tokens", type=int, default=256)
    parser.add_argument("--llm-context-chars", type=int, default=1800)
    parser.add_argument("--answer-sources", type=int, default=4)
    parser.add_argument("--fetch-pages", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_id = args.run_id or uuid.uuid4().hex[:8]
    if args.provider == "tavily" and not os.environ.get("TAVILY_API_KEY"):
        print("WARNING: TAVILY_API_KEY is not set; Tavily web search will fail.", file=sys.stderr)

    cases = load_loose_json_records(args.questions)
    if args.limit > 0:
        cases = cases[: args.limit]
    print(f"Loaded {len(cases)} question(s) from {args.questions}")

    llm_bundle = None
    if args.llm_answer or args.judge_with_llm:
        from src.generate_answer import load_llm

        print(f"Loading LLM from {args.model_path}", flush=True)
        llm_bundle = load_llm(args.model_path)

    results: list[dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        question = str(case.get("question") or "")
        expected_answer = str(case.get("answer") or "")
        expected_source_url = str(case.get("source_url") or "")

        suggestions = web_suggestions(
            question=question,
            tiers={int(value) for value in args.tiers.split(",") if value.strip()},
            per_domain=args.per_domain,
            sleep_seconds=0.1,
            provider=args.provider,
            broad=args.broad,
            allow_curated_fallback=False,
            source_policy=args.source_policy,
            min_relevance=args.min_relevance,
            category=str(case.get("category") or ""),
        )
        if args.best_local_context:
            local_suggestions, _config = best_local_rag_suggestions(question, limit=args.best_local_limit)
            suggestions.extend(local_suggestions)

        suggestions.sort(
            key=lambda item: (
                99 if item.get("source_tier") is None else int(item.get("source_tier") or 99),
                -float(item.get("relevance") or 0.0),
            )
        )
        suggestions = suggestions[: args.max_suggestions]

        if args.llm_answer:
            answer_obj = make_llm_answer(
                question=question,
                suggestions=suggestions,
                model_path=args.model_path,
                fetch_pages=args.fetch_pages,
                max_sources=args.answer_sources,
                max_chars_per_source=args.llm_context_chars,
                max_new_tokens=args.max_new_tokens,
                llm_bundle=llm_bundle,
            )
        elif args.extractive_answer:
            answer_obj = make_extractive_answer(
                question=question,
                suggestions=suggestions,
                fetch_pages=args.fetch_pages,
                max_sources=args.answer_sources,
                max_sentences=4,
            )
        else:
            answer_obj = {"method": "none", "text": ""}

        predicted = str(answer_obj.get("text") or "")
        lexical_recall = answer_overlap(expected_answer, predicted)
        lexical_f1 = token_f1(expected_answer, predicted)
        source_hit = contains_expected_source(expected_source_url, suggestions)
        judge = None
        if args.judge_with_llm and llm_bundle:
            judge = llm_judge(
                question=question,
                expected_answer=expected_answer,
                predicted_answer=predicted,
                source_summaries=source_summary_text(suggestions),
                llm_bundle=llm_bundle,
                max_new_tokens=args.judge_max_new_tokens,
            )

        result = {
            "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "run_id": run_id,
            "id": case.get("id"),
            "question": question,
            "expected_answer": expected_answer,
            "expected_source_title": case.get("source_title"),
            "expected_source_url": expected_source_url,
            "answer": answer_obj,
            "scores": {
                "token_recall_vs_gt": round(lexical_recall, 4),
                "token_f1_vs_gt": round(lexical_f1, 4),
                "expected_source_domain_hit": source_hit,
                "official_source_count": official_source_count(suggestions),
                "mean_source_relevance": round(mean([float(item.get("relevance") or 0.0) for item in suggestions]), 4),
            },
            "llm_judge": judge,
            "suggestions": suggestions,
        }
        results.append(result)
        judge_bits = ""
        if judge and not judge.get("parse_error"):
            judge_bits = (
                f" judge_relevance={judge.get('answer_relevance')} "
                f"judge_correctness={judge.get('answer_correctness_vs_ground_truth')}"
            )
        print(
            f"[{index}/{len(cases)}] id={case.get('id')} "
            f"f1={lexical_f1:.3f} source_hit={source_hit}{judge_bits}"
        )

    write_jsonl(results, args.results_jsonl)

    judge_correctness = [
        float(result["llm_judge"].get("answer_correctness_vs_ground_truth"))
        for result in results
        if result.get("llm_judge") and not result["llm_judge"].get("parse_error")
    ]
    summary = {
        "run_id": run_id,
        "questions": len(results),
        "results_jsonl": str(args.results_jsonl),
        "mean_token_recall_vs_gt": round(mean([result["scores"]["token_recall_vs_gt"] for result in results]), 4),
        "mean_token_f1_vs_gt": round(mean([result["scores"]["token_f1_vs_gt"] for result in results]), 4),
        "expected_source_domain_hit_rate": round(mean([float(result["scores"]["expected_source_domain_hit"]) for result in results]), 4),
        "mean_official_source_count": round(mean([float(result["scores"]["official_source_count"]) for result in results]), 4),
        "mean_source_relevance": round(mean([result["scores"]["mean_source_relevance"] for result in results]), 4),
        "mean_llm_judge_correctness": round(mean(judge_correctness), 4) if judge_correctness else None,
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote results to {args.results_jsonl}")
    print(f"Wrote summary to {args.summary_json}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
