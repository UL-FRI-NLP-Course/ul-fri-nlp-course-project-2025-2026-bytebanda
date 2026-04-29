"""Evaluation runner for retrieval and generated tax answers."""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .build_index import DEFAULT_EMBEDDING_MODEL, DEFAULT_INDEX_CHUNKS, DEFAULT_INDEX_PATH
from .generate_answer import (
    DEFAULT_LOCAL_MODEL_PATH,
    DEFAULT_SYSTEM_PROMPT,
    append_sources,
    build_messages,
    generate_from_prompt,
    load_llm,
    load_system_prompt,
    looks_slovenian,
    render_prompt,
)
from .retrieve import (
    DEFAULT_CANDIDATE_K,
    DEFAULT_LEXICAL_WEIGHT,
    DEFAULT_RETRIEVAL_MODE,
    DEFAULT_SOURCE_BOOST,
    DEFAULT_TOP_K,
    RETRIEVAL_MODES,
    RetrievalEngine,
    tokenize_for_matching,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUESTIONS = PROJECT_ROOT / "evaluation" / "tax_eval_questions.jsonl"
DEFAULT_RESULTS = PROJECT_ROOT / "logs" / "rag-eval-results.jsonl"


def normalize_text(text: str) -> str:
    """Return lowercase text without combining accents for tolerant matching."""
    normalized = unicodedata.normalize("NFKD", text.lower())
    return "".join(char for char in normalized if not unicodedata.combining(char))


def load_questions(path: Path) -> List[Dict[str, Any]]:
    """Load JSONL evaluation questions."""
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(records: Iterable[Dict[str, Any]], path: Path) -> None:
    """Write JSONL evaluation results."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def extract_expected_articles(case: Dict[str, Any]) -> List[str]:
    """Extract expected article numbers from human-readable expected locations."""
    articles: List[str] = []
    for location in case.get("expected_locations") or []:
        normalized = normalize_text(str(location))
        for match in re.finditer(r"(\d+(?:\.[a-z])?)\.?\s*clen", normalized):
            article = match.group(1).rstrip(".")
            if article not in articles:
                articles.append(article)
    return articles


def chunk_article_number(chunk: Dict[str, Any]) -> str | None:
    """Return normalized article metadata for one chunk if available."""
    metadata = chunk.get("metadata") or {}
    article = metadata.get("article_number")
    if article is not None:
        return str(article).rstrip(".")

    text_head = normalize_text(chunk.get("text", "")[:500])
    match = re.search(r"(\d+(?:\.[a-z])?)\.?\s*clen", text_head)
    if match:
        return match.group(1).rstrip(".")
    return None


def evaluate_hits(case: Dict[str, Any], chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compare retrieved chunks with expected sources, chunk ids, articles, and phrases."""
    expected_sources = set(case.get("expected_sources") or [])
    expected_chunks = set(case.get("expected_chunks") or [])
    expected_phrases = case.get("expected_phrases") or []
    expected_articles = set(extract_expected_articles(case))
    retrieved_sources = {chunk.get("source") for chunk in chunks}
    retrieved_chunk_ids = {chunk.get("chunk_id") for chunk in chunks}
    combined_text = "\n".join(chunk.get("text", "") for chunk in chunks)
    normalized_text = normalize_text(combined_text)

    phrase_hits = [
        phrase for phrase in expected_phrases if normalize_text(phrase) in normalized_text
    ]
    article_hits = []
    for chunk in chunks:
        source_ok = not expected_sources or chunk.get("source") in expected_sources
        article = chunk_article_number(chunk)
        article_ok = not expected_articles or article in expected_articles
        if source_ok and article_ok:
            article_hits.append(
                {
                    "source": chunk.get("source"),
                    "chunk_id": chunk.get("chunk_id"),
                    "article_number": article,
                }
            )

    return {
        "source_hit": not expected_sources or bool(expected_sources & retrieved_sources),
        "chunk_hit": not expected_chunks or bool(expected_chunks & retrieved_chunk_ids),
        "article_hit": not expected_articles or bool(article_hits),
        "article_hits": article_hits,
        "expected_articles": sorted(expected_articles),
        "phrase_hits": phrase_hits,
        "phrase_hit_count": len(phrase_hits),
        "phrase_count": len(expected_phrases),
        "all_phrases_hit": len(phrase_hits) == len(expected_phrases),
    }


def answer_body(answer: str | None) -> str:
    """Remove deterministic source appendix from an answer before scoring text."""
    if not answer:
        return ""
    for marker in ("\n\nViri:", "\n\nSources:"):
        if marker in answer:
            return answer.split(marker, 1)[0]
    return answer


def answer_says_not_found(answer: str | None) -> bool:
    """Detect fallback answers that say the retrieved context did not contain the answer."""
    normalized = normalize_text(answer_body(answer))
    markers = (
        "ne najdem",
        "ni najden",
        "ni bilo najden",
        "not found",
        "not contain",
        "provided sources do not",
    )
    return any(marker in normalized for marker in markers)


def token_overlap(reference: str, candidate: str) -> float:
    """Return fraction of reference tokens covered by candidate tokens."""
    reference_tokens = set(tokenize_for_matching(reference))
    if not reference_tokens:
        return 0.0
    candidate_tokens = set(tokenize_for_matching(candidate))
    return len(reference_tokens & candidate_tokens) / len(reference_tokens)


def score_context_relevance(hits: Dict[str, Any]) -> int:
    """Score retrieved context quality on a 0-2 scale."""
    if hits["article_hit"] or hits["all_phrases_hit"]:
        return 2
    if hits["source_hit"] or hits["phrase_hit_count"] > 0:
        return 1
    return 0


def score_answer_correctness(case: Dict[str, Any], answer: str | None, answer_error: str | None) -> int:
    """Score whether generated answer covers the expected answer on a 0-2 scale."""
    if answer_error or not answer:
        return 0
    if answer_says_not_found(answer):
        return 0

    body = answer_body(answer)
    expected_phrases = case.get("expected_phrases") or []
    phrase_hits = [
        phrase for phrase in expected_phrases if normalize_text(phrase) in normalize_text(body)
    ]
    expected_answer = case.get("expected_answer") or " ".join(expected_phrases)
    overlap = token_overlap(expected_answer, body)

    if expected_phrases and len(phrase_hits) == len(expected_phrases):
        return 2
    if overlap >= 0.65:
        return 2
    if phrase_hits or overlap >= 0.35:
        return 1
    return 0


def score_faithfulness(answer: str | None, answer_error: str | None, chunks: List[Dict[str, Any]], context_score: int) -> int:
    """Score whether generated answer is supported by retrieved context on a 0-2 scale."""
    if answer_error or not answer:
        return 0
    if answer_says_not_found(answer):
        return 2 if context_score == 0 else 1

    context_text = "\n".join(chunk.get("text", "") for chunk in chunks)
    overlap = token_overlap(answer_body(answer), context_text)
    if overlap >= 0.75:
        return 2
    if overlap >= 0.45:
        return 1
    return 0


def evaluate_scores(
    case: Dict[str, Any],
    chunks: List[Dict[str, Any]],
    hits: Dict[str, Any],
    answer: str | None,
    answer_error: str | None,
) -> Dict[str, Any]:
    """Compute automatic 0-2 evaluation scores for one case."""
    context_relevance = score_context_relevance(hits)
    return {
        "context_relevance": context_relevance,
        "faithfulness": score_faithfulness(answer, answer_error, chunks, context_relevance),
        "answer_correctness": score_answer_correctness(case, answer, answer_error),
    }


def print_case_report(
    index: int,
    total: int,
    case: Dict[str, Any],
    chunks: List[Dict[str, Any]],
    hits: Dict[str, Any],
    scores: Dict[str, Any],
    answer: str | None,
    answer_error: str | None,
    chunk_chars: int,
) -> None:
    """Print a readable case report to stdout, intended for SLURM logs."""
    print("=" * 100)
    print(f"CASE {index}/{total}: {case.get('id')} [{case.get('category')}]")
    print(f"QUESTION: {case.get('question')}")
    print()
    print("EXPECTED LOCATIONS:")
    for location in case.get("expected_locations") or []:
        print(f"- {location}")
    print(f"EXPECTED ARTICLES: {', '.join(hits.get('expected_articles') or [])}")
    print(f"EXPECTED CHUNKS: {', '.join(case.get('expected_chunks') or [])}")
    print(f"EXPECTED ANSWER: {case.get('expected_answer')}")
    print()
    print(
        "HITS: "
        f"source={hits['source_hit']} "
        f"article={hits['article_hit']} "
        f"chunk={hits['chunk_hit']} "
        f"phrases={hits['phrase_hit_count']}/{hits['phrase_count']} "
        f"all_phrases={hits['all_phrases_hit']}"
    )
    print(
        "SCORES: "
        f"context_relevance={scores['context_relevance']}/2 "
        f"faithfulness={scores['faithfulness']}/2 "
        f"answer_correctness={scores['answer_correctness']}/2"
    )
    if hits["phrase_hits"]:
        print("PHRASES FOUND IN RETRIEVED CONTEXT:")
        for phrase in hits["phrase_hits"]:
            print(f"- {phrase}")
    print()
    print("RETRIEVED CHUNKS:")
    for chunk in chunks:
        text = chunk.get("text", "")
        if chunk_chars > 0 and len(text) > chunk_chars:
            text = text[:chunk_chars] + " ..."
        metadata = chunk.get("metadata") or {}
        article = metadata.get("article_number")
        law = metadata.get("law_id")
        print("-" * 100)
        print(
            f"rank={chunk.get('rank')} score={chunk.get('score'):.4f} "
            f"dense={chunk.get('dense_score', chunk.get('score')):.4f} "
            f"lexical={chunk.get('lexical_score', 0.0):.4f} "
            f"boost={chunk.get('source_boost', 0.0):.4f} "
            f"source={chunk.get('source')} law={law} article={article} "
            f"chunk_id={chunk.get('chunk_id')}"
        )
        print(text)
    print()
    print("GENERATED ANSWER:")
    if answer_error:
        print(f"[generation failed] {answer_error}")
    elif answer:
        print(answer)
    else:
        print("[generation disabled]")
    print()


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""
    parser = argparse.ArgumentParser(description="Run retrieval/generation evaluation cases.")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--results-jsonl", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--index-path", type=Path, default=DEFAULT_INDEX_PATH)
    parser.add_argument("--chunks-path", type=Path, default=DEFAULT_INDEX_CHUNKS)
    parser.add_argument(
        "--retrieval-mode",
        choices=sorted(RETRIEVAL_MODES),
        default=DEFAULT_RETRIEVAL_MODE,
    )
    parser.add_argument("--candidate-k", type=int, default=DEFAULT_CANDIDATE_K)
    parser.add_argument("--lexical-weight", type=float, default=DEFAULT_LEXICAL_WEIGHT)
    parser.add_argument("--source-boost", type=float, default=DEFAULT_SOURCE_BOOST)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_LOCAL_MODEL_PATH)
    parser.add_argument("--system-prompt", type=Path, default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--prompt-label", default="default")
    parser.add_argument("--run-label", default="rag-eval")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--chunk-chars", type=int, default=1600)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--generate", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the evaluation."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.top_k <= 0:
        parser.error("--top-k must be greater than zero")
    if args.candidate_k <= 0:
        parser.error("--candidate-k must be greater than zero")

    questions = load_questions(args.questions)
    if args.limit > 0:
        questions = questions[: args.limit]

    print(f"Loaded {len(questions)} evaluation question(s) from {args.questions}")
    print(
        f"run_label={args.run_label} prompt_label={args.prompt_label} "
        f"top_k={args.top_k} retrieval_mode={args.retrieval_mode} "
        f"candidate_k={args.candidate_k} lexical_weight={args.lexical_weight} "
        f"source_boost={args.source_boost} generate={args.generate} "
        f"max_new_tokens={args.max_new_tokens}"
    )
    print(f"index_path={args.index_path}")
    print(f"chunks_path={args.chunks_path}")
    print(f"results_jsonl={args.results_jsonl}")

    retriever = RetrievalEngine(
        index_path=args.index_path,
        chunks_path=args.chunks_path,
        embedding_model=args.embedding_model,
    )

    tokenizer = model = torch_module = system_prompt = None
    generation_load_error = None
    if args.generate:
        try:
            system_prompt = load_system_prompt(args.system_prompt)
            tokenizer, model, torch_module = load_llm(args.model_path)
        except Exception as exc:  # Keep retrieval evaluation useful if generation fails.
            generation_load_error = str(exc)
            print(f"WARNING: generation disabled because LLM loading failed: {exc}", file=sys.stderr)

    results: List[Dict[str, Any]] = []
    totals = {
        "source_hits": 0,
        "article_hits": 0,
        "chunk_hits": 0,
        "all_phrase_hits": 0,
        "generation_failures": 0,
        "context_relevance": 0,
        "faithfulness": 0,
        "answer_correctness": 0,
    }
    category_totals: Dict[str, Dict[str, int]] = {}

    for index, case in enumerate(questions, start=1):
        chunks = retriever.retrieve(
            case["question"],
            top_k=args.top_k,
            retrieval_mode=args.retrieval_mode,
            candidate_k=args.candidate_k,
            lexical_weight=args.lexical_weight,
            source_boost=args.source_boost,
        )
        hits = evaluate_hits(case, chunks)
        totals["source_hits"] += int(hits["source_hit"])
        totals["article_hits"] += int(hits["article_hit"])
        totals["chunk_hits"] += int(hits["chunk_hit"])
        totals["all_phrase_hits"] += int(hits["all_phrases_hit"])
        category = str(case.get("category") or "unknown")
        category_totals.setdefault(
            category,
            {
                "questions": 0,
                "source_hits": 0,
                "article_hits": 0,
                "chunk_hits": 0,
                "all_phrase_hits": 0,
                "context_relevance": 0,
                "faithfulness": 0,
                "answer_correctness": 0,
            },
        )
        category_totals[category]["questions"] += 1
        category_totals[category]["source_hits"] += int(hits["source_hit"])
        category_totals[category]["article_hits"] += int(hits["article_hit"])
        category_totals[category]["chunk_hits"] += int(hits["chunk_hit"])
        category_totals[category]["all_phrase_hits"] += int(hits["all_phrases_hit"])

        answer = None
        answer_error = generation_load_error
        if args.generate and not generation_load_error:
            try:
                messages = build_messages(case["question"], chunks, system_prompt)
                prompt = render_prompt(tokenizer, messages)
                generated = generate_from_prompt(
                    prompt,
                    tokenizer,
                    model,
                    torch_module,
                    max_new_tokens=args.max_new_tokens,
                )
                if not generated.strip():
                    generated = "V pridobljenih virih tega podatka ne najdem."
                answer = append_sources(
                    generated,
                    chunks,
                    slovenian=looks_slovenian(case["question"]),
                )
                answer_error = None
            except Exception as exc:
                answer_error = str(exc)
                totals["generation_failures"] += 1

        scores = evaluate_scores(case, chunks, hits, answer, answer_error)
        for key in ("context_relevance", "faithfulness", "answer_correctness"):
            totals[key] += int(scores[key])
            category_totals[category][key] += int(scores[key])

        print_case_report(
            index=index,
            total=len(questions),
            case=case,
            chunks=chunks,
            hits=hits,
            scores=scores,
            answer=answer,
            answer_error=answer_error,
            chunk_chars=args.chunk_chars,
        )

        results.append(
            {
                "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                "run_label": args.run_label,
                "prompt_label": args.prompt_label,
                "retrieval_mode": args.retrieval_mode,
                "top_k": args.top_k,
                "candidate_k": args.candidate_k,
                "lexical_weight": args.lexical_weight,
                "source_boost": args.source_boost,
                "id": case.get("id"),
                "category": case.get("category"),
                "question": case.get("question"),
                "expected_sources": case.get("expected_sources"),
                "expected_chunks": case.get("expected_chunks"),
                "expected_locations": case.get("expected_locations"),
                "expected_articles": hits.get("expected_articles"),
                "expected_answer": case.get("expected_answer"),
                "hits": hits,
                "scores": scores,
                "retrieved": [
                    {
                        "rank": chunk.get("rank"),
                        "score": chunk.get("score"),
                        "dense_score": chunk.get("dense_score"),
                        "lexical_score": chunk.get("lexical_score"),
                        "source_boost": chunk.get("source_boost"),
                        "source": chunk.get("source"),
                        "chunk_id": chunk.get("chunk_id"),
                        "metadata": chunk.get("metadata"),
                        "text": chunk.get("text"),
                    }
                    for chunk in chunks
                ],
                "answer": answer,
                "answer_error": answer_error,
            }
        )

    write_jsonl(results, args.results_jsonl)

    total = len(questions) or 1
    max_score = 2 * total
    print("=" * 100)
    print("SUMMARY")
    print(f"questions={len(questions)}")
    print(f"source_hit_at_{args.top_k}={totals['source_hits']}/{len(questions)}")
    print(f"article_hit_at_{args.top_k}={totals['article_hits']}/{len(questions)}")
    print(f"chunk_hit_at_{args.top_k}={totals['chunk_hits']}/{len(questions)}")
    print(f"all_expected_phrases_hit_at_{args.top_k}={totals['all_phrase_hits']}/{len(questions)}")
    print(f"generation_failures={totals['generation_failures']}")
    print(f"source_hit_rate={totals['source_hits'] / total:.3f}")
    print(f"article_hit_rate={totals['article_hits'] / total:.3f}")
    print(f"chunk_hit_rate={totals['chunk_hits'] / total:.3f}")
    print(f"all_phrase_hit_rate={totals['all_phrase_hits'] / total:.3f}")
    print(f"context_relevance_mean={totals['context_relevance'] / total:.3f}/2")
    print(f"faithfulness_mean={totals['faithfulness'] / total:.3f}/2")
    print(f"answer_correctness_mean={totals['answer_correctness'] / total:.3f}/2")
    print(f"context_relevance_total={totals['context_relevance']}/{max_score}")
    print(f"faithfulness_total={totals['faithfulness']}/{max_score}")
    print(f"answer_correctness_total={totals['answer_correctness']}/{max_score}")
    print()
    print("CATEGORY SUMMARY")
    for category, stats in sorted(category_totals.items()):
        count = stats["questions"] or 1
        print(
            f"{category}: "
            f"n={stats['questions']} "
            f"source_hit_rate={stats['source_hits'] / count:.3f} "
            f"article_hit_rate={stats['article_hits'] / count:.3f} "
            f"chunk_hit_rate={stats['chunk_hits'] / count:.3f} "
            f"all_phrase_hit_rate={stats['all_phrase_hits'] / count:.3f} "
            f"context={stats['context_relevance'] / count:.3f}/2 "
            f"faithfulness={stats['faithfulness'] / count:.3f}/2 "
            f"correctness={stats['answer_correctness'] / count:.3f}/2"
        )
    print(f"Wrote structured results to {args.results_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
