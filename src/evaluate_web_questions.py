"""Evaluate JSON web questions against the local RAG pipeline."""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

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
    DEFAULT_ARTICLE_BOOST,
    DEFAULT_CANDIDATE_K,
    DEFAULT_LEXICAL_WEIGHT,
    DEFAULT_RETRIEVAL_MODE,
    DEFAULT_SOURCE_BOOST,
    DEFAULT_TITLE_WEIGHT,
    DEFAULT_TOP_K,
    RETRIEVAL_MODES,
    RetrievalEngine,
    infer_query_law_ids,
    tokenize_for_matching,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUESTIONS = PROJECT_ROOT / "evaluation" / "generated_web_questions.json"
DEFAULT_RESULTS = PROJECT_ROOT / "logs" / "web-question-eval-results.jsonl"


def append_run_id(path: Path, run_id: str) -> Path:
    """Append a run id before the file suffix."""
    return path.with_name(f"{path.stem}-{run_id}{path.suffix}")


def normalize_text(text: str) -> str:
    """Normalize text for tolerant lexical comparison."""
    normalized = unicodedata.normalize("NFKD", text.lower())
    without_accents = "".join(char for char in normalized if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", " ", without_accents).strip()


def load_web_questions(path: Path) -> list[dict[str, Any]]:
    """Load a JSON array or JSONL file with question/answer/source_url fields."""
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return []
    if raw.startswith("["):
        records = json.loads(raw)
    else:
        records = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if not isinstance(records, list):
        raise ValueError(f"Expected a list of questions in {path}")
    return [record for record in records if str(record.get("question", "")).strip()]


def write_jsonl(records: Iterable[dict[str, Any]], path: Path) -> None:
    """Write JSON Lines results."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def answer_overlap(expected: str, actual: str) -> float:
    """Return a simple token-overlap score between expected and generated answers."""
    expected_tokens = set(tokenize_for_matching(expected))
    actual_tokens = set(tokenize_for_matching(actual))
    if not expected_tokens:
        return 0.0
    return len(expected_tokens & actual_tokens) / len(expected_tokens)


def source_domain(url: str) -> str:
    """Extract a rough domain from a URL for reporting."""
    match = re.search(r"https?://([^/]+)", url)
    return match.group(1).lower() if match else ""


def gpu_report() -> dict[str, Any]:
    """Return a tiny PyTorch CUDA report for logs."""
    try:
        import torch
    except ImportError as exc:
        return {"torch_available": False, "error": str(exc)}

    report: dict[str, Any] = {
        "torch_available": True,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
    }
    if torch.cuda.is_available():
        report["cuda_device_name"] = torch.cuda.get_device_name(0)
        report["cuda_capability"] = list(torch.cuda.get_device_capability(0))
    return report


def build_parser() -> argparse.ArgumentParser:
    """Create CLI parser."""
    parser = argparse.ArgumentParser(description="Evaluate generated_web_questions.json with local RAG.")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--results-jsonl", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--append-run-id", action="store_true")
    parser.add_argument("--run-id", default=None)
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
    parser.add_argument("--article-boost", type=float, default=DEFAULT_ARTICLE_BOOST)
    parser.add_argument("--title-weight", type=float, default=DEFAULT_TITLE_WEIGHT)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_LOCAL_MODEL_PATH)
    parser.add_argument("--system-prompt", type=Path, default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--generate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gpu-smoke-test", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run web-question evaluation."""
    parser = build_parser()
    args = parser.parse_args(argv)
    run_id = args.run_id or uuid.uuid4().hex[:8]
    if args.append_run_id:
        args.results_jsonl = append_run_id(args.results_jsonl, run_id)

    if args.gpu_smoke_test:
        print("GPU report:", json.dumps(gpu_report(), ensure_ascii=False))

    questions = load_web_questions(args.questions)
    if args.limit > 0:
        questions = questions[: args.limit]
    print(f"Loaded {len(questions)} web question(s) from {args.questions}")
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
        except Exception as exc:
            generation_load_error = str(exc)
            print(f"WARNING: generation disabled because LLM loading failed: {exc}", file=sys.stderr)

    results: list[dict[str, Any]] = []
    overlap_total = 0.0
    source_domain_hits = 0

    for index, case in enumerate(questions, start=1):
        question = str(case["question"])
        expected_answer = str(case.get("answer") or "")
        expected_source_url = str(case.get("source_url") or "")
        expected_domain = source_domain(expected_source_url)
        query_law_ids = sorted(infer_query_law_ids(question))

        chunks = retriever.retrieve(
            question,
            top_k=args.top_k,
            retrieval_mode=args.retrieval_mode,
            candidate_k=args.candidate_k,
            lexical_weight=args.lexical_weight,
            source_boost=args.source_boost,
            article_boost=args.article_boost,
            title_weight=args.title_weight,
            query_law_ids=query_law_ids,
        )

        retrieved_sources = [str(chunk.get("source") or "") for chunk in chunks]
        domain_hit = bool(expected_domain and any(expected_domain in source for source in retrieved_sources))
        source_domain_hits += int(domain_hit)

        answer = None
        answer_error = generation_load_error
        if args.generate and not generation_load_error:
            try:
                messages = build_messages(question, chunks, system_prompt)
                prompt = render_prompt(tokenizer, messages)
                generated = generate_from_prompt(
                    prompt,
                    tokenizer,
                    model,
                    torch_module,
                    max_new_tokens=args.max_new_tokens,
                )
                if not generated.strip():
                    generated = "I did not find this information in the retrieved sources."
                answer = append_sources(generated, chunks, slovenian=looks_slovenian(question))
                answer_error = None
            except Exception as exc:
                answer_error = str(exc)

        overlap = answer_overlap(expected_answer, answer or "")
        overlap_total += overlap
        print(
            f"[{index}/{len(questions)}] id={case.get('id')} "
            f"overlap={overlap:.3f} source_domain_hit={domain_hit} question={question}"
        )

        results.append(
            {
                "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                "run_id": run_id,
                "id": case.get("id"),
                "question": question,
                "expected_answer": expected_answer,
                "expected_source_title": case.get("source_title"),
                "expected_source_url": expected_source_url,
                "expected_source_domain": expected_domain,
                "retrieval_mode": args.retrieval_mode,
                "top_k": args.top_k,
                "query_law_ids": query_law_ids,
                "retrieved": [
                    {
                        "rank": chunk.get("rank"),
                        "score": chunk.get("score"),
                        "source": chunk.get("source"),
                        "chunk_id": chunk.get("chunk_id"),
                        "metadata": chunk.get("metadata"),
                        "text": str(chunk.get("text") or "")[:1200],
                    }
                    for chunk in chunks
                ],
                "retrieved_sources": retrieved_sources,
                "source_domain_hit": domain_hit,
                "answer_overlap": overlap,
                "answer": answer,
                "answer_error": answer_error,
            }
        )

    write_jsonl(results, args.results_jsonl)
    count = max(len(results), 1)
    print(
        "Summary: "
        f"mean_answer_overlap={overlap_total / count:.3f} "
        f"source_domain_hits={source_domain_hits}/{len(results)}"
    )
    print(f"Wrote results to {args.results_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
