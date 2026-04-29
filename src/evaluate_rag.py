"""Evaluation runner for retrieval and generated tax answers."""

from __future__ import annotations

import argparse
import json
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
from .retrieve import DEFAULT_TOP_K, ensure_index_exists, read_jsonl


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


class Retriever:
    """Reusable FAISS retriever for an evaluation run."""

    def __init__(
        self,
        index_path: Path = DEFAULT_INDEX_PATH,
        chunks_path: Path = DEFAULT_INDEX_CHUNKS,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    ) -> None:
        try:
            import faiss
            import numpy as np
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Evaluation requires faiss-cpu, numpy, and sentence-transformers. "
                "Install requirements.txt first."
            ) from exc

        ensure_index_exists(index_path, chunks_path)
        self.np = np
        self.index = faiss.read_index(str(index_path))
        self.chunks = read_jsonl(chunks_path)
        self.model = SentenceTransformer(embedding_model)

        if self.index.ntotal != len(self.chunks):
            raise ValueError(
                f"Index/chunk mismatch: FAISS has {self.index.ntotal} vectors, "
                f"but {chunks_path} contains {len(self.chunks)} chunks."
            )

    def retrieve(self, question: str, top_k: int) -> List[Dict[str, Any]]:
        """Return top-k chunks for a question."""
        query = self.model.encode(
            [question],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        query = self.np.asarray(query, dtype="float32")
        search_k = min(top_k, len(self.chunks))
        scores, indices = self.index.search(query, search_k)

        results: List[Dict[str, Any]] = []
        for rank, (score, chunk_index) in enumerate(zip(scores[0], indices[0]), start=1):
            if chunk_index < 0:
                continue
            chunk = dict(self.chunks[int(chunk_index)])
            chunk["rank"] = rank
            chunk["score"] = float(score)
            results.append(chunk)
        return results


def evaluate_hits(case: Dict[str, Any], chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compare retrieved chunks with expected sources, chunk ids, and phrases."""
    expected_sources = set(case.get("expected_sources") or [])
    expected_chunks = set(case.get("expected_chunks") or [])
    expected_phrases = case.get("expected_phrases") or []
    retrieved_sources = {chunk.get("source") for chunk in chunks}
    retrieved_chunk_ids = {chunk.get("chunk_id") for chunk in chunks}
    combined_text = "\n".join(chunk.get("text", "") for chunk in chunks)
    normalized_text = normalize_text(combined_text)

    phrase_hits = [
        phrase for phrase in expected_phrases if normalize_text(phrase) in normalized_text
    ]

    return {
        "source_hit": not expected_sources or bool(expected_sources & retrieved_sources),
        "chunk_hit": not expected_chunks or bool(expected_chunks & retrieved_chunk_ids),
        "phrase_hits": phrase_hits,
        "phrase_hit_count": len(phrase_hits),
        "phrase_count": len(expected_phrases),
        "all_phrases_hit": len(phrase_hits) == len(expected_phrases),
    }


def print_case_report(
    index: int,
    total: int,
    case: Dict[str, Any],
    chunks: List[Dict[str, Any]],
    hits: Dict[str, Any],
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
    print(f"EXPECTED CHUNKS: {', '.join(case.get('expected_chunks') or [])}")
    print(f"EXPECTED ANSWER: {case.get('expected_answer')}")
    print()
    print(
        "HITS: "
        f"source={hits['source_hit']} "
        f"chunk={hits['chunk_hit']} "
        f"phrases={hits['phrase_hit_count']}/{hits['phrase_count']} "
        f"all_phrases={hits['all_phrases_hit']}"
    )
    if hits["phrase_hits"]:
        print("PHRASES FOUND:")
        for phrase in hits["phrase_hits"]:
            print(f"- {phrase}")
    print()
    print("RETRIEVED CHUNKS:")
    for chunk in chunks:
        text = chunk.get("text", "")
        if chunk_chars > 0 and len(text) > chunk_chars:
            text = text[:chunk_chars] + " ..."
        print("-" * 100)
        print(
            f"rank={chunk.get('rank')} score={chunk.get('score'):.4f} "
            f"source={chunk.get('source')} chunk_id={chunk.get('chunk_id')}"
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
    parser.add_argument("--model-path", type=Path, default=DEFAULT_LOCAL_MODEL_PATH)
    parser.add_argument("--system-prompt", type=Path, default=DEFAULT_SYSTEM_PROMPT)
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

    questions = load_questions(args.questions)
    if args.limit > 0:
        questions = questions[: args.limit]

    print(f"Loaded {len(questions)} evaluation question(s) from {args.questions}")
    print(f"top_k={args.top_k} generate={args.generate} max_new_tokens={args.max_new_tokens}")
    print(f"results_jsonl={args.results_jsonl}")

    retriever = Retriever(
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
        "chunk_hits": 0,
        "all_phrase_hits": 0,
        "generation_failures": 0,
    }
    category_totals: Dict[str, Dict[str, int]] = {}

    for index, case in enumerate(questions, start=1):
        chunks = retriever.retrieve(case["question"], top_k=args.top_k)
        hits = evaluate_hits(case, chunks)
        totals["source_hits"] += int(hits["source_hit"])
        totals["chunk_hits"] += int(hits["chunk_hit"])
        totals["all_phrase_hits"] += int(hits["all_phrases_hit"])
        category = str(case.get("category") or "unknown")
        category_totals.setdefault(
            category,
            {"questions": 0, "source_hits": 0, "chunk_hits": 0, "all_phrase_hits": 0},
        )
        category_totals[category]["questions"] += 1
        category_totals[category]["source_hits"] += int(hits["source_hit"])
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

        print_case_report(
            index=index,
            total=len(questions),
            case=case,
            chunks=chunks,
            hits=hits,
            answer=answer,
            answer_error=answer_error,
            chunk_chars=args.chunk_chars,
        )

        results.append(
            {
                "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                "id": case.get("id"),
                "category": case.get("category"),
                "question": case.get("question"),
                "expected_sources": case.get("expected_sources"),
                "expected_chunks": case.get("expected_chunks"),
                "expected_locations": case.get("expected_locations"),
                "expected_answer": case.get("expected_answer"),
                "hits": hits,
                "retrieved": [
                    {
                        "rank": chunk.get("rank"),
                        "score": chunk.get("score"),
                        "source": chunk.get("source"),
                        "chunk_id": chunk.get("chunk_id"),
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
    print("=" * 100)
    print("SUMMARY")
    print(f"questions={len(questions)}")
    print(f"source_hit_at_{args.top_k}={totals['source_hits']}/{len(questions)}")
    print(f"chunk_hit_at_{args.top_k}={totals['chunk_hits']}/{len(questions)}")
    print(f"all_expected_phrases_hit_at_{args.top_k}={totals['all_phrase_hits']}/{len(questions)}")
    print(f"generation_failures={totals['generation_failures']}")
    print(f"source_hit_rate={totals['source_hits'] / total:.3f}")
    print(f"chunk_hit_rate={totals['chunk_hits'] / total:.3f}")
    print(f"all_phrase_hit_rate={totals['all_phrase_hits'] / total:.3f}")
    print()
    print("CATEGORY SUMMARY")
    for category, stats in sorted(category_totals.items()):
        count = stats["questions"] or 1
        print(
            f"{category}: "
            f"n={stats['questions']} "
            f"source_hit_rate={stats['source_hits'] / count:.3f} "
            f"chunk_hit_rate={stats['chunk_hits'] / count:.3f} "
            f"all_phrase_hit_rate={stats['all_phrase_hits'] / count:.3f}"
        )
    print(f"Wrote structured results to {args.results_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
