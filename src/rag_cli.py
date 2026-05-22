"""Command-line interface for the minimal Slovenian tax RAG pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .build_index import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_INDEX_CHUNKS,
    DEFAULT_INDEX_PATH,
    DEFAULT_PROCESSED_CHUNKS,
)
from .chunking import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_STRATEGY,
    DEFAULT_OVERLAP,
    LEGAL_CHUNK_STRATEGY,
)
from .generate_answer import DEFAULT_LOCAL_MODEL_PATH, DEFAULT_SYSTEM_PROMPT
from .ingest import DEFAULT_RAW_DIR
from .retrieve import (
    DEFAULT_CANDIDATE_K,
    DEFAULT_ARTICLE_BOOST,
    DEFAULT_LEXICAL_WEIGHT,
    DEFAULT_RETRIEVAL_MODE,
    DEFAULT_SOURCE_BOOST,
    DEFAULT_TITLE_WEIGHT,
    DEFAULT_TOP_K,
    RETRIEVAL_MODES,
)


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(description="Minimal RAG CLI for Slovenian tax documents.")
    parser.add_argument("--build-index", action="store_true", help="Build the FAISS index.")
    parser.add_argument("--ask", type=str, help="Ask a question using the RAG pipeline.")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Number of chunks to retrieve.")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help="Directory containing raw .txt, .md, .pdf, and .html files.",
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help="SentenceTransformers model used for embeddings.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_LOCAL_MODEL_PATH,
        help="Path to the local Hugging Face causal language model.",
    )
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP)
    parser.add_argument(
        "--chunk-strategy",
        choices=[DEFAULT_CHUNK_STRATEGY, LEGAL_CHUNK_STRATEGY],
        default=DEFAULT_CHUNK_STRATEGY,
        help="Chunking strategy: fixed baseline chunks or legal article-aware chunks.",
    )
    parser.add_argument("--processed-chunks-path", type=Path, default=DEFAULT_PROCESSED_CHUNKS)
    parser.add_argument("--index-path", type=Path, default=DEFAULT_INDEX_PATH)
    parser.add_argument("--index-chunks-path", type=Path, default=DEFAULT_INDEX_CHUNKS)
    parser.add_argument(
        "--chunks-path",
        type=Path,
        default=None,
        help="Chunk metadata path used for retrieval. Defaults to --index-chunks-path.",
    )
    parser.add_argument(
        "--retrieval-mode",
        choices=sorted(RETRIEVAL_MODES),
        default=DEFAULT_RETRIEVAL_MODE,
        help="Use dense FAISS scores only, or rerank dense candidates with lexical/legal-source signals.",
    )
    parser.add_argument("--candidate-k", type=int, default=DEFAULT_CANDIDATE_K)
    parser.add_argument("--lexical-weight", type=float, default=DEFAULT_LEXICAL_WEIGHT)
    parser.add_argument("--source-boost", type=float, default=DEFAULT_SOURCE_BOOST)
    parser.add_argument("--article-boost", type=float, default=DEFAULT_ARTICLE_BOOST)
    parser.add_argument("--title-weight", type=float, default=DEFAULT_TITLE_WEIGHT)
    parser.add_argument("--system-prompt", type=Path, default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.build_index and not args.ask:
        parser.print_help()
        return 1

    try:
        if args.build_index:
            from .build_index import build_index

            build_index(
                raw_dir=args.raw_dir,
                processed_chunks_path=args.processed_chunks_path,
                index_path=args.index_path,
                index_chunks_path=args.index_chunks_path,
                embedding_model=args.embedding_model,
                chunk_size=args.chunk_size,
                overlap=args.overlap,
                chunk_strategy=args.chunk_strategy,
                batch_size=args.batch_size,
            )

        if args.ask:
            from .generate_answer import answer_question

            answer = answer_question(
                args.ask,
                top_k=args.top_k,
                index_path=args.index_path,
                chunks_path=args.chunks_path or args.index_chunks_path,
                model_path=args.model_path,
                system_prompt_path=args.system_prompt,
                embedding_model=args.embedding_model,
                retrieval_mode=args.retrieval_mode,
                candidate_k=args.candidate_k,
                lexical_weight=args.lexical_weight,
                source_boost=args.source_boost,
                article_boost=args.article_boost,
                title_weight=args.title_weight,
                max_new_tokens=args.max_new_tokens,
            )
            print(answer)

    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
