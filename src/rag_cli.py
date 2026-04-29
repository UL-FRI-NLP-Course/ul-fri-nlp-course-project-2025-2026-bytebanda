"""Command-line interface for the minimal Slovenian tax RAG pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .build_index import DEFAULT_EMBEDDING_MODEL
from .chunking import DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP
from .generate_answer import DEFAULT_LOCAL_MODEL_PATH
from .ingest import DEFAULT_RAW_DIR
from .retrieve import DEFAULT_TOP_K


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
                embedding_model=args.embedding_model,
                chunk_size=args.chunk_size,
                overlap=args.overlap,
                batch_size=args.batch_size,
            )

        if args.ask:
            from .generate_answer import answer_question

            answer = answer_question(
                args.ask,
                top_k=args.top_k,
                model_path=args.model_path,
                embedding_model=args.embedding_model,
                max_new_tokens=args.max_new_tokens,
            )
            print(answer)

    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
