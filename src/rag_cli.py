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
from .generate_answer import (
    DEFAULT_CHAT_SYSTEM_PROMPT,
    DEFAULT_LOCAL_MODEL_PATH,
    DEFAULT_SYSTEM_PROMPT,
)
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
    parser.add_argument(
        "--chat",
        action="store_true",
        help="Start an interactive RAG chat. Each turn retrieves sources before generation.",
    )
    parser.add_argument(
        "--direct-chat",
        action="store_true",
        help="Start a plain model chat without retrieval or source citations.",
    )
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
    parser.add_argument(
        "--chat-system-prompt",
        default=DEFAULT_CHAT_SYSTEM_PROMPT,
        help="System prompt text used by --chat --direct-chat.",
    )
    parser.add_argument(
        "--chat-history-messages",
        type=int,
        default=6,
        help="Recent chat messages included in RAG generation for follow-up context.",
    )
    parser.add_argument(
        "--chat-retrieval-history-turns",
        type=int,
        default=2,
        help="Recent user turns included in the retrieval query for follow-up context.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    return parser


def run_direct_chat(args: argparse.Namespace) -> None:
    """Run an interactive chat loop without retrieval."""
    from .generate_answer import generate_chat_reply, load_llm

    tokenizer, model, torch_module = load_llm(args.model_path)
    history = [{"role": "system", "content": args.chat_system_prompt}]

    print("Direct chat mode. Type /exit, /quit, or press Ctrl-D to stop.")
    while True:
        try:
            user_message = input("\nYou: ").strip()
        except EOFError:
            print()
            return

        if not user_message:
            continue
        if user_message.lower() in {"/exit", "/quit"}:
            return

        history.append({"role": "user", "content": user_message})
        answer = generate_chat_reply(
            history,
            tokenizer,
            model,
            torch_module,
            max_new_tokens=args.max_new_tokens,
        )
        if not answer:
            answer = "Ne znam sestaviti odgovora na to sporocilo."
        history.append({"role": "assistant", "content": answer})
        print(f"\nAssistant: {answer}")


def build_chat_retrieval_query(
    history: list[dict[str, str]],
    user_message: str,
    history_turns: int,
) -> str:
    """Build a retrieval query that keeps short follow-up context."""
    if history_turns <= 0:
        return user_message

    previous_user_messages = [
        message["content"].strip()
        for message in history
        if message.get("role") == "user" and message.get("content", "").strip()
    ][-history_turns:]
    if not previous_user_messages:
        return user_message

    previous = "\n".join(f"- {message}" for message in previous_user_messages)
    return (
        "Prejsnja vprasanja v pogovoru:\n"
        f"{previous}\n\n"
        "Trenutno vprasanje:\n"
        f"{user_message}"
    )


def print_source_summary(chunks: list[dict]) -> None:
    """Print compact source metadata for the last RAG chat turn."""
    if not chunks:
        print("No retrieved sources yet.")
        return

    print("Retrieved sources:")
    for chunk in chunks:
        metadata = chunk.get("metadata") or {}
        law = metadata.get("law_id") or "-"
        article = metadata.get("article_number") or "-"
        title = metadata.get("article_title") or ""
        title_text = f" ({title})" if title else ""
        print(
            f"- rank={chunk.get('rank')} source={chunk.get('source')} "
            f"law={law} article={article}{title_text} "
            f"chunk_id={chunk.get('chunk_id')} score={chunk.get('score'):.4f}"
        )


def run_rag_chat(args: argparse.Namespace) -> None:
    """Run an interactive chat loop with retrieval before every answer."""
    from .generate_answer import (
        append_sources,
        build_chat_messages,
        generate_from_prompt,
        load_llm,
        load_system_prompt,
        looks_slovenian,
        render_prompt,
        strip_source_appendix,
    )
    from .retrieve import RetrievalEngine

    chunks_path = args.chunks_path or args.index_chunks_path
    system_prompt = load_system_prompt(args.system_prompt)
    retriever = RetrievalEngine(
        index_path=args.index_path,
        chunks_path=chunks_path,
        embedding_model=args.embedding_model,
    )
    tokenizer, model, torch_module = load_llm(args.model_path)

    history: list[dict[str, str]] = []
    last_chunks: list[dict] = []

    print("RAG chat mode. Each turn retrieves sources before generation.")
    print("Commands: /sources, /clear, /exit, /quit. Press Ctrl-D to stop.")
    while True:
        try:
            user_message = input("\nYou: ").strip()
        except EOFError:
            print()
            return

        if not user_message:
            continue

        command = user_message.lower()
        if command in {"/exit", "/quit"}:
            return
        if command == "/clear":
            history.clear()
            last_chunks = []
            print("Chat history cleared.")
            continue
        if command == "/sources":
            print_source_summary(last_chunks)
            continue

        retrieval_query = build_chat_retrieval_query(
            history,
            user_message,
            args.chat_retrieval_history_turns,
        )
        chunks = retriever.retrieve(
            retrieval_query,
            top_k=args.top_k,
            retrieval_mode=args.retrieval_mode,
            candidate_k=args.candidate_k,
            lexical_weight=args.lexical_weight,
            source_boost=args.source_boost,
            article_boost=args.article_boost,
            title_weight=args.title_weight,
        )

        if not chunks:
            answer = (
                "V indeksu nisem nasel relevantnih virov za to vprasanje."
                if looks_slovenian(user_message)
                else "I did not find relevant sources in the index for this question."
            )
            print(f"\nAssistant: {answer}")
            history.append({"role": "user", "content": user_message})
            history.append({"role": "assistant", "content": answer})
            last_chunks = []
            continue

        messages = build_chat_messages(
            user_message,
            chunks,
            system_prompt,
            history=history,
            max_history_messages=args.chat_history_messages,
        )
        prompt = render_prompt(tokenizer, messages)
        generated = generate_from_prompt(
            prompt,
            tokenizer,
            model,
            torch_module,
            max_new_tokens=args.max_new_tokens,
        )
        if not generated.strip():
            generated = (
                "V pridobljenih virih tega podatka ne najdem."
                if looks_slovenian(user_message)
                else "I did not find this information in the retrieved sources."
            )

        answer = append_sources(
            generated,
            chunks,
            slovenian=looks_slovenian(user_message),
        )
        print(f"\nAssistant: {answer}")

        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": strip_source_appendix(generated)})
        last_chunks = chunks


def main(argv: list[str] | None = None) -> int:
    """Run the CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.build_index and not args.ask and not args.chat and not args.direct_chat:
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

        if args.direct_chat:
            run_direct_chat(args)
        elif args.chat:
            run_rag_chat(args)

    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
