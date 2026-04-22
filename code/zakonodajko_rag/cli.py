from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .answering import answer_question
from .classla_support import annotate_jsonl, lemmatize_query_text, lemmatize_query_texts
from .constants import (
    DEFAULT_ANNOTATION_PATH,
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_BM25_PATH,
    DEFAULT_CHROMA_DIR,
    DEFAULT_CHUNK_PATH,
    DEFAULT_DOWNLOAD_REPORT,
    DEFAULT_EMBEDDING_PROFILE,
    DEFAULT_FURS_DDV_TOPIC_URL,
    DEFAULT_EVAL_PATH,
    DEFAULT_FURS_DOWNLOAD_DIR,
    DEFAULT_FURS_GUIDANCE_URL,
    DEFAULT_FURS_MIN_YEAR,
    DEFAULT_FURS_PORTAL_MAX_PAGES,
    DEFAULT_MAX_CHUNK_CHARS,
    DEFAULT_LOCAL_GENERATOR_MAX_NEW_TOKENS,
    DEFAULT_LOCAL_GENERATOR_MODEL,
    DEFAULT_PARSED_PATH,
    DEFAULT_REAL_EVAL_DOWNLOAD_DIR,
    DEFAULT_REAL_EVAL_PATH,
    DEFAULT_SPLIT_TRIGGER_CHARS,
    DEFAULT_UNIT_PATH,
    DEFAULT_OVERLAP_CHARS,
)
from .furs import (
    build_furs_annotation_units,
    fetch_and_parse_furs_ddv_technical_resources,
    fetch_and_parse_furs_guidance,
    fetch_and_parse_furs_portal_resources,
)
from .pisrs import build_annotation_units, chunk_units, parse_all_documents
from .real_eval import build_furs_real_eval_dataset, evaluate_real_answers, load_real_eval_rows
from .retrieval import (
    build_bm25_index,
    build_dense_index,
    compare_embedding_profiles,
    ensure_dense_index,
    evaluate_retrieval,
    hybrid_search,
    load_chunk_map,
)
from .router import build_query_profile, route_query
from .text_utils import read_jsonl, write_jsonl
from .webapp import WebAppSettings, create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Zakonodajko PISRS ingestion and retrieval baseline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build-all", help="Parse documents, annotate, chunk, and build indexes")
    build_parser.add_argument("--download-report", type=Path, default=DEFAULT_DOWNLOAD_REPORT)
    build_parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    build_parser.add_argument("--classla-python", type=str, default=None)
    build_parser.add_argument("--split-trigger-chars", type=int, default=DEFAULT_SPLIT_TRIGGER_CHARS)
    build_parser.add_argument("--max-chunk-chars", type=int, default=DEFAULT_MAX_CHUNK_CHARS)
    build_parser.add_argument("--overlap-chars", type=int, default=DEFAULT_OVERLAP_CHARS)
    build_parser.add_argument("--include-furs", action="store_true")
    build_parser.add_argument("--furs-guidance-url", type=str, default=DEFAULT_FURS_GUIDANCE_URL)
    build_parser.add_argument("--furs-ddv-topic-url", type=str, default=DEFAULT_FURS_DDV_TOPIC_URL)
    build_parser.add_argument("--furs-download-dir", type=Path, default=DEFAULT_FURS_DOWNLOAD_DIR)
    build_parser.add_argument("--furs-min-year", type=int, default=DEFAULT_FURS_MIN_YEAR)
    build_parser.add_argument("--include-furs-ddv-technical", action=argparse.BooleanOptionalAction, default=True)
    build_parser.add_argument("--include-furs-portal-pages", action=argparse.BooleanOptionalAction, default=True)
    build_parser.add_argument("--furs-portal-max-pages", type=int, default=DEFAULT_FURS_PORTAL_MAX_PAGES)
    build_parser.add_argument("--preserve-real-eval-holdout", action=argparse.BooleanOptionalAction, default=True)
    build_parser.add_argument("--embedding-profile", type=str, default=DEFAULT_EMBEDDING_PROFILE)
    build_parser.add_argument("--embedding-model-name", type=str, default=None)

    query_parser = subparsers.add_parser("query", help="Run hybrid retrieval for a single query")
    query_parser.add_argument("query", type=str)
    query_parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNK_PATH)
    query_parser.add_argument("--chroma-dir", type=Path, default=DEFAULT_CHROMA_DIR)
    query_parser.add_argument("--bm25-path", type=Path, default=DEFAULT_BM25_PATH)
    query_parser.add_argument("--classla-python", type=str, default=None)
    query_parser.add_argument("--top-k", type=int, default=5)
    query_parser.add_argument("--embedding-profile", type=str, default=DEFAULT_EMBEDDING_PROFILE)
    query_parser.add_argument("--embedding-model-name", type=str, default=None)
    query_parser.add_argument("--reranker-model", type=str, default=None)

    answer_parser = subparsers.add_parser("answer", help="Answer a question from retrieved legal chunks")
    answer_parser.add_argument("query", type=str)
    answer_parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNK_PATH)
    answer_parser.add_argument("--chroma-dir", type=Path, default=DEFAULT_CHROMA_DIR)
    answer_parser.add_argument("--bm25-path", type=Path, default=DEFAULT_BM25_PATH)
    answer_parser.add_argument("--classla-python", type=str, default=None)
    answer_parser.add_argument("--top-k", type=int, default=5)
    answer_parser.add_argument("--embedding-profile", type=str, default=DEFAULT_EMBEDDING_PROFILE)
    answer_parser.add_argument("--embedding-model-name", type=str, default=None)
    answer_parser.add_argument("--generator-model", type=str, default=DEFAULT_LOCAL_GENERATOR_MODEL)
    answer_parser.add_argument("--reranker-model", type=str, default=None)
    answer_parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_LOCAL_GENERATOR_MAX_NEW_TOKENS)
    answer_parser.add_argument("--extractive-only", action="store_true")

    serve_parser = subparsers.add_parser("serve", help="Run the local chat web app")
    serve_parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNK_PATH)
    serve_parser.add_argument("--chroma-dir", type=Path, default=DEFAULT_CHROMA_DIR)
    serve_parser.add_argument("--bm25-path", type=Path, default=DEFAULT_BM25_PATH)
    serve_parser.add_argument("--classla-python", type=str, default=None)
    serve_parser.add_argument("--top-k", type=int, default=5)
    serve_parser.add_argument("--embedding-profile", type=str, default=DEFAULT_EMBEDDING_PROFILE)
    serve_parser.add_argument("--embedding-model-name", type=str, default=None)
    serve_parser.add_argument("--generator-model", type=str, default=DEFAULT_LOCAL_GENERATOR_MODEL)
    serve_parser.add_argument("--reranker-model", type=str, default=None)
    serve_parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_LOCAL_GENERATOR_MAX_NEW_TOKENS)
    serve_parser.add_argument("--extractive-only", action="store_true")
    serve_parser.add_argument("--host", type=str, default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate dense and hybrid retrieval")
    eval_parser.add_argument("--evaluation-path", type=Path, default=DEFAULT_EVAL_PATH)
    eval_parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNK_PATH)
    eval_parser.add_argument("--chroma-dir", type=Path, default=DEFAULT_CHROMA_DIR)
    eval_parser.add_argument("--bm25-path", type=Path, default=DEFAULT_BM25_PATH)
    eval_parser.add_argument("--classla-python", type=str, default=None)
    eval_parser.add_argument("--top-k", type=int, default=5)
    eval_parser.add_argument("--embedding-profile", type=str, default=DEFAULT_EMBEDDING_PROFILE)
    eval_parser.add_argument("--embedding-model-name", type=str, default=None)
    eval_parser.add_argument("--reranker-model", type=str, default=None)

    compare_parser = subparsers.add_parser("compare-embeddings", help="A/B compare embedding profiles")
    compare_parser.add_argument("--evaluation-path", type=Path, default=DEFAULT_EVAL_PATH)
    compare_parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNK_PATH)
    compare_parser.add_argument("--chroma-dir", type=Path, default=DEFAULT_CHROMA_DIR)
    compare_parser.add_argument("--bm25-path", type=Path, default=DEFAULT_BM25_PATH)
    compare_parser.add_argument("--classla-python", type=str, default=None)
    compare_parser.add_argument("--top-k", type=int, default=5)
    compare_parser.add_argument(
        "--embedding-profiles",
        nargs="+",
        default=[DEFAULT_EMBEDDING_PROFILE, "e5_large_instruct"],
    )
    compare_parser.add_argument("--reranker-model", type=str, default=None)

    build_real_eval_parser = subparsers.add_parser("build-real-eval", help="Build real-world eval set from public FURS Q&A")
    build_real_eval_parser.add_argument("--output-path", type=Path, default=DEFAULT_REAL_EVAL_PATH)
    build_real_eval_parser.add_argument("--download-dir", type=Path, default=DEFAULT_REAL_EVAL_DOWNLOAD_DIR)
    build_real_eval_parser.add_argument("--limit", type=int, default=60)

    eval_real_parser = subparsers.add_parser("evaluate-real", help="Evaluate answer quality on FURS-derived real questions")
    eval_real_parser.add_argument("--real-eval-path", type=Path, default=DEFAULT_REAL_EVAL_PATH)
    eval_real_parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNK_PATH)
    eval_real_parser.add_argument("--chroma-dir", type=Path, default=DEFAULT_CHROMA_DIR)
    eval_real_parser.add_argument("--bm25-path", type=Path, default=DEFAULT_BM25_PATH)
    eval_real_parser.add_argument("--classla-python", type=str, default=None)
    eval_real_parser.add_argument("--top-k", type=int, default=5)
    eval_real_parser.add_argument("--embedding-profile", type=str, default=DEFAULT_EMBEDDING_PROFILE)
    eval_real_parser.add_argument("--embedding-model-name", type=str, default=None)
    eval_real_parser.add_argument("--generator-model", type=str, default=DEFAULT_LOCAL_GENERATOR_MODEL)
    eval_real_parser.add_argument("--reranker-model", type=str, default=None)
    eval_real_parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_LOCAL_GENERATOR_MAX_NEW_TOKENS)
    eval_real_parser.add_argument("--extractive-only", action="store_true")

    args = parser.parse_args()
    if args.command == "build-all":
        build_all(args)
    elif args.command == "query":
        run_query(args)
    elif args.command == "answer":
        run_answer(args)
    elif args.command == "serve":
        run_server(args)
    elif args.command == "evaluate":
        run_evaluation(args)
    elif args.command == "compare-embeddings":
        run_embedding_comparison(args)
    elif args.command == "build-real-eval":
        run_build_real_eval(args)
    elif args.command == "evaluate-real":
        run_evaluate_real(args)


def build_all(args: argparse.Namespace) -> None:
    artifact_dir = args.artifact_dir
    parsed_path = artifact_dir / DEFAULT_PARSED_PATH.name
    unit_path = artifact_dir / DEFAULT_UNIT_PATH.name
    annotation_path = artifact_dir / DEFAULT_ANNOTATION_PATH.name
    chunk_path = artifact_dir / DEFAULT_CHUNK_PATH.name
    bm25_path = artifact_dir / DEFAULT_BM25_PATH.name
    chroma_dir = artifact_dir / DEFAULT_CHROMA_DIR.name

    parsed_documents = parse_all_documents(args.download_report)
    if args.include_furs:
        parsed_documents.extend(
            fetch_and_parse_furs_guidance(
                download_dir=args.furs_download_dir,
                index_url=args.furs_guidance_url,
                min_year=args.furs_min_year,
            )
        )
        if args.include_furs_ddv_technical:
            parsed_documents.extend(
                fetch_and_parse_furs_ddv_technical_resources(
                    download_dir=args.furs_download_dir / "ddv_topic",
                    topic_url=args.furs_ddv_topic_url,
                    preserve_real_eval_holdout=args.preserve_real_eval_holdout,
                )
            )
        if args.include_furs_portal_pages:
            parsed_documents.extend(
                fetch_and_parse_furs_portal_resources(
                    download_dir=args.furs_download_dir / "portal_pages",
                    topic_url=args.furs_ddv_topic_url,
                    max_pages=args.furs_portal_max_pages,
                )
            )
    parsed_documents = dedupe_rows_by_key(parsed_documents, "doc_id")
    write_jsonl(parsed_path, parsed_documents)

    units = build_annotation_units([doc for doc in parsed_documents if doc.get("source_type", "pisrs") == "pisrs"])
    if args.include_furs:
        units.extend(build_furs_annotation_units([doc for doc in parsed_documents if doc.get("source_type") == "furs_guidance"]))
    units = dedupe_rows_by_key(units, "unit_id")
    write_jsonl(unit_path, units)

    annotate_jsonl(unit_path, annotation_path, classla_python=args.classla_python)
    annotations = read_jsonl(annotation_path)
    annotations_by_unit = {row["unit_id"]: row for row in annotations}

    chunks = chunk_units(
        units,
        annotations_by_unit,
        split_trigger_chars=args.split_trigger_chars,
        max_chunk_chars=args.max_chunk_chars,
        overlap_chars=args.overlap_chars,
    )
    write_jsonl(chunk_path, chunks)
    build_bm25_index(chunks, bm25_path)
    build_dense_index(
        chunks,
        chroma_dir,
        embedding_profile=args.embedding_profile,
        embedding_model_name=args.embedding_model_name,
    )

    print(
        json.dumps(
            {
                "parsed_documents": str(parsed_path),
                "annotation_units": str(unit_path),
                "classla_annotations": str(annotation_path),
                "chunks": str(chunk_path),
                "bm25_index": str(bm25_path),
                "chroma_dir": str(chroma_dir),
                "document_count": len(parsed_documents),
                "unit_count": len(units),
                "chunk_count": len(chunks),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def run_query(args: argparse.Namespace) -> None:
    chunk_map = load_chunk_map(args.chunks)
    ensure_dense_index(
        list(chunk_map.values()),
        args.chroma_dir,
        embedding_profile=args.embedding_profile,
        embedding_model_name=args.embedding_model_name,
    )
    query_lemmas = lemmatize_query_text(args.query, classla_python=args.classla_python)
    route = route_query(build_query_profile(args.query, query_lemmas))
    results = hybrid_search(
        args.query,
        query_lemmas,
        chunk_map,
        args.chroma_dir,
        args.bm25_path,
        args.top_k,
        embedding_profile=args.embedding_profile,
        embedding_model_name=args.embedding_model_name,
        source_policy=route.source_policy,
        reranker_model=args.reranker_model,
    )
    printable = []
    for item in results:
        chunk = item["chunk"]
        printable.append(
            {
                "score": item["score"],
                "law_id": chunk["law_id"],
                "title": chunk["title"],
                "article_number": chunk["article_number"],
                "article_title": chunk["article_title"],
                "section_path": chunk["section_path"],
                "source_url": chunk["source_url"],
                "text_preview": chunk["raw_chunk_text"][:500],
            }
        )
    print(json.dumps(printable, ensure_ascii=False, indent=2))


def run_answer(args: argparse.Namespace) -> None:
    chunk_map = load_chunk_map(args.chunks)
    ensure_dense_index(
        list(chunk_map.values()),
        args.chroma_dir,
        embedding_profile=args.embedding_profile,
        embedding_model_name=args.embedding_model_name,
    )
    query_lemmas = lemmatize_query_text(args.query, classla_python=args.classla_python)
    payload = answer_question(
        args.query,
        query_lemmas,
        chunk_map,
        args.chroma_dir,
        args.bm25_path,
        top_k=args.top_k,
        generator_model=None if args.extractive_only else args.generator_model,
        max_new_tokens=args.max_new_tokens,
        embedding_profile=args.embedding_profile,
        embedding_model_name=args.embedding_model_name,
        reranker_model=args.reranker_model,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def run_evaluation(args: argparse.Namespace) -> None:
    chunk_map = load_chunk_map(args.chunks)
    ensure_dense_index(
        list(chunk_map.values()),
        args.chroma_dir,
        embedding_profile=args.embedding_profile,
        embedding_model_name=args.embedding_model_name,
    )
    evaluation_rows = read_jsonl(args.evaluation_path)
    query_lemmas = lemmatize_query_texts(
        {row["query_id"]: row["query"] for row in evaluation_rows},
        classla_python=args.classla_python,
    )
    report = evaluate_retrieval(
        evaluation_rows,
        chunk_map,
        args.chroma_dir,
        args.bm25_path,
        query_lemmas,
        top_k=args.top_k,
        embedding_profile=args.embedding_profile,
        embedding_model_name=args.embedding_model_name,
        reranker_model=args.reranker_model,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


def run_embedding_comparison(args: argparse.Namespace) -> None:
    chunk_map = load_chunk_map(args.chunks)
    chunks = read_jsonl(args.chunks)
    evaluation_rows = read_jsonl(args.evaluation_path)
    query_lemmas = lemmatize_query_texts(
        {row["query_id"]: row["query"] for row in evaluation_rows},
        classla_python=args.classla_python,
    )
    report = compare_embedding_profiles(
        evaluation_rows,
        chunks,
        chunk_map,
        args.chroma_dir,
        args.bm25_path,
        query_lemmas,
        embedding_profiles=args.embedding_profiles,
        top_k=args.top_k,
        reranker_model=args.reranker_model,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


def run_build_real_eval(args: argparse.Namespace) -> None:
    rows = build_furs_real_eval_dataset(
        output_path=args.output_path,
        download_dir=args.download_dir,
        limit=args.limit,
    )
    print(
        json.dumps(
            {
                "output_path": str(args.output_path),
                "download_dir": str(args.download_dir),
                "query_count": len(rows),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def run_evaluate_real(args: argparse.Namespace) -> None:
    chunk_map = load_chunk_map(args.chunks)
    ensure_dense_index(
        list(chunk_map.values()),
        args.chroma_dir,
        embedding_profile=args.embedding_profile,
        embedding_model_name=args.embedding_model_name,
    )
    rows = load_real_eval_rows(args.real_eval_path)
    query_lemmas = lemmatize_query_texts(
        {row["query_id"]: row["query"] for row in rows},
        classla_python=args.classla_python,
    )
    report = evaluate_real_answers(
        rows,
        chunk_map,
        args.chroma_dir,
        args.bm25_path,
        lemmatize_query_text,
        classla_python=args.classla_python,
        query_lemmas=query_lemmas,
        top_k=args.top_k,
        generator_model=None if args.extractive_only else args.generator_model,
        max_new_tokens=args.max_new_tokens,
        embedding_profile=args.embedding_profile,
        embedding_model_name=args.embedding_model_name,
        reranker_model=args.reranker_model,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


def run_server(args: argparse.Namespace) -> None:
    import uvicorn

    app = create_app(
        WebAppSettings(
            chunks_path=args.chunks,
            chroma_dir=args.chroma_dir,
            bm25_path=args.bm25_path,
            classla_python=args.classla_python,
            top_k=args.top_k,
            embedding_profile=args.embedding_profile,
            embedding_model_name=args.embedding_model_name,
            generator_model=None if args.extractive_only else args.generator_model,
            reranker_model=args.reranker_model,
            max_new_tokens=args.max_new_tokens,
        )
    )
    uvicorn.run(app, host=args.host, port=args.port)


def dedupe_rows_by_key(rows: list[dict[str, Any]], key_name: str) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        key = str(row.get(key_name) or "")
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


if __name__ == "__main__":
    main()
