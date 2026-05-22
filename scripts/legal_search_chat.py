#!/usr/bin/env python3
"""Interactive terminal chatbot for legal web search + grounded answers."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from legal_web_suggestions import (  # noqa: E402
    DEFAULT_LOCAL_MODEL_PATH,
    make_extractive_answer,
    make_llm_answer,
    source_role,
    web_suggestions,
    best_local_rag_suggestions,
)


def warning_messages(question: str, suggestions: list[dict[str, Any]]) -> list[str]:
    """Return human-readable warnings about source quality."""
    warnings = []
    if not suggestions:
        return ["Ni najdenih virov. Odgovor morda ni mogoč."]

    if any(str(item.get("retrieval_method") or "").startswith(("tavily", "brave", "duckduckgo")) for item in suggestions):
        warnings.append(
            "Viri so pridobljeni s spletnim iskanjem. To niso nujno preverjeni ali popolni rezultati; "
            "prednost imajo uradni viri (PISRS, FURS, GOV.SI, e-Uprava, AJPES)."
        )

    strong = [item for item in suggestions if float(item.get("relevance") or 0.0) >= 0.8]
    if not strong:
        warnings.append("Ni najdenega zelo relevantnega vira; odgovor obravnavaj kot nezanesljiv.")

    official = [
        item
        for item in suggestions
        if source_role(item)
        in {"primary_legal_source", "official_guidance", "official_register_guidance", "official_public_source"}
    ]
    if not official:
        warnings.append("Ni najdenega uradnega vira.")

    noisy_titles = [
        str(item.get("title") or "")
        for item in suggestions
        if float(item.get("relevance") or 0.0) < 0.35
    ]
    if noisy_titles:
        warnings.append(f"{len(noisy_titles)} manj relevantnih virov je še vedno v kontekstu.")

    if any("pdf" in str(item.get("url") or "").lower() for item in suggestions):
        warnings.append("Nekateri viri so PDF; morda je na voljo samo povzetek iz iskanja.")

    return warnings


def reliable_sources(suggestions: list[dict[str, Any]], min_relevance: float, min_official: int) -> list[dict[str, Any]]:
    """Return sources strong enough to support an answer."""
    official_roles = {
        "primary_legal_source",
        "official_guidance",
        "official_register_guidance",
        "official_public_source",
        "local_legal_context",
    }
    strong = [
        item
        for item in suggestions
        if float(item.get("relevance") or 0.0) >= min_relevance and source_role(item) in official_roles
    ]
    if len(strong) >= min_official:
        return strong
    return []


def unsupported_answer(answer_text: str) -> bool:
    """Detect answers that admit missing support or look empty."""
    text = answer_text.strip().lower()
    if len(text) < 40:
        return True
    markers = (
        "no answer could be",
        "not found",
        "not enough information",
        "provided sources do not",
        "ni najden",
        "ni dovolj podatkov",
        "viri ne vsebujejo",
        "ne morem odgovoriti",
    )
    return any(marker in text for marker in markers)


def print_sources(suggestions: list[dict[str, Any]], max_sources: int) -> None:
    print("\nViri:")
    for index, item in enumerate(suggestions[:max_sources], start=1):
        title = item.get("title") or "(untitled)"
        url = item.get("url") or "(local/no URL)"
        role = source_role(item)
        relevance = float(item.get("relevance") or 0.0)
        print(f"[{index}] {title}")
        print(f"    {url}")
        print(f"    role={role} relevance={relevance:.3f} method={item.get('retrieval_method')}")


def answer_once(question: str, args: argparse.Namespace, llm_bundle: tuple[Any, Any, Any] | None) -> None:
    suggestions: list[dict[str, Any]] = []
    if args.web:
        suggestions.extend(
            web_suggestions(
                question=question,
                tiers={int(value) for value in args.tiers.split(",") if value.strip()},
                per_domain=args.per_domain,
                sleep_seconds=args.sleep,
                provider=args.provider,
                broad=args.broad,
                allow_curated_fallback=False,
                source_policy=args.source_policy,
                min_relevance=args.min_relevance,
                category="",
            )
        )

    best_config = None
    if args.best_local_context:
        local_suggestions, best_config = best_local_rag_suggestions(
            question,
            limit=args.best_local_limit,
        )
        suggestions.extend(local_suggestions)

    suggestions.sort(
        key=lambda item: (
            99 if item.get("source_tier") is None else int(item.get("source_tier") or 99),
            -float(item.get("relevance") or 0.0),
        )
    )
    suggestions = suggestions[: args.max_suggestions]

    print("\nOpozorila:")
    for warning in warning_messages(question, suggestions):
        print(f"- {warning}")

    answer_suggestions = suggestions
    if args.require_reliable_sources:
        answer_suggestions = reliable_sources(
            suggestions,
            min_relevance=args.answer_min_relevance,
            min_official=args.min_official_sources,
        )
        if not answer_suggestions:
            print("\nOdgovor:")
            print(
                "Zanesljivega odgovora nisem našel. Spletno iskanje ni vrnilo dovolj "
                "relevantnih uradnih virov, zato bi bil odgovor lahko zavajajoč."
            )
            print_sources(suggestions, args.max_sources_print)
            print()
            return

    if args.llm_answer:
        answer = make_llm_answer(
            question=question,
            suggestions=answer_suggestions,
            model_path=args.model_path,
            fetch_pages=args.fetch_pages,
            max_sources=args.answer_sources,
            max_chars_per_source=args.llm_context_chars,
            max_new_tokens=args.max_new_tokens,
            llm_bundle=llm_bundle,
        )
    else:
        answer = make_extractive_answer(
            question=question,
            suggestions=answer_suggestions,
            fetch_pages=args.fetch_pages,
            max_sources=args.answer_sources,
            max_sentences=args.answer_sentences,
        )

    print("\nOdgovor:")
    if answer.get("error"):
        print(f"ERROR: {answer['error']}")
    answer_text = answer.get("text") or "(no answer)"
    if args.require_supported_answer and unsupported_answer(answer_text):
        print(
            "Zanesljivega odgovora nisem našel. Najdeni viri ne vsebujejo dovolj jasne "
            "podlage za odgovor."
        )
    else:
        print(answer_text)
    print_sources(suggestions, args.max_sources_print)

    if best_config:
        print(
            "\nLocal RAG config: "
            f"{best_config['chunk_config']} | {best_config['embedding_model']} | "
            f"{best_config['retrieval_config']} | score={best_config['score']:.3f}"
        )
    print()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interactive legal search chatbot.")
    parser.add_argument("--provider", choices=["duckduckgo", "brave", "tavily"], default="tavily")
    parser.add_argument("--tiers", default="1")
    parser.add_argument("--source-policy", choices=["auto", "all"], default="auto")
    parser.add_argument("--min-relevance", type=float, default=0.25)
    parser.add_argument("--per-domain", type=int, default=1)
    parser.add_argument("--max-suggestions", type=int, default=5)
    parser.add_argument("--max-sources-print", type=int, default=5)
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--broad", action="store_true")
    parser.add_argument("--no-web", dest="web", action="store_false")
    parser.set_defaults(web=True)
    parser.add_argument("--best-local-context", action="store_true")
    parser.add_argument("--best-local-limit", type=int, default=3)
    parser.add_argument("--llm-answer", action="store_true")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_LOCAL_MODEL_PATH)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--llm-context-chars", type=int, default=1800)
    parser.add_argument("--fetch-pages", action="store_true")
    parser.add_argument("--answer-sources", type=int, default=4)
    parser.add_argument("--answer-sentences", type=int, default=4)
    parser.add_argument(
        "--require-reliable-sources",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Only answer when enough relevant official/local sources are available.",
    )
    parser.add_argument(
        "--require-supported-answer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Replace unsupported/empty answers with a not-found message.",
    )
    parser.add_argument("--answer-min-relevance", type=float, default=0.55)
    parser.add_argument("--min-official-sources", type=int, default=1)
    parser.add_argument("--question", default=None, help="Ask one question and exit.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.provider == "tavily" and args.web and not os.environ.get("TAVILY_API_KEY"):
        print("WARNING: TAVILY_API_KEY is not set; Tavily web search will fail.", file=sys.stderr)

    llm_bundle = None
    if args.llm_answer:
        from src.generate_answer import load_llm

        print(f"Loading LLM from {args.model_path} ...", flush=True)
        llm_bundle = load_llm(args.model_path)
        print("LLM loaded. Ask questions, or type 'exit'.", flush=True)

    if args.question:
        answer_once(args.question, args, llm_bundle)
        return 0

    print("Legal search chat. Vpiši vprašanje ali 'exit' za izhod.")
    while True:
        try:
            question = input("\nlegal-search> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if question.lower() in {"exit", "quit", "q"}:
            return 0
        if not question:
            continue
        answer_once(question, args, llm_bundle)


if __name__ == "__main__":
    raise SystemExit(main())
