#!/usr/bin/env python3
"""Create a lightweight evaluation JSONL from indexed legal chunks."""

from __future__ import annotations

import argparse
import json
import random
import re
import uuid
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHUNKS = PROJECT_ROOT / "data" / "index" / "chunks.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "evaluation" / "generated_tax_eval_questions.jsonl"
QUESTION_STYLES = ("citation", "natural", "mixed")
PAIRING_MODES = ("any", "same-law", "different-law")


def append_run_id(path: Path, run_id: str) -> Path:
    """Append a run id before the file suffix."""
    return path.with_name(f"{path.stem}-{run_id}{path.suffix}")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def clean_title(title: str | None) -> str:
    if not title:
        return "vsebino tega člena"
    title = re.sub(r"^[\s(]+|[\s)]+$", "", title)
    return title[:1].lower() + title[1:] if title else "vsebino tega člena"


def title_without_boilerplate(title: str) -> str:
    title = clean_title(title)
    title = re.sub(r"\bDDV\b", "davka na dodano vrednost", title, flags=re.IGNORECASE)
    return title


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def text_without_heading(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines and lines[0].startswith("Law:"):
        lines = lines[1:]
    if lines and re.match(r"^\d+[.\w]*\s*\.?\s*člen$", lines[0], flags=re.IGNORECASE):
        lines = lines[1:]
    if lines and lines[0].startswith("(") and lines[0].endswith(")"):
        lines = lines[1:]
    return normalize_space(" ".join(lines))


def split_candidate_phrases(text: str) -> list[str]:
    body = text_without_heading(text)
    pieces = re.split(r"(?<=[.!?])\s+|;\s+|\s+-\s+", body)
    phrases = []
    for piece in pieces:
        piece = normalize_space(re.sub(r"^\(?\d+[.)]\s*", "", piece))
        if 35 <= len(piece) <= 220:
            phrases.append(piece)
        if len(phrases) == 3:
            break
    if not phrases and body:
        phrases.append(body[:180])
    return phrases


def first_content_words(text: str, limit: int = 9) -> str:
    body = text_without_heading(text)
    words = re.findall(r"[A-Za-zČŠŽčšž0-9][A-Za-zČŠŽčšž0-9.-]*", body)
    stop = {
        "clen",
        "člen",
        "odstavek",
        "zakona",
        "tega",
        "tem",
        "tretji",
        "prvi",
        "drugi",
        "mora",
        "lahko",
    }
    kept = [word for word in words if word.lower() not in stop and len(word) > 2]
    return " ".join(kept[:limit])


def is_good_article_chunk(chunk: dict[str, Any], min_chars: int) -> bool:
    metadata = chunk.get("metadata") or {}
    text = text_without_heading(chunk.get("text", ""))
    return (
        metadata.get("article_number")
        and metadata.get("law_id")
        and metadata.get("document_role") == "statute"
        and len(text) >= min_chars
        and "črtan" not in text.lower()[:80]
    )


def chunk_law_id(chunk: dict[str, Any]) -> str:
    return str((chunk.get("metadata") or {}).get("law_id") or "unknown")


def chunk_article(chunk: dict[str, Any]) -> str:
    return str((chunk.get("metadata") or {}).get("article_number") or "unknown")


def chunk_title(chunk: dict[str, Any]) -> str:
    return clean_title((chunk.get("metadata") or {}).get("article_title"))


def make_question(chunk: dict[str, Any], style: str, rng: random.Random) -> str:
    metadata = chunk.get("metadata") or {}
    law_id = chunk_law_id(chunk)
    article = chunk_article(chunk)
    title = title_without_boilerplate(metadata.get("article_title"))
    topic_hint = first_content_words(chunk.get("text", ""))

    if style == "citation":
        return f"Kaj določa {article}. člen {law_id} glede {title}?"

    templates = [
        f"Kaj velja glede {title}?",
        f"Kako so urejena pravila za {title}?",
        f"Kakšne obveznosti ali pogoji veljajo pri temi: {title}?",
        f"Kdaj oziroma kako se uporablja pravilo o temi: {title}?",
    ]
    if topic_hint:
        templates.append(f"Kaj pomeni oziroma določa pravilo, ki omenja: {topic_hint}?")
    return rng.choice(templates)


def make_dual_question(chunks: list[dict[str, Any]], style: str, rng: random.Random) -> str:
    first, second = chunks
    first_law = chunk_law_id(first)
    second_law = chunk_law_id(second)
    first_article = chunk_article(first)
    second_article = chunk_article(second)
    first_title = title_without_boilerplate(chunk_title(first))
    second_title = title_without_boilerplate(chunk_title(second))

    if style == "citation":
        return (
            f"Primerjaj ali povzemite, kaj določata {first_article}. člen {first_law} "
            f"glede {first_title} in {second_article}. člen {second_law} glede {second_title}."
        )

    templates = [
        f"Kako sta urejeni temi {first_title} in {second_title}?",
        f"Kaj velja glede {first_title} ter glede {second_title}?",
        f"Kakšne obveznosti ali pogoji veljajo pri temah {first_title} in {second_title}?",
        f"Povzemi pravili o temah {first_title} in {second_title}.",
    ]
    return rng.choice(templates)


def expected_location(chunk: dict[str, Any]) -> str:
    source = chunk.get("source")
    law_id = chunk_law_id(chunk)
    article = chunk_article(chunk)
    title = chunk_title(chunk)
    return f"{source}: {law_id}, {article}. člen ({title})"


def make_case(index: int, chunk: dict[str, Any], style: str, rng: random.Random) -> dict[str, Any]:
    law_id = chunk_law_id(chunk)
    phrases = split_candidate_phrases(chunk.get("text", ""))
    source = chunk.get("source")
    chunk_id = chunk.get("chunk_id")

    return {
        "id": f"generated_{index:02d}",
        "category": f"{law_id.lower()}_generated",
        "question": make_question(chunk, style, rng),
        "expected_sources": [source],
        "expected_chunks": [chunk_id],
        "expected_locations": [expected_location(chunk)],
        "expected_phrases": phrases,
        "expected_answer": " ".join(phrases),
    }


def make_dual_case(
    index: int,
    chunks: list[dict[str, Any]],
    style: str,
    rng: random.Random,
) -> dict[str, Any]:
    laws = [chunk_law_id(chunk) for chunk in chunks]
    phrases: list[str] = []
    for chunk in chunks:
        phrases.extend(split_candidate_phrases(chunk.get("text", ""))[:2])
    sources = [chunk.get("source") for chunk in chunks]
    chunk_ids = [chunk.get("chunk_id") for chunk in chunks]
    category_laws = "-".join(sorted(set(law.lower() for law in laws)))

    return {
        "id": f"generated_dual_{index:02d}",
        "category": f"{category_laws}_dual_generated",
        "question": make_dual_question(chunks, style, rng),
        "expected_sources": sources,
        "expected_chunks": chunk_ids,
        "expected_locations": [expected_location(chunk) for chunk in chunks],
        "expected_phrases": phrases,
        "expected_answer": " ".join(phrases),
        "source_count": len(chunks),
    }


def pick_pair(
    anchor: dict[str, Any],
    candidates: list[dict[str, Any]],
    rng: random.Random,
    pairing: str,
) -> list[dict[str, Any]]:
    anchor_law = chunk_law_id(anchor)
    if pairing == "same-law":
        pool = [
            chunk for chunk in candidates
            if chunk.get("chunk_id") != anchor.get("chunk_id") and chunk_law_id(chunk) == anchor_law
        ]
    elif pairing == "different-law":
        pool = [
            chunk for chunk in candidates
            if chunk.get("chunk_id") != anchor.get("chunk_id") and chunk_law_id(chunk) != anchor_law
        ]
    else:
        pool = [chunk for chunk in candidates if chunk.get("chunk_id") != anchor.get("chunk_id")]
    if not pool:
        raise ValueError(f"No pair candidate found for {anchor.get('chunk_id')} with pairing={pairing}")
    return [anchor, rng.choice(pool)]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks-path", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--append-run-id",
        action="store_true",
        help="Append a short random run id to --output before writing.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Run id used with --append-run-id. Defaults to a random 8-character id.",
    )
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-chars", type=int, default=350)
    parser.add_argument(
        "--sources-per-question",
        type=int,
        choices=[1, 2],
        default=1,
        help="Use 1 normal source chunk or 2 source chunks per generated question.",
    )
    parser.add_argument(
        "--pairing",
        choices=PAIRING_MODES,
        default="any",
        help="For dual questions, choose whether the two chunks come from any, same, or different laws.",
    )
    parser.add_argument(
        "--style",
        choices=QUESTION_STYLES,
        default="citation",
        help="citation includes law/article ids; natural hides them; mixed alternates both.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or uuid.uuid4().hex[:8]
    if args.append_run_id:
        args.output = append_run_id(args.output, run_id)
    chunks = load_jsonl(args.chunks_path)
    candidates = [chunk for chunk in chunks if is_good_article_chunk(chunk, args.min_chars)]
    if len(candidates) < args.n:
        raise SystemExit(
            f"Only found {len(candidates)} usable article chunks in {args.chunks_path}; "
            "build a legal index first or lower --min-chars."
        )

    rng = random.Random(args.seed)
    selected = rng.sample(candidates, args.n)
    records = []
    for index, chunk in enumerate(selected, start=1):
        style = args.style
        if style == "mixed":
            style = "citation" if index % 2 else "natural"
        if args.sources_per_question == 2:
            pair = pick_pair(chunk, candidates, rng, args.pairing)
            record = make_dual_case(index, pair, style, rng)
            record["pairing"] = args.pairing
        else:
            record = make_case(index, chunk, style, rng)
        record["run_id"] = run_id
        records.append(record)
    write_jsonl(records, args.output)
    print(f"Wrote {len(records)} questions to {args.output} with run_id={run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
