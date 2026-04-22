from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any

from .regexes import extract_query_citations
from .text_utils import normalize_text


SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?;])\s+|\n+")
REFERRAL_SENTENCE_RE = re.compile(
    r"\b(?:velja|uporablja|smiselno uporablja|v skladu z|skladno z|po)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ResolutionStep:
    from_chunk_id: str
    from_citation: str
    to_chunk_id: str
    to_citation: str
    trigger_sentence: str
    depth: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_referral_results(
    results: list[dict[str, Any]],
    chunk_map: dict[str, dict[str, Any]],
    route: Any,
    max_hops: int = 2,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not route.follow_referrals or not results:
        return results, []

    article_index = build_article_index(chunk_map)
    law_alias_index = build_law_alias_index(chunk_map)
    additions: dict[str, dict[str, Any]] = {}
    chains: list[ResolutionStep] = []

    for item in results[:3]:
        start_chunk = item["chunk"]
        if start_chunk.get("source_type", "pisrs") != "pisrs":
            continue
        seed_score = float(item.get("score", 0.0))
        for resolved_item, step in follow_referral_chain(
            start_chunk,
            seed_score,
            article_index,
            law_alias_index,
            max_hops=max_hops,
        ):
            existing = additions.get(resolved_item["chunk"]["chunk_id"])
            if existing is None or resolved_item["score"] > existing["score"]:
                additions[resolved_item["chunk"]["chunk_id"]] = resolved_item
            chains.append(step)

    if not additions:
        return results, []

    merged: dict[str, dict[str, Any]] = {item["chunk"]["chunk_id"]: dict(item) for item in results}
    for chunk_id, item in additions.items():
        if chunk_id in merged:
            merged[chunk_id]["score"] = max(float(merged[chunk_id].get("score", 0.0)), float(item["score"]))
            merged[chunk_id]["resolved_via"] = item.get("resolved_via")
        else:
            merged[chunk_id] = item

    ranked = sorted(merged.values(), key=lambda row: float(row.get("score", 0.0)), reverse=True)
    return ranked, [step.as_dict() for step in chains]


def follow_referral_chain(
    chunk: dict[str, Any],
    base_score: float,
    article_index: dict[tuple[str, str], dict[str, Any]],
    law_alias_index: dict[str, set[str]],
    max_hops: int,
    depth: int = 0,
    visited: set[tuple[str, str]] | None = None,
) -> list[tuple[dict[str, Any], ResolutionStep]]:
    if visited is None:
        visited = set()
    current_key = ((chunk.get("law_id") or ""), (chunk.get("article_number") or "").lower())
    if current_key in visited or depth >= max_hops:
        return []
    visited = set(visited)
    visited.add(current_key)

    resolved: list[tuple[dict[str, Any], ResolutionStep]] = []
    for sentence in extract_chunk_sentences(chunk):
        if not is_referral_sentence(sentence):
            continue
        citations = extract_query_citations(sentence)
        target_articles = citations["articles"]
        if not target_articles:
            continue
        target_law_ids = resolve_target_law_ids(citations["law_refs"], law_alias_index, default_law_id=chunk["law_id"])
        for law_id in target_law_ids:
            for article_ref in target_articles:
                target_key = (law_id, article_ref.lower())
                target_chunk = article_index.get(target_key)
                if target_chunk is None or target_chunk["chunk_id"] == chunk["chunk_id"]:
                    continue
                score = base_score + (0.8 / (depth + 1))
                step = ResolutionStep(
                    from_chunk_id=chunk["chunk_id"],
                    from_citation=format_chunk_citation(chunk),
                    to_chunk_id=target_chunk["chunk_id"],
                    to_citation=format_chunk_citation(target_chunk),
                    trigger_sentence=normalize_text(sentence),
                    depth=depth + 1,
                )
                resolved.append(
                    (
                        {
                            "chunk": target_chunk,
                            "score": score,
                            "resolved_via": chunk["chunk_id"],
                            "resolution_depth": depth + 1,
                        },
                        step,
                    )
                )
                resolved.extend(
                    follow_referral_chain(
                        target_chunk,
                        score,
                        article_index,
                        law_alias_index,
                        max_hops=max_hops,
                        depth=depth + 1,
                        visited=visited,
                    )
                )
    return resolved


def build_article_index(chunk_map: dict[str, dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    article_index: dict[tuple[str, str], dict[str, Any]] = {}
    for chunk in chunk_map.values():
        article_number = normalize_text(chunk.get("article_number") or "").lower()
        law_id = chunk.get("law_id") or ""
        if not article_number or not law_id:
            continue
        key = (law_id, article_number)
        current = article_index.get(key)
        if current is None or len(chunk.get("raw_chunk_text", "")) > len(current.get("raw_chunk_text", "")):
            article_index[key] = chunk
    return article_index


def build_law_alias_index(chunk_map: dict[str, dict[str, Any]]) -> dict[str, set[str]]:
    alias_index: dict[str, set[str]] = {}
    for chunk in chunk_map.values():
        aliases = {
            normalize_text(chunk.get("law_id") or "").lower(),
            normalize_text(chunk.get("title") or "").lower(),
            *[normalize_text(item).lower() for item in (chunk.get("legal_refs") or {}).get("law_refs", [])],
            *[normalize_text(item).lower() for item in (chunk.get("legal_refs") or {}).get("act_ids", [])],
        }
        law_id = chunk.get("law_id") or ""
        for alias in aliases:
            if alias:
                alias_index.setdefault(alias, set()).add(law_id)
    return alias_index


def resolve_target_law_ids(
    law_refs: list[str],
    law_alias_index: dict[str, set[str]],
    default_law_id: str,
) -> list[str]:
    if not law_refs:
        return [default_law_id]
    resolved: list[str] = []
    for ref in law_refs:
        for law_id in sorted(law_alias_index.get(normalize_text(ref).lower(), set())):
            if law_id not in resolved:
                resolved.append(law_id)
    return resolved or [default_law_id]


def extract_chunk_sentences(chunk: dict[str, Any]) -> list[str]:
    text = chunk_body_text(chunk)
    spans = chunk.get("sentence_spans") or []
    extracted: list[str] = []
    for span in spans:
        sentence = normalize_text(span.get("text", ""))
        if sentence:
            extracted.append(sentence)
    if extracted:
        return dedupe_preserve_order(extracted)
    return dedupe_preserve_order([part for part in SENTENCE_SPLIT_RE.split(text) if normalize_text(part)])


def chunk_body_text(chunk: dict[str, Any]) -> str:
    raw = chunk.get("raw_chunk_text", "")
    if "\n" not in raw:
        return normalize_text(raw)
    return normalize_text(raw.split("\n", 1)[1])


def dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        normalized = normalize_text(value)
        if not normalized or normalized in seen:
            continue
        ordered.append(normalized)
        seen.add(normalized)
    return ordered


def is_referral_sentence(sentence: str) -> bool:
    normalized = normalize_text(sentence)
    return bool(normalized and REFERRAL_SENTENCE_RE.search(normalized) and extract_query_citations(normalized)["articles"])


def format_chunk_citation(chunk: dict[str, Any]) -> str:
    law_refs = (chunk.get("legal_refs") or {}).get("law_refs") or []
    law_ref = law_refs[0] if law_refs else chunk.get("law_id") or chunk.get("title") or "Vir"
    article_number = normalize_text(chunk.get("article_number") or "")
    article_title = normalize_text(chunk.get("article_title") or "")
    if article_number and article_title:
        return f"{law_ref}, {article_number} {article_title}"
    if article_number:
        return f"{law_ref}, {article_number}"
    return law_ref
