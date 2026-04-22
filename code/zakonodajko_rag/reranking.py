from __future__ import annotations

from functools import lru_cache
from typing import Any

from .text_utils import normalize_text


def rerank_results(
    query: str,
    results: list[dict[str, Any]],
    reranker_model: str | None,
    top_n: int = 15,
    route: Any | None = None,
) -> list[dict[str, Any]]:
    if not reranker_model or len(results) <= 1:
        return results

    reranker = load_reranker_model(reranker_model)
    candidate_count = min(top_n, len(results))
    head = [dict(item) for item in results[:candidate_count]]
    tail = [dict(item) for item in results[candidate_count:]]
    pairs = [(format_reranker_query(query, route), truncate_chunk_for_reranking(item["chunk"])) for item in head]
    scores = reranker.predict(pairs)
    for item, score in zip(head, scores, strict=True):
        item["rerank_score"] = float(score)
        item["score"] = float(item.get("score", 0.0)) + 0.35 * float(score)
    head.sort(key=lambda item: item["score"], reverse=True)
    return head + tail


@lru_cache(maxsize=3)
def load_reranker_model(reranker_model: str):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(reranker_model)


def truncate_chunk_for_reranking(chunk: dict[str, Any], max_chars: int = 1200) -> str:
    text = normalize_text(chunk.get("raw_chunk_text", ""))
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip(" ;,:") + " …"


def format_reranker_query(query: str, route: Any | None = None) -> str:
    if route is None:
        return query
    if getattr(route, "intent", "") == "practical_guidance":
        return f"Praktično davčno vprašanje: {query}"
    if getattr(route, "intent", "") == "explicit_article":
        return f"Pravni člen: {query}"
    return query
