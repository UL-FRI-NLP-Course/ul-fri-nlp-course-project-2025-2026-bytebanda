from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .constants import DEFAULT_EMBEDDING_PROFILE, RRF_K
from .embeddings import (
    load_embedding_model,
    prepare_documents_for_embedding,
    prepare_query_for_embedding,
    resolve_embedding_profile,
)
from .regexes import extract_query_citations
from .reranking import rerank_results
from .router import TECHNICAL_PROCESS_HINTS, build_query_profile, route_query
from .text_utils import ensure_parent_dir, normalize_text, read_jsonl

SOURCE_AWARE_QUERY_HINTS = ("furs", "pojasnilo", "pojasnila", "navodilo", "navodila", "smernice", "smernica")
LOW_SIGNAL_FURS_TERMS = {
    "davcni",
    "davčni",
    "evidenca",
    "evidenc",
    "evidenci",
    "furs",
    "navodilo",
    "navodila",
    "obrazec",
    "oddaja",
    "oddati",
    "podatki",
    "pojasnilo",
    "pojasnila",
    "polje",
    "predložiti",
    "predloziti",
    "prijava",
    "servis",
    "spletni",
    "spletnega",
    "zavezanec",
}
DISTINCTIVE_TECHNICAL_TERMS = TECHNICAL_PROCESS_HINTS.intersection(
    {"api", "certifikat", "client", "edavki", "erp", "excel", "identifikator", "klient", "oauth", "testni", "testnem", "xml"}
)


def configure_chromadb_logging() -> None:
    logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)


def build_bm25_index(chunks: list[dict[str, Any]], output_path: Path) -> dict[str, Any]:
    payload = {
        "chunk_ids": [chunk["chunk_id"] for chunk in chunks],
        "tokenized_corpus": [chunk["lemma_chunk_text"].split() for chunk in chunks],
    }
    ensure_parent_dir(output_path)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def load_bm25_index(path: Path) -> tuple[list[str], Any]:
    from rank_bm25 import BM25Okapi

    payload = json.loads(path.read_text(encoding="utf-8"))
    corpus = payload["tokenized_corpus"]
    return payload["chunk_ids"], BM25Okapi(corpus)


def build_dense_index(
    chunks: list[dict[str, Any]],
    chroma_dir: Path,
    embedding_profile: str = DEFAULT_EMBEDDING_PROFILE,
    embedding_model_name: str | None = None,
    collection_name: str | None = None,
) -> None:
    import chromadb
    from chromadb.config import Settings

    profile = resolve_embedding_profile(embedding_profile, embedding_model_name)
    active_collection = collection_name or profile.collection_name
    configure_chromadb_logging()
    chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=str(chroma_dir),
        settings=Settings(anonymized_telemetry=False),
    )
    try:
        client.delete_collection(active_collection)
    except Exception:
        pass
    collection = client.create_collection(name=active_collection, metadata={"hnsw:space": "cosine"})
    model = load_embedding_model(
        profile.model_name,
        max_seq_length=profile.max_seq_length,
        use_eager_attention=profile.use_eager_attention,
    )
    texts = prepare_documents_for_embedding([chunk["raw_chunk_text"] for chunk in chunks], profile)
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=16,
    )
    metadatas = [
        {
            "chunk_id": chunk["chunk_id"],
            "doc_id": chunk["doc_id"],
            "law_id": chunk["law_id"],
            "title": chunk["title"],
            "section_path": chunk["section_path"] or "",
            "article_number": chunk["article_number"] or "",
            "article_title": chunk["article_title"] or "",
            "chunk_type": chunk["chunk_type"],
            "source_url": chunk["source_url"],
            "selected_npb": int(chunk["selected_npb"]) if chunk.get("selected_npb") is not None else -1,
            "source_type": chunk.get("source_type", "pisrs"),
        }
        for chunk in chunks
    ]
    collection.add(
        ids=[chunk["chunk_id"] for chunk in chunks],
        documents=texts,
        embeddings=embeddings.tolist(),
        metadatas=metadatas,
    )


def load_chunk_map(chunks_path: Path) -> dict[str, dict[str, Any]]:
    return {row["chunk_id"]: row for row in read_jsonl(chunks_path)}


def search_dense(
    query: str,
    chunk_map: dict[str, dict[str, Any]],
    chroma_dir: Path,
    top_k: int,
    embedding_profile: str = DEFAULT_EMBEDDING_PROFILE,
    embedding_model_name: str | None = None,
    collection_name: str | None = None,
) -> list[dict[str, Any]]:
    import chromadb
    from chromadb.config import Settings

    profile = resolve_embedding_profile(embedding_profile, embedding_model_name)
    active_collection = collection_name or profile.collection_name
    configure_chromadb_logging()
    client = chromadb.PersistentClient(
        path=str(chroma_dir),
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.get_collection(active_collection)
    model = load_embedding_model(
        profile.model_name,
        max_seq_length=profile.max_seq_length,
        use_eager_attention=profile.use_eager_attention,
    )
    embedding_query = prepare_query_for_embedding(query, profile)
    embedding = model.encode([embedding_query], normalize_embeddings=True)
    results = collection.query(query_embeddings=embedding.tolist(), n_results=top_k, include=["distances", "metadatas"])
    ranked = []
    for index, chunk_id in enumerate(results["ids"][0]):
        ranked.append(
            {
                "chunk": chunk_map[chunk_id],
                "rank": index + 1,
                "distance": results["distances"][0][index],
                "source": "dense",
            }
        )
    return ranked


def search_sparse(
    query_tokens: list[str],
    chunk_map: dict[str, dict[str, Any]],
    bm25_path: Path,
    top_k: int,
) -> list[dict[str, Any]]:
    chunk_ids, bm25 = load_bm25_index(bm25_path)
    scores = bm25.get_scores(query_tokens)
    ranked_ids = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)[:top_k]
    return [
        {
            "chunk": chunk_map[chunk_ids[idx]],
            "rank": rank + 1,
            "score": float(scores[idx]),
            "source": "sparse",
        }
        for rank, idx in enumerate(ranked_ids)
    ]


def hybrid_search(
    query: str,
    query_tokens: list[str],
    chunk_map: dict[str, dict[str, Any]],
    chroma_dir: Path,
    bm25_path: Path,
    top_k: int,
    embedding_profile: str = DEFAULT_EMBEDDING_PROFILE,
    embedding_model_name: str | None = None,
    source_policy: str = "pisrs_first",
    reranker_model: str | None = None,
    reranker_top_n: int = 15,
) -> list[dict[str, Any]]:
    query_profile = build_query_profile(query, query_tokens)
    route = route_query(query_profile)
    citations = extract_query_citations(query)
    candidate_pool = max(top_k, 25)
    dense_results = search_dense(
        query,
        chunk_map,
        chroma_dir,
        candidate_pool,
        embedding_profile=embedding_profile,
        embedding_model_name=embedding_model_name,
    )
    if not citations["articles"] and route.intent not in {"practical_guidance", "follow_up"} and not should_blend_sparse_without_article_citation(query):
        ranked_dense = []
        for item in dense_results:
            base_score = 1.0 / (RRF_K + item["rank"])
            source_bonus = source_rank_bonus(item["chunk"], citations)
            preference_bonus = query_source_preference_bonus(query, item["chunk"])
            policy_bonus = source_policy_bonus(item["chunk"], source_policy, query_profile)
            guidance_bonus = source_specific_query_bonus(item["chunk"], query_profile, source_policy)
            ranked_dense.append(
                {
                    "chunk": item["chunk"],
                    "dense_rank": item["rank"],
                    "sparse_rank": None,
                    "rrf_score": base_score,
                    "citation_boost": 0.0,
                    "source_bonus": source_bonus,
                    "query_preference_bonus": preference_bonus,
                    "source_policy_bonus": policy_bonus,
                    "source_specific_bonus": guidance_bonus,
                    "score": base_score + source_bonus + preference_bonus + policy_bonus + guidance_bonus,
                }
            )
        ranked_dense.sort(key=lambda row: row["score"], reverse=True)
        filtered = apply_source_policy(ranked_dense, source_policy, query_profile)
        reranked = rerank_results(query, filtered, reranker_model=reranker_model, top_n=reranker_top_n, route=route)
        return reranked[:top_k]
    sparse_results = search_sparse(query_tokens, chunk_map, bm25_path, candidate_pool)
    combined: dict[str, dict[str, Any]] = {}

    for result_set, key in ((dense_results, "dense_rank"), (sparse_results, "sparse_rank")):
        for item in result_set:
            chunk_id = item["chunk"]["chunk_id"]
            combined.setdefault(
                chunk_id,
                {
                    "chunk": item["chunk"],
                    "dense_rank": None,
                    "sparse_rank": None,
                    "rrf_score": 0.0,
                    "citation_boost": 0.0,
                },
            )
            combined[chunk_id][key] = item["rank"]
            combined[chunk_id]["rrf_score"] += 1.0 / (RRF_K + item["rank"])

    for rank, chunk in enumerate(direct_citation_candidates(chunk_map, citations), start=1):
        chunk_id = chunk["chunk_id"]
        combined.setdefault(
            chunk_id,
            {
                "chunk": chunk,
                "dense_rank": None,
                "sparse_rank": None,
                "rrf_score": 0.0,
                "citation_boost": 0.0,
            },
        )
        combined[chunk_id]["rrf_score"] += 2.0 / (RRF_K + rank)

    if source_policy in {"furs_allowed", "furs_preferred"} or route.intent in {"practical_guidance", "follow_up"}:
        for rank, candidate in enumerate(direct_furs_guidance_candidates(chunk_map, query_profile), start=1):
            chunk = candidate["chunk"]
            chunk_id = chunk["chunk_id"]
            combined.setdefault(
                chunk_id,
                {
                    "chunk": chunk,
                    "dense_rank": None,
                    "sparse_rank": None,
                    "rrf_score": 0.0,
                    "citation_boost": 0.0,
                },
            )
            combined[chunk_id]["rrf_score"] += 1.5 / (RRF_K + rank)
            combined[chunk_id]["source_specific_bonus"] = max(
                combined[chunk_id].get("source_specific_bonus", 0.0),
                candidate["bonus"],
            )

    for item in combined.values():
        item["citation_boost"] = citation_boost(item["chunk"], citations)
        item["source_bonus"] = source_rank_bonus(item["chunk"], citations)
        item["query_preference_bonus"] = query_source_preference_bonus(query, item["chunk"])
        item["source_policy_bonus"] = source_policy_bonus(item["chunk"], source_policy, query_profile)
        item["source_specific_bonus"] = max(
            item.get("source_specific_bonus", 0.0),
            source_specific_query_bonus(item["chunk"], query_profile, source_policy),
        )
        item["score"] = (
            item["rrf_score"]
            + item["citation_boost"]
            + item["source_bonus"]
            + item["query_preference_bonus"]
            + item["source_policy_bonus"]
            + item["source_specific_bonus"]
        )

    ranked = sorted(combined.values(), key=lambda row: row["score"], reverse=True)
    filtered = apply_source_policy(ranked, source_policy, query_profile)
    reranked = rerank_results(query, filtered, reranker_model=reranker_model, top_n=reranker_top_n, route=route)
    return reranked[:top_k]


def should_blend_sparse_without_article_citation(query: str) -> bool:
    lowered = query.lower()
    return any(hint in lowered for hint in SOURCE_AWARE_QUERY_HINTS)


def citation_boost(chunk: dict[str, Any], citations: dict[str, list[str]]) -> float:
    law_refs = {ref.lower() for ref in citations["law_refs"]}
    article_refs = {ref.lower() for ref in citations["articles"]}
    chunk_law_aliases = {
        chunk["law_id"].lower(),
        chunk["title"].lower(),
        *[ref.lower() for ref in chunk["legal_refs"]["law_refs"]],
        *[ref.lower() for ref in chunk["legal_refs"]["act_ids"]],
    }
    chunk_article = (chunk["article_number"] or "").lower()
    score = 0.0
    law_match = bool(law_refs and law_refs.intersection(chunk_law_aliases))
    article_match = bool(article_refs and chunk_article and chunk_article in article_refs)
    if law_match:
        score += 1.0
    if article_match:
        score += 2.5
    if law_match and article_match:
        score += 4.0
    return score


def source_rank_bonus(chunk: dict[str, Any], citations: dict[str, list[str]]) -> float:
    source_type = chunk.get("source_type", "pisrs")
    explicit_legal_query = bool(citations["law_refs"] or citations["articles"])
    if source_type == "pisrs":
        return 0.04 if explicit_legal_query else 0.02
    if source_type == "furs_guidance":
        return -0.06 if citations["articles"] else 0.0
    return 0.0


def query_source_preference_bonus(query: str, chunk: dict[str, Any]) -> float:
    lowered = query.lower()
    if "furs" not in lowered:
        return 0.0
    if chunk.get("source_type") == "furs_guidance":
        return 0.04
    if chunk.get("source_type") == "pisrs":
        return -0.01
    return 0.0


def source_policy_bonus(chunk: dict[str, Any], source_policy: str, query_profile=None) -> float:
    source_type = chunk.get("source_type", "pisrs")
    furs_score = furs_guidance_lexical_score(chunk, query_profile) if query_profile is not None else 0.0
    if source_policy == "pisrs_only":
        return 0.1 if source_type == "pisrs" else -1.0
    if source_policy == "pisrs_first":
        return 0.05 if source_type == "pisrs" else -0.05
    if source_policy == "furs_preferred":
        if source_type == "furs_guidance":
            if furs_score >= 3.0:
                return 0.18
            if furs_score >= 2.0:
                return 0.05
            return -0.04
        return -0.05
    if source_policy == "furs_allowed":
        if source_type == "furs_guidance" and furs_score >= 2.0:
            return 0.08
        return 0.0
    return 0.0


def apply_source_policy(results: list[dict[str, Any]], source_policy: str, query_profile=None) -> list[dict[str, Any]]:
    if source_policy != "pisrs_only":
        if source_policy == "furs_preferred":
            furs_results = [item for item in results if item["chunk"].get("source_type") == "furs_guidance"]
            if furs_results:
                best_furs = furs_results[0]
                best_overall = results[0] if results else None
                best_furs_score = furs_guidance_lexical_score(best_furs["chunk"], query_profile)
                if best_furs_score >= 2.5 and (best_overall is None or best_furs["score"] >= best_overall["score"] - 0.08):
                    return [best_furs] + [item for item in results if item["chunk"]["chunk_id"] != best_furs["chunk"]["chunk_id"]]
        return results
    filtered = [item for item in results if item["chunk"].get("source_type", "pisrs") == "pisrs"]
    return filtered or results


def direct_furs_guidance_candidates(
    chunk_map: dict[str, dict[str, Any]],
    query_profile,
    limit: int = 10,
) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    for chunk in chunk_map.values():
        if chunk.get("source_type") != "furs_guidance":
            continue
        score = furs_guidance_lexical_score(chunk, query_profile)
        if score < 2.5:
            continue
        ranked.append(
            {
                "chunk": chunk,
                "score": score,
                "bonus": min(0.28, 0.04 * score),
            }
        )
    ranked.sort(key=lambda item: (item["score"], item["chunk"].get("section_path") == "VPRAŠANJA IN ODGOVORI"), reverse=True)
    return ranked[:limit]


def source_specific_query_bonus(chunk: dict[str, Any], query_profile, source_policy: str) -> float:
    if chunk.get("source_type") == "furs_guidance" and (
        source_policy in {"furs_allowed", "furs_preferred"} or query_profile.expects_practical_guidance
    ):
        return min(0.22, 0.03 * furs_guidance_lexical_score(chunk, query_profile))
    if chunk.get("source_type", "pisrs") == "pisrs" and query_profile.has_technical_markers and source_policy == "furs_preferred":
        return -0.02
    return 0.0


def furs_guidance_lexical_score(chunk: dict[str, Any], query_profile) -> float:
    if chunk.get("source_type") != "furs_guidance":
        return 0.0
    haystack = normalize_text(
        " ".join(
            part
            for part in (
                chunk.get("title"),
                chunk.get("section_path"),
                chunk.get("raw_chunk_text", "")[:1800],
            )
            if part
        )
    ).lower()
    technical_terms = {term for term in query_profile.keywords if term in DISTINCTIVE_TECHNICAL_TERMS}
    if technical_terms and not any(term in haystack for term in technical_terms):
        return 0.0
    score = 0.0
    surface_hits = 0
    for term in query_profile.keywords:
        if len(term) < 4 or term in LOW_SIGNAL_FURS_TERMS:
            continue
        if term in haystack:
            surface_hits += 1
    score += float(surface_hits)
    if "vprašanja in odgovori" in haystack or "vprasanja in odgovori" in haystack:
        score += 1.0
    if "pojasnilo furs" in haystack:
        score += 0.5
    if query_profile.mentions_furs:
        score += 0.5
    if query_profile.has_technical_markers and any(marker in haystack for marker in ("xml", "edavki", "oauth", "certifikat", "spletni servis")):
        score += 1.5
    return score


def evaluate_retrieval(
    evaluation_rows: list[dict[str, Any]],
    chunk_map: dict[str, dict[str, Any]],
    chroma_dir: Path,
    bm25_path: Path,
    query_lemmas: dict[str, list[str]],
    top_k: int = 5,
    embedding_profile: str = DEFAULT_EMBEDDING_PROFILE,
    embedding_model_name: str | None = None,
    reranker_model: str | None = None,
) -> dict[str, Any]:
    dense_hits = 0
    hybrid_hits = 0
    dense_top1 = 0
    hybrid_top1 = 0
    explicit_citation_total = 0
    hybrid_explicit_top1 = 0
    per_query = []

    for row in evaluation_rows:
        query = row["query"]
        targets = row["targets"]
        lemmas = query_lemmas[row["query_id"]]
        route = route_query(build_query_profile(query, lemmas))
        dense_results = search_dense(
            query,
            chunk_map,
            chroma_dir,
            top_k,
            embedding_profile=embedding_profile,
            embedding_model_name=embedding_model_name,
        )
        hybrid_results = hybrid_search(
            query,
            lemmas,
            chunk_map,
            chroma_dir,
            bm25_path,
            top_k,
            embedding_profile=embedding_profile,
            embedding_model_name=embedding_model_name,
            source_policy=route.source_policy,
            reranker_model=reranker_model,
        )

        dense_hit = any(matches_target(item["chunk"], targets) for item in dense_results)
        hybrid_hit = any(matches_target(item["chunk"], targets) for item in hybrid_results)
        dense_hits += int(dense_hit)
        hybrid_hits += int(hybrid_hit)
        dense_top1 += int(bool(dense_results) and matches_target(dense_results[0]["chunk"], targets))
        hybrid_top1 += int(bool(hybrid_results) and matches_target(hybrid_results[0]["chunk"], targets))
        if row.get("category") == "explicit_citation":
            explicit_citation_total += 1
            hybrid_explicit_top1 += int(bool(hybrid_results) and matches_target(hybrid_results[0]["chunk"], targets))
        per_query.append(
            {
                "query_id": row["query_id"],
                "query": query,
                "dense_hit@5": dense_hit,
                "hybrid_hit@5": hybrid_hit,
                "dense_top1": bool(dense_results) and matches_target(dense_results[0]["chunk"], targets),
                "hybrid_top1": bool(hybrid_results) and matches_target(hybrid_results[0]["chunk"], targets),
            }
        )

    total = len(evaluation_rows) or 1
    return {
        "query_count": len(evaluation_rows),
        "embedding_profile": embedding_profile,
        "embedding_model_name": resolve_embedding_profile(embedding_profile, embedding_model_name).model_name,
        "reranker_model": reranker_model,
        "recall_at_5_dense": dense_hits / total,
        "recall_at_5_hybrid": hybrid_hits / total,
        "top1_dense": dense_top1 / total,
        "top1_hybrid": hybrid_top1 / total,
        "explicit_citation_top1_hybrid": (
            hybrid_explicit_top1 / explicit_citation_total if explicit_citation_total else None
        ),
        "per_query": per_query,
    }


def compare_embedding_profiles(
    evaluation_rows: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    chunk_map: dict[str, dict[str, Any]],
    chroma_dir: Path,
    bm25_path: Path,
    query_lemmas: dict[str, list[str]],
    embedding_profiles: list[str],
    top_k: int = 5,
    reranker_model: str | None = None,
) -> dict[str, Any]:
    report: dict[str, Any] = {"profiles": []}
    for profile_name in embedding_profiles:
        ensure_dense_index(chunks, chroma_dir, embedding_profile=profile_name)
        profile_report = evaluate_retrieval(
            evaluation_rows,
            chunk_map,
            chroma_dir,
            bm25_path,
            query_lemmas,
            top_k=top_k,
            embedding_profile=profile_name,
            reranker_model=reranker_model,
        )
        report["profiles"].append(profile_report)
    return report


def matches_target(chunk: dict[str, Any], targets: list[dict[str, Any]]) -> bool:
    for target in targets:
        if target.get("law_id") and chunk["law_id"] != target["law_id"]:
            continue
        if target.get("article_number") and (chunk["article_number"] or "") != target["article_number"]:
            continue
        return True
    return False


def direct_citation_candidates(
    chunk_map: dict[str, dict[str, Any]],
    citations: dict[str, list[str]],
) -> list[dict[str, Any]]:
    law_refs = {ref.lower() for ref in citations["law_refs"]}
    article_refs = {ref.lower() for ref in citations["articles"]}
    if not law_refs and not article_refs:
        return []

    ranked: list[tuple[int, dict[str, Any]]] = []
    for chunk in chunk_map.values():
        chunk_law_aliases = {
            chunk["law_id"].lower(),
            chunk["title"].lower(),
            *[ref.lower() for ref in chunk["legal_refs"]["law_refs"]],
            *[ref.lower() for ref in chunk["legal_refs"]["act_ids"]],
        }
        chunk_article = (chunk["article_number"] or "").lower()
        law_match = bool(law_refs and law_refs.intersection(chunk_law_aliases))
        article_match = bool(article_refs and chunk_article and chunk_article in article_refs)
        if not law_match and not article_match:
            continue
        rank_key = 0 if law_match and article_match else 1 if article_match else 2
        ranked.append((rank_key, chunk))

    ranked.sort(key=lambda item: (item[0], item[1]["law_id"], item[1]["article_number"] or ""))
    return [chunk for _, chunk in ranked[:25]]


def ensure_dense_index(
    chunks: list[dict[str, Any]],
    chroma_dir: Path,
    embedding_profile: str = DEFAULT_EMBEDDING_PROFILE,
    embedding_model_name: str | None = None,
) -> None:
    import chromadb
    from chromadb.config import Settings

    profile = resolve_embedding_profile(embedding_profile, embedding_model_name)
    configure_chromadb_logging()
    chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=str(chroma_dir),
        settings=Settings(anonymized_telemetry=False),
    )
    existing = {
        collection if isinstance(collection, str) else collection.name
        for collection in client.list_collections()
    }
    if profile.collection_name in existing:
        return
    build_dense_index(
        chunks,
        chroma_dir,
        embedding_profile=profile.name,
        embedding_model_name=profile.model_name,
        collection_name=profile.collection_name,
    )
