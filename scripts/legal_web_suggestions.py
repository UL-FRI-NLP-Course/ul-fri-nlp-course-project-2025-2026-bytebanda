#!/usr/bin/env python3
"""Create related-source suggestions for Slovenian legal RAG questions.

This is intentionally separate from the main RAG pipeline. By default it searches
local downloaded data first; with --mode web or --mode both it also tries a small
DuckDuckGo HTML search over curated Slovenian legal/source tiers.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from bs4 import BeautifulSoup
except ImportError:  # pragma: no cover - script still works with regex fallback.
    BeautifulSoup = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DEFAULT_INPUT = PROJECT_ROOT / "evaluation" / "web_suggestion_seed_questions.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "logs" / "web_suggestions.jsonl"
DEFAULT_LOCAL_DIR = PROJECT_ROOT / "downloads" / "pisrs"
DEFAULT_CHUNKS = PROJECT_ROOT / "data" / "index" / "chunks.jsonl"
DEFAULT_EXPERIMENT_CSV = (
    PROJECT_ROOT
    / "experiments"
    / "retrieval"
    / "four_dataset_medium_mig_01"
    / "reports"
    / "retrieval_experiments.csv"
)
DEFAULT_LOCAL_MODEL_PATH = Path("/d/hpc/projects/onj_fri/models/intent")


@dataclass(frozen=True)
class Source:
    tier: int
    tier_name: str
    name: str
    domain: str
    note: str


SOURCES: tuple[Source, ...] = (
    Source(1, "authoritative", "PISRS", "pisrs.si", "Official Slovenian legal texts"),
    Source(1, "authoritative", "FURS", "fu.gov.si", "Slovenian Financial Administration guidance"),
    Source(1, "authoritative", "GOV.SI", "gov.si", "Official Slovenian government pages"),
    Source(1, "authoritative", "e-Uprava", "e-uprava.gov.si", "Official public services portal"),
    Source(1, "authoritative", "AJPES", "ajpes.si", "Official business registry and filings"),
    Source(2, "expert_commentary", "IUS-INFO", "iusinfo.si", "Legal commentary and case-law portal"),
    Source(2, "expert_commentary", "Tax-Fin-Lex", "tax-fin-lex.si", "Tax/legal commentary and news"),
    Source(2, "expert_commentary", "Pravna praksa", "pravna-praksa.si", "Legal practice commentary"),
    Source(3, "natural_language", "Data.si", "data.si", "Practical business and tax explanations"),
    Source(3, "natural_language", "Mladi podjetnik", "mladipodjetnik.si", "Entrepreneur-oriented explanations"),
    Source(3, "natural_language", "Legal forums", "pravniki.info", "Forum-style legal phrasing"),
    Source(3, "natural_language", "Reddit Slovenia", "reddit.com/r/Slovenia", "Natural user questions"),
)

BLOCKED_WEB_DOMAINS: set[str] = set()
WARNED_WEB_DOMAINS: set[str] = set()

SLO_STOPWORDS = {
    "ali",
    "bolj",
    "bodo",
    "brez",
    "biti",
    "clen",
    "člen",
    "dela",
    "glede",
    "kako",
    "kaj",
    "kdaj",
    "kdo",
    "kjer",
    "kateri",
    "katere",
    "lahko",
    "mora",
    "moraš",
    "mora",
    "naj",
    "oziroma",
    "pod",
    "pri",
    "pravila",
    "pravilo",
    "sem",
    "sta",
    "ter",
    "tega",
    "tem",
    "to",
    "velja",
    "za",
    "zakaj",
    "zakona",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def tokens(text: str) -> set[str]:
    found = re.findall(r"[A-Za-zČŠŽčšž0-9][A-Za-zČŠŽčšž0-9.-]*", text.lower())
    return {tok for tok in found if len(tok) >= 3 and tok not in SLO_STOPWORDS}


def short_summary(text: str, query: str, max_chars: int = 320) -> str:
    text = normalize_text(text)
    if not text:
        return ""
    query_terms = tokens(query)
    sentences = re.split(r"(?<=[.!?])\s+", text)
    ranked = sorted(
        sentences,
        key=lambda sentence: len(tokens(sentence) & query_terms),
        reverse=True,
    )
    summary = ranked[0] if ranked else text
    if len(summary) > max_chars:
        summary = summary[: max_chars - 3].rsplit(" ", 1)[0] + "..."
    return summary


def strip_html(raw: str) -> str:
    if BeautifulSoup is not None:
        soup = BeautifulSoup(raw, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        return normalize_text(soup.get_text(" "))
    return normalize_text(re.sub(r"<[^>]+>", " ", raw))


def split_sentences(text: str) -> list[str]:
    """Split context into sentence-like units for extractive answering."""
    cleaned = normalize_text(text)
    if not cleaned:
        return []
    pieces = re.split(r"(?<=[.!?])\s+|\n+", cleaned)
    return [piece.strip() for piece in pieces if 40 <= len(piece.strip()) <= 700]


def read_index_chunks(chunks_path: Path) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    if not chunks_path.exists():
        return records
    for chunk in load_jsonl(chunks_path):
        metadata = chunk.get("metadata") or {}
        article = metadata.get("article_number")
        law_id = metadata.get("law_id")
        title = metadata.get("article_title") or ""
        heading = f"{law_id or ''} {article or ''}. clen {title}".strip()
        source = str(chunk.get("source") or metadata.get("raw_path") or "indexed chunk")
        records.append(
            {
                "title": normalize_text(f"{heading} [{chunk.get('chunk_id')}]"),
                "url": source,
                "text": str(chunk.get("text") or ""),
                "source_name": "local indexed legal chunk",
                "domain": "local",
                "tier": "1",
                "tier_name": "authoritative",
            }
        )
    return records


def read_local_documents(local_dir: Path) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    if not local_dir.exists():
        return records
    for path in sorted(local_dir.rglob("*")):
        if path.suffix.lower() not in {".html", ".txt", ".md"}:
            continue
        try:
            raw = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        text = strip_html(raw) if path.suffix.lower() == ".html" else normalize_text(raw)
        if text:
            records.append(
                {
                    "title": path.stem,
                    "url": str(path),
                    "text": text,
                    "source_name": "local PISRS download",
                    "domain": "local",
                    "tier": "1",
                    "tier_name": "authoritative",
                }
            )
    return records


def score_text(query: str, text: str) -> float:
    query_terms = tokens(query)
    if not query_terms:
        return 0.0
    text_terms = tokens(text)
    overlap = len(query_terms & text_terms)
    law_bonus = 0.0
    for law in ("zddv-1", "zdoh-2", "zddpo-2", "zdavp-2"):
        if law in query.lower() and law in text.lower():
            law_bonus += 2.0
    return overlap / max(len(query_terms), 1) + law_bonus


def source_affinity(question: str, source: Source | None, result: dict[str, str] | None = None) -> float:
    """Boost sources that are a natural fit for the question."""
    if source is None:
        return 0.0

    haystack = " ".join(
        [
            question,
            source.name,
            source.domain,
            (result or {}).get("title", ""),
            (result or {}).get("snippet", ""),
        ]
    ).lower()
    score = 0.0

    if any(term in haystack for term in ("ddv", "zddv")):
        if source.name in {"FURS", "PISRS", "GOV.SI"}:
            score += 0.6
    if any(term in haystack for term in ("zdoh", "dohodnina", "normiran", "normirane")):
        if source.name in {"PISRS", "FURS", "GOV.SI"}:
            score += 0.6
    if any(term in haystack for term in ("zddpo", "pravnih oseb", "ddpo")):
        if source.name in {"PISRS", "FURS", "GOV.SI"}:
            score += 0.6
    if any(term in haystack for term in ("zdavp", "davčni obračun", "davcni obracun", "zamudi")):
        if source.name in {"FURS", "PISRS", "GOV.SI"}:
            score += 0.6
    if any(term in haystack for term in ("ajpes", "letno poročilo", "letno porocilo", "bilanca")):
        if source.name == "AJPES":
            score += 1.2
        elif source.name in {"PISRS", "GOV.SI"}:
            score += 0.4

    if source.name == "e-Uprava" and not any(
        term in haystack for term in ("e-uprava", "prebivali", "vloga", "portal", "dovoljenje")
    ):
        score -= 0.4
    return score


def result_relevance(question: str, title: str, summary: str, source: Source | None = None) -> float:
    """Score one result with lexical overlap plus source/domain fit."""
    result = {"title": title, "snippet": summary}
    return score_text(question, f"{title} {summary}") + source_affinity(question, source, result)


def is_noisy_result(question: str, title: str, summary: str, source: Source | None, min_relevance: float) -> bool:
    """Filter obviously unrelated or boilerplate-heavy results."""
    relevance = result_relevance(question, title, summary, source)
    if relevance < min_relevance:
        return True

    lower = f"{title} {summary}".lower()
    boilerplate_markers = (
        "pisava",
        "velikost",
        "velike/male",
        "koristnost strani",
        "čez nekaj sekund vas bomo",
        "filtriranje se bo izvedlo",
        "pozabljeno geslo",
    )
    if any(marker in lower for marker in boilerplate_markers):
        query_overlap = len(tokens(question) & tokens(f"{title} {summary}"))
        return query_overlap < 3
    return False


def source_role(suggestion: dict[str, Any]) -> str:
    """Classify a suggestion's role for answer/report readability."""
    source_name = str(suggestion.get("source_name") or "").lower()
    domain = str(suggestion.get("domain") or "").lower()
    if suggestion.get("retrieval_method") == "best_local_rag":
        return "local_legal_context"
    if "pisrs" in source_name or "pisrs" in domain:
        return "primary_legal_source"
    if "furs" in source_name or "fu.gov.si" in domain:
        return "official_guidance"
    if "ajpes" in source_name or "ajpes" in domain:
        return "official_register_guidance"
    if "gov" in source_name or "gov.si" in domain:
        return "official_public_source"
    return "supporting_source"


def selected_sources(question: str, category: str, tiers: set[int], policy: str) -> list[Source]:
    """Choose source domains to search without hard-coding result links."""
    eligible = [source for source in SOURCES if source.tier in tiers]
    if policy == "all":
        return eligible

    text = f"{question} {category}".lower()
    names: list[str] = []
    if any(term in text for term in ("ajpes", "letno poročilo", "letno porocilo", "bilanca")):
        names.extend(["AJPES", "PISRS", "GOV.SI", "FURS"])
    if any(term in text for term in ("ddv", "zddv")):
        names.extend(["FURS", "PISRS", "GOV.SI"])
    if any(term in text for term in ("zdoh", "dohodnina", "normiran", "normirane")):
        names.extend(["PISRS", "FURS", "GOV.SI"])
    if any(term in text for term in ("zddpo", "pravnih oseb", "ddpo")):
        names.extend(["PISRS", "FURS", "GOV.SI"])
    if any(term in text for term in ("zdavp", "davčni obračun", "davcni obracun", "zamudi")):
        names.extend(["FURS", "PISRS", "GOV.SI"])
    if any(term in text for term in ("e-uprava", "prebivali", "vloga", "dovoljenje")):
        names.extend(["e-Uprava", "GOV.SI"])

    if 2 in tiers:
        names.extend(["IUS-INFO", "Tax-Fin-Lex", "Pravna praksa"])
    if 3 in tiers:
        names.extend(["Data.si", "Mladi podjetnik", "Legal forums", "Reddit Slovenia"])

    if not names:
        names = [source.name for source in eligible]

    seen = set()
    ordered_names = []
    for name in names:
        if name not in seen:
            seen.add(name)
            ordered_names.append(name)
    return [source for name in ordered_names for source in eligible if source.name == name]


def local_suggestions(question: str, local_docs: list[dict[str, str]], limit: int) -> list[dict[str, Any]]:
    ranked = []
    for doc in local_docs:
        score = score_text(question, f"{doc['title']} {doc['text'][:8000]}")
        if score <= 0:
            continue
        ranked.append((score, doc))
    ranked.sort(key=lambda item: item[0], reverse=True)

    suggestions = []
    for score, doc in ranked[:limit]:
        suggestions.append(
            {
                "source_tier": int(doc["tier"]),
                "tier_name": doc["tier_name"],
                "source_name": doc["source_name"],
                "domain": doc["domain"],
                "title": doc["title"],
                "url": doc["url"],
                "summary": short_summary(doc["text"], question),
                "relevance": round(score, 4),
                "retrieval_method": "local_downloads",
            }
        )
    return suggestions


def load_best_retrieval_config(csv_path: Path = DEFAULT_EXPERIMENT_CSV) -> dict[str, Any] | None:
    """Read the best retrieval config from the experiment CSV."""
    if not csv_path.exists():
        return None
    with csv_path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return None

    def score(row: dict[str, str]) -> float:
        try:
            return float(row.get("score") or 0.0)
        except ValueError:
            return 0.0

    best = max(rows, key=score)
    embedding_slug = str(best["embedding"]).replace("/", "-")
    index_dir = (
        PROJECT_ROOT
        / "experiments"
        / "retrieval"
        / "four_dataset_medium_mig_01"
        / "indexes"
        / best["chunk_config"]
        / embedding_slug
    )
    return {
        "experiment_csv": str(csv_path),
        "score": float(best["score"]),
        "dataset": best["dataset"],
        "chunk_config": best["chunk_config"],
        "embedding_model": best["embedding"],
        "retrieval_config": best["retrieval_config"],
        "retrieval_mode": best["retrieval_mode"],
        "top_k": int(float(best["top_k"])),
        "candidate_k": int(float(best["candidate_k"])),
        "lexical_weight": float(best["lexical_weight"]),
        "source_boost": float(best["source_boost"]),
        "article_boost": float(best["article_boost"]),
        "title_weight": float(best["title_weight"]),
        "index_path": str(index_dir / "faiss.index"),
        "chunks_path": str(index_dir / "chunks.jsonl"),
    }


def best_local_rag_suggestions(question: str, limit: int = 4) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Retrieve local legal chunks using the best experiment config."""
    config = load_best_retrieval_config()
    if not config:
        return [], None

    index_path = Path(config["index_path"])
    chunks_path = Path(config["chunks_path"])
    if not index_path.exists() or not chunks_path.exists():
        return [], config

    try:
        from src.retrieve import RetrievalEngine, infer_query_law_ids
    except ImportError as exc:
        print(f"WARNING: cannot import local retriever: {exc}", file=sys.stderr)
        return [], config

    try:
        retriever = RetrievalEngine(
            index_path=index_path,
            chunks_path=chunks_path,
            embedding_model=config["embedding_model"],
        )
        chunks = retriever.retrieve(
            question,
            top_k=min(limit, config["top_k"]),
            retrieval_mode=config["retrieval_mode"],
            candidate_k=config["candidate_k"],
            lexical_weight=config["lexical_weight"],
            source_boost=config["source_boost"],
            article_boost=config["article_boost"],
            title_weight=config["title_weight"],
            query_law_ids=sorted(infer_query_law_ids(question)),
        )
    except Exception as exc:
        print(f"WARNING: best local retrieval failed: {exc}", file=sys.stderr)
        return [], config

    suggestions = []
    for chunk in chunks[:limit]:
        metadata = chunk.get("metadata") or {}
        title_parts = [
            metadata.get("law_id"),
            f"{metadata.get('article_number')}. clen" if metadata.get("article_number") else None,
            metadata.get("article_title"),
            f"[{chunk.get('chunk_id')}]",
        ]
        title = normalize_text(" ".join(str(part) for part in title_parts if part))
        suggestions.append(
            {
                "source_tier": 1,
                "tier_name": "authoritative",
                "source_name": "best local RAG config",
                "domain": "local",
                "title": title,
                "url": str(chunk.get("source") or ""),
                "summary": short_summary(str(chunk.get("text") or ""), question, max_chars=700),
                "relevance": round(float(chunk.get("score") or 0.0), 4),
                "retrieval_method": "best_local_rag",
                "source_note": "Local legal index using the best experiment retrieval configuration.",
                "chunk_id": chunk.get("chunk_id"),
                "metadata": metadata,
            }
        )
    return suggestions, config


def fetch_url(url: str, timeout: int = 12) -> str:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (compatible; legal-rag-source-suggestions/0.1; "
                "+https://example.invalid)"
            )
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="ignore")


def fetch_page_text(url: str, timeout: int = 12, max_chars: int = 12000) -> str:
    """Fetch simple HTML/text pages for answer context; skip PDFs and office docs."""
    lowered = url.lower()
    if any(lowered.endswith(suffix) for suffix in (".pdf", ".doc", ".docx", ".xls", ".xlsx")):
        return ""
    try:
        raw = fetch_url(url, timeout=timeout)
    except Exception:
        return ""
    return strip_html(raw)[:max_chars]


def fetch_json(
    url: str,
    headers: dict[str, str] | None = None,
    timeout: int = 12,
) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "legal-rag-source-suggestions/0.1",
            **(headers or {}),
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8", errors="ignore"))


def search_hint(source: Source, question: str, query: str) -> dict[str, Any]:
    """Return a source/query hint without inventing a search-result URL."""
    return {
        "source_tier": source.tier,
        "tier_name": source.tier_name,
        "source_name": source.name,
        "domain": source.domain,
        "title": f"Search {source.name}",
        "summary": (
            f"Curated {source.tier_name.replace('_', ' ')} source. "
            f"Suggested query for this source: {query}"
        ),
        "relevance": 0.0,
        "retrieval_method": "search_hint",
        "source_note": source.note,
        "search_query": query,
    }


def parse_duckduckgo_results(raw_html: str) -> list[dict[str, str]]:
    if BeautifulSoup is None:
        return []
    soup = BeautifulSoup(raw_html, "html.parser")
    results: list[dict[str, str]] = []
    for result in soup.select(".result"):
        link = result.select_one(".result__a")
        if not link:
            continue
        href = link.get("href", "")
        parsed = urllib.parse.urlparse(href)
        params = urllib.parse.parse_qs(parsed.query)
        url = params.get("uddg", [href])[0]
        snippet_node = result.select_one(".result__snippet")
        results.append(
            {
                "title": normalize_text(link.get_text(" ")),
                "url": html.unescape(url),
                "snippet": normalize_text(snippet_node.get_text(" ")) if snippet_node else "",
            }
        )
    return results


def brave_results(query: str, limit: int) -> list[dict[str, str]]:
    """Return Brave Search API results. Requires BRAVE_SEARCH_API_KEY."""
    api_key = os.environ.get("BRAVE_SEARCH_API_KEY")
    if not api_key:
        raise RuntimeError("BRAVE_SEARCH_API_KEY is not set")

    url = "https://api.search.brave.com/res/v1/web/search?" + urllib.parse.urlencode(
        {"q": query, "count": max(1, min(limit, 20))}
    )
    payload = fetch_json(
        url,
        headers={
            "Accept": "application/json",
            "X-Subscription-Token": api_key,
        },
    )
    web_payload = payload.get("web") or {}
    records = []
    for item in web_payload.get("results") or []:
        records.append(
            {
                "title": normalize_text(str(item.get("title") or "")),
                "url": str(item.get("url") or ""),
                "snippet": normalize_text(str(item.get("description") or "")),
            }
        )
    return [record for record in records if record["url"]]


def tavily_results(query: str, limit: int) -> list[dict[str, str]]:
    """Return Tavily Search API results. Requires TAVILY_API_KEY."""
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY is not set")

    request = urllib.request.Request(
        "https://api.tavily.com/search",
        data=json.dumps(
            {
                "query": query,
                "max_results": max(1, min(limit, 20)),
                "search_depth": "basic",
                "include_answer": False,
            }
        ).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "legal-rag-source-suggestions/0.1",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        payload = json.loads(response.read().decode("utf-8", errors="ignore"))

    records = []
    for item in payload.get("results") or []:
        records.append(
            {
                "title": normalize_text(str(item.get("title") or "")),
                "url": str(item.get("url") or ""),
                "snippet": normalize_text(str(item.get("content") or "")),
            }
        )
    return [record for record in records if record["url"]]


def duckduckgo_results(query: str, limit: int) -> list[dict[str, str]]:
    search_url = "https://html.duckduckgo.com/html/?" + urllib.parse.urlencode({"q": query})
    raw = fetch_url(search_url)
    return parse_duckduckgo_results(raw)[:limit]


def broad_web_suggestions(
    question: str,
    limit: int,
    provider: str,
    sleep_seconds: float,
    min_relevance: float,
) -> list[dict[str, Any]]:
    query = question
    try:
        if provider == "brave":
            results = brave_results(query, limit)
        elif provider == "tavily":
            results = tavily_results(query, limit)
        elif provider == "duckduckgo":
            results = duckduckgo_results(query, limit)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    except Exception as exc:
        print(f"WARNING: broad web search failed with provider={provider}: {exc}", file=sys.stderr)
        return []

    suggestions = []
    for result in results:
        relevance = result_relevance(question, result["title"], result["snippet"])
        if is_noisy_result(question, result["title"], result["snippet"], None, min_relevance):
            continue
        suggestions.append(
            {
                "source_tier": None,
                "tier_name": "broad_web",
                "source_name": provider,
                "domain": urllib.parse.urlparse(result["url"]).netloc.lower(),
                "title": result["title"],
                "url": result["url"],
                "summary": result["snippet"],
                "relevance": round(relevance, 4),
                "retrieval_method": f"{provider}_broad_search",
                "search_query": query,
            }
        )
    if sleep_seconds:
        time.sleep(sleep_seconds)
    return suggestions


def web_suggestions(
    question: str,
    tiers: set[int],
    per_domain: int,
    sleep_seconds: float,
    provider: str,
    broad: bool,
    allow_curated_fallback: bool,
    source_policy: str,
    min_relevance: float,
    category: str = "",
) -> list[dict[str, Any]]:
    if broad:
        return broad_web_suggestions(
            question=question,
            limit=per_domain,
            provider=provider,
            sleep_seconds=sleep_seconds,
            min_relevance=min_relevance,
        )

    suggestions: list[dict[str, Any]] = []
    for source in selected_sources(question, category, tiers, source_policy):
        query = f"{question} site:{source.domain}"
        if source.domain in BLOCKED_WEB_DOMAINS:
            if allow_curated_fallback:
                suggestions.append(search_hint(source, question, query))
            continue
        try:
            if provider == "brave":
                results = brave_results(query, per_domain)
            elif provider == "tavily":
                results = tavily_results(query, per_domain)
            elif provider == "duckduckgo":
                results = duckduckgo_results(query, per_domain)
            else:
                raise ValueError(f"Unsupported provider: {provider}")
        except Exception as exc:
            message = str(exc)
            if "HTTP Error 403" in message:
                BLOCKED_WEB_DOMAINS.add(source.domain)
            if source.domain not in WARNED_WEB_DOMAINS:
                print(
                    f"WARNING: web search failed for {source.domain}: {exc}. "
                    "No synthetic search-result links will be emitted unless fallback hints are enabled.",
                    file=sys.stderr,
                )
                WARNED_WEB_DOMAINS.add(source.domain)
            results = []
        if not results and allow_curated_fallback:
            suggestions.append(search_hint(source, question, query))
        for result in results:
            relevance = result_relevance(question, result["title"], result["snippet"], source)
            if is_noisy_result(question, result["title"], result["snippet"], source, min_relevance):
                continue
            suggestions.append(
                {
                    "source_tier": source.tier,
                    "tier_name": source.tier_name,
                    "source_name": source.name,
                    "domain": source.domain,
                    "title": result["title"],
                    "url": result["url"],
                    "summary": result["snippet"],
                    "relevance": round(relevance, 4),
                    "retrieval_method": f"{provider}_site_search",
                    "source_note": source.note,
                    "search_query": query,
                }
            )
        if sleep_seconds:
            time.sleep(sleep_seconds)
    suggestions.sort(key=lambda item: (item["source_tier"], -item["relevance"]))
    return suggestions


def make_extractive_answer(
    question: str,
    suggestions: list[dict[str, Any]],
    fetch_pages: bool,
    max_sources: int,
    max_sentences: int,
) -> dict[str, Any]:
    """Build a short answer from retrieved snippets/page text with citations."""
    contexts: list[dict[str, Any]] = []
    for index, suggestion in enumerate(suggestions[:max_sources], start=1):
        summary = str(suggestion.get("summary") or "")
        page_text = fetch_page_text(str(suggestion.get("url") or "")) if fetch_pages else ""
        text = normalize_text(f"{summary} {page_text}")
        if not text:
            continue
        contexts.append(
            {
                "ref": index,
                "title": suggestion.get("title"),
                "url": suggestion.get("url"),
                "source_name": suggestion.get("source_name"),
                "role": source_role(suggestion),
                "text": text,
            }
        )

    ranked_sentences: list[tuple[float, int, str]] = []
    for context in contexts:
        for sentence in split_sentences(context["text"]):
            ranked_sentences.append(
                (
                    result_relevance(question, str(context["title"] or ""), sentence),
                    int(context["ref"]),
                    sentence,
                )
            )
    ranked_sentences.sort(key=lambda item: item[0], reverse=True)

    selected: list[tuple[int, str]] = []
    seen = set()
    for score, ref, sentence in ranked_sentences:
        key = normalize_text(sentence).lower()[:180]
        if key in seen or score <= 0:
            continue
        seen.add(key)
        selected.append((ref, sentence))
        if len(selected) >= max_sentences:
            break

    sources = [
        {
            "ref": context["ref"],
            "title": context["title"],
            "url": context["url"],
            "source_name": context["source_name"],
            "role": context["role"],
        }
        for context in contexts
    ]
    if not selected:
        return {
            "method": "extractive_web_context",
            "text": "No answer could be extracted from the retrieved context.",
            "sources": sources,
        }

    return {
        "method": "extractive_web_context",
        "text": " ".join(f"{sentence} [{ref}]" for ref, sentence in selected),
        "sources": sources,
    }


def format_llm_context(
    suggestions: list[dict[str, Any]],
    fetch_pages: bool,
    max_sources: int,
    max_chars_per_source: int,
) -> tuple[str, list[dict[str, Any]]]:
    """Format retrieved web/local suggestions as cited LLM context."""
    blocks = []
    sources = []
    for index, suggestion in enumerate(suggestions[:max_sources], start=1):
        summary = str(suggestion.get("summary") or "")
        page_text = fetch_page_text(str(suggestion.get("url") or "")) if fetch_pages else ""
        context_text = normalize_text(f"{summary}\n{page_text}")[:max_chars_per_source]
        if not context_text:
            continue
        role = source_role(suggestion)
        title = str(suggestion.get("title") or "")
        url = str(suggestion.get("url") or "")
        source_name = str(suggestion.get("source_name") or "")
        blocks.append(
            f"[{index}] source_name={source_name}\n"
            f"role={role}\n"
            f"title={title}\n"
            f"url={url}\n"
            f"context={context_text}"
        )
        sources.append(
            {
                "ref": index,
                "title": title,
                "url": url,
                "source_name": source_name,
                "role": role,
            }
        )
    return "\n\n".join(blocks), sources


def make_llm_answer(
    question: str,
    suggestions: list[dict[str, Any]],
    model_path: Path,
    fetch_pages: bool,
    max_sources: int,
    max_chars_per_source: int,
    max_new_tokens: int,
    llm_bundle: tuple[Any, Any, Any] | None = None,
) -> dict[str, Any]:
    """Generate a grounded answer from retrieved web/local context using the local LLM."""
    try:
        from src.generate_answer import generate_from_prompt, load_llm
    except ImportError as exc:
        return {"method": "llm_context_answer", "text": "", "error": f"Cannot import LLM helpers: {exc}"}

    context, sources = format_llm_context(
        suggestions,
        fetch_pages=fetch_pages,
        max_sources=max_sources,
        max_chars_per_source=max_chars_per_source,
    )
    if not context:
        return {
            "method": "llm_context_answer",
            "text": "No answer could be generated because no context was available.",
            "sources": sources,
        }

    system = (
        "You are a careful Slovenian legal/tax research assistant. Answer in the same "
        "language as the user's question. If the question is in Slovenian, answer in "
        "Slovenian. Answer only from the provided sources. Prefer authoritative sources "
        "such as PISRS, FURS, GOV.SI, e-Uprava, and AJPES. Treat web search snippets as "
        "unverified leads unless they come from official sources. If the sources do not "
        "contain enough information, say so. Cite sources using bracket numbers like [1], [2]."
    )
    user = f"""Question:
{question}

Retrieved sources:
{context}

Write a concise but complete answer in the same language as the question. Explain the rule in plain language, mention important thresholds or deadlines when present, and cite the source numbers."""

    prompt = f"<s>[INST] {system}\n\n{user} [/INST]"
    try:
        tokenizer, model, torch_module = llm_bundle or load_llm(model_path)
        answer = generate_from_prompt(
            prompt,
            tokenizer,
            model,
            torch_module,
            max_new_tokens=max_new_tokens,
        )
    except Exception as exc:
        return {
            "method": "llm_context_answer",
            "text": "",
            "sources": sources,
            "error": str(exc),
        }

    return {
        "method": "llm_context_answer",
        "text": answer.strip(),
        "sources": sources,
    }


def build_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    questions = load_jsonl(args.questions)
    if args.limit > 0:
        questions = questions[: args.limit]

    tiers = {int(value) for value in args.tiers.split(",") if value.strip()}
    local_docs: list[dict[str, str]] = []
    if args.mode in {"local", "both"}:
        local_docs = read_index_chunks(args.chunks_path)
        if not local_docs:
            local_docs = read_local_documents(args.local_dir)
    records: list[dict[str, Any]] = []
    llm_bundle = None
    if args.llm_answer:
        try:
            from src.generate_answer import load_llm

            llm_bundle = load_llm(args.model_path)
        except Exception as exc:
            print(f"WARNING: LLM loading failed: {exc}", file=sys.stderr)

    for case in questions:
        question = str(case.get("question", "")).strip()
        if not question:
            continue

        suggestions: list[dict[str, Any]] = []
        best_config = None
        if args.mode in {"local", "both"}:
            suggestions.extend(local_suggestions(question, local_docs, args.local_limit))
        if args.mode in {"web", "both"}:
            suggestions.extend(
                web_suggestions(
                    question,
                    tiers,
                    args.per_domain,
                    args.sleep,
                    provider=args.provider,
                    broad=args.broad,
                    allow_curated_fallback=args.allow_curated_fallback,
                    source_policy=args.source_policy,
                    min_relevance=args.min_relevance,
                    category=str(case.get("category") or ""),
                )
            )
        if args.best_local_context:
            local_rag_suggestions, best_config = best_local_rag_suggestions(
                question,
                limit=args.best_local_limit,
            )
            suggestions.extend(local_rag_suggestions)

        suggestions.sort(
            key=lambda item: (
                99 if item["source_tier"] is None else item["source_tier"],
                -float(item["relevance"]),
            )
        )
        record = {
            "id": case.get("id"),
            "question": question,
            "category": case.get("category"),
            "suggestions": suggestions[: args.max_suggestions],
        }
        if args.answer:
            record["answer"] = make_extractive_answer(
                question,
                record["suggestions"],
                fetch_pages=args.fetch_pages,
                max_sources=args.answer_sources,
                max_sentences=args.answer_sentences,
            )
        if args.llm_answer:
            record["llm_answer"] = make_llm_answer(
                question,
                record["suggestions"],
                model_path=args.model_path,
                fetch_pages=args.fetch_pages,
                max_sources=args.answer_sources,
                max_chars_per_source=args.llm_context_chars,
                max_new_tokens=args.max_new_tokens,
                llm_bundle=llm_bundle,
            )
        if best_config:
            record["best_local_config"] = best_config
        records.append(record)
    return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Suggest related web/local sources for legal RAG questions.")
    parser.add_argument("--questions", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--local-dir", type=Path, default=DEFAULT_LOCAL_DIR)
    parser.add_argument("--chunks-path", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--mode", choices=["local", "web", "both"], default="local")
    parser.add_argument("--provider", choices=["duckduckgo", "brave", "tavily"], default="duckduckgo")
    parser.add_argument(
        "--broad",
        action="store_true",
        help="Search the whole web instead of searching one curated site at a time.",
    )
    parser.add_argument(
        "--allow-curated-fallback",
        action="store_true",
        help="Emit source/query hints when automated search results cannot be fetched.",
    )
    parser.add_argument(
        "--source-policy",
        choices=["auto", "all"],
        default="auto",
        help="auto searches only likely source domains; all searches every selected tier domain.",
    )
    parser.add_argument(
        "--min-relevance",
        type=float,
        default=0.25,
        help="Drop web results below this internal relevance score.",
    )
    parser.add_argument("--tiers", default="1,2,3", help="Comma-separated source tiers to use for web mode.")
    parser.add_argument("--limit", type=int, default=0, help="Only process the first N questions.")
    parser.add_argument("--local-limit", type=int, default=6)
    parser.add_argument(
        "--best-local-context",
        action="store_true",
        help="Add local legal chunks retrieved with the best experiment config.",
    )
    parser.add_argument("--best-local-limit", type=int, default=4)
    parser.add_argument("--answer", action="store_true", help="Add an extractive answer from retrieved context.")
    parser.add_argument("--llm-answer", action="store_true", help="Generate an answer with the local LLM.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_LOCAL_MODEL_PATH)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--llm-context-chars", type=int, default=1800)
    parser.add_argument(
        "--fetch-pages",
        action="store_true",
        help="Fetch HTML result pages and use their text as extra answer context.",
    )
    parser.add_argument("--answer-sources", type=int, default=5)
    parser.add_argument("--answer-sentences", type=int, default=4)
    parser.add_argument("--per-domain", type=int, default=1)
    parser.add_argument("--max-suggestions", type=int, default=8)
    parser.add_argument("--sleep", type=float, default=0.4, help="Delay between web search requests.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    records = build_records(args)
    write_jsonl(records, args.output)
    print(f"Wrote {len(records)} suggestion record(s) to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
