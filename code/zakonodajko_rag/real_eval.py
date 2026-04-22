from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

import requests

from .answering import answer_question
from .constants import DEFAULT_LOCAL_GENERATOR_MAX_NEW_TOKENS
from .regexes import extract_query_citations
from .router import SLOVENE_STOPWORDS, TERM_RE
from .text_utils import ensure_parent_dir, normalize_text, read_jsonl, stable_id_fragment, write_jsonl


FURS_REAL_EVAL_SEEDS = [
    {
        "title": "Kratka vprašanja in odgovori - evidenci obračunanega DDV in odbitka DDV ter predizpolnitev obračuna DDV",
        "url": "https://www.fu.gov.si/fileadmin/Internet/Davki_in_druge_dajatve/Podrocja/Davek_na_dodano_vrednost/Opis/Kratka_vprasanja_in_odgovori_Evidenca_obracunanega_DDV_in_Evidenca_odbitka_DDV.doc",
        "category": "furs_ddv_qna",
    },
    {
        "title": "Poročanje po državah (Country-by-Country Reporting) - vprašanja in odgovori",
        "url": "https://www.fu.gov.si/fileadmin/Internet/Nadzor/Podrocja/CbCR/Opis/Porocanje_po_drzavah_CbCR.docx",
        "category": "furs_cbcr_qna",
    },
]

NUMBERED_QA_RE = re.compile(
    r"(?ms)^\s*(?P<label>\d+\.\d+\.?)\s+(?P<question>.+?)\n+\s*ODGOVOR:\s*(?P<answer>.+?)(?=^\s*\d+\.\d+\.?\s+|\Z)"
)
VPRAŠANJE_QA_RE = re.compile(
    r"(?ms)^\s*Vprašanje\s+(?P<label>\d+):\s*(?P<question>.+?)\n+\s*(?P<answer>.+?)(?=^\s*Vprašanje\s+\d+:|\Z)"
)
DATE_TRAIL_RE = re.compile(r"\s*\((?:\d{1,2}\.\s*\d{1,2}\.\s*\d{4}[^()]*)\)\s*$")
MULTISPACE_RE = re.compile(r"\s+")


def build_furs_real_eval_dataset(
    output_path: Path,
    download_dir: Path,
    limit: int = 60,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in FURS_REAL_EVAL_SEEDS:
        local_path = download_eval_seed(seed, download_dir)
        text = extract_text_from_office_document(local_path)
        pairs = extract_furs_qa_pairs(text, seed["title"], seed["url"], seed["category"])
        rows.extend(pairs)
    rows = dedupe_eval_rows(rows)
    rows = rows[:limit]
    write_jsonl(output_path, rows)
    return rows


def evaluate_real_answers(
    eval_rows: list[dict[str, Any]],
    chunk_map: dict[str, dict[str, Any]],
    chroma_dir: Path,
    bm25_path: Path,
    lemmatize_fn,
    classla_python: str | None = None,
    query_lemmas: dict[str, list[str]] | None = None,
    top_k: int = 5,
    generator_model: str | None = None,
    max_new_tokens: int = DEFAULT_LOCAL_GENERATOR_MAX_NEW_TOKENS,
    embedding_profile: str = "bge_m3",
    embedding_model_name: str | None = None,
    reranker_model: str | None = None,
) -> dict[str, Any]:
    per_query: list[dict[str, Any]] = []
    citation_source_matches = 0
    citation_title_matches = 0
    law_signal_matches = 0
    article_signal_matches = 0
    heuristic_useful = 0

    for row in eval_rows:
        query_tokens = (query_lemmas or {}).get(row["query_id"])
        if query_tokens is None:
            query_tokens = lemmatize_fn(row["query"], classla_python)
        payload = answer_question(
            row["query"],
            query_tokens,
            chunk_map,
            chroma_dir,
            bm25_path,
            top_k=top_k,
            generator_model=generator_model,
            max_new_tokens=max_new_tokens,
            embedding_profile=embedding_profile,
            embedding_model_name=embedding_model_name,
            reranker_model=reranker_model,
        )
        metrics = score_real_eval_row(row, payload)
        citation_source_matches += int(metrics["top_citation_source_type_match"])
        citation_title_matches += int(metrics["top_citation_title_match"])
        law_signal_matches += int(metrics["law_signal_match"])
        article_signal_matches += int(metrics["article_signal_match"])
        heuristic_useful += int(metrics["heuristic_useful"])
        per_query.append(
            {
                "query_id": row["query_id"],
                "query": row["query"],
                "top_citation_source_type_match": metrics["top_citation_source_type_match"],
                "top_citation_title_match": metrics["top_citation_title_match"],
                "law_signal_match": metrics["law_signal_match"],
                "article_signal_match": metrics["article_signal_match"],
                "reference_keyword_recall": metrics["reference_keyword_recall"],
                "heuristic_useful": metrics["heuristic_useful"],
                "top_citation": payload["citations"][0] if payload.get("citations") else None,
                "answer_preview": payload["answer"][:400],
            }
        )

    total = len(eval_rows) or 1
    return {
        "query_count": len(eval_rows),
        "top_citation_source_type_accuracy": citation_source_matches / total,
        "top_citation_title_accuracy": citation_title_matches / total,
        "law_signal_accuracy": law_signal_matches / total,
        "article_signal_accuracy": article_signal_matches / total,
        "heuristic_usefulness": heuristic_useful / total,
        "per_query": per_query,
    }


def score_real_eval_row(row: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    top_citation = (payload.get("citations") or [{}])[0]
    expected_source_type = row.get("expected_source_type")
    expected_source_title = normalize_text(row.get("expected_source_title", "")).lower()
    answer_text = normalize_text(payload.get("answer", "")).lower()
    citation_source_type_match = bool(expected_source_type and top_citation.get("source_type") == expected_source_type)
    citation_title = normalize_text(top_citation.get("title", "")).lower()
    citation_title_match = bool(expected_source_title and citation_title and expected_source_title in citation_title)

    expected_law_refs = {normalize_text(item).lower() for item in row.get("expected_law_refs", [])}
    expected_articles = {normalize_text(item).lower() for item in row.get("expected_articles", [])}
    citation_blob = " ".join(
        normalize_text(value)
        for citation in payload.get("citations") or []
        for value in (
            citation.get("law_ref", ""),
            citation.get("title", ""),
            citation.get("article_number", ""),
            citation.get("article_title", ""),
        )
    ).lower()
    law_signal_match = not expected_law_refs or any(ref in answer_text or ref in citation_blob for ref in expected_law_refs)
    article_signal_match = not expected_articles or any(
        article in answer_text or article in citation_blob for article in expected_articles
    )

    reference_keyword_recall = keyword_recall(row.get("reference_answer", ""), payload.get("answer", ""))
    grounding_match = citation_source_type_match or citation_title_match or law_signal_match or article_signal_match
    heuristic_useful = grounding_match and reference_keyword_recall >= 0.25
    return {
        "top_citation_source_type_match": citation_source_type_match,
        "top_citation_title_match": citation_title_match,
        "law_signal_match": law_signal_match,
        "article_signal_match": article_signal_match,
        "reference_keyword_recall": round(reference_keyword_recall, 4),
        "heuristic_useful": heuristic_useful,
    }


def download_eval_seed(seed: dict[str, Any], download_dir: Path) -> Path:
    suffix = Path(seed["url"]).suffix.lower() or ".bin"
    target = download_dir / f"{stable_id_fragment(seed['url'])}_{slug_filename(seed['title'])}{suffix}"
    if target.exists():
        return target
    ensure_parent_dir(target)
    response = requests.get(seed["url"], timeout=60)
    response.raise_for_status()
    target.write_bytes(response.content)
    return target


def extract_text_from_office_document(path: Path) -> str:
    result = subprocess.run(
        ["textutil", "-convert", "txt", "-stdout", str(path)],
        check=True,
        capture_output=True,
    )
    return result.stdout.decode("utf-8", errors="ignore")


def extract_furs_qa_pairs(
    text: str,
    source_title: str,
    source_url: str,
    category: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pattern in (NUMBERED_QA_RE, VPRAŠANJE_QA_RE):
        for match in pattern.finditer(text):
            question = clean_question_text(match.group("question"))
            answer = clean_answer_text(match.group("answer"))
            if not question or not answer:
                continue
            if len(question) < 12 or len(answer) < 20:
                continue
            citations = extract_query_citations(answer)
            row = {
                "query_id": f"real_{stable_id_fragment(source_url + question, length=10)}",
                "query": question,
                "category": category,
                "expected_source_type": "furs_guidance",
                "expected_source_title": source_title,
                "expected_source_url": source_url,
                "expected_law_refs": citations["law_refs"],
                "expected_articles": citations["articles"],
                "reference_answer": answer,
            }
            rows.append(row)
    return rows


def clean_question_text(text: str) -> str:
    normalized = normalize_text(text)
    normalized = DATE_TRAIL_RE.sub("", normalized).strip()
    normalized = MULTISPACE_RE.sub(" ", normalized)
    return normalized.rstrip(".") + "?" if not normalized.endswith("?") else normalized


def clean_answer_text(text: str) -> str:
    normalized = normalize_text(text)
    normalized = re.sub(r'HYPERLINK\s+"[^"]+"', "", normalized)
    normalized = normalized.replace("ODGOVOR:", "")
    normalized = MULTISPACE_RE.sub(" ", normalized)
    return normalized.strip()


def keyword_recall(reference_answer: str, model_answer: str) -> float:
    reference_terms = meaningful_terms(reference_answer)
    if not reference_terms:
        return 0.0
    answer_terms = meaningful_terms(model_answer)
    overlap = reference_terms.intersection(answer_terms)
    return len(overlap) / len(reference_terms)


def meaningful_terms(text: str) -> set[str]:
    terms = set()
    for token in TERM_RE.findall(normalize_text(text).lower()):
        if token in SLOVENE_STOPWORDS or len(token) < 4:
            continue
        terms.add(token)
    return terms


def dedupe_eval_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        query = normalize_text(row["query"]).lower()
        if query in seen:
            continue
        ordered.append(row)
        seen.add(query)
    return ordered


def slug_filename(value: str) -> str:
    normalized = normalize_text(value).lower()
    normalized = re.sub(r"[^0-9a-zčšž]+", "_", normalized, flags=re.IGNORECASE).strip("_")
    return normalized[:80] or "furs_eval"


def load_real_eval_rows(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path)
