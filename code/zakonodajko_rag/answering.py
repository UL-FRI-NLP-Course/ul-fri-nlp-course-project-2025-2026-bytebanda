from __future__ import annotations

import json
import re
import warnings
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from .constants import DEFAULT_EMBEDDING_PROFILE, DEFAULT_LOCAL_GENERATOR_MAX_NEW_TOKENS
from .regexes import extract_regex_features
from .resolver import resolve_referral_results
from .retrieval import citation_boost, hybrid_search
from .router import TECHNICAL_PROCESS_HINTS, TERM_RE, QueryProfile, QueryRoute, build_query_profile, is_useful_query_term, route_query
from .text_utils import normalize_text


SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?;])\s+|\n+")
TITLE_LAW_REF_RE = re.compile(r"\(([^()]+)\)\s*$")
DEADLINE_PHRASE_HINTS = ("najpozneje", "najkasneje", "na dan", "v roku", "v petih dneh", "v 15 dneh", "rok")
EURO_AMOUNT_RE = re.compile(r"\d{1,3}(?:\.\d{3})*,\d{2}")
PERCENT_VALUE_RE = re.compile(r"\d{1,2}\s?%")
REFERRAL_SENTENCE_RE = re.compile(
    r"\b(?:velja|uporablja|smiselno uporablja|v skladu z)\s+\d+\.(?:[a-zčšž])?\s*člen\b",
    re.IGNORECASE,
)
DISTINCTIVE_TECHNICAL_TERMS = TECHNICAL_PROCESS_HINTS.intersection(
    {"api", "certifikat", "client", "edavki", "erp", "excel", "identifikator", "klient", "oauth", "testni", "testnem", "xml"}
)
QUERY_UNDERSTANDING_INTENTS = {
    "explicit_article",
    "deadline",
    "definition",
    "amount_percentage",
    "calculation",
    "practical_guidance",
    "comparison",
    "follow_up",
    "general",
}


def suppress_generation_warnings() -> None:
    warnings.filterwarnings(
        "ignore",
        message=r"`do_sample` is set to `False`.*top_k.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"To copy construct from a tensor.*",
        category=UserWarning,
    )


@dataclass(frozen=True)
class QueryUnderstanding:
    standalone_query: str
    intent: str
    use_context: bool
    confidence: float
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def answer_question(
    query: str,
    query_tokens: list[str],
    chunk_map: dict[str, dict[str, Any]],
    chroma_dir: Path,
    bm25_path: Path,
    top_k: int = 5,
    generator_model: str | None = None,
    max_new_tokens: int = DEFAULT_LOCAL_GENERATOR_MAX_NEW_TOKENS,
    embedding_profile: str = DEFAULT_EMBEDDING_PROFILE,
    embedding_model_name: str | None = None,
    reranker_model: str | None = None,
    agent_plan: Any | None = None,
) -> dict[str, Any]:
    query_profile = build_query_profile(query, query_tokens)
    route = apply_agent_plan_to_route(route_query(query_profile), agent_plan)
    results = hybrid_search(
        query,
        query_tokens,
        chunk_map,
        chroma_dir,
        bm25_path,
        top_k,
        embedding_profile=embedding_profile,
        embedding_model_name=embedding_model_name,
        source_policy=route.source_policy,
        reranker_model=reranker_model,
    )
    resolved_results, resolution_chain = resolve_referral_results(results, chunk_map, route)
    payload = compose_answer_from_results(
        query,
        query_tokens,
        resolved_results,
        max_sentences=min(3, top_k),
        route=route,
        resolution_chain=resolution_chain,
    )
    if generator_model and not payload["insufficient_evidence"] and should_use_local_generation(query_profile, payload, route):
        generation_results = select_generation_results(resolved_results, payload["citations"])
        generated = generate_with_local_transformer(
            query,
            generation_results,
            generator_model=generator_model,
            max_new_tokens=max_new_tokens,
            draft_answer=payload["answer"],
        )
        if generated and generated_answer_is_usable(generated, query_profile, generation_results, payload["answer"]):
            payload["answer"] = generated
            payload["backend"] = "local_transformer"
            payload["citations"] = collect_citations([item["chunk"] for item in generation_results]) or payload["citations"]
    payload["route"] = route.__dict__
    payload["resolution_chain"] = resolution_chain
    payload["agent_plan"] = agent_plan_to_dict(agent_plan)
    payload["citation_verification"] = verify_primary_citation(payload, query_profile, route, agent_plan)
    return payload


def apply_agent_plan_to_route(route: QueryRoute, agent_plan: Any | None) -> QueryRoute:
    if agent_plan is None:
        return route
    source_policy = normalize_plan_value(agent_plan, "source_policy", route.source_policy)
    answer_style = normalize_plan_value(agent_plan, "answer_style", route.answer_style)
    follow_referrals = bool(normalize_plan_value(agent_plan, "follow_referrals", route.follow_referrals))
    preserve_origin_article = bool(
        normalize_plan_value(agent_plan, "preserve_origin_article", route.preserve_origin_article)
    )
    actions = set(plan_actions(agent_plan))
    if actions and "resolve_referrals" not in actions:
        follow_referrals = False
    if actions and "compose_article_overview" in actions:
        answer_style = "article_overview"
    elif actions and "compose_guided_explanation" in actions:
        answer_style = "guided_explanation"
    elif actions and "compose_structured_answer" in actions:
        answer_style = "structured_rule"
    return QueryRoute(
        intent=normalize_plan_value(agent_plan, "intent", route.intent),
        source_policy=source_policy,
        answer_style=answer_style,
        allow_generation=route.allow_generation,
        follow_referrals=follow_referrals,
        preserve_origin_article=preserve_origin_article,
    )


def normalize_plan_value(agent_plan: Any, field: str, default: Any) -> Any:
    if agent_plan is None:
        return default
    if isinstance(agent_plan, dict):
        return agent_plan.get(field, default)
    return getattr(agent_plan, field, default)


def plan_actions(agent_plan: Any | None) -> list[str]:
    raw_actions = normalize_plan_value(agent_plan, "actions", [])
    if isinstance(raw_actions, tuple):
        return [str(action) for action in raw_actions]
    if isinstance(raw_actions, list):
        return [str(action) for action in raw_actions]
    return []


def agent_plan_to_dict(agent_plan: Any | None) -> dict[str, Any] | None:
    if agent_plan is None:
        return None
    if hasattr(agent_plan, "as_dict"):
        return agent_plan.as_dict()
    if isinstance(agent_plan, dict):
        return agent_plan
    return None


def verify_primary_citation(
    payload: dict[str, Any],
    query_profile: QueryProfile,
    route: QueryRoute,
    agent_plan: Any | None,
) -> dict[str, Any]:
    top_citation = (payload.get("citations") or [None])[0]
    if top_citation is None:
        return {"status": "missing", "score": 0.0, "checks": {"has_citation": False}}
    checks: dict[str, bool] = {"has_citation": True}
    if query_profile.citations["articles"]:
        checks["article_match"] = str(top_citation.get("article_number", "")).lower() in {
            article.lower() for article in query_profile.citations["articles"]
        }
    if query_profile.citations["law_refs"]:
        citation_aliases = {
            str(top_citation.get("law_ref", "")).lower(),
            str(top_citation.get("law_id", "")).lower(),
            str(top_citation.get("title", "")).lower(),
        }
        checks["law_match"] = bool(citation_aliases.intersection({item.lower() for item in query_profile.citations["law_refs"]}))
    if route.source_policy == "furs_preferred":
        checks["source_match"] = top_citation.get("source_type") == "furs_guidance"
    elif route.source_policy == "pisrs_only":
        checks["source_match"] = top_citation.get("source_type", "pisrs") == "pisrs"
    elif "prefer_furs_guidance" in plan_actions(agent_plan):
        checks["source_match"] = top_citation.get("source_type") == "furs_guidance"
    elif "prefer_pisrs" in plan_actions(agent_plan):
        checks["source_match"] = top_citation.get("source_type", "pisrs") == "pisrs"
    score = sum(1 for passed in checks.values() if passed) / max(1, len(checks))
    status = "verified" if score >= 0.75 else "weak"
    return {"status": status, "score": round(score, 4), "checks": checks}


def understand_chat_query_with_local_model(
    message: str,
    history: list[dict[str, Any]],
    generator_model: str | None,
    max_new_tokens: int = 160,
) -> QueryUnderstanding | None:
    normalized_message = normalize_text(message)
    if not normalized_message or not generator_model or not history:
        return None
    tokenizer, model, device = load_generator_model(generator_model)
    messages = [
        {
            "role": "system",
            "content": (
                "Si query-router za slovenskega davčnega pomočnika. "
                "Tvoja naloga je iz zgodovine pogovora in trenutnega vprašanja sestaviti samostojno poizvedbo za retrieval. "
                "Vrni izključno veljaven JSON brez markdowna ali dodatnega besedila."
            ),
        },
        {"role": "user", "content": build_query_understanding_prompt(normalized_message, history)},
    ]
    prompt = render_generation_prompt(tokenizer, messages)
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=3072)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    input_length = encoded["input_ids"].shape[-1]
    with warnings.catch_warnings():
        suppress_generation_warnings()
        generated = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.pad_token_id,
        )
    output = tokenizer.decode(generated[0][input_length:], skip_special_tokens=True).strip()
    return parse_query_understanding_output(output, normalized_message)


def compose_answer_from_results(
    query: str,
    query_tokens: list[str],
    results: list[dict[str, Any]],
    max_sentences: int = 3,
    route: QueryRoute | None = None,
    resolution_chain: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    query_profile = build_query_profile(query, query_tokens)
    active_route = route or route_query(query_profile)
    prebuilt_answer: str | None = None
    answer_sections: list[dict[str, str]] = []
    explicit_article_chunk = first_explicit_article_match(results, query_profile.citations)
    structured_match = first_structured_answer_match(results, query_profile, active_route)
    prefer_structured_over_article = structured_match is not None and (
        query_profile.expects_definition
        or query_profile.expects_deadline
        or query_profile.expects_amount
        or query_profile.expects_percentage
        or query_profile.asks_extreme_rate
        or query_profile.asks_bracket_threshold
    )
    if explicit_article_chunk is not None and not prefer_structured_over_article:
        evidence = build_article_evidence(explicit_article_chunk, max_sentences=max_sentences)
        citations = collect_citations([explicit_article_chunk])
        insufficient = False
        prebuilt_answer = f"{format_short_citation(explicit_article_chunk)}: {build_article_overview(explicit_article_chunk)}"
    elif structured_match is not None:
        evidence = [
            {
                "chunk": structured_match["chunk"],
                "sentence": structured_match["evidence"],
                "support_score": structured_match["support_score"],
                "result_rank": 1,
            }
        ]
        citations = collect_citations([structured_match["chunk"]])
        insufficient = False
        prebuilt_answer = structured_match["answer"]
    else:
        evidence = select_supporting_sentences(results, query_profile, max_sentences=max_sentences, route=active_route)
        citations = collect_citations([item["chunk"] for item in results[:max(3, max_sentences)]])
        insufficient = is_insufficient_evidence(query_profile, results, evidence)
    citations = prioritize_citations_for_route(citations, active_route, evidence)
    if insufficient:
        answer = build_insufficient_evidence_answer(citations)
        answer_sections = [{"label": "Omejena podlaga", "text": answer}]
    elif prebuilt_answer is not None:
        answer_sections = build_answer_sections(active_route, citations, evidence, query_profile)
        answer = render_answer_sections(answer_sections) if answer_sections else prebuilt_answer
    else:
        answer = compose_extractive_answer(query_profile, evidence)
        citations = collect_citations([evidence[0]["chunk"], *[item["chunk"] for item in results[:max(3, max_sentences)]]]) or citations
        citations = prioritize_citations_for_route(citations, active_route, evidence)
        answer_sections = build_answer_sections(active_route, citations, evidence, query_profile)
        if answer_sections:
            answer = render_answer_sections(answer_sections)

    if resolution_chain:
        answer_sections = append_resolution_section(answer_sections, resolution_chain)
        answer = render_answer_sections(answer_sections)

    return {
        "query": query,
        "backend": "extractive",
        "insufficient_evidence": insufficient,
        "answer": answer,
        "answer_sections": answer_sections,
        "citations": citations,
        "supporting_sentences": [
            {
                "text": item["sentence"],
                "support_score": round(item["support_score"], 3),
                "citation": format_short_citation(item["chunk"]),
                "chunk_id": item["chunk"]["chunk_id"],
            }
            for item in evidence
        ],
        "used_chunks": [
            {
                "rank": rank,
                "score": round(float(item.get("score", 0.0)), 6),
                "law_id": item["chunk"]["law_id"],
                "law_ref": law_display_name(item["chunk"]),
                "title": item["chunk"]["title"],
                "section_path": item["chunk"]["section_path"],
                "article_number": item["chunk"]["article_number"],
                "article_title": item["chunk"]["article_title"],
                "source_url": item["chunk"]["source_url"],
                "chunk_id": item["chunk"]["chunk_id"],
                "text_preview": item["chunk"]["raw_chunk_text"][:500],
            }
            for rank, item in enumerate(results[:max(3, max_sentences)], start=1)
        ],
        "route": active_route.__dict__,
        "resolution_chain": resolution_chain or [],
    }
def select_supporting_sentences(
    results: list[dict[str, Any]],
    query_profile: QueryProfile,
    max_sentences: int = 3,
    route: QueryRoute | None = None,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for result_rank, item in enumerate(results[: max(5, max_sentences)], start=1):
        chunk = item["chunk"]
        best_candidate: dict[str, Any] | None = None
        for sentence in extract_chunk_sentences(chunk):
            support_score = score_sentence(sentence, chunk, query_profile, result_rank, route=route)
            candidate = {
                "chunk": chunk,
                "sentence": sentence,
                "support_score": support_score,
                "result_rank": result_rank,
            }
            if best_candidate is None or candidate["support_score"] > best_candidate["support_score"]:
                best_candidate = candidate
        if best_candidate is not None:
            candidates.append(best_candidate)
    candidates.sort(key=lambda item: (item["support_score"], -item["result_rank"]), reverse=True)
    return candidates[:max_sentences]


def extract_chunk_sentences(chunk: dict[str, Any]) -> list[str]:
    body_text = chunk_body_text(chunk)
    spans = chunk.get("sentence_spans") or []
    extracted: list[str] = []
    for span in spans:
        start = max(0, min(len(body_text), int(span.get("start", 0))))
        end = max(start, min(len(body_text), int(span.get("end", len(body_text)))))
        sentence = normalize_text(body_text[start:end]) or normalize_text(span.get("text", ""))
        if sentence:
            extracted.append(sentence)
    if extracted:
        return dedupe_preserve_order(extracted)
    return dedupe_preserve_order(
        [normalize_text(part) for part in SENTENCE_SPLIT_RE.split(body_text) if normalize_text(part)]
    )


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


def chunk_mentions_terms(chunk: dict[str, Any], terms: set[str]) -> bool:
    if not terms:
        return False
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
    return any(term in haystack for term in terms)


def score_sentence(
    sentence: str,
    chunk: dict[str, Any],
    query_profile: QueryProfile,
    result_rank: int,
    route: QueryRoute | None = None,
) -> float:
    sentence_terms = {token.lower() for token in TERM_RE.findall(normalize_text(sentence))}
    overlap = query_profile.keywords.intersection(sentence_terms)
    header_overlap = query_profile.keywords.intersection(chunk_header_terms(chunk))
    regex_features = extract_regex_features(sentence)
    score = float(len(overlap))
    score += min(3.0, 1.5 * len(header_overlap))
    score += min(citation_boost(chunk, query_profile.citations), 4.0)
    score += max(0.0, 1.0 - 0.12 * (result_rank - 1))
    if query_profile.expects_deadline and (regex_features["deadlines"] or regex_features["dates"]):
        score += 2.0
    if query_profile.expects_amount and regex_features["amounts"]:
        score += 2.0
    if query_profile.expects_percentage and regex_features["percentages"]:
        score += 2.0
    if sentence.startswith("[OPOMBA]"):
        score -= 1.0
    if len(sentence) < 30:
        score -= 0.5
    if route and route.source_policy in {"furs_allowed", "furs_preferred"}:
        if chunk.get("source_type") == "furs_guidance":
            score += 1.5
        elif route.intent == "practical_guidance" and query_profile.has_technical_markers:
            score -= 0.35
    return score


def is_insufficient_evidence(
    query_profile: QueryProfile,
    results: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
) -> bool:
    if not results or not evidence:
        return True
    technical_terms = {term for term in query_profile.keywords if term in DISTINCTIVE_TECHNICAL_TERMS}
    if technical_terms and not any(chunk_mentions_terms(item["chunk"], technical_terms) for item in results[:5]):
        return True
    if query_profile.citations["articles"]:
        return not any(chunk_matches_query_citations(item["chunk"], query_profile.citations) for item in results[:3])
    best_score = evidence[0]["support_score"]
    minimum_support = 1.5 if len(query_profile.keywords) <= 3 else 2.0
    if query_profile.expects_deadline or query_profile.expects_amount or query_profile.expects_percentage:
        minimum_support -= 0.25
    return best_score < minimum_support


def chunk_matches_query_citations(chunk: dict[str, Any], citations: dict[str, list[str]]) -> bool:
    chunk_law_aliases = direct_chunk_law_aliases(chunk)
    query_law_refs = {ref.lower() for ref in citations["law_refs"]}
    query_articles = {ref.lower() for ref in citations["articles"]}
    law_match = not query_law_refs or bool(query_law_refs.intersection(chunk_law_aliases))
    article_match = not query_articles or (chunk.get("article_number") or "").lower() in query_articles
    return law_match and article_match


def compose_extractive_answer(
    query_profile: QueryProfile,
    evidence: list[dict[str, Any]],
) -> str:
    lead = evidence[0]
    lead_citation = format_short_citation(lead["chunk"])
    if query_profile.citations["articles"] and lead["chunk"].get("article_number"):
        return f"{lead_citation} določa: {lead['sentence']}"
    else:
        return f"Po {lead_citation}: {lead['sentence']}"


def build_insufficient_evidence_answer(citations: list[dict[str, Any]]) -> str:
    if not citations:
        return "V zbranih odlomkih nisem našel dovolj neposredne pravne podlage za zanesljiv odgovor."
    labels = ", ".join(format_short_citation(citation) for citation in citations[:2])
    return (
        "V zbranih odlomkih nisem našel dovolj neposredne pravne podlage za zanesljiv odgovor. "
        f"Najbližji relevantni vir{i_suffix(len(citations[:2]))}: {labels}."
    )


def i_suffix(count: int) -> str:
    return "i" if count != 1 else ""


def build_answer_sections(
    route: QueryRoute,
    citations: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
    query_profile: QueryProfile,
) -> list[dict[str, str]]:
    if not evidence and not citations:
        return []
    sections: list[dict[str, str]] = []
    short_answer = derive_short_answer(route, evidence, query_profile)
    if short_answer:
        sections.append({"label": "Kratek odgovor", "text": short_answer})

    legal_citations = [citation for citation in citations if citation.get("source_type", "pisrs") == "pisrs"]
    furs_citations = [citation for citation in citations if citation.get("source_type") == "furs_guidance"]
    if route.source_policy in {"furs_allowed", "furs_preferred"} and furs_citations:
        sections.append({"label": "Pojasnilo FURS", "text": format_short_citation(furs_citations[0])})
    if legal_citations:
        legal_basis = format_short_citation(legal_citations[0])
        sections.append({"label": "Pravna podlaga", "text": legal_basis})

    return sections


def derive_short_answer(
    route: QueryRoute,
    evidence: list[dict[str, Any]],
    query_profile: QueryProfile,
) -> str:
    if not evidence:
        return ""
    if route.intent == "explicit_article" and not (
        query_profile.expects_definition
        or query_profile.expects_deadline
        or query_profile.expects_amount
        or query_profile.expects_percentage
        or query_profile.asks_extreme_rate
        or query_profile.asks_bracket_threshold
    ):
        return build_article_overview(evidence[0]["chunk"])
    lead = evidence[0]["sentence"]
    if query_profile.expects_percentage and not lead.endswith("."):
        return lead + "."
    return lead


def append_resolution_section(
    sections: list[dict[str, str]],
    resolution_chain: list[dict[str, Any]],
) -> list[dict[str, str]]:
    if not resolution_chain:
        return sections
    chain_labels = [f"{item['from_citation']} -> {item['to_citation']}" for item in resolution_chain[:2]]
    text = "; ".join(chain_labels)
    if sections and sections[-1]["label"] == "Uporabljena napotitev":
        return sections
    return [*sections, {"label": "Uporabljena napotitev", "text": text}]


def render_answer_sections(sections: list[dict[str, str]]) -> str:
    return "\n\n".join(f"{section['label']}: {section['text']}" for section in sections if section.get("text"))


def prioritize_citations_for_route(
    citations: list[dict[str, Any]],
    route: QueryRoute,
    evidence: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not citations or route.source_policy not in {"furs_allowed", "furs_preferred"}:
        return citations
    furs_evidence_ids = {item["chunk"]["chunk_id"] for item in evidence if item["chunk"].get("source_type") == "furs_guidance"}
    if not furs_evidence_ids and route.source_policy != "furs_preferred":
        return citations

    def sort_key(citation: dict[str, Any]) -> tuple[int, int]:
        source_rank = 0 if citation.get("source_type") == "furs_guidance" else 1
        evidence_rank = 0 if citation.get("chunk_id") in furs_evidence_ids else 1
        return (source_rank, evidence_rank)

    return sorted(citations, key=sort_key)


def collect_citations(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered: list[dict[str, Any]] = []
    seen: set[tuple[str, str | None]] = set()
    for chunk in chunks:
        key = (chunk["law_id"], chunk.get("article_number"))
        if key in seen:
            continue
        citation = {
            "law_id": chunk["law_id"],
            "law_ref": law_display_name(chunk),
            "title": chunk["title"],
            "section_path": chunk["section_path"],
            "article_number": chunk["article_number"],
            "article_title": chunk["article_title"],
            "source_url": chunk["source_url"],
            "chunk_id": chunk["chunk_id"],
            "source_type": chunk.get("source_type", "pisrs"),
        }
        ordered.append(citation)
        seen.add(key)
    return ordered


def first_structured_answer_match(
    results: list[dict[str, Any]],
    query_profile: QueryProfile,
    route: QueryRoute | None = None,
) -> dict[str, Any] | None:
    if not results:
        return None
    if route and route.intent in {"practical_guidance", "follow_up"} and any(
        item["chunk"].get("source_type") == "furs_guidance" for item in results[:5]
    ):
        return None
    if query_profile.expects_definition:
        best_definition_match: dict[str, Any] | None = None
        for result_rank, item in enumerate(results[:5], start=1):
            chunk = item["chunk"]
            score = score_definition_chunk(chunk, query_profile, result_rank)
            if score <= 0:
                continue
            excerpt = extract_definition_excerpt(chunk)
            citation = format_short_citation(chunk)
            candidate = {
                "chunk": chunk,
                "answer": f"Po {citation}: {excerpt}",
                "evidence": excerpt,
                "support_score": score,
            }
            if best_definition_match is None or candidate["support_score"] > best_definition_match["support_score"]:
                best_definition_match = candidate
        if best_definition_match is not None:
            return best_definition_match

    for result_rank, item in enumerate(results[:3], start=1):
        chunk = item["chunk"]
        header_overlap = query_profile.keywords.intersection(chunk_header_terms(chunk))
        if query_profile.asks_bracket_threshold or query_profile.asks_extreme_rate:
            bracket_answer = extract_rate_bracket_answer(chunk, query_profile, result_rank)
            if bracket_answer is not None:
                return bracket_answer
        if query_profile.expects_deadline:
            deadline_sentence = extract_deadline_excerpt(chunk)
            if deadline_sentence and (header_overlap or result_rank <= 2):
                citation = format_short_citation(chunk)
                return {
                    "chunk": chunk,
                    "answer": f"Po {citation}: {deadline_sentence}",
                    "evidence": deadline_sentence,
                    "support_score": 9.35 - 0.15 * (result_rank - 1),
                }
        if query_profile.expects_percentage:
            percentages = normalize_percentages(chunk.get("percentages") or [])
            if percentages and (header_overlap or result_rank == 1):
                citation = format_short_citation(chunk)
                joined = join_list_natural(percentages)
                return {
                    "chunk": chunk,
                    "answer": f"Po {citation} so stopnje dohodnine: {joined}.",
                    "evidence": f"Stopnje dohodnine za davčno leto so: {joined}.",
                    "support_score": 9.5 - 0.15 * (result_rank - 1),
                }
    return None


def score_definition_chunk(
    chunk: dict[str, Any],
    query_profile: QueryProfile,
    result_rank: int,
) -> float:
    focus_phrase = extract_definition_focus_phrase(query_profile.query)
    header_text = normalize_text(
        " ".join(
            part
            for part in (
                chunk.get("title"),
                chunk.get("section_path"),
                chunk.get("article_number"),
                chunk.get("article_title"),
            )
            if part
        )
    ).lower()
    body_text = chunk_body_text(chunk).lower()
    header_overlap = query_profile.keywords.intersection(chunk_header_terms(chunk))
    score = 7.2 - 0.25 * (result_rank - 1)
    score += min(2.0, 1.0 * len(header_overlap))
    if focus_phrase:
        if focus_phrase in header_text:
            score += 4.5
        elif focus_phrase in body_text:
            score += 1.0
    article_title = clean_article_title(chunk.get("article_title") or "").lower()
    if article_title == "pomen izrazov":
        score -= 3.0
    if "pomen izrazov" in header_text:
        score -= 1.5
    return score


def extract_definition_focus_phrase(query: str) -> str:
    normalized = normalize_text(query).lower().rstrip("?.!:; ")
    for prefix in ("kaj pomeni", "kaj je", "kaj so"):
        if normalized.startswith(prefix):
            remainder = normalized[len(prefix) :].strip(" ,:-")
            return remainder
    return normalized


def first_explicit_article_match(
    results: list[dict[str, Any]],
    citations: dict[str, list[str]],
) -> dict[str, Any] | None:
    if not citations["articles"]:
        return None
    for item in results[:5]:
        if chunk_matches_query_citations(item["chunk"], citations):
            return item["chunk"]
    return None


def direct_chunk_law_aliases(chunk: dict[str, Any]) -> set[str]:
    aliases = {
        (chunk.get("law_id") or "").lower(),
        (chunk.get("title") or "").lower(),
        law_display_name(chunk).lower(),
    }
    return {alias for alias in aliases if alias}


def chunk_header_terms(chunk: dict[str, Any]) -> set[str]:
    text = " ".join(
        part
        for part in (
            chunk.get("title"),
            chunk.get("section_path"),
            chunk.get("article_number"),
            chunk.get("article_title"),
        )
        if part
    )
    return {token.lower() for token in TERM_RE.findall(normalize_text(text))}


def law_display_name(chunk: dict[str, Any]) -> str:
    if chunk.get("source_type") == "furs_guidance":
        return "FURS"
    legal_refs = chunk.get("legal_refs") or {}
    if legal_refs.get("law_refs"):
        return legal_refs["law_refs"][0]
    title = chunk.get("title") or ""
    match = TITLE_LAW_REF_RE.search(title)
    if match:
        return match.group(1)
    return chunk.get("law_id") or title


def format_short_citation(chunk_or_citation: dict[str, Any]) -> str:
    if chunk_or_citation.get("source_type") == "furs_guidance":
        title = normalize_text(chunk_or_citation.get("title") or "")
        section_path = normalize_text(chunk_or_citation.get("section_path") or "")
        if section_path and section_path not in {"Celotno pojasnilo", "Uvod"}:
            return f"FURS, {title} ({section_path})"
        return f"FURS, {title}"
    law_ref = chunk_or_citation.get("law_ref") or law_display_name(chunk_or_citation)
    article_number = chunk_or_citation.get("article_number")
    article_title = normalize_text(chunk_or_citation.get("article_title") or "")
    if article_number and article_title:
        return f"{law_ref}, {article_number} {article_title}"
    if article_number:
        return f"{law_ref}, {article_number}"
    section_path = normalize_text(chunk_or_citation.get("section_path") or "")
    if section_path:
        return f"{law_ref}, {section_path}"
    return law_ref


def extract_article_excerpt(chunk: dict[str, Any], max_chars: int = 420) -> str:
    body = chunk_body_text(chunk)
    preferred_cut_patterns = [
        re.compile(r"\s2\.\s"),
        re.compile(r"\s3\.\s"),
        re.compile(r"\s[a-zčšž]\)\s", re.IGNORECASE),
    ]
    for pattern in preferred_cut_patterns:
        match = pattern.search(body)
        if match and match.start() <= max_chars:
            return normalize_text(body[: match.start()].rstrip(" ;,:")) + "."
    if len(body) <= max_chars:
        return body
    truncated = body[:max_chars]
    last_boundary = max(truncated.rfind(". "), truncated.rfind("; "), truncated.rfind(", "))
    if last_boundary > 80:
        truncated = truncated[:last_boundary]
    return normalize_text(truncated.rstrip(" ;,:")) + " …"


def build_article_overview(chunk: dict[str, Any], max_items: int = 3) -> str:
    body = chunk_body_text(chunk)
    topic = clean_article_title(chunk.get("article_title") or "")
    lead = extract_article_intro(body)
    numbered_items = extract_numbered_items(body)

    parts: list[str] = []
    if topic:
        parts.append(f"Tema člena: {topic}.")
    if lead:
        parts.append(f"Ključno pravilo: {lead}.")
    if numbered_items:
        highlights = [summarize_numbered_item(item) for item in numbered_items[:max_items]]
        highlights = [item for item in highlights if item]
        if highlights:
            parts.append(f"Med glavnimi primeri so: {'; '.join(highlights)}.")
        remaining = len(numbered_items) - len(highlights)
        if remaining > 0:
            parts.append(f"Člen našteva še {remaining} dodatn{slovene_plural_suffix(remaining, 'i primer', 'a primera', 'e primere', 'ih primerov')}.")
    if not parts:
        return extract_article_excerpt(chunk)
    return " ".join(parts)


def build_article_evidence(chunk: dict[str, Any], max_sentences: int = 3) -> list[dict[str, Any]]:
    body = chunk_body_text(chunk)
    evidence_sentences: list[str] = []
    lead = extract_article_intro(body)
    if lead:
        evidence_sentences.append(lead if lead.endswith(".") else f"{lead}.")
    for item in extract_numbered_items(body):
        summary = summarize_numbered_item(item, max_words=24)
        if summary:
            evidence_sentences.append(summary[0].upper() + summary[1:] if len(summary) > 1 else summary.upper())
        if len(evidence_sentences) >= max_sentences:
            break
    if not evidence_sentences:
        evidence_sentences.append(extract_article_excerpt(chunk))
    return [
        {
            "chunk": chunk,
            "sentence": sentence,
            "support_score": 10.0 - 0.4 * index,
            "result_rank": 1,
        }
        for index, sentence in enumerate(evidence_sentences[:max_sentences])
    ]


def clean_article_title(article_title: str) -> str:
    cleaned = normalize_text(article_title)
    if cleaned.startswith("(") and cleaned.endswith(")"):
        cleaned = cleaned[1:-1].strip()
    return cleaned


def extract_article_intro(body: str) -> str:
    normalized = normalize_text(re.sub(r"^\(\d+\)\s*", "", body))
    if not normalized:
        return ""
    item_positions = list(re.finditer(r"(?:^|[:;])\s*\d+\.\s+", normalized))
    intro = normalized[: item_positions[0].start()].strip(" ;,:") if item_positions else normalized
    sentences = dedupe_preserve_order(
        [normalize_text(part) for part in SENTENCE_SPLIT_RE.split(intro) if normalize_text(part)]
    )
    lead = sentences[0] if sentences else intro
    lead = lead.rstrip(" ;,:")
    if lead.lower().endswith(" če"):
        lead = lead[:-3].rstrip(" ,;:") + " ob določenih kršitvah"
    if lead.lower().endswith(" če so"):
        lead = lead[:-6].rstrip(" ,;:") + " ob določenih pogojih"
    return normalize_text(lead)


def extract_numbered_items(body: str) -> list[str]:
    normalized = normalize_text(body)
    matches = list(re.finditer(r"(?:^|[:;])\s*(\d+)\.\s+", normalized))
    if not matches:
        return []
    items: list[str] = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(normalized)
        item = normalize_text(normalized[start:end].strip(" ;,:"))
        if item:
            items.append(item)
    return dedupe_preserve_order(items)


def summarize_numbered_item(item: str, max_words: int = 18) -> str:
    cleaned = normalize_text(item)
    cleaned = re.sub(r"\([^()]{8,}\)$", "", cleaned).strip(" ;,:")
    if not cleaned:
        return ""
    words = cleaned.split()
    if len(words) <= max_words:
        return cleaned
    shortened = " ".join(words[:max_words]).rstrip(" ;,:")
    return f"{shortened} …"


def slovene_plural_suffix(count: int, singular: str, dual: str, paucal: str, plural: str) -> str:
    if count % 100 == 1:
        return singular
    if count % 100 == 2:
        return dual
    if count % 100 in {3, 4}:
        return paucal
    return plural


def extract_definition_excerpt(chunk: dict[str, Any], max_chars: int = 360) -> str:
    sentences = extract_chunk_sentences(chunk)
    if not sentences:
        return extract_article_excerpt(chunk, max_chars=max_chars)
    combined = " ".join(sentences[:2])
    if len(combined) <= max_chars:
        return combined
    truncated = combined[:max_chars]
    boundary = max(truncated.rfind(". "), truncated.rfind("; "), truncated.rfind(", "))
    if boundary > 100:
        truncated = truncated[:boundary]
    return normalize_text(truncated.rstrip(" ;,:")) + " …"


def extract_deadline_excerpt(chunk: dict[str, Any]) -> str | None:
    best_sentence: str | None = None
    best_score = float("-inf")
    for sentence in extract_chunk_sentences(chunk):
        score = score_deadline_sentence(sentence)
        if score > best_score:
            best_score = score
            best_sentence = sentence
    if best_sentence is None or best_score <= 0:
        return None
    return best_sentence


def score_deadline_sentence(sentence: str) -> float:
    normalized = normalize_text(sentence)
    lowered = normalized.lower()
    if not normalized:
        return -10.0
    if REFERRAL_SENTENCE_RE.search(normalized):
        return -5.0
    score = 0.0
    if "najpozneje" in lowered or "najkasneje" in lowered:
        score += 3.0
    if "na dan" in lowered:
        score += 2.0
    if "v roku" in lowered:
        score += 1.5
    if any(hint in lowered for hint in DEADLINE_PHRASE_HINTS):
        score += 1.0
    regex_features = extract_regex_features(normalized)
    if regex_features["dates"] or regex_features["deadlines"]:
        score += 1.0
    if "predložiti obračun" in lowered or "predložiti" in lowered:
        score += 0.75
    if len(normalized) < 28:
        score -= 0.5
    return score


def normalize_percentages(values: list[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = normalize_text(value).replace(" %", "%")
        if cleaned and cleaned not in seen:
            ordered.append(cleaned)
            seen.add(cleaned)
    return ordered


def extract_rate_bracket_answer(
    chunk: dict[str, Any],
    query_profile: QueryProfile,
    result_rank: int,
) -> dict[str, Any] | None:
    if not is_dohodnina_rate_chunk(chunk):
        return None
    brackets = extract_progressive_tax_brackets(chunk)
    if not brackets:
        return None
    normalized_query = query_profile.normalized_query
    target = min(brackets, key=bracket_sort_key) if any(marker in normalized_query for marker in ("najnižj", "najnizj", "najmanj")) else max(brackets, key=bracket_sort_key)
    range_text = format_bracket_range(target)
    rate_text = target["rate"]
    citation = format_short_citation(chunk)

    if query_profile.asks_bracket_threshold or query_profile.asks_extreme_rate:
        lead = f"{'Najvišja' if target is max(brackets, key=bracket_sort_key) else 'Najnižja'} stopnja dohodnine je {rate_text}"
        if range_text:
            lead += f" za del neto letne osnove {range_text}"
        answer = f"Po {citation} je {lead.lower()}."
        evidence = f"{lead}."
        return {
            "chunk": chunk,
            "answer": answer,
            "evidence": evidence,
            "support_score": 9.6 - 0.15 * (result_rank - 1),
        }
    return None


def is_dohodnina_rate_chunk(chunk: dict[str, Any]) -> bool:
    haystack = normalize_text(
        " ".join(
            part
            for part in (
                chunk.get("title"),
                chunk.get("article_title"),
                chunk.get("section_path"),
                chunk.get("raw_chunk_text", "")[:1200],
            )
            if part
        )
    ).lower()
    return "dohodnin" in haystack and "stopnje dohodnine" in haystack


def extract_progressive_tax_brackets(chunk: dict[str, Any]) -> list[dict[str, str | None]]:
    body = chunk_body_text(chunk)
    if "tabela:" not in body.lower() or "neto letna osnova" not in body.lower():
        return []
    parts = [normalize_text(part) for part in body.split("|") if normalize_text(part)]
    brackets: list[dict[str, str | None]] = []

    if len(parts) >= 4:
        upper_match = re.search(r"do\s+(" + EURO_AMOUNT_RE.pattern + r")", parts[2], re.IGNORECASE)
        rate_match = PERCENT_VALUE_RE.search(parts[3])
        if upper_match and rate_match:
            brackets.append({"lower": None, "upper": upper_match.group(1), "rate": normalize_text(rate_match.group(0)).replace(" %", "%")})

    for index, part in enumerate(parts):
        rate = normalize_text(part).replace(" %", "%")
        if not PERCENT_VALUE_RE.fullmatch(rate):
            continue
        if index + 2 >= len(parts) or parts[index + 1].lower() != "nad":
            continue
        amounts = EURO_AMOUNT_RE.findall(parts[index + 2])
        if not amounts:
            continue
        lower = amounts[0]
        upper = amounts[1] if len(amounts) > 1 else None
        brackets.append({"lower": lower, "upper": upper, "rate": rate})

    deduped: list[dict[str, str | None]] = []
    seen: set[tuple[str | None, str | None, str]] = set()
    for bracket in brackets:
        key = (bracket["lower"], bracket["upper"], str(bracket["rate"]))
        if key in seen:
            continue
        deduped.append(bracket)
        seen.add(key)
    return deduped


def bracket_sort_key(bracket: dict[str, str | None]) -> tuple[float, float]:
    lower = euro_amount_to_float(bracket.get("lower"))
    upper = euro_amount_to_float(bracket.get("upper"))
    return (lower if lower is not None else float("-inf"), upper if upper is not None else float("inf"))


def euro_amount_to_float(value: str | None) -> float | None:
    if not value:
        return None
    normalized = value.replace(".", "").replace(",", ".")
    try:
        return float(normalized)
    except ValueError:
        return None


def format_bracket_range(bracket: dict[str, str | None]) -> str:
    lower = bracket.get("lower")
    upper = bracket.get("upper")
    if lower and upper:
        return f"nad {lower} EUR do {upper} EUR"
    if upper:
        return f"do {upper} EUR"
    if lower:
        return f"nad {lower} EUR"
    return ""


def join_list_natural(values: list[str]) -> str:
    if not values:
        return ""
    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        return f"{values[0]} in {values[1]}"
    return f"{', '.join(values[:-1])} in {values[-1]}"


def build_query_understanding_prompt(message: str, history: list[dict[str, Any]]) -> str:
    history_lines: list[str] = []
    for item in history[-6:]:
        role = item.get("role", "user").upper()
        content = normalize_text(item.get("content", ""))
        if content:
            history_lines.append(f"{role}: {content}")
        topic = normalize_text(item.get("memory_topic", ""))
        if role == "ASSISTANT" and topic:
            history_lines.append(f"{role}_TOPIC: {topic}")
    joined_history = "\n".join(history_lines) if history_lines else "(brez zgodovine)"
    return (
        "Vrni JSON oblike:\n"
        '{'
        '"use_context": true, '
        '"standalone_query": "samostojna poizvedba v slovenščini", '
        '"intent": "explicit_article|deadline|definition|amount_percentage|practical_guidance|comparison|follow_up|general", '
        '"confidence": 0.0, '
        '"reason": "kratek razlog"'
        "}\n\n"
        "Pravila:\n"
        "- Če je trenutno vprašanje samostojno, naj bo use_context false in standalone_query naj bo enak trenutnemu vprašanju.\n"
        "- Če je follow-up, uporabi samo nujni kontekst iz zgodovine in napiši samostojno poizvedbo.\n"
        "- Ohrani zakon, člen, pravno temo in ključno uporabniško namero.\n"
        "- Ne odgovarjaj na vprašanje, samo preoblikuj poizvedbo.\n"
        "- Vrni samo JSON.\n\n"
        f"ZGODOVINA:\n{joined_history}\n\n"
        f"TRENUTNO VPRAŠANJE:\n{message}"
    )


def parse_query_understanding_output(output: str, fallback_message: str) -> QueryUnderstanding | None:
    payload = extract_first_json_object(output)
    if payload is None:
        return None
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return None
    standalone_query = normalize_text(str(data.get("standalone_query", "")))
    intent = normalize_text(str(data.get("intent", "general"))).lower()
    if intent not in QUERY_UNDERSTANDING_INTENTS:
        intent = "general"
    try:
        confidence = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    use_context = bool(data.get("use_context", False))
    reason = normalize_text(str(data.get("reason", "")))
    if not standalone_query:
        return None
    if not use_context:
        standalone_query = fallback_message
    return QueryUnderstanding(
        standalone_query=standalone_query,
        intent=intent,
        use_context=use_context,
        confidence=confidence,
        reason=reason,
    )


def extract_first_json_object(output: str) -> str | None:
    normalized = normalize_text(output)
    if not normalized:
        return None
    fenced = re.sub(r"^```(?:json)?\s*|\s*```$", "", normalized, flags=re.IGNORECASE).strip()
    if fenced.startswith("{") and fenced.endswith("}"):
        return fenced
    match = re.search(r"\{.*\}", fenced)
    return match.group(0) if match else None


def build_grounded_prompt(query: str, results: list[dict[str, Any]]) -> str:
    context_blocks = []
    for index, item in enumerate(results, start=1):
        chunk = item["chunk"]
        citation = format_short_citation(chunk)
        excerpt = truncate_chunk_for_generation(chunk)
        context_blocks.append(f"[Vir {index}] {citation}\n{excerpt}")
    joined_context = "\n\n".join(context_blocks)
    return (
        "Odgovarjaj v slovenščini kot pravni asistent za slovensko davčno zakonodajo.\n"
        "Uporabi izključno podane pravne vire.\n"
        "Če v virih ni dovolj podlage, to jasno povej in ne ugibaj.\n"
        "Odgovori jedrnato, navadno v 2 do 4 stavkih.\n"
        "Če vprašanje sprašuje po stopnjah, rokih, globah ali zneskih, jih navedi eksplicitno.\n"
        "Na koncu dodaj vrstico 'CITATI:' in navedi uporabljene vire v obliki 'Zakon, člen'.\n\n"
        f"VPRAŠANJE:\n{query}\n\n"
        f"VIRI:\n{joined_context}\n\n"
        "ODGOVOR:"
    )


def build_grounded_prompt_with_draft(query: str, results: list[dict[str, Any]], draft_answer: str | None) -> str:
    prompt = build_grounded_prompt(query, results)
    if not draft_answer:
        return prompt
    return (
        f"{prompt}\n\n"
        "DELOVNI OSNUTEK ODGOVORA:\n"
        f"{draft_answer}\n\n"
        "Osnutek izboljšaj v naraven, kratek slovenski odgovor. Ne dodajaj novih dejstev."
    )


def generate_with_local_transformer(
    query: str,
    results: list[dict[str, Any]],
    generator_model: str,
    max_new_tokens: int = DEFAULT_LOCAL_GENERATOR_MAX_NEW_TOKENS,
    draft_answer: str | None = None,
) -> str | None:
    if not results:
        return None
    tokenizer, model, device = load_generator_model(generator_model)
    messages = [
        {
            "role": "system",
            "content": (
                "Si Zakonodajko, lokalni pomočnik za slovensko davčno zakonodajo. "
                "Odgovarjaj samo na podlagi podanih virov, v slovenščini, jasno in brez ugibanja."
            ),
        },
        {"role": "user", "content": build_grounded_prompt_with_draft(query, results, draft_answer)},
    ]
    prompt = render_generation_prompt(tokenizer, messages)
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    input_length = encoded["input_ids"].shape[-1]
    with warnings.catch_warnings():
        suppress_generation_warnings()
        generated = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.pad_token_id,
        )
    output = tokenizer.decode(generated[0][input_length:], skip_special_tokens=True).strip()
    return clean_generated_answer(output)


def should_use_local_generation(query_profile: QueryProfile, payload: dict[str, Any], route: QueryRoute) -> bool:
    if not route.allow_generation:
        return False
    if query_profile.citations["articles"]:
        return False
    if query_profile.expects_percentage or query_profile.expects_amount or query_profile.expects_deadline:
        return False
    if payload.get("supporting_sentences"):
        top_support = payload["supporting_sentences"][0].get("support_score", 0.0)
        if top_support >= 9.0:
            return False
    return True


def select_generation_results(
    results: list[dict[str, Any]],
    citations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    citation_chunk_ids = {citation["chunk_id"] for citation in citations if citation.get("chunk_id")}
    selected = [item for item in results if item["chunk"]["chunk_id"] in citation_chunk_ids]
    return selected[:2] or results[:2]


def generated_answer_is_usable(
    generated: str,
    query_profile: QueryProfile,
    generation_results: list[dict[str, Any]],
    draft_answer: str,
) -> bool:
    normalized = normalize_text(generated)
    if not normalized or len(normalized) < 40:
        return False
    if normalized.count(":") >= 4 and len(normalized.split()) <= 16:
        return False
    banned_fragments = {"pravna vira", "dodatni vorec", "vir:", "zakon, člen:"}
    lowered = normalized.lower()
    if any(fragment in lowered for fragment in banned_fragments):
        return False
    if lexical_overlap(normalized, draft_answer) < 2:
        return False
    if query_profile.expects_percentage:
        expected_percentages = {
            percentage
            for item in generation_results
            for percentage in normalize_percentages(item["chunk"].get("percentages") or [])
        }
        generated_percentages = set(normalize_percentages(extract_regex_features(normalized)["percentages"]))
        return bool(expected_percentages) and expected_percentages.issubset(generated_percentages)
    return True


@lru_cache(maxsize=2)
def load_generator_model(generator_model: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(generator_model)
    device = detect_generation_device()
    torch_dtype = choose_torch_dtype(device)
    model = AutoModelForCausalLM.from_pretrained(generator_model, torch_dtype=torch_dtype)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.to(device)
    model.eval()
    return tokenizer, model, device


def detect_generation_device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def choose_torch_dtype(device: str):
    import torch

    if device == "cpu":
        return torch.float32
    return torch.float16


def render_generation_prompt(tokenizer, messages: list[dict[str, str]]) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return "\n\n".join(f"{message['role'].upper()}:\n{message['content']}" for message in messages) + "\n\nASSISTANT:\n"


def truncate_chunk_for_generation(chunk: dict[str, Any], max_chars: int = 1400) -> str:
    text = normalize_text(chunk.get("raw_chunk_text", ""))
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    boundary = max(truncated.rfind(". "), truncated.rfind("; "), truncated.rfind(") "))
    if boundary > 200:
        truncated = truncated[: boundary + 1]
    return normalize_text(truncated.rstrip(" ;,:")) + " …"


def clean_generated_answer(output: str) -> str | None:
    normalized = normalize_text(output)
    if not normalized:
        return None
    normalized = normalized.replace("**", "")
    normalized = re.sub(r"\bCITATI\s*:\s*.*$", "", normalized, flags=re.IGNORECASE).strip()
    normalized = re.sub(
        r"^(?:Odgovor|ODGOVOR|Zakon,\s*člen|Pravna\s+vira|Vir|Dodatni\s+\w+)\s*:\s*",
        "",
        normalized,
        flags=re.IGNORECASE,
    ).strip()
    while re.match(r"^(?:Odgovor|ODGOVOR|Zakon,\s*člen|Pravna\s+vira|Vir|Dodatni\s+\w+)\s*:", normalized, re.IGNORECASE):
        normalized = re.sub(
            r"^(?:Odgovor|ODGOVOR|Zakon,\s*člen|Pravna\s+vira|Vir|Dodatni\s+\w+)\s*:\s*",
            "",
            normalized,
            flags=re.IGNORECASE,
        ).strip()
    return normalized or None


def lexical_overlap(text_a: str, text_b: str) -> int:
    terms_a = useful_terms(text_a)
    terms_b = useful_terms(text_b)
    return len(terms_a.intersection(terms_b))


def useful_terms(text: str) -> set[str]:
    return {
        token.lower()
        for token in TERM_RE.findall(normalize_text(text))
        if is_useful_query_term(token.lower())
    }
