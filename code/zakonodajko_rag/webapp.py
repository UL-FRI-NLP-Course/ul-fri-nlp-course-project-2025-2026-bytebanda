from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .answering import QueryUnderstanding, answer_question, format_short_citation, understand_chat_query_with_local_model
from .calculators import maybe_handle_calculator_turn
from .classla_support import lemmatize_query_text
from .constants import (
    DEFAULT_BM25_PATH,
    DEFAULT_CHROMA_DIR,
    DEFAULT_CHUNK_PATH,
    DEFAULT_EMBEDDING_PROFILE,
    DEFAULT_LOCAL_GENERATOR_MAX_NEW_TOKENS,
    DEFAULT_LOCAL_GENERATOR_MODEL,
)
from .planner import LegalAgentPlan, plan_legal_chat_actions
from .retrieval import ensure_dense_index, load_chunk_map
from .text_utils import normalize_text


FOLLOW_UP_PREFIXES = (
    "kaj pa",
    "in pa",
    "kaj če",
    "kaj pa če",
    "koliko pa",
    "kdaj pa",
    "kaj potem",
    "in za",
    "za pravno osebo",
    "za s.p.",
)
FOLLOW_UP_PRONOUNS = {
    "to",
    "ta",
    "tega",
    "teh",
    "tem",
    "tistih",
    "tisti",
    "tisto",
    "njih",
    "njega",
    "njem",
    "njo",
    "nje",
    "takem",
    "tako",
    "potem",
}
SHORT_FOLLOW_UP_STARTERS = {"in", "pa", "potem", "torej"}
EXPLANATION_HINTS = (
    "razloži",
    "pojasni",
    "podrobneje",
    "bolj podrobno",
    "več podrobnosti",
    "daljši odgovor",
    "daljše pojasnilo",
    "bolj na dolgo",
    "razširi odgovor",
    "razširi to",
)
META_REFERENCE_HINTS = ("prejšnje vprašanje", "prejšnji odgovor", "zadnji odgovor", "to vprašanje", "ta odgovor")


@dataclass(frozen=True)
class WebAppSettings:
    chunks_path: Path = DEFAULT_CHUNK_PATH
    chroma_dir: Path = DEFAULT_CHROMA_DIR
    bm25_path: Path = DEFAULT_BM25_PATH
    classla_python: str | None = None
    top_k: int = 5
    embedding_profile: str = DEFAULT_EMBEDDING_PROFILE
    embedding_model_name: str | None = None
    generator_model: str | None = DEFAULT_LOCAL_GENERATOR_MODEL
    reranker_model: str | None = None
    max_new_tokens: int = DEFAULT_LOCAL_GENERATOR_MAX_NEW_TOKENS


class ChatHistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=8000)
    citations: list[dict[str, Any]] = Field(default_factory=list)
    retrieval_query: str | None = Field(default=None, max_length=8000)
    memory_topic: str | None = Field(default=None, max_length=500)
    calculator_context: dict[str, Any] | None = None


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    history: list[ChatHistoryMessage] = Field(default_factory=list)


def create_app(
    settings: WebAppSettings,
    answer_pipeline: Callable[..., dict[str, Any]] = answer_question,
    lemmatize_fn: Callable[[str, str | None], list[str]] = lemmatize_query_text,
    query_understanding_fn: Callable[[str, list[dict[str, Any]], str | None, int], QueryUnderstanding | None] = understand_chat_query_with_local_model,
    agent_planner_fn: Callable[[str, list[dict[str, Any]], str | None, int], LegalAgentPlan] = plan_legal_chat_actions,
) -> FastAPI:
    if not settings.chunks_path.exists():
        raise FileNotFoundError(f"Missing chunks file: {settings.chunks_path}")
    if not settings.bm25_path.exists():
        raise FileNotFoundError(f"Missing BM25 index: {settings.bm25_path}")
    if not settings.chroma_dir.exists():
        raise FileNotFoundError(f"Missing Chroma directory: {settings.chroma_dir}")

    static_dir = Path(__file__).resolve().parent / "static"
    chunk_map = load_chunk_map(settings.chunks_path)
    ensure_dense_index(
        list(chunk_map.values()),
        settings.chroma_dir,
        embedding_profile=settings.embedding_profile,
        embedding_model_name=settings.embedding_model_name,
    )
    document_count = len({chunk["doc_id"] for chunk in chunk_map.values()})

    app = FastAPI(title="Zakonodajko", version="0.1.0")
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "chunk_count": len(chunk_map),
            "document_count": document_count,
            "embedding_profile": settings.embedding_profile,
            "embedding_model_name": settings.embedding_model_name,
            "generator_model": settings.generator_model,
            "reranker_model": settings.reranker_model,
        }

    @app.post("/api/chat")
    def chat(request: ChatRequest) -> dict[str, Any]:
        message = normalize_text(request.message)
        if not message:
            raise HTTPException(status_code=400, detail="Sporočilo ne sme biti prazno.")

        history = [item.model_dump() for item in request.history[-8:]]
        heuristic_query = contextualize_chat_query(message, history)
        query_understanding = None
        retrieval_query = heuristic_query
        if settings.generator_model and history:
            query_understanding = query_understanding_fn(
                message,
                history,
                settings.generator_model,
                min(settings.max_new_tokens, 160),
            )
            if should_use_query_understanding(query_understanding, message):
                retrieval_query = query_understanding.standalone_query
        planning_model = settings.generator_model if history else None
        agent_plan = agent_planner_fn(
            retrieval_query,
            history,
            planning_model,
            min(settings.max_new_tokens, 192),
        )
        calculator_payload = maybe_handle_calculator_turn(
            message,
            history,
            chunk_map,
            agent_plan=agent_plan.as_dict(),
        )
        if calculator_payload is not None:
            calculator_payload["retrieval_query"] = retrieval_query
            calculator_payload["contextualized"] = retrieval_query != message
            calculator_payload["heuristic_query"] = heuristic_query
            calculator_payload["query_understanding"] = query_understanding.as_dict() if query_understanding is not None else None
            calculator_payload["agent_plan"] = agent_plan.as_dict()
            calculator_payload["processing_trace"] = build_calculator_processing_trace(
                message=message,
                heuristic_query=heuristic_query,
                retrieval_query=retrieval_query,
                query_understanding=query_understanding,
                agent_plan=agent_plan.as_dict(),
                payload=calculator_payload,
            )
            return calculator_payload
        query_tokens = lemmatize_fn(retrieval_query, settings.classla_python)
        payload = answer_pipeline(
            retrieval_query,
            query_tokens,
            chunk_map,
            settings.chroma_dir,
            settings.bm25_path,
            top_k=settings.top_k,
            embedding_profile=settings.embedding_profile,
            embedding_model_name=settings.embedding_model_name,
            generator_model=settings.generator_model,
            reranker_model=settings.reranker_model,
            max_new_tokens=settings.max_new_tokens,
            agent_plan=agent_plan,
        )
        processing_trace = build_processing_trace(
            message=message,
            heuristic_query=heuristic_query,
            retrieval_query=retrieval_query,
            query_understanding=query_understanding,
            agent_plan=payload.get("agent_plan") or agent_plan.as_dict(),
            payload=payload,
        )
        return {
            "message": payload["answer"],
            "answer_sections": payload.get("answer_sections", []),
            "citations": payload["citations"],
            "used_chunks": payload["used_chunks"],
            "supporting_sentences": payload["supporting_sentences"],
            "insufficient_evidence": payload["insufficient_evidence"],
            "backend": payload["backend"],
            "retrieval_query": retrieval_query,
            "contextualized": retrieval_query != message,
            "heuristic_query": heuristic_query,
            "query_understanding": query_understanding.as_dict() if query_understanding is not None else None,
            "agent_plan": payload.get("agent_plan") or agent_plan.as_dict(),
            "memory_topic": derive_memory_topic(payload),
            "generator_model": settings.generator_model,
            "route": payload.get("route"),
            "resolution_chain": payload.get("resolution_chain", []),
            "citation_verification": payload.get("citation_verification"),
            "processing_trace": processing_trace,
        }

    return app


def contextualize_chat_query(message: str, history: list[dict[str, Any]], max_user_turns: int = 2) -> str:
    normalized = normalize_text(message)
    if not normalized:
        return ""
    if is_self_contained_question(normalized):
        return normalized

    prior_user_turns = latest_user_turns(history)
    if not prior_user_turns:
        return normalized

    context_turns = [turn for turn in prior_user_turns[:max_user_turns] if turn != normalized]
    last_assistant_topic = latest_assistant_topic(history)
    if not context_turns and not last_assistant_topic:
        return normalized
    if is_explanatory_follow_up(normalized):
        return build_explanatory_follow_up_query(normalized, context_turns, last_assistant_topic)
    segments: list[str] = []
    if context_turns:
        context_turns.reverse()
        segments.append(f"Prejšnje vprašanje: {' | '.join(context_turns)}")
    if last_assistant_topic:
        segments.append(f"Zadnja pravna tema: {last_assistant_topic}")
    segments.append(f"Trenutno vprašanje: {normalized}")
    return ". ".join(segments)


def is_self_contained_question(message: str) -> bool:
    lowered = normalize_text(message).lower()
    return not should_contextualize_follow_up(lowered)


def should_contextualize_follow_up(message: str) -> bool:
    lowered = normalize_text(message).lower()
    tokens = [token.strip("?.!,;:") for token in lowered.split() if token.strip("?.!,;:")]
    if any(lowered.startswith(prefix) for prefix in FOLLOW_UP_PREFIXES):
        return True
    if is_explanatory_follow_up(lowered):
        return True
    if not tokens:
        return False
    if len(tokens) <= 4 and tokens[0] in SHORT_FOLLOW_UP_STARTERS:
        return True
    if any(token in FOLLOW_UP_PRONOUNS for token in tokens) and any(
        marker in lowered for marker in ("od teh", "od tega", "od njih", "v njo", "vanjo", "v njem", "med njimi")
    ):
        return True
    if len(tokens) <= 5 and any(token in FOLLOW_UP_PRONOUNS for token in tokens):
        return True
    return False


def latest_assistant_topic(history: list[dict[str, Any]]) -> str:
    for item in reversed(history):
        if item.get("role") != "assistant":
            continue
        topic = normalize_text(item.get("memory_topic", ""))
        if topic:
            return topic
        citations = item.get("citations") or []
        if citations:
            return format_short_citation(citations[0])
    return ""


def latest_user_turns(history: list[dict[str, Any]]) -> list[str]:
    return [
        normalize_text(item["content"])
        for item in reversed(history)
        if item.get("role") == "user" and normalize_text(item.get("content", ""))
    ]


def is_explanatory_follow_up(message: str) -> bool:
    lowered = normalize_text(message).lower()
    has_explanation_hint = any(hint in lowered for hint in EXPLANATION_HINTS)
    if not has_explanation_hint:
        return False
    has_meta_reference = any(hint in lowered for hint in META_REFERENCE_HINTS)
    tokens = [token.strip("?.!,;:") for token in lowered.split() if token.strip("?.!,;:")]
    has_pronoun_reference = any(token in FOLLOW_UP_PRONOUNS for token in tokens)
    short_expansion_request = len(tokens) <= 5
    return has_meta_reference or has_pronoun_reference or short_expansion_request


def build_explanatory_follow_up_query(
    message: str,
    context_turns: list[str],
    last_assistant_topic: str,
) -> str:
    latest_user_question = context_turns[0] if context_turns else ""
    segments: list[str] = []
    if latest_user_question:
        segments.append(f"Podrobneje razloži vprašanje: {latest_user_question}")
    if last_assistant_topic:
        segments.append(f"Pravna tema: {last_assistant_topic}")
    segments.append(f"Navodilo uporabnika: {message}")
    return ". ".join(segments)


def derive_memory_topic(payload: dict[str, Any]) -> str:
    citations = payload.get("citations") or []
    if citations:
        return format_short_citation(citations[0])
    answer = normalize_text(payload.get("answer", ""))
    if not answer:
        return ""
    return answer[:180]


def should_use_query_understanding(understanding: QueryUnderstanding | None, message: str) -> bool:
    if understanding is None:
        return False
    if understanding.use_context:
        return understanding.confidence >= 0.6 and normalize_text(understanding.standalone_query) != normalize_text(message)
    return understanding.confidence >= 0.75


def build_processing_trace(
    message: str,
    heuristic_query: str,
    retrieval_query: str,
    query_understanding: QueryUnderstanding | None,
    agent_plan: dict[str, Any] | None,
    payload: dict[str, Any],
) -> list[dict[str, str]]:
    citations = payload.get("citations") or []
    resolution_chain = payload.get("resolution_chain") or []
    verification = payload.get("citation_verification") or {}
    backend = payload.get("backend") or "extractive"
    route = payload.get("route") or {}
    top_citation = citations[0] if citations else None
    trace: list[dict[str, str]] = []

    understanding_detail = "Vprašanje je obravnavano kot samostojno."
    if retrieval_query != message:
        if query_understanding and normalize_text(query_understanding.standalone_query) == normalize_text(retrieval_query):
            understanding_detail = "Lokalni model je iz zgodovine pogovora sestavil samostojno poizvedbo za retrieval."
        elif heuristic_query == retrieval_query:
            understanding_detail = "Uporabljena je bila heuristična interpretacija follow-up vprašanja."
        else:
            understanding_detail = "Vprašanje je bilo razširjeno z dodatnim kontekstom pred iskanjem virov."
    trace.append({"label": "Razumevanje vprašanja", "status": "completed", "detail": understanding_detail})

    if agent_plan:
        trace.append(
            {
                "label": "Načrt odgovora",
                "status": "completed",
                "detail": describe_agent_plan(agent_plan, route),
            }
        )

    retrieval_detail = "Glavni vir še ni bil potrjen."
    if top_citation:
        retrieval_detail = f"Kot primarni vir je bil izbran {format_short_citation(top_citation)}."
    trace.append({"label": "Iskanje virov", "status": "completed", "detail": retrieval_detail})

    if resolution_chain:
        chain_preview = " → ".join(step["to_citation"] for step in resolution_chain[:2])
        trace.append(
            {
                "label": "Napotitve med členi",
                "status": "completed",
                "detail": f"Sistem je sledil napotitvi do: {chain_preview}.",
            }
        )
    elif route.get("follow_referrals"):
        trace.append(
            {
                "label": "Napotitve med členi",
                "status": "completed",
                "detail": "Napotitvena veriga ni bila potrebna ali ni spremenila glavnega vira.",
            }
        )

    answer_detail = (
        "Odgovor je sestavil lokalni model na podlagi izbranih virov."
        if backend == "local_transformer"
        else "Odgovor je bil sestavljen deterministično iz najrelevantnejših pravnih odlomkov."
    )
    trace.append({"label": "Sestava odgovora", "status": "completed", "detail": answer_detail})

    verification_status = str(verification.get("status") or "unknown")
    verification_detail = "Primarni citat ni bil preverjen."
    if verification_status == "verified":
        verification_detail = "Primarni citat podpira odgovor po internih preverjanjih."
    elif verification_status == "weak":
        verification_detail = "Primarni citat je le delno skladen; odgovor preveri tudi ročno."
    elif verification_status == "missing":
        verification_detail = "Odgovor nima dovolj močnega primarnega citata."
    trace.append({"label": "Preverjanje citata", "status": verification_status, "detail": verification_detail})
    return trace


def build_calculator_processing_trace(
    message: str,
    heuristic_query: str,
    retrieval_query: str,
    query_understanding: QueryUnderstanding | None,
    agent_plan: dict[str, Any] | None,
    payload: dict[str, Any],
) -> list[dict[str, str]]:
    trace: list[dict[str, str]] = []
    understanding_detail = "Vprašanje je obravnavano kot samostojno."
    if retrieval_query != message:
        if query_understanding and normalize_text(query_understanding.standalone_query) == normalize_text(retrieval_query):
            understanding_detail = "Lokalni model je najprej sestavil samostojno poizvedbo za izračun."
        elif heuristic_query == retrieval_query:
            understanding_detail = "Vprašanje je bilo razširjeno s heurističnim chat kontekstom."
        else:
            understanding_detail = "Uporabljena je bila razširjena interpretacija vprašanja pred izračunom."
    trace.append({"label": "Razumevanje vprašanja", "status": "completed", "detail": understanding_detail})
    if agent_plan:
        trace.append({"label": "Načrt odgovora", "status": "completed", "detail": describe_agent_plan(agent_plan, {"intent": "calculation"})})
    calculator_result = payload.get("calculator_result") or {}
    trace.append(
        {
            "label": "Uporaba kalkulatorja",
            "status": "completed",
            "detail": "Za ta odgovor je uporabljen deterministični kalkulator, ne prosto LLM računanje.",
        }
    )
    if calculator_result.get("status") == "pending":
        trace.append({"label": "Vhodni podatki", "status": "weak", "detail": payload.get("message", "Manjka podatek za izračun.")})
    else:
        trace.append({"label": "Vhodni podatki", "status": "completed", "detail": "Vhodni podatki so bili izluščeni iz vprašanja in potrjeni za izračun."})
        trace.append({"label": "Izračun", "status": "completed", "detail": "Rezultat je bil izračunan deterministično, brez LLM računanja."})
    verification = payload.get("citation_verification") or {}
    verification_status = str(verification.get("status") or "missing")
    verification_detail = "Pravna podlaga za formulo ni bila potrjena."
    if verification_status == "verified":
        verification_detail = "Pravna podlaga za uporabljeno formulo je bila potrjena."
    trace.append({"label": "Preverjanje citata", "status": verification_status, "detail": verification_detail})
    return trace


def describe_agent_plan(agent_plan: dict[str, Any], route: dict[str, Any]) -> str:
    intent = str(agent_plan.get("intent") or route.get("intent") or "general")
    source_policy = str(agent_plan.get("source_policy") or route.get("source_policy") or "pisrs_first")
    actions = [str(action) for action in (agent_plan.get("actions") or [])]
    source_phrase = {
        "pisrs_only": "iskanje samo po PISRS zakonodaji",
        "pisrs_first": "prednost PISRS, po potrebi tudi drugi viri",
        "furs_allowed": "prednost zakonodaji, FURS pojasnila so dovoljena",
        "furs_preferred": "prednost FURS pojasnilom ob podpori zakonodaje",
    }.get(source_policy, "uravnoteženo iskanje virov")
    action_phrase = "z referral korakom" if "resolve_referrals" in actions else "brez dodatne napotitvene verige"
    intent_phrase = {
        "explicit_article": "eksplicitni člen",
        "deadline": "rok ali obveznost",
        "definition": "definicijsko vprašanje",
        "amount_percentage": "znesek, stopnja ali prag",
        "practical_guidance": "praktično navodilo",
        "comparison": "primerjava pravil",
        "follow_up": "follow-up na prejšnjo temo",
        "general": "splošno pravno vprašanje",
    }.get(intent, "splošno pravno vprašanje")
    return f"Plan je vprašanje prepoznal kot {intent_phrase}; uporablja {source_phrase} in dela {action_phrase}."
