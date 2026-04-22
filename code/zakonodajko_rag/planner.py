from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass
from typing import Any, Literal

from .answering import extract_first_json_object, load_generator_model, render_generation_prompt, suppress_generation_warnings
from .router import AnswerStyle, Intent, QueryRoute, SourcePolicy, build_query_profile, route_query
from .text_utils import normalize_text


AgentAction = Literal[
    "retrieve_chunks",
    "resolve_referrals",
    "prefer_pisrs",
    "prefer_furs_guidance",
    "run_calculator",
    "compose_article_overview",
    "compose_structured_answer",
    "compose_guided_explanation",
    "verify_primary_citation",
]

KNOWN_AGENT_ACTIONS: set[str] = {
    "retrieve_chunks",
    "resolve_referrals",
    "prefer_pisrs",
    "prefer_furs_guidance",
    "run_calculator",
    "compose_article_overview",
    "compose_structured_answer",
    "compose_guided_explanation",
    "verify_primary_citation",
}
PLANNING_INTENTS: set[str] = {
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
SOURCE_POLICIES: set[str] = {"pisrs_only", "pisrs_first", "furs_allowed", "furs_preferred"}
ANSWER_STYLES: set[str] = {"article_overview", "structured_rule", "guided_explanation", "comparative", "extractive"}


@dataclass(frozen=True)
class LegalAgentPlan:
    query: str
    intent: Intent
    source_policy: SourcePolicy
    answer_style: AnswerStyle
    follow_referrals: bool
    preserve_origin_article: bool
    actions: tuple[AgentAction, ...]
    confidence: float
    reason: str = ""
    backend: str = "heuristic"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def plan_legal_chat_actions(
    query: str,
    history: list[dict[str, Any]] | None,
    generator_model: str | None,
    max_new_tokens: int = 192,
) -> LegalAgentPlan:
    heuristic_plan = build_heuristic_legal_agent_plan(query)
    model_plan = plan_with_local_model(query, history or [], generator_model, max_new_tokens=max_new_tokens)
    return merge_plans(heuristic_plan, model_plan)


def build_heuristic_legal_agent_plan(query: str) -> LegalAgentPlan:
    if looks_like_calculation_query(query):
        return LegalAgentPlan(
            query=normalize_text(query),
            intent="calculation",
            source_policy="pisrs_first",
            answer_style="structured_rule",
            follow_referrals=False,
            preserve_origin_article=False,
            actions=("retrieve_chunks", "prefer_pisrs", "run_calculator", "verify_primary_citation"),
            confidence=0.6,
            reason="Hevristično prepoznano vprašanje za kalkulator.",
            backend="heuristic",
        )
    profile = build_query_profile(query, [])
    route = route_query(profile)
    return LegalAgentPlan(
        query=normalize_text(query),
        intent=route.intent,
        source_policy=route.source_policy,
        answer_style=route.answer_style,
        follow_referrals=route.follow_referrals,
        preserve_origin_article=route.preserve_origin_article,
        actions=default_actions_for_route(route),
        confidence=0.55,
        reason="Hevristični plan na podlagi query routerja.",
        backend="heuristic",
    )


def plan_with_local_model(
    query: str,
    history: list[dict[str, Any]],
    generator_model: str | None,
    max_new_tokens: int = 192,
) -> LegalAgentPlan | None:
    normalized_query = normalize_text(query)
    if not normalized_query or not generator_model:
        return None
    tokenizer, model, device = load_generator_model(generator_model)
    messages = [
        {
            "role": "system",
            "content": (
                "Si planner za slovenskega davčnega pomočnika. "
                "Vrni samo veljaven JSON brez markdowna. "
                "Izberi le dovoljene akcije in ne izmišljuj novih."
            ),
        },
        {
            "role": "user",
            "content": build_planning_prompt(normalized_query, history),
        },
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
    return parse_model_plan_output(output, normalized_query)


def build_planning_prompt(query: str, history: list[dict[str, Any]]) -> str:
    history_lines: list[str] = []
    for item in history[-6:]:
        role = "uporabnik" if item.get("role") == "user" else "asistent"
        content = normalize_text(item.get("content", ""))
        if not content:
            continue
        line = f"- {role}: {content}"
        memory_topic = normalize_text(item.get("memory_topic", ""))
        if role == "asistent" and memory_topic:
            line += f" [tema: {memory_topic}]"
        history_lines.append(line)
    history_block = "\n".join(history_lines) if history_lines else "- ni prejšnje zgodovine"
    return (
        "Naloga: pripravi omejen plan za odgovor na davčno-pravno vprašanje.\n"
        "Dovoljene vrednosti:\n"
        f"- intent: {sorted(PLANNING_INTENTS)}\n"
        f"- source_policy: {sorted(SOURCE_POLICIES)}\n"
        f"- answer_style: {sorted(ANSWER_STYLES)}\n"
        f"- actions: {sorted(KNOWN_AGENT_ACTIONS)}\n"
        "Pravila:\n"
        "- Vedno vključi retrieve_chunks in verify_primary_citation.\n"
        "- Če je vprašanje o konkretnem členu, preferiraj pisrs_only.\n"
        "- Če vprašanje zahteva praktično uporabo ali FURS pojasnilo, lahko preferiraš furs_allowed ali furs_preferred.\n"
        "- Če je vprašanje predvsem računsko, uporabi intent calculation in akcijo run_calculator.\n"
        "- resolve_referrals uporabi le, če je treba slediti napotitvam med členi.\n"
        "- confidence naj bo med 0 in 1.\n"
        "Vrni JSON oblike:\n"
        '{'
        '"intent":"...",'
        '"source_policy":"...",'
        '"answer_style":"...",'
        '"follow_referrals":true,'
        '"preserve_origin_article":false,'
        '"actions":["retrieve_chunks","verify_primary_citation"],'
        '"confidence":0.0,'
        '"reason":"..."'
        '}\n'
        f"Zgodovina pogovora:\n{history_block}\n"
        f"Trenutna poizvedba:\n{query}"
    )


def parse_model_plan_output(output: str, fallback_query: str) -> LegalAgentPlan | None:
    json_fragment = extract_first_json_object(output)
    if json_fragment is None:
        return None
    try:
        payload = json.loads(json_fragment)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    intent = str(payload.get("intent", "")).strip()
    source_policy = str(payload.get("source_policy", "")).strip()
    answer_style = str(payload.get("answer_style", "")).strip()
    if intent not in PLANNING_INTENTS or source_policy not in SOURCE_POLICIES or answer_style not in ANSWER_STYLES:
        return None
    raw_actions = payload.get("actions", [])
    if not isinstance(raw_actions, list):
        return None
    cleaned_actions: list[AgentAction] = []
    for action in raw_actions:
        normalized = str(action).strip()
        if normalized not in KNOWN_AGENT_ACTIONS:
            return None
        if normalized not in cleaned_actions:
            cleaned_actions.append(normalized)  # type: ignore[arg-type]
    if "retrieve_chunks" not in cleaned_actions:
        cleaned_actions.insert(0, "retrieve_chunks")
    if "verify_primary_citation" not in cleaned_actions:
        cleaned_actions.append("verify_primary_citation")
    try:
        confidence = float(payload.get("confidence", 0.0))
    except (TypeError, ValueError):
        return None
    confidence = max(0.0, min(1.0, confidence))
    return LegalAgentPlan(
        query=fallback_query,
        intent=intent,  # type: ignore[arg-type]
        source_policy=source_policy,  # type: ignore[arg-type]
        answer_style=answer_style,  # type: ignore[arg-type]
        follow_referrals=bool(payload.get("follow_referrals", False)),
        preserve_origin_article=bool(payload.get("preserve_origin_article", False)),
        actions=tuple(cleaned_actions),
        confidence=confidence,
        reason=normalize_text(str(payload.get("reason", ""))),
        backend="local_model",
    )


def merge_plans(heuristic_plan: LegalAgentPlan, model_plan: LegalAgentPlan | None) -> LegalAgentPlan:
    if model_plan is None or model_plan.confidence < 0.6:
        return heuristic_plan
    if heuristic_plan.intent == "explicit_article":
        return sanitize_plan(heuristic_plan, model_plan.confidence, backend=model_plan.backend, reason=model_plan.reason)
    return sanitize_plan(
        model_plan,
        model_plan.confidence,
        backend=model_plan.backend,
        reason=model_plan.reason or heuristic_plan.reason,
        fallback=heuristic_plan,
    )


def sanitize_plan(
    candidate: LegalAgentPlan,
    confidence: float,
    backend: str,
    reason: str,
    fallback: LegalAgentPlan | None = None,
) -> LegalAgentPlan:
    base = fallback or candidate
    actions = [action for action in candidate.actions if action in KNOWN_AGENT_ACTIONS]
    if "retrieve_chunks" not in actions:
        actions.insert(0, "retrieve_chunks")
    if candidate.follow_referrals and "resolve_referrals" not in actions:
        actions.append("resolve_referrals")
    if candidate.source_policy in {"pisrs_only", "pisrs_first"}:
        if "prefer_pisrs" not in actions:
            actions.append("prefer_pisrs")
        actions = [action for action in actions if action != "prefer_furs_guidance"]
    else:
        if "prefer_furs_guidance" not in actions:
            actions.append("prefer_furs_guidance")
    if candidate.answer_style == "article_overview":
        if "compose_article_overview" not in actions:
            actions.append("compose_article_overview")
    elif candidate.answer_style == "guided_explanation":
        if "compose_guided_explanation" not in actions:
            actions.append("compose_guided_explanation")
    else:
        if "compose_structured_answer" not in actions:
            actions.append("compose_structured_answer")
    if candidate.intent == "calculation" and "run_calculator" not in actions:
        actions.append("run_calculator")
    if "verify_primary_citation" not in actions:
        actions.append("verify_primary_citation")
    unique_actions: list[AgentAction] = []
    for action in actions:
        if action not in unique_actions:
            unique_actions.append(action)  # type: ignore[arg-type]
    return LegalAgentPlan(
        query=candidate.query or base.query,
        intent=candidate.intent,
        source_policy=candidate.source_policy,
        answer_style=candidate.answer_style,
        follow_referrals=candidate.follow_referrals,
        preserve_origin_article=candidate.preserve_origin_article,
        actions=tuple(unique_actions),
        confidence=max(0.0, min(1.0, confidence)),
        reason=reason,
        backend=backend,
    )


def default_actions_for_route(route: QueryRoute) -> tuple[AgentAction, ...]:
    actions: list[AgentAction] = ["retrieve_chunks"]
    if route.follow_referrals:
        actions.append("resolve_referrals")
    if route.source_policy in {"furs_allowed", "furs_preferred"}:
        actions.append("prefer_furs_guidance")
    else:
        actions.append("prefer_pisrs")
    if route.answer_style == "article_overview":
        actions.append("compose_article_overview")
    elif route.answer_style == "guided_explanation":
        actions.append("compose_guided_explanation")
    else:
        actions.append("compose_structured_answer")
    actions.append("verify_primary_citation")
    return tuple(actions)


def looks_like_calculation_query(query: str) -> bool:
    lowered = normalize_text(query).lower()
    mentions_vat = "ddv" in lowered
    mentions_income_tax = "dohodnin" in lowered
    has_currency = "eur" in lowered or "€" in lowered
    has_percent = "%" in lowered
    calc_verbs = ("izračun", "izracun", "izračunaj", "izracunaj", "koliko znaša", "koliko znasa", "kolikšen je", "koliksen je")
    asks_bracket = any(token in lowered for token in ("razred", "padem", "spadam", "stopnja"))
    return (mentions_vat and (has_currency or has_percent or any(verb in lowered for verb in calc_verbs))) or (
        mentions_income_tax and (has_currency or asks_bracket or any(verb in lowered for verb in calc_verbs))
    )
