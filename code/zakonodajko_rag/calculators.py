from __future__ import annotations

from datetime import date
from decimal import Decimal, ROUND_HALF_UP
import re
from typing import Any

from .answering import collect_citations, extract_progressive_tax_brackets
from .text_utils import normalize_text


EURO_WITH_UNIT_RE = re.compile(
    r"(?P<amount>\d{1,3}(?:[.\s]\d{3})*(?:,\d+)?|\d+(?:,\d+)?)\s*(?:€|eur|eurov|eura|evrov|evra)\b",
    re.IGNORECASE,
)
NUMBER_RE = re.compile(r"\d{1,3}(?:[.\s]\d{3})*(?:,\d+)?|\d+(?:,\d+)?")
PERCENT_RE = re.compile(r"(?P<rate>\d{1,2}(?:,\d+)?)\s*%")
YEAR_RE = re.compile(r"(?:davčn(?:o|ega)\s+leto|za leto|v letu|leto)\s*(20\d{2})", re.IGNORECASE)
CALCULATION_HINTS = ("izračun", "izracun", "koliko je ddv", "koliko znaša ddv", "koliko znaša dohodnina", "izračunaj", "izracunaj")
INCOME_BRACKET_HINTS = ("razred", "stopnja", "stopnjo", "stopnji", "padem", "spadam", "najvišja", "najvisja", "najnižja", "najnizja")
CALCULATOR_REFERENCE_HINTS = ("v tem primeru", "pri tej osnovi", "tega", "takrat", "potem", "skupno", "skupaj", "koliko pa")
VAT_FOLLOW_UP_HINTS = ("ddv", "osnova", "bruto", "neto", "skupaj", "skupno", "plačati", "placati")
INCOME_TAX_FOLLOW_UP_HINTS = (
    "dohodnin",
    "razred",
    "stopnj",
    "koliko",
    "skupno",
    "plačati",
    "placati",
    "plačam",
    "placam",
)
CURRENT_SUPPORTED_TAX_YEAR = date.today().year


def maybe_handle_calculator_turn(
    message: str,
    history: list[dict[str, Any]],
    chunk_map: dict[str, dict[str, Any]],
    agent_plan: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    normalized = normalize_text(message)
    if not normalized:
        return None
    pending_context = latest_pending_calculator_context(history)
    if pending_context is not None and looks_like_calculator_follow_up(normalized, pending_context):
        merged_context = merge_calculator_context(pending_context, normalized)
        return build_calculator_payload(normalized, merged_context, chunk_map, contextualized=True)
    recent_context = latest_calculator_context(history, statuses={"completed"})
    if recent_context is not None and looks_like_calculator_follow_up(normalized, recent_context):
        merged_context = merge_calculator_context(recent_context, normalized)
        return build_calculator_payload(normalized, merged_context, chunk_map, contextualized=True)
    if not should_try_calculator(normalized, agent_plan):
        return None
    context = build_new_calculator_context(normalized)
    if context is None:
        return None
    return build_calculator_payload(normalized, context, chunk_map, contextualized=False)


def should_try_calculator(message: str, agent_plan: dict[str, Any] | None) -> bool:
    normalized = normalize_text(message).lower()
    actions = [str(action) for action in ((agent_plan or {}).get("actions") or [])]
    intent = str((agent_plan or {}).get("intent") or "")
    if intent == "calculation" or "run_calculator" in actions:
        return True
    return ("ddv" in normalized or "dohodnin" in normalized) and (
        "izračun" in normalized
        or "izracun" in normalized
        or "izračunaj" in normalized
        or "izracunaj" in normalized
        or "koliko znaša" in normalized
        or "koliko znasa" in normalized
        or "razred" in normalized
        or "padem" in normalized
        or "spadam" in normalized
    )


def looks_like_calculator_follow_up(message: str, context: dict[str, Any]) -> bool:
    lowered = normalize_text(message).lower()
    if context.get("calculator") == "vat":
        has_reference = any(marker in lowered for marker in CALCULATOR_REFERENCE_HINTS)
        return bool(
            extract_percent(message) is not None
            or extract_amount(message) is not None
            or any(marker in lowered for marker in ("neto", "bruto", "z ddv", "brez ddv", "uporabi"))
            or (
                context.get("status") == "completed"
                and has_reference
                and any(marker in lowered for marker in VAT_FOLLOW_UP_HINTS)
            )
        )
    if context.get("calculator") == "income_tax_brackets":
        has_reference = any(marker in lowered for marker in CALCULATOR_REFERENCE_HINTS)
        return bool(
            extract_amount(message) is not None
            or extract_tax_year(message) is not None
            or any(marker in lowered for marker in ("neto", "bruto", "letna osnova", "davčna osnova"))
            or (
                context.get("status") == "completed"
                and has_reference
                and any(marker in lowered for marker in INCOME_TAX_FOLLOW_UP_HINTS)
            )
        )
    return False


def build_new_calculator_context(message: str) -> dict[str, Any] | None:
    lowered = normalize_text(message).lower()
    has_calc_hint = any(hint in lowered for hint in CALCULATION_HINTS)
    has_amount = extract_amount(message) is not None
    has_rate = extract_percent(message) is not None
    if "ddv" in lowered and (has_calc_hint or (has_amount and has_rate)):
        return parse_vat_context(message)
    if "dohodnin" in lowered and has_amount and (
        has_calc_hint
        or "koliko znaša dohodnina" in lowered
        or "koliko znasa dohodnina" in lowered
        or any(hint in lowered for hint in INCOME_BRACKET_HINTS)
    ):
        return parse_income_tax_context(message)
    return None


def latest_pending_calculator_context(history: list[dict[str, Any]]) -> dict[str, Any] | None:
    return latest_calculator_context(history, statuses={"pending"})


def latest_calculator_context(
    history: list[dict[str, Any]],
    statuses: set[str] | None = None,
) -> dict[str, Any] | None:
    allowed_statuses = statuses or {"pending", "ready", "completed"}
    for item in reversed(history):
        if item.get("role") != "assistant":
            continue
        context = item.get("calculator_context")
        if not isinstance(context, dict):
            continue
        if context.get("status") in allowed_statuses:
            return context
    return None


def merge_calculator_context(context: dict[str, Any], message: str) -> dict[str, Any]:
    calculator = context.get("calculator")
    if calculator == "vat":
        merged = parse_vat_context(message, base_context=context)
    elif calculator == "income_tax_brackets":
        merged = parse_income_tax_context(message, base_context=context)
    else:
        merged = dict(context)
    merged["prompt"] = context.get("prompt", "")
    return merged


def parse_vat_context(message: str, base_context: dict[str, Any] | None = None) -> dict[str, Any]:
    params = dict((base_context or {}).get("params") or {})
    assumptions = list((base_context or {}).get("assumptions") or [])
    amount = extract_amount(message)
    if amount is not None:
        params["amount"] = amount
    rate = extract_percent(message)
    if rate is not None:
        params["vat_rate"] = rate
    lowered = normalize_text(message).lower()
    if "bruto" in lowered or "z ddv" in lowered:
        params["amount_type"] = "gross"
    elif "neto" in lowered or "brez ddv" in lowered or "davčna osnova" in lowered:
        params["amount_type"] = "net"
    elif "amount_type" not in params and "ddv na" in lowered:
        params["amount_type"] = "net"
        assumption = "Ker vprašanje uporablja obliko 'DDV na znesek', je znesek obravnavan kot neto osnova."
        if assumption not in assumptions:
            assumptions.append(assumption)

    missing: list[str] = []
    if "amount" not in params:
        missing.append("amount")
    if "vat_rate" not in params:
        missing.append("vat_rate")
    if "amount_type" not in params:
        missing.append("amount_type")

    prompt = ""
    if missing:
        if missing == ["vat_rate"]:
            prompt = "Katero stopnjo DDV naj uporabim, na primer 22% ali 9,5%?"
        elif missing == ["amount_type"]:
            prompt = "Ali gre za neto znesek brez DDV ali za bruto znesek z DDV?"
        else:
            prompt = "Za izračun DDV potrebujem znesek, stopnjo DDV in podatek, ali gre za neto ali bruto znesek."

    return {
        "calculator": "vat",
        "title": "DDV kalkulator",
        "params": params,
        "missing_params": missing,
        "assumptions": assumptions,
        "status": "pending" if missing else "ready",
        "prompt": prompt,
    }


def parse_income_tax_context(message: str, base_context: dict[str, Any] | None = None) -> dict[str, Any]:
    params = dict((base_context or {}).get("params") or {})
    assumptions = list((base_context or {}).get("assumptions") or [])
    amount = extract_amount(message)
    if amount is not None:
        params["annual_tax_base"] = amount
    year = extract_tax_year(message)
    if year is not None:
        params["tax_year"] = year
    elif "tax_year" not in params:
        params["tax_year"] = CURRENT_SUPPORTED_TAX_YEAR
        assumption = f"Uporabljena je trenutno indeksirana lestvica za davčno leto {CURRENT_SUPPORTED_TAX_YEAR}."
        if assumption not in assumptions:
            assumptions.append(assumption)
    lowered = normalize_text(message).lower()
    if "neto letn" in lowered or "neto davčn" in lowered:
        params["tax_base_type"] = "net_annual"
    elif "bruto" in lowered:
        params["tax_base_type"] = "gross"

    missing: list[str] = []
    if "annual_tax_base" not in params:
        missing.append("annual_tax_base")
    if "tax_base_type" not in params:
        missing.append("tax_base_type")

    prompt = ""
    if missing:
        if missing == ["tax_base_type"]:
            prompt = "Ali gre za neto letno davčno osnovo? Trenutni kalkulator podpira izračun po neto letni osnovi."
        elif missing == ["annual_tax_base"]:
            prompt = "Kolikšna je neto letna davčna osnova v EUR?"
        else:
            prompt = "Za izračun dohodnine potrebujem neto letno davčno osnovo v EUR."

    return {
        "calculator": "income_tax_brackets",
        "title": "Kalkulator dohodninske lestvice",
        "params": params,
        "missing_params": missing,
        "assumptions": assumptions,
        "status": "pending" if missing else "ready",
        "prompt": prompt,
    }


def build_calculator_payload(
    message: str,
    context: dict[str, Any],
    chunk_map: dict[str, dict[str, Any]],
    contextualized: bool,
) -> dict[str, Any]:
    calculator = context.get("calculator")
    if context.get("status") == "pending":
        return build_calculator_clarification_payload(message, context, contextualized)
    if calculator == "vat":
        return build_vat_payload(message, context, chunk_map, contextualized)
    if calculator == "income_tax_brackets":
        return build_income_tax_payload(message, context, chunk_map, contextualized)
    return build_calculator_clarification_payload(message, context, contextualized)


def build_calculator_clarification_payload(
    message: str,
    context: dict[str, Any],
    contextualized: bool,
) -> dict[str, Any]:
    title = context.get("title") or "Davčni kalkulator"
    prompt = normalize_text(context.get("prompt") or "Potrebujem še nekaj vhodnih podatkov za izračun.")
    trace = [
        {"label": "Prepoznan izračun", "status": "completed", "detail": f"Sistem je vprašanje prepoznal kot {title.lower()}."},
        {"label": "Vhodni podatki", "status": "weak", "detail": prompt},
    ]
    return {
        "message": prompt,
        "answer_sections": [{"label": "Potreben podatek", "text": prompt}],
        "citations": [],
        "used_chunks": [],
        "supporting_sentences": [],
        "insufficient_evidence": False,
        "backend": "calculator_clarification",
        "retrieval_query": message,
        "contextualized": contextualized,
        "heuristic_query": message,
        "query_understanding": None,
        "agent_plan": None,
        "memory_topic": title,
        "route": {"intent": "calculation"},
        "resolution_chain": [],
        "citation_verification": {"status": "missing", "score": 0.0, "checks": {"has_citation": False}},
        "processing_trace": trace,
        "calculator_result": {
            "calculator": context.get("calculator"),
            "title": title,
            "status": "pending",
            "inputs": format_inputs(context.get("params") or {}),
            "missing_params": context.get("missing_params") or [],
            "assumptions": context.get("assumptions") or [],
            "breakdown": [],
            "result_summary": None,
        },
        "calculator_context": context,
    }


def build_vat_payload(
    message: str,
    context: dict[str, Any],
    chunk_map: dict[str, dict[str, Any]],
    contextualized: bool,
) -> dict[str, Any]:
    params = context["params"]
    amount = Decimal(str(params["amount"]))
    rate = Decimal(str(params["vat_rate"]))
    amount_type = params["amount_type"]
    if amount_type == "gross":
        gross = quantize_money(amount)
        base = quantize_money(gross / (Decimal("1") + (rate / Decimal("100"))))
        vat = quantize_money(gross - base)
    else:
        base = quantize_money(amount)
        vat = quantize_money(base * rate / Decimal("100"))
        gross = quantize_money(base + vat)

    chunk = find_article_chunk(chunk_map, "ZDDV-1", "41. člen")
    citations = collect_citations([chunk]) if chunk else []
    used_chunks = [build_used_chunk(chunk)] if chunk else []
    result_summary = (
        f"DDV pri stopnji {format_percent(rate)} znaša {format_money(vat)}, "
        f"skupni znesek pa {format_money(gross)}."
        if amount_type == "net"
        else f"Osnova brez DDV znaša {format_money(base)}, DDV pa {format_money(vat)}."
    )
    if citations:
        message_text = (
            f"Rezultat: {result_summary}\n\n"
            f"Razčlenitev: osnova {format_money(base)}, DDV {format_money(vat)}, skupaj {format_money(gross)}.\n\n"
            f"Predpostavke: {render_assumptions(context.get('assumptions') or ['Uporabljena je podana stopnja DDV.'])}\n\n"
            f"Pravna podlaga: {citations[0]['law_ref']}, {citations[0]['article_number']} ({citations[0]['article_title'].strip('()')})"
        )
    else:
        message_text = f"Rezultat: {result_summary}"
    trace = [
        {"label": "Prepoznan izračun", "status": "completed", "detail": "Vprašanje je bilo prepoznano kot DDV izračun."},
        {"label": "Vhodni podatki", "status": "completed", "detail": f"Uporabljeni so znesek {format_money(amount)} in stopnja {format_percent(rate)}."},
        {"label": "Izračun", "status": "completed", "detail": f"DDV je izračunan deterministično iz osnove in stopnje."},
        {"label": "Pravna podlaga", "status": "verified", "detail": "Stopnja DDV je vezana na 41. člen ZDDV-1."},
    ]
    return {
        "message": message_text,
        "answer_sections": [
            {"label": "Rezultat", "text": result_summary},
            {"label": "Pravna podlaga", "text": "ZDDV-1, 41. člen (stopnja DDV)"},
        ],
        "citations": citations,
        "used_chunks": used_chunks,
        "supporting_sentences": [],
        "insufficient_evidence": False,
        "backend": "calculator",
        "retrieval_query": message,
        "contextualized": contextualized,
        "heuristic_query": message,
        "query_understanding": None,
        "agent_plan": None,
        "memory_topic": "DDV kalkulator",
        "route": {"intent": "calculation"},
        "resolution_chain": [],
        "citation_verification": {"status": "verified", "score": 1.0, "checks": {"has_citation": bool(citations)}},
        "processing_trace": trace,
        "calculator_result": {
            "calculator": "vat",
            "title": "DDV kalkulator",
            "status": "completed",
            "inputs": format_inputs({"amount": amount, "amount_type": amount_type, "vat_rate": rate}),
            "missing_params": [],
            "assumptions": context.get("assumptions") or ["Uporabljena je podana stopnja DDV."],
            "breakdown": [
                {"label": "Davčna osnova", "value": format_money(base)},
                {"label": "DDV", "value": format_money(vat)},
                {"label": "Skupaj", "value": format_money(gross)},
            ],
            "result_summary": result_summary,
        },
        "calculator_context": build_completed_calculator_context(context, result_summary),
    }


def build_income_tax_payload(
    message: str,
    context: dict[str, Any],
    chunk_map: dict[str, dict[str, Any]],
    contextualized: bool,
) -> dict[str, Any]:
    params = context["params"]
    if params.get("tax_base_type") != "net_annual":
        context = dict(context)
        context["status"] = "pending"
        context["missing_params"] = ["tax_base_type"]
        context["prompt"] = "Trenutni kalkulator podpira izračun po neto letni davčni osnovi. Sporoči neto letno davčno osnovo."
        return build_calculator_clarification_payload(message, context, contextualized)
    tax_year = int(params["tax_year"])
    if tax_year != CURRENT_SUPPORTED_TAX_YEAR:
        title = context.get("title") or "Kalkulator dohodninske lestvice"
        note = f"Trenutno lahko izračunam dohodnino le po trenutno indeksirani lestvici za davčno leto {CURRENT_SUPPORTED_TAX_YEAR}."
        return {
            "message": note,
            "answer_sections": [{"label": "Omejitev kalkulatorja", "text": note}],
            "citations": [],
            "used_chunks": [],
            "supporting_sentences": [],
            "insufficient_evidence": False,
            "backend": "calculator_clarification",
            "retrieval_query": message,
            "contextualized": contextualized,
            "heuristic_query": message,
            "query_understanding": None,
            "agent_plan": None,
            "memory_topic": title,
            "route": {"intent": "calculation"},
            "resolution_chain": [],
            "citation_verification": {"status": "missing", "score": 0.0, "checks": {"has_citation": False}},
            "processing_trace": [
                {"label": "Prepoznan izračun", "status": "completed", "detail": "Vprašanje je bilo prepoznano kot izračun dohodnine."},
                {"label": "Omejitev podatkov", "status": "weak", "detail": note},
            ],
            "calculator_result": {
                "calculator": "income_tax_brackets",
                "title": title,
                "status": "unsupported_year",
                "inputs": format_inputs(params),
                "missing_params": [],
                "assumptions": context.get("assumptions") or [],
                "breakdown": [],
                "result_summary": None,
            },
            "calculator_context": context,
        }

    base = Decimal(str(params["annual_tax_base"]))
    chunk = find_article_chunk(chunk_map, "ZDoh-2", "122. člen")
    if chunk is None:
        return build_calculator_clarification_payload(message, context, contextualized)
    brackets = extract_progressive_tax_brackets(chunk)
    breakdown, total_tax, top_rate = compute_progressive_tax(base, brackets)
    citations = collect_citations([chunk])
    used_chunks = [build_used_chunk(chunk)]
    lowered_message = normalize_text(message).lower()
    active_bracket = find_active_bracket(base, brackets)
    bracket_summary = ""
    if active_bracket is not None:
        bracket_summary = (
            f"Pri tej osnovi padeš v razred {format_bracket_range_for_summary(active_bracket)}"
            f" s stopnjo {format_percent(decimal_from_percent(str(active_bracket.get('rate') or '0%')))}."
        )
    result_summary = f"Ocenjena dohodnina po lestvici znaša {format_money(total_tax)}."
    if any(hint in lowered_message for hint in ("razred", "padem", "spadam")) and bracket_summary:
        result_summary = bracket_summary
    top_rate_summary = f"Najvišja uporabljena stopnja v izračunu je {format_percent(top_rate)}." if top_rate is not None else ""
    if any(hint in lowered_message for hint in ("najvišja stopnja", "najvisja stopnja")) and top_rate is not None:
        result_summary = f"Najvišja uporabljena stopnja pri tej osnovi je {format_percent(top_rate)}."
    message_text = (
        f"Rezultat: {result_summary} {top_rate_summary}\n\n"
        f"Predpostavke: {render_assumptions(context.get('assumptions') or [])}\n\n"
        f"Pravna podlaga: ZDoh-2, 122. člen (stopnje dohodnine)"
    ).strip()
    trace = [
        {"label": "Prepoznan izračun", "status": "completed", "detail": "Vprašanje je bilo prepoznano kot izračun po dohodninski lestvici."},
        {"label": "Vhodni podatki", "status": "completed", "detail": f"Uporabljena je neto letna davčna osnova {format_money(base)}."},
        {"label": "Izračun", "status": "completed", "detail": "Dohodnina je izračunana po progresivnih razredih iz 122. člena ZDoh-2."},
        {"label": "Pravna podlaga", "status": "verified", "detail": "Rezultat je vezan na trenutno indeksirano lestvico iz 122. člena ZDoh-2."},
    ]
    return {
        "message": message_text,
        "answer_sections": [
            {"label": "Rezultat", "text": result_summary},
            {"label": "Pravna podlaga", "text": "ZDoh-2, 122. člen (stopnje dohodnine)"},
        ],
        "citations": citations,
        "used_chunks": used_chunks,
        "supporting_sentences": [],
        "insufficient_evidence": False,
        "backend": "calculator",
        "retrieval_query": message,
        "contextualized": contextualized,
        "heuristic_query": message,
        "query_understanding": None,
        "agent_plan": None,
        "memory_topic": "Kalkulator dohodninske lestvice",
        "route": {"intent": "calculation"},
        "resolution_chain": [],
        "citation_verification": {"status": "verified", "score": 1.0, "checks": {"has_citation": True}},
        "processing_trace": trace,
        "calculator_result": {
            "calculator": "income_tax_brackets",
            "title": "Kalkulator dohodninske lestvice",
            "status": "completed",
            "inputs": format_inputs(params),
            "missing_params": [],
            "assumptions": context.get("assumptions") or [],
            "breakdown": breakdown,
            "result_summary": f"{result_summary} {top_rate_summary}".strip(),
        },
        "calculator_context": build_completed_calculator_context(
            context,
            f"{result_summary} {top_rate_summary}".strip(),
        ),
    }


def build_completed_calculator_context(context: dict[str, Any], result_summary: str) -> dict[str, Any]:
    return {
        "calculator": context.get("calculator"),
        "title": context.get("title"),
        "params": dict(context.get("params") or {}),
        "missing_params": [],
        "assumptions": list(context.get("assumptions") or []),
        "status": "completed",
        "prompt": "",
        "result_summary": normalize_text(result_summary),
    }


def compute_progressive_tax(
    base: Decimal,
    brackets: list[dict[str, str | None]],
) -> tuple[list[dict[str, str]], Decimal, Decimal | None]:
    ordered = sorted(brackets, key=sort_bracket_for_calc)
    total = Decimal("0")
    top_rate: Decimal | None = None
    breakdown: list[dict[str, str]] = []
    for bracket in ordered:
        lower = decimal_from_optional_amount(bracket.get("lower")) or Decimal("0")
        upper = decimal_from_optional_amount(bracket.get("upper"))
        rate = decimal_from_percent(str(bracket.get("rate") or "0%"))
        if base <= lower:
            continue
        taxable = (base if upper is None else min(base, upper)) - lower
        if taxable <= 0:
            continue
        tax = quantize_money(taxable * rate / Decimal("100"))
        total += tax
        top_rate = rate
        breakdown.append(
            {
                "label": format_bracket_label(lower, upper, rate),
                "value": f"osnova {format_money(taxable)} → davek {format_money(tax)}",
            }
        )
    return breakdown, quantize_money(total), top_rate


def find_active_bracket(base: Decimal, brackets: list[dict[str, str | None]]) -> dict[str, str | None] | None:
    ordered = sorted(brackets, key=sort_bracket_for_calc)
    active: dict[str, str | None] | None = None
    for bracket in ordered:
        lower = decimal_from_optional_amount(bracket.get("lower")) or Decimal("0")
        upper = decimal_from_optional_amount(bracket.get("upper"))
        if base <= lower:
            continue
        if upper is None or base <= upper:
            active = bracket
            break
        active = bracket
    return active


def format_bracket_label(lower: Decimal, upper: Decimal | None, rate: Decimal) -> str:
    if upper is None:
        return f"Nad {format_money(lower)} ({format_percent(rate)})"
    if lower == 0:
        return f"Do {format_money(upper)} ({format_percent(rate)})"
    return f"Nad {format_money(lower)} do {format_money(upper)} ({format_percent(rate)})"


def format_bracket_range_for_summary(bracket: dict[str, str | None]) -> str:
    lower = decimal_from_optional_amount(bracket.get("lower")) or Decimal("0")
    upper = decimal_from_optional_amount(bracket.get("upper"))
    if upper is None:
        return f"nad {format_money(lower)}"
    if lower == 0:
        return f"do {format_money(upper)}"
    return f"nad {format_money(lower)} do {format_money(upper)}"


def sort_bracket_for_calc(bracket: dict[str, str | None]) -> tuple[Decimal, Decimal]:
    lower = decimal_from_optional_amount(bracket.get("lower")) or Decimal("0")
    upper = decimal_from_optional_amount(bracket.get("upper")) or Decimal("999999999")
    return lower, upper


def find_article_chunk(chunk_map: dict[str, dict[str, Any]], law_ref: str, article_number: str) -> dict[str, Any] | None:
    for chunk in chunk_map.values():
        law_refs = (chunk.get("legal_refs") or {}).get("law_refs") or []
        if law_ref in law_refs and chunk.get("article_number") == article_number:
            return chunk
    return None


def build_used_chunk(chunk: dict[str, Any]) -> dict[str, Any]:
    return {
        "rank": 1,
        "score": 1.0,
        "law_id": chunk.get("law_id"),
        "law_ref": ((chunk.get("legal_refs") or {}).get("law_refs") or [chunk.get("law_id")])[0],
        "title": chunk.get("title"),
        "section_path": chunk.get("section_path"),
        "article_number": chunk.get("article_number"),
        "article_title": chunk.get("article_title"),
        "source_url": chunk.get("source_url"),
        "chunk_id": chunk.get("chunk_id"),
        "text_preview": normalize_text(chunk.get("raw_chunk_text", ""))[:800],
    }


def extract_amount(text: str) -> float | None:
    year = extract_tax_year(text)
    match = EURO_WITH_UNIT_RE.search(text)
    if match:
        return float(decimal_from_amount(match.group("amount")))
    candidates: list[str] = []
    for item in NUMBER_RE.finditer(text):
        following = text[item.end() : item.end() + 1]
        if following == "%":
            continue
        candidates.append(item.group(0))
    filtered: list[str] = []
    for candidate in candidates:
        stripped = candidate.replace(".", "").replace(" ", "").replace(",", "")
        if year is not None and stripped == str(year):
            continue
        filtered.append(candidate)
    if len(filtered) == 1:
        return float(decimal_from_amount(filtered[0]))
    return None


def extract_percent(text: str) -> float | None:
    match = PERCENT_RE.search(text)
    if match is None:
        return None
    return float(decimal_from_percent(match.group("rate")))


def extract_tax_year(text: str) -> int | None:
    match = YEAR_RE.search(text)
    if match is None:
        return None
    return int(match.group(1))


def decimal_from_amount(value: str) -> Decimal:
    normalized = value.replace(" ", "").replace(".", "").replace(",", ".")
    return Decimal(normalized)


def decimal_from_percent(value: str) -> Decimal:
    normalized = value.replace("%", "").replace(" ", "").replace(",", ".")
    return Decimal(normalized)


def decimal_from_optional_amount(value: str | None) -> Decimal | None:
    if not value:
        return None
    return decimal_from_amount(value)


def quantize_money(value: Decimal) -> Decimal:
    return value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def format_money(value: Decimal) -> str:
    quantized = quantize_money(value)
    sign = "-" if quantized < 0 else ""
    absolute = f"{abs(quantized):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"{sign}{absolute} EUR"


def format_percent(value: Decimal) -> str:
    normalized = value.normalize()
    if normalized == normalized.to_integral():
        return f"{int(normalized)}%"
    return f"{str(normalized).replace('.', ',')}%"


def render_assumptions(assumptions: list[str]) -> str:
    cleaned = [normalize_text(item) for item in assumptions if normalize_text(item)]
    if not cleaned:
        return "brez dodatnih predpostavk."
    return "; ".join(cleaned)


def format_inputs(params: dict[str, Any]) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    for key, label in (
        ("amount", "Znesek"),
        ("amount_type", "Tip zneska"),
        ("vat_rate", "Stopnja DDV"),
        ("annual_tax_base", "Neto letna davčna osnova"),
        ("tax_year", "Davčno leto"),
        ("tax_base_type", "Vrsta osnove"),
    ):
        if key not in params:
            continue
        value = params[key]
        if key in {"amount", "annual_tax_base"}:
            rendered = format_money(Decimal(str(value)))
        elif key == "vat_rate":
            rendered = format_percent(Decimal(str(value)))
        elif key == "amount_type":
            rendered = "neto" if value == "net" else "bruto"
        elif key == "tax_base_type":
            rendered = "neto letna osnova" if value == "net_annual" else str(value)
        else:
            rendered = str(value)
        items.append({"label": label, "value": rendered})
    return items
