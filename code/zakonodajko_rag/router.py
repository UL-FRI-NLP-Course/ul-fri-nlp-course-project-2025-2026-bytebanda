from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any, Literal

from .regexes import extract_query_citations
from .text_utils import normalize_text


TERM_RE = re.compile(r"[0-9A-Za-zČŠŽčšž]+(?:[.-][0-9A-Za-zČŠŽčšž]+)*")
SLOVENE_STOPWORDS = {
    "a",
    "ali",
    "biti",
    "bo",
    "da",
    "do",
    "ga",
    "in",
    "iz",
    "je",
    "jo",
    "kaj",
    "kako",
    "kateri",
    "katere",
    "katero",
    "ki",
    "kje",
    "ko",
    "kot",
    "na",
    "nad",
    "naj",
    "najpozneje",
    "najkasneje",
    "ne",
    "o",
    "ob",
    "od",
    "pa",
    "po",
    "pod",
    "pri",
    "se",
    "so",
    "ta",
    "taen",
    "te",
    "to",
    "v",
    "za",
}
GENERIC_LEGAL_TERMS = {"zakon", "pravilnik", "člen", "odstavek", "točka", "del", "poglavje"}
DEADLINE_HINTS = {"rok", "kdaj", "datum", "najpozneje", "najkasneje"}
AMOUNT_HINTS = {"globa", "kazen", "znesek", "koliko", "plača", "plačati", "eurov", "eur"}
PERCENTAGE_HINTS = {"stopnja", "stopnje", "stopenj", "odstotek", "odstotki", "delež", "delezi", "procent"}
PRACTICAL_GUIDANCE_HINTS = {
    "kako",
    "oddam",
    "oddati",
    "uveljavljam",
    "uveljaviti",
    "prijavim",
    "prijaviti",
    "izpolnim",
    "izpolniti",
    "furs",
    "pojasnilo",
    "pojasnila",
    "navodilo",
    "navodila",
}
TECHNICAL_PROCESS_HINTS = {
    "api",
    "certifikat",
    "certifikata",
    "certifikatom",
    "certifikatu",
    "client",
    "datoteka",
    "datoteke",
    "edavki",
    "edavkov",
    "erp",
    "evidenca",
    "evidenc",
    "evidenci",
    "excel",
    "identifikator",
    "identifikatorja",
    "klient",
    "klienta",
    "oauth",
    "obrazec",
    "obrazca",
    "obravnave",
    "polje",
    "polju",
    "portal",
    "prijava",
    "servis",
    "servisa",
    "spletni",
    "spletnega",
    "strukturirani",
    "testni",
    "testnem",
    "xml",
}
PRACTICAL_QUERY_PATTERNS = (
    "ali bo mogoče",
    "ali bo mozno",
    "ali se lahko",
    "kje so objavljene",
    "v povezavi z navodili",
    "kako pridobim",
    "kako predložim",
    "kako predlozim",
    "testnem okolju",
    "spletni servis",
    "identifikator klienta",
    "ročnim nalaganjem",
    "rocnim nalaganjem",
)
COMPARISON_HINTS = {"razlika", "primerjava", "primerjaj", "ali", "namesto"}
FOLLOW_UP_MARKERS = (
    "prejšnje vprašanje:",
    "zadnja pravna tema:",
    "podrobneje razloži vprašanje:",
    "navodilo uporabnika:",
)


Intent = Literal[
    "explicit_article",
    "deadline",
    "definition",
    "amount_percentage",
    "calculation",
    "practical_guidance",
    "comparison",
    "follow_up",
    "general",
]
SourcePolicy = Literal["pisrs_only", "pisrs_first", "furs_allowed", "furs_preferred"]
AnswerStyle = Literal["article_overview", "structured_rule", "guided_explanation", "comparative", "extractive"]


@dataclass(frozen=True)
class QueryProfile:
    query: str
    normalized_query: str
    citations: dict[str, list[str]]
    keywords: set[str]
    expects_definition: bool
    expects_deadline: bool
    expects_amount: bool
    expects_percentage: bool
    mentions_furs: bool
    has_follow_up_context: bool
    practical_signal_count: int
    has_technical_markers: bool
    expects_practical_guidance: bool
    asks_extreme_rate: bool
    asks_bracket_threshold: bool

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["keywords"] = sorted(self.keywords)
        return payload


@dataclass(frozen=True)
class QueryRoute:
    intent: Intent
    source_policy: SourcePolicy
    answer_style: AnswerStyle
    allow_generation: bool
    follow_referrals: bool
    preserve_origin_article: bool = False


def build_query_profile(query: str, query_tokens: list[str]) -> QueryProfile:
    normalized_query = normalize_text(query).lower()
    surface_terms = [token.lower() for token in TERM_RE.findall(normalized_query)]
    citations = extract_query_citations(query)
    keyword_pool: set[str] = set()
    for token in [*query_tokens, *surface_terms]:
        normalized = normalize_text(token).lower()
        if is_useful_query_term(normalized):
            keyword_pool.add(normalized)
    for citation_values in citations.values():
        for value in citation_values:
            for token in TERM_RE.findall(normalize_text(value).lower()):
                if is_useful_query_term(token):
                    keyword_pool.add(token)
    practical_signal_count = 0
    practical_signal_count += count_hint_matches(keyword_pool, PRACTICAL_GUIDANCE_HINTS)
    practical_signal_count += count_hint_matches(keyword_pool, TECHNICAL_PROCESS_HINTS)
    practical_signal_count += count_query_pattern_matches(normalized_query, PRACTICAL_QUERY_PATTERNS)
    has_technical_markers = contains_hint(keyword_pool, TECHNICAL_PROCESS_HINTS) or any(
        pattern in normalized_query for pattern in PRACTICAL_QUERY_PATTERNS
    )
    if "»" in query or '"' in query:
        practical_signal_count += 1
    if "furs" in normalized_query:
        practical_signal_count += 2
    expects_practical_guidance = "furs" in normalized_query or practical_signal_count >= 3
    has_percentage_hint = contains_hint(keyword_pool, PERCENTAGE_HINTS) or "%" in normalized_query
    asks_extreme_rate = any(
        marker in normalized_query
        for marker in ("najvišj", "najvisj", "najnižj", "najnizj", "največj", "najvecj", "najmanj")
    )
    asks_bracket_threshold = (
        has_percentage_hint
        and (
            asks_extreme_rate
            or any(marker in normalized_query for marker in ("padem", "padeš", "pades", "pade", "pademo", "spadam", "spada", "spadamo"))
            or "v njo" in normalized_query
            or "vanjo" in normalized_query
        )
    )
    return QueryProfile(
        query=query,
        normalized_query=normalized_query,
        citations=citations,
        keywords=keyword_pool,
        expects_definition=normalized_query.startswith("kaj pomeni") or normalized_query.startswith("kaj je"),
        expects_deadline=(contains_hint(keyword_pool, DEADLINE_HINTS) or "do kdaj" in normalized_query)
        and not asks_bracket_threshold,
        expects_amount=contains_hint(keyword_pool, AMOUNT_HINTS),
        expects_percentage=has_percentage_hint,
        mentions_furs=("furs" in normalized_query),
        has_follow_up_context=any(marker in normalized_query for marker in FOLLOW_UP_MARKERS),
        practical_signal_count=practical_signal_count,
        has_technical_markers=has_technical_markers,
        expects_practical_guidance=expects_practical_guidance,
        asks_extreme_rate=asks_extreme_rate,
        asks_bracket_threshold=asks_bracket_threshold,
    )


def route_query(query_profile: QueryProfile) -> QueryRoute:
    query = query_profile.normalized_query
    if query_profile.citations["articles"]:
        return QueryRoute(
            intent="explicit_article",
            source_policy="pisrs_only",
            answer_style="article_overview",
            allow_generation=False,
            follow_referrals=False,
            preserve_origin_article=True,
        )
    if query_profile.expects_deadline:
        return QueryRoute(
            intent="deadline",
            source_policy="pisrs_first",
            answer_style="structured_rule",
            allow_generation=False,
            follow_referrals=True,
        )
    if query_profile.expects_definition:
        return QueryRoute(
            intent="definition",
            source_policy="pisrs_first",
            answer_style="structured_rule",
            allow_generation=False,
            follow_referrals=False,
        )
    if query_profile.expects_amount or query_profile.expects_percentage:
        return QueryRoute(
            intent="amount_percentage",
            source_policy="pisrs_first",
            answer_style="structured_rule",
            allow_generation=False,
            follow_referrals=True,
        )
    if query_profile.has_follow_up_context:
        source_policy: SourcePolicy = (
            "furs_preferred"
            if "pravna tema: furs" in query or query_profile.expects_practical_guidance
            else "pisrs_first"
        )
        return QueryRoute(
            intent="follow_up",
            source_policy=source_policy,
            answer_style="guided_explanation",
            allow_generation=True,
            follow_referrals=True,
        )
    if query_profile.expects_practical_guidance:
        return QueryRoute(
            intent="practical_guidance",
            source_policy="furs_preferred" if query_profile.mentions_furs or query_profile.has_technical_markers else "furs_allowed",
            answer_style="guided_explanation",
            allow_generation=True,
            follow_referrals=True,
        )
    if contains_hint(query_profile.keywords, COMPARISON_HINTS):
        return QueryRoute(
            intent="comparison",
            source_policy="pisrs_first",
            answer_style="comparative",
            allow_generation=True,
            follow_referrals=True,
        )
    return QueryRoute(
        intent="general",
        source_policy="pisrs_first",
        answer_style="extractive",
        allow_generation=True,
        follow_referrals=True,
    )


def contains_hint(keywords: set[str], hints: set[str]) -> bool:
    return bool(keywords.intersection(hints))


def count_hint_matches(keywords: set[str], hints: set[str]) -> int:
    return len(keywords.intersection(hints))


def count_query_pattern_matches(query: str, patterns: tuple[str, ...]) -> int:
    return sum(1 for pattern in patterns if pattern in query)


def is_useful_query_term(token: str) -> bool:
    if not token:
        return False
    if token in SLOVENE_STOPWORDS or token in GENERIC_LEGAL_TERMS:
        return False
    if len(token) < 3 and not any(char.isdigit() for char in token):
        return False
    return True
