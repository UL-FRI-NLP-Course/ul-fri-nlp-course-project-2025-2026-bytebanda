from __future__ import annotations

import re
from typing import Any

from .constants import ORDINAL_WORDS
from .text_utils import normalize_text


LAW_REF_RE = re.compile(r"\b[A-ZŽŠČ][A-Za-zŽŠČžšč]+-\d+[A-ZŽŠČ]?\b")
ACT_ID_RE = re.compile(r"\b(?:ZAKO|PRAV|ODLU)\d+\b")
ARTICLE_RE = re.compile(r"\b\d+\.(?:[a-zčšž])?\s*člen\b", re.IGNORECASE)
ITEM_RE = re.compile(r"\b\d+\.\s*točka\b", re.IGNORECASE)
PARAGRAPH_RE = re.compile(
    rf"\b(?:{'|'.join(ORDINAL_WORDS)})\s+odstavek\b",
    re.IGNORECASE,
)
DATE_RE = re.compile(
    r"\b\d{1,2}\.\s?\d{1,2}\.\s?\d{4}\b"
    r"|\b\d{1,2}\.\s*(?:januarja|februarja|marca|aprila|maja|junija|julija|avgusta|septembra|oktobra|novembra|decembra)\s+\d{4}\b",
    re.IGNORECASE,
)
AMOUNT_RE = re.compile(
    r"\b\d{1,3}(?:[.\s]\d{3})*(?:,\d+)?\s*(?:eurov|euro|eur)\b",
    re.IGNORECASE,
)
PERCENT_RE = re.compile(r"\b\d+(?:,\d+)?\s*%")
DEADLINE_RE = re.compile(
    r"\b(?:rok|najpozneje|najkasneje|do)\b[^.!?\n]{0,80}?\b\d{1,2}\.\s?(?:\d{1,2}\.\s?\d{4}|"
    r"januarja|februarja|marca|aprila|maja|junija|julija|avgusta|septembra|oktobra|novembra|decembra)\b",
    re.IGNORECASE,
)


def _unique(matches: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for match in matches:
        normalized = normalize_text(match)
        if normalized and normalized not in seen:
            ordered.append(normalized)
            seen.add(normalized)
    return ordered


def extract_regex_features(text: str) -> dict[str, Any]:
    normalized = normalize_text(text)
    law_refs = _unique(LAW_REF_RE.findall(normalized))
    act_ids = _unique(ACT_ID_RE.findall(normalized))
    articles = _unique(ARTICLE_RE.findall(normalized))
    paragraphs = _unique(PARAGRAPH_RE.findall(normalized))
    items = _unique(ITEM_RE.findall(normalized))
    dates = _unique(DATE_RE.findall(normalized))
    amounts = _unique(AMOUNT_RE.findall(normalized))
    percentages = _unique(PERCENT_RE.findall(normalized))
    deadlines = _unique(DEADLINE_RE.findall(normalized))
    return {
        "law_refs": law_refs,
        "act_ids": act_ids,
        "articles": articles,
        "paragraphs": paragraphs,
        "items": items,
        "dates": dates,
        "amounts": amounts,
        "percentages": percentages,
        "deadlines": deadlines,
    }


def extract_query_citations(text: str) -> dict[str, list[str]]:
    features = extract_regex_features(text)
    return {
        "law_refs": features["law_refs"] + features["act_ids"],
        "articles": features["articles"],
        "paragraphs": features["paragraphs"],
        "items": features["items"],
    }
