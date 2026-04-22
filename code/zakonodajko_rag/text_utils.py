from __future__ import annotations

import json
import re
import unicodedata
from hashlib import sha1
from pathlib import Path
from typing import Any, Iterable


WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "")
    text = text.replace("\xa0", " ")
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text


def normalize_multiline(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "")
    lines = [WHITESPACE_RE.sub(" ", line).strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines)


def slugify(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", normalized).strip("-").lower()
    return cleaned or "item"


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalize_header_token(token: str) -> str:
    token = normalize_text(token).lower()
    return re.sub(r"^[^\wčšž]+|[^\wčšž]+$", "", token, flags=re.IGNORECASE)


def header_tokens(text: str) -> list[str]:
    tokens = [normalize_header_token(token) for token in re.split(r"\s+", text)]
    return [token for token in tokens if token]


def stable_id_fragment(value: str, length: int = 8) -> str:
    normalized = unicodedata.normalize("NFC", value or "")
    return sha1(normalized.encode("utf-8")).hexdigest()[:length]
