"""Text chunking utilities for the tax RAG pipeline."""

from __future__ import annotations

import re
from typing import Dict, Iterable, List


DEFAULT_CHUNK_SIZE = 1200
DEFAULT_OVERLAP = 200
DEFAULT_CHUNK_STRATEGY = "fixed"
LEGAL_CHUNK_STRATEGY = "legal"

ARTICLE_HEADING_RE = re.compile(r"^\s*(\d+\.(?:[a-z])?)\s*člen\s*$", re.IGNORECASE)


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> List[str]:
    """Split text into overlapping character-based chunks.

    The splitter is intentionally simple for a baseline system. It prefers a
    whitespace split near the end of each chunk when possible, while preserving
    the requested character overlap between adjacent chunks.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0:
        raise ValueError("overlap must be zero or greater")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    text = text.strip()
    if not text:
        return []

    chunks: List[str] = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        if end < len(text):
            split_at = text.rfind(" ", start + (chunk_size // 2), end)
            if split_at > start:
                end = split_at

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        next_start = end - overlap
        start = next_start if next_start > start else start + 1

    return chunks


def source_law_id(source: str | None) -> str | None:
    """Infer the main legal act identifier from a source filename."""
    if not source:
        return None

    lowered = source.lower()
    if "zddv-1" in lowered:
        return "ZDDV-1"
    if "zdoh-2" in lowered:
        return "ZDoh-2"
    if "zddpo-2" in lowered:
        return "ZDDPO-2"
    if "zdavp-2" in lowered:
        return "ZDavP-2"
    return None


def source_document_role(source: str | None) -> str:
    """Classify a source as a statute, rulebook, or other document."""
    if not source:
        return "unknown"
    return "rulebook" if source.upper().startswith("PRAV") else "statute"


def article_sections(text: str) -> List[Dict]:
    """Split legal text into article sections when article headings are present."""
    lines = text.splitlines()
    starts = []

    for index, line in enumerate(lines):
        match = ARTICLE_HEADING_RE.match(line)
        if match:
            starts.append((index, match.group(1).rstrip(".")))

    if not starts:
        return [{"text": text.strip(), "section_type": "document"}] if text.strip() else []

    sections: List[Dict] = []
    first_start = starts[0][0]
    preamble = "\n".join(lines[:first_start]).strip()
    if preamble:
        sections.append({"text": preamble, "section_type": "preamble"})

    for position, (start, article_number) in enumerate(starts):
        end = starts[position + 1][0] if position + 1 < len(starts) else len(lines)
        section_lines = lines[start:end]
        article_title = None
        if len(section_lines) > 1 and section_lines[1].strip().startswith("("):
            article_title = section_lines[1].strip()

        section_text = "\n".join(section_lines).strip()
        if section_text:
            sections.append(
                {
                    "text": section_text,
                    "section_type": "article",
                    "article_number": article_number,
                    "article_title": article_title,
                }
            )

    return sections


def chunk_legal_document(
    document: Dict,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> List[Dict]:
    """Create article-aware chunks for one legal document."""
    source = document.get("source")
    law_id = source_law_id(source)
    document_role = source_document_role(source)
    chunks: List[Dict] = []

    for section_number, section in enumerate(article_sections(document.get("text", ""))):
        section_text = section["text"]
        section_chunks = chunk_text(section_text, chunk_size=chunk_size, overlap=overlap)

        for section_part, chunk in enumerate(section_chunks):
            article_number = section.get("article_number")
            article_title = section.get("article_title")
            prefix_parts = []
            if law_id:
                prefix_parts.append(f"Law: {law_id}")
            if article_number:
                prefix_parts.append(f"Article: {article_number}. člen")
            if article_title:
                prefix_parts.append(f"Title: {article_title}")

            prefixed_chunk = chunk
            if prefix_parts and section_part > 0:
                prefixed_chunk = " | ".join(prefix_parts) + "\n" + chunk

            metadata = dict(document.get("metadata") or {})
            metadata.update(
                {
                    "document_type": document.get("document_type"),
                    "document_role": document_role,
                    "law_id": law_id,
                    "section_type": section.get("section_type"),
                    "section_number": section_number,
                    "section_part": section_part,
                    "chunking_strategy": LEGAL_CHUNK_STRATEGY,
                }
            )
            if article_number:
                metadata["article_number"] = article_number
            if article_title:
                metadata["article_title"] = article_title

            chunks.append(
                {
                    "text": prefixed_chunk,
                    "source": source,
                    "page": document.get("page"),
                    "metadata": metadata,
                }
            )

    return chunks


def make_chunks(
    documents: Iterable[Dict],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    strategy: str = DEFAULT_CHUNK_STRATEGY,
) -> List[Dict]:
    """Create chunk records with stable metadata from extracted documents."""
    if strategy not in {DEFAULT_CHUNK_STRATEGY, LEGAL_CHUNK_STRATEGY}:
        raise ValueError(f"Unsupported chunking strategy: {strategy}")

    records: List[Dict] = []
    chunk_counter = 0

    for document in documents:
        if strategy == LEGAL_CHUNK_STRATEGY:
            chunk_inputs = chunk_legal_document(
                document,
                chunk_size=chunk_size,
                overlap=overlap,
            )
        else:
            chunk_inputs = []
            for chunk_number, chunk in enumerate(
                chunk_text(
                    document.get("text", ""),
                    chunk_size=chunk_size,
                    overlap=overlap,
                )
            ):
                metadata = dict(document.get("metadata") or {})
                metadata.update(
                    {
                        "document_type": document.get("document_type"),
                        "document_role": source_document_role(document.get("source")),
                        "law_id": source_law_id(document.get("source")),
                        "chunk_number": chunk_number,
                        "chunking_strategy": DEFAULT_CHUNK_STRATEGY,
                    }
                )
                chunk_inputs.append(
                    {
                        "text": chunk,
                        "source": document.get("source"),
                        "page": document.get("page"),
                        "metadata": metadata,
                    }
                )

        for chunk_number, chunk_record in enumerate(chunk_inputs):
            metadata = dict(chunk_record.get("metadata") or {})
            metadata["chunk_number"] = chunk_number
            records.append(
                {
                    "chunk_id": f"chunk_{chunk_counter:06d}",
                    "text": chunk_record.get("text", ""),
                    "source": chunk_record.get("source"),
                    "page": chunk_record.get("page"),
                    "metadata": metadata,
                }
            )
            chunk_counter += 1

    return records
