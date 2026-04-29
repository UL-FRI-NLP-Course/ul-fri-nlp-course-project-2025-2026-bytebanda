"""Text chunking utilities for the tax RAG pipeline."""

from __future__ import annotations

from typing import Dict, Iterable, List


DEFAULT_CHUNK_SIZE = 1200
DEFAULT_OVERLAP = 200


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


def make_chunks(
    documents: Iterable[Dict],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> List[Dict]:
    """Create chunk records with stable metadata from extracted documents."""
    records: List[Dict] = []
    chunk_counter = 0

    for document in documents:
        text_chunks = chunk_text(
            document.get("text", ""),
            chunk_size=chunk_size,
            overlap=overlap,
        )

        for chunk_number, chunk in enumerate(text_chunks):
            metadata = dict(document.get("metadata") or {})
            metadata.update(
                {
                    "document_type": document.get("document_type"),
                    "chunk_number": chunk_number,
                }
            )

            records.append(
                {
                    "chunk_id": f"chunk_{chunk_counter:06d}",
                    "text": chunk,
                    "source": document.get("source"),
                    "page": document.get("page"),
                    "metadata": metadata,
                }
            )
            chunk_counter += 1

    return records
