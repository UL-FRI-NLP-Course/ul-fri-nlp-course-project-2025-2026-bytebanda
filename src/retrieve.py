"""Top-k retrieval over the saved FAISS index."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from .build_index import DEFAULT_EMBEDDING_MODEL, DEFAULT_INDEX_CHUNKS, DEFAULT_INDEX_PATH


DEFAULT_TOP_K = 5


def read_jsonl(path: Path) -> List[Dict]:
    """Read JSON Lines records."""
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def ensure_index_exists(index_path: Path, chunks_path: Path) -> None:
    """Raise a helpful error if the index has not been built yet."""
    missing = [str(path) for path in (index_path, chunks_path) if not path.exists()]
    if missing:
        joined = ", ".join(missing)
        raise FileNotFoundError(
            f"Missing retrieval files: {joined}. Build the index with "
            "python -m src.rag_cli --build-index"
        )


def retrieve(
    question: str,
    top_k: int = DEFAULT_TOP_K,
    index_path: Path = DEFAULT_INDEX_PATH,
    chunks_path: Path = DEFAULT_INDEX_CHUNKS,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> List[Dict]:
    """Return the most relevant chunks for a question."""
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    try:
        import faiss
        import numpy as np
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "Retrieval requires faiss-cpu, numpy, and sentence-transformers. "
            "Install requirements.txt first."
        ) from exc

    index_path = Path(index_path)
    chunks_path = Path(chunks_path)
    ensure_index_exists(index_path, chunks_path)

    chunks = read_jsonl(chunks_path)
    if not chunks:
        raise ValueError(f"No chunk metadata found in {chunks_path}")

    index = faiss.read_index(str(index_path))
    if index.ntotal != len(chunks):
        raise ValueError(
            f"Index/chunk mismatch: FAISS has {index.ntotal} vectors, "
            f"but {chunks_path} contains {len(chunks)} chunks."
        )

    model = SentenceTransformer(embedding_model)
    query = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    query = np.asarray(query, dtype="float32")

    search_k = min(top_k, len(chunks))
    scores, indices = index.search(query, search_k)

    results: List[Dict] = []
    for rank, (score, chunk_index) in enumerate(zip(scores[0], indices[0]), start=1):
        if chunk_index < 0:
            continue
        chunk = dict(chunks[int(chunk_index)])
        chunk["rank"] = rank
        chunk["score"] = float(score)
        results.append(chunk)

    return results
