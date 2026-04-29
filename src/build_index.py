"""Build a FAISS vector index from local tax documents."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .chunking import DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP, make_chunks
from .ingest import DEFAULT_RAW_DIR, load_documents


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROCESSED_CHUNKS = PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"
DEFAULT_INDEX_PATH = PROJECT_ROOT / "data" / "index" / "faiss.index"
DEFAULT_INDEX_CHUNKS = PROJECT_ROOT / "data" / "index" / "chunks.jsonl"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def write_jsonl(records: Iterable[Dict], path: Path) -> None:
    """Write records as UTF-8 JSON Lines."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_index(
    raw_dir: Optional[Path] = None,
    processed_chunks_path: Path = DEFAULT_PROCESSED_CHUNKS,
    index_path: Path = DEFAULT_INDEX_PATH,
    index_chunks_path: Path = DEFAULT_INDEX_CHUNKS,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    batch_size: int = 32,
) -> Dict:
    """Ingest documents, chunk text, embed chunks, and save a FAISS index."""
    try:
        import faiss
        import numpy as np
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "Index building requires faiss-cpu, numpy, and sentence-transformers. "
            "Install requirements.txt first."
        ) from exc

    raw_dir = Path(raw_dir or DEFAULT_RAW_DIR)
    print(f"Loading raw documents from {raw_dir}")
    documents = load_documents(raw_dir)

    chunks = make_chunks(documents, chunk_size=chunk_size, overlap=overlap)
    if not chunks:
        raise ValueError("No chunks were created. Check that raw documents contain extractable text.")

    print(f"Created {len(chunks)} chunk(s) from {len(documents)} document record(s)")
    write_jsonl(chunks, processed_chunks_path)
    write_jsonl(chunks, index_chunks_path)

    print(f"Loading embedding model: {embedding_model}")
    model = SentenceTransformer(embedding_model)
    texts: List[str] = [chunk["text"] for chunk in chunks]

    print("Encoding chunks")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    embeddings = np.asarray(embeddings, dtype="float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))

    print(f"Saved processed chunks to {processed_chunks_path}")
    print(f"Saved FAISS index to {index_path}")
    print(f"Saved indexed chunk metadata to {index_chunks_path}")

    return {
        "documents": len(documents),
        "chunks": len(chunks),
        "index_path": str(index_path),
        "chunks_path": str(index_chunks_path),
    }
