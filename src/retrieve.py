"""Top-k retrieval over the saved FAISS index."""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set

from .build_index import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_INDEX_CHUNKS,
    DEFAULT_INDEX_PATH,
    embedding_texts_for_model,
)


DEFAULT_TOP_K = 3
DEFAULT_RETRIEVAL_MODE = "dense"
HYBRID_RETRIEVAL_MODE = "hybrid"
DEFAULT_CANDIDATE_K = 30
DEFAULT_LEXICAL_WEIGHT = 0.20
DEFAULT_SOURCE_BOOST = 0.18
DEFAULT_ARTICLE_BOOST = 0.35
DEFAULT_TITLE_WEIGHT = 0.25

RETRIEVAL_MODES = {DEFAULT_RETRIEVAL_MODE, HYBRID_RETRIEVAL_MODE}

STOPWORDS = {
    "a",
    "ali",
    "bi",
    "bil",
    "bila",
    "bilo",
    "bo",
    "do",
    "in",
    "iz",
    "je",
    "jih",
    "kaj",
    "kako",
    "katera",
    "katere",
    "kateri",
    "kdo",
    "ki",
    "ko",
    "kot",
    "lahko",
    "na",
    "nad",
    "naj",
    "ne",
    "ni",
    "o",
    "ob",
    "od",
    "po",
    "pri",
    "se",
    "so",
    "sta",
    "su",
    "ta",
    "ter",
    "to",
    "v",
    "za",
    "z",
}


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
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


def normalize_for_matching(text: str) -> str:
    """Lowercase text and remove accents for robust lexical matching."""
    normalized = unicodedata.normalize("NFKD", text.lower())
    without_accents = "".join(
        char for char in normalized if not unicodedata.combining(char)
    )
    return re.sub(r"[^a-z0-9]+", " ", without_accents)


def tokenize_for_matching(text: str) -> List[str]:
    """Tokenize normalized text and drop short/common words."""
    return [
        token
        for token in normalize_for_matching(text).split()
        if len(token) > 1 and token not in STOPWORDS
    ]


def infer_query_law_ids(question: str) -> Set[str]:
    """Infer which legal act is named or strongly implied by a question."""
    normalized = normalize_for_matching(question)
    laws: Set[str] = set()
    if "zddv" in normalized or re.search(r"\bddv\b", normalized):
        laws.add("ZDDV-1")
    if "zdoh" in normalized or "dohodnina" in normalized:
        laws.add("ZDoh-2")
    if "zddpo" in normalized or "dobicek" in normalized or "pravnih oseb" in normalized:
        laws.add("ZDDPO-2")
    if "zdavp" in normalized or "davcni postopek" in normalized:
        laws.add("ZDavP-2")
    return laws


def infer_query_articles(question: str) -> Set[str]:
    """Infer article numbers explicitly named in a question."""
    normalized = normalize_for_matching(question)
    return {
        normalize_article_number(match.group(1))
        for match in re.finditer(r"\b(\d+(?:\s*[a-z])?)\s*clen\b", normalized)
    }


def normalize_article_number(article: Any) -> str:
    """Normalize article ids such as 86.b, 86 b, or 86.b. for matching."""
    return re.sub(r"[^a-z0-9]+", "", str(article).lower())


def chunk_match_text(chunk: Dict[str, Any]) -> str:
    """Return chunk text plus selected metadata used by lexical reranking."""
    metadata = chunk.get("metadata") or {}
    metadata_parts = [
        chunk.get("source") or "",
        metadata.get("law_id") or "",
        metadata.get("article_number") or "",
        metadata.get("article_title") or "",
    ]
    return "\n".join(str(part) for part in metadata_parts) + "\n" + chunk.get("text", "")


def lexical_overlap_score(question_tokens: Iterable[str], chunk: Dict[str, Any]) -> float:
    """Score how much query vocabulary appears in a candidate chunk."""
    query_counts = Counter(question_tokens)
    if not query_counts:
        return 0.0

    chunk_counts = Counter(tokenize_for_matching(chunk_match_text(chunk)))
    overlap = sum(min(count, chunk_counts.get(token, 0)) for token, count in query_counts.items())
    return overlap / max(sum(query_counts.values()), 1)


def title_overlap_score(question_tokens: Iterable[str], chunk: Dict[str, Any]) -> float:
    """Score how much query vocabulary appears in the article title."""
    metadata = chunk.get("metadata") or {}
    title = str(metadata.get("article_title") or "")
    title_tokens = set(tokenize_for_matching(title))
    query_tokens = list(question_tokens)
    if not title_tokens or not query_tokens:
        return 0.0
    overlap = sum(1 for token in query_tokens if token in title_tokens)
    return overlap / max(len(query_tokens), 1)


def law_source_boost(question_law_ids: Set[str], chunk: Dict[str, Any], source_boost: float) -> float:
    """Boost chunks from the legal act named by the question."""
    if not question_law_ids:
        return 0.0
    metadata = chunk.get("metadata") or {}
    return source_boost if metadata.get("law_id") in question_law_ids else 0.0


def article_number_boost(question_articles: Set[str], chunk: Dict[str, Any], article_boost: float) -> float:
    """Boost chunks whose article number is explicitly named by the question."""
    if not question_articles:
        return 0.0
    metadata = chunk.get("metadata") or {}
    article = metadata.get("article_number")
    if article is None:
        return 0.0
    return article_boost if normalize_article_number(article) in question_articles else 0.0


class RetrievalEngine:
    """Reusable FAISS retriever for CLI and evaluation runs."""

    def __init__(
        self,
        index_path: Path = DEFAULT_INDEX_PATH,
        chunks_path: Path = DEFAULT_INDEX_CHUNKS,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    ) -> None:
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

        self.np = np
        self.embedding_model_name = embedding_model
        self.index = faiss.read_index(str(index_path))
        self.chunks = read_jsonl(chunks_path)
        self.model = SentenceTransformer(embedding_model)

        if self.index.ntotal != len(self.chunks):
            raise ValueError(
                f"Index/chunk mismatch: FAISS has {self.index.ntotal} vectors, "
                f"but {chunks_path} contains {len(self.chunks)} chunks."
            )

    def retrieve(
        self,
        question: str,
        top_k: int = DEFAULT_TOP_K,
        retrieval_mode: str = DEFAULT_RETRIEVAL_MODE,
        candidate_k: int = DEFAULT_CANDIDATE_K,
        lexical_weight: float = DEFAULT_LEXICAL_WEIGHT,
        source_boost: float = DEFAULT_SOURCE_BOOST,
        article_boost: float = DEFAULT_ARTICLE_BOOST,
        title_weight: float = DEFAULT_TITLE_WEIGHT,
        query_law_ids: Iterable[str] | None = None,
    ) -> List[Dict[str, Any]]:
        """Return ranked chunks for a question."""
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        if retrieval_mode not in RETRIEVAL_MODES:
            raise ValueError(f"Unsupported retrieval mode: {retrieval_mode}")

        query_text = embedding_texts_for_model(
            [question],
            self.embedding_model_name,
            role="query",
        )
        query = self.model.encode(
            query_text,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        query = self.np.asarray(query, dtype="float32")

        if retrieval_mode == HYBRID_RETRIEVAL_MODE:
            search_k = min(max(top_k, candidate_k), len(self.chunks))
        else:
            search_k = min(top_k, len(self.chunks))

        scores, indices = self.index.search(query, search_k)
        candidates: List[Dict[str, Any]] = []
        question_tokens = tokenize_for_matching(question)
        question_law_ids = infer_query_law_ids(question)
        if query_law_ids:
            question_law_ids.update(str(law_id) for law_id in query_law_ids)
        question_articles = infer_query_articles(question)

        for score, chunk_index in zip(scores[0], indices[0]):
            if chunk_index < 0:
                continue
            chunk = dict(self.chunks[int(chunk_index)])
            dense_score = float(score)
            lexical_score = 0.0
            title_score = 0.0
            boost = 0.0
            article_match_boost = 0.0

            if retrieval_mode == HYBRID_RETRIEVAL_MODE:
                lexical_score = lexical_overlap_score(question_tokens, chunk)
                title_score = title_overlap_score(question_tokens, chunk)
                boost = law_source_boost(question_law_ids, chunk, source_boost)
                article_match_boost = article_number_boost(question_articles, chunk, article_boost)

            chunk["dense_score"] = dense_score
            chunk["lexical_score"] = lexical_score
            chunk["title_score"] = title_score
            chunk["source_boost"] = boost
            chunk["article_boost"] = article_match_boost
            chunk["score"] = (
                dense_score
                + (lexical_weight * lexical_score)
                + (title_weight * title_score)
                + boost
                + article_match_boost
            )
            candidates.append(chunk)

        if retrieval_mode == HYBRID_RETRIEVAL_MODE:
            candidates.sort(key=lambda chunk: chunk.get("score", 0.0), reverse=True)

        results = candidates[:top_k]
        for rank, chunk in enumerate(results, start=1):
            chunk["rank"] = rank
        return results


def retrieve(
    question: str,
    top_k: int = DEFAULT_TOP_K,
    index_path: Path = DEFAULT_INDEX_PATH,
    chunks_path: Path = DEFAULT_INDEX_CHUNKS,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    retrieval_mode: str = DEFAULT_RETRIEVAL_MODE,
    candidate_k: int = DEFAULT_CANDIDATE_K,
    lexical_weight: float = DEFAULT_LEXICAL_WEIGHT,
    source_boost: float = DEFAULT_SOURCE_BOOST,
    article_boost: float = DEFAULT_ARTICLE_BOOST,
    title_weight: float = DEFAULT_TITLE_WEIGHT,
    query_law_ids: Iterable[str] | None = None,
) -> List[Dict[str, Any]]:
    """Return the most relevant chunks for a question."""
    engine = RetrievalEngine(
        index_path=index_path,
        chunks_path=chunks_path,
        embedding_model=embedding_model,
    )
    return engine.retrieve(
        question,
        top_k=top_k,
        retrieval_mode=retrieval_mode,
        candidate_k=candidate_k,
        lexical_weight=lexical_weight,
        source_boost=source_boost,
        article_boost=article_boost,
        title_weight=title_weight,
        query_law_ids=query_law_ids,
    )
