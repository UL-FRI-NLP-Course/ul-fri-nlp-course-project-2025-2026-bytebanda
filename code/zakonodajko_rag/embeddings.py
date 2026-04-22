from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import re


@dataclass(frozen=True)
class EmbeddingProfile:
    name: str
    model_name: str
    collection_name: str
    query_instruction: str | None = None
    max_seq_length: int = 512
    use_eager_attention: bool = False


EMBEDDING_PROFILES: dict[str, EmbeddingProfile] = {
    "bge_m3": EmbeddingProfile(
        name="bge_m3",
        model_name="BAAI/bge-m3",
        collection_name="pisrs_bge_m3",
        max_seq_length=512,
        use_eager_attention=True,
    ),
    "e5_large_instruct": EmbeddingProfile(
        name="e5_large_instruct",
        model_name="intfloat/multilingual-e5-large-instruct",
        collection_name="pisrs_e5_large_instruct",
        query_instruction=(
            "Given a Slovenian tax-law question, retrieve legal and official guidance passages "
            "that answer it."
        ),
        max_seq_length=512,
    ),
}

DEFAULT_EMBEDDING_PROFILE = "bge_m3"


def resolve_embedding_profile(
    embedding_profile: str | None = None,
    embedding_model_name: str | None = None,
) -> EmbeddingProfile:
    if embedding_model_name:
        profile_name = sanitize_profile_name(embedding_profile or embedding_model_name)
        base_profile = EMBEDDING_PROFILES.get(embedding_profile or "", EMBEDDING_PROFILES[DEFAULT_EMBEDDING_PROFILE])
        return EmbeddingProfile(
            name=profile_name,
            model_name=embedding_model_name,
            collection_name=f"pisrs_{profile_name}",
            query_instruction=base_profile.query_instruction if embedding_profile == "e5_large_instruct" else None,
            max_seq_length=base_profile.max_seq_length,
            use_eager_attention=base_profile.use_eager_attention,
        )
    if embedding_profile:
        if embedding_profile not in EMBEDDING_PROFILES:
            raise ValueError(f"Unknown embedding profile: {embedding_profile}")
        return EMBEDDING_PROFILES[embedding_profile]
    return EMBEDDING_PROFILES[DEFAULT_EMBEDDING_PROFILE]


def prepare_query_for_embedding(query: str, profile: EmbeddingProfile) -> str:
    if not profile.query_instruction:
        return query
    return f"Instruct: {profile.query_instruction}\nQuery: {query}"


def prepare_documents_for_embedding(texts: list[str], profile: EmbeddingProfile) -> list[str]:
    _ = profile
    return texts


@lru_cache(maxsize=4)
def load_embedding_model(
    model_name: str,
    max_seq_length: int = 512,
    use_eager_attention: bool = False,
):
    from sentence_transformers import SentenceTransformer

    if use_eager_attention:
        try:
            model = SentenceTransformer(
                model_name,
                device="cpu",
                model_kwargs={"attn_implementation": "eager"},
            )
        except TypeError:
            model = SentenceTransformer(model_name, device="cpu")
    else:
        model = SentenceTransformer(model_name, device="cpu")
    model.max_seq_length = max_seq_length
    return model


def sanitize_profile_name(value: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z]+", "_", value).strip("_").lower()
    return cleaned or DEFAULT_EMBEDDING_PROFILE
