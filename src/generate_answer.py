"""Answer generation with local Mistral using retrieved context."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, List

from .build_index import DEFAULT_EMBEDDING_MODEL, DEFAULT_INDEX_CHUNKS, DEFAULT_INDEX_PATH
from .retrieve import (
    DEFAULT_ARTICLE_BOOST,
    DEFAULT_CANDIDATE_K,
    DEFAULT_LEXICAL_WEIGHT,
    DEFAULT_RETRIEVAL_MODE,
    DEFAULT_SOURCE_BOOST,
    DEFAULT_TITLE_WEIGHT,
    DEFAULT_TOP_K,
    retrieve,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SYSTEM_PROMPT = PROJECT_ROOT / "prompts" / "tax_assistant_system_prompt.txt"
DEFAULT_MISTRAL_MODEL_PATH = Path("/d/hpc/projects/onj_fri/models/intent")
DEFAULT_GAMS9_MODEL_PATH = Path("/d/hpc/projects/onj_fri/brainstorm/models/GaMS-9B-Instruct")
DEFAULT_LOCAL_MODEL_PATH = DEFAULT_GAMS9_MODEL_PATH
DEFAULT_CHAT_SYSTEM_PROMPT = (
    "Si prijazen, jasen in koristen asistent. Odgovarjaj v jeziku uporabnika. "
    "Ce nisi preprican, to povej odkrito."
)


def load_system_prompt(path: Path = DEFAULT_SYSTEM_PROMPT) -> str:
    """Load the system prompt used for grounded tax answers."""
    if not path.exists():
        raise FileNotFoundError(f"System prompt not found: {path}")
    return path.read_text(encoding="utf-8").strip()


def looks_slovenian(text: str) -> bool:
    """Lightweight language hint used only for source heading text."""
    lowered = f" {text.lower()} "
    markers = (
        " kaj ",
        " kdaj ",
        " katere ",
        " kateri ",
        " lahko ",
        " mora ",
        " zavezanec ",
        " dohodnina ",
        " davčni ",
        " davcni ",
        " stroške ",
        " stroske ",
    )
    return any(marker in lowered for marker in markers) or any(ch in lowered for ch in "čšž")


def format_context(chunks: Iterable[Dict]) -> str:
    """Format retrieved chunks for the generation prompt."""
    blocks = []
    for chunk in chunks:
        page = chunk.get("page")
        page_text = f", page {page}" if page is not None else ""
        metadata = chunk.get("metadata") or {}
        metadata_parts = []
        if metadata.get("law_id"):
            metadata_parts.append(f"law={metadata.get('law_id')}")
        if metadata.get("document_role"):
            metadata_parts.append(f"role={metadata.get('document_role')}")
        if metadata.get("article_number"):
            metadata_parts.append(f"article={metadata.get('article_number')}. člen")
        if metadata.get("article_title"):
            metadata_parts.append(f"title={metadata.get('article_title')}")
        metadata_text = f", {'; '.join(metadata_parts)}" if metadata_parts else ""
        blocks.append(
            f"[{chunk['rank']}] source={chunk.get('source')}, "
            f"chunk_id={chunk.get('chunk_id')}{page_text}, score={chunk.get('score'):.4f}"
            f"{metadata_text}\n"
            f"{chunk.get('text', '')}"
        )
    return "\n\n".join(blocks)


def build_messages(question: str, chunks: List[Dict], system_prompt: str) -> List[Dict[str, str]]:
    """Build chat messages for a Mistral Instruct model."""
    context = format_context(chunks)
    user_prompt = f"""Retrieved context:
{context}

User question:
{question}

Answer using only the retrieved context. Prefer the chunk whose legal act, article,
and wording directly answer the question. Ignore merely related chunks if they do
not contain the exact rule being asked about. If the retrieved context does not
contain the answer, say that the answer was not found in the provided sources.
Cite source filenames, articles when available, and chunk IDs."""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def strip_source_appendix(answer: str) -> str:
    """Remove deterministic source lists before reusing an answer as chat history."""
    for marker in ("\n\nViri:", "\n\nSources:"):
        if marker in answer:
            return answer.split(marker, 1)[0].strip()
    return answer.strip()


def format_chat_history(
    history: List[Dict[str, str]],
    max_messages: int = 6,
    max_chars_per_message: int = 900,
) -> str:
    """Render recent chat messages as non-authoritative conversation context."""
    if max_messages <= 0:
        return ""

    selected = [
        message
        for message in history
        if message.get("role") in {"user", "assistant"} and message.get("content")
    ][-max_messages:]
    lines = []
    for message in selected:
        role = "User" if message.get("role") == "user" else "Assistant"
        content = strip_source_appendix(str(message.get("content", "")))
        if len(content) > max_chars_per_message:
            content = content[:max_chars_per_message].rstrip() + " ..."
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def build_chat_messages(
    question: str,
    chunks: List[Dict],
    system_prompt: str,
    history: List[Dict[str, str]] | None = None,
    max_history_messages: int = 6,
) -> List[Dict[str, str]]:
    """Build grounded chat messages using retrieved context plus recent chat history."""
    context = format_context(chunks)
    history_text = format_chat_history(history or [], max_messages=max_history_messages)
    history_block = (
        f"\nConversation so far, for resolving references only:\n{history_text}\n"
        if history_text
        else ""
    )
    user_prompt = f"""Retrieved context for the current question:
{context}
{history_block}
Current user question:
{question}

Answer the current user question using only the retrieved context as the factual
source. Use the conversation history only to understand follow-up references, not
as a legal source. Prefer the chunk whose legal act, article, and wording
directly answer the question. If the retrieved context does not contain the
answer, say that the answer was not found in the provided sources. Cite source
filenames, articles when available, and chunk IDs."""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def preferred_torch_dtype(torch_module):
    """Use bfloat16 on supported CUDA devices, then fp16 CUDA, then float32."""
    if torch_module.cuda.is_available():
        major, _minor = torch_module.cuda.get_device_capability()
        if major >= 8 and torch_module.cuda.is_bf16_supported():
            return torch_module.bfloat16
        return torch_module.float16
    return torch_module.float32


def load_llm(model_path: Path = DEFAULT_LOCAL_MODEL_PATH):
    """Load the local Mistral model from the HPC model path."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Answer generation requires torch and transformers. Install requirements.txt first."
        ) from exc

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Local model path not found: {model_path}. "
            "On the HPC, this should point to the GaMS-9B or Mistral model path."
        )

    dtype = preferred_torch_dtype(torch)
    print(f"Loading local LLM from {model_path} with dtype={dtype}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), use_fast=True)
    except Exception as fast_exc:
        print(f"Fast tokenizer failed, falling back to slow tokenizer: {fast_exc}")
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    requested_cache = os.environ.get("RAG_CACHE_IMPLEMENTATION")
    current_cache = getattr(model.generation_config, "cache_implementation", None)
    if requested_cache:
        model.generation_config.cache_implementation = requested_cache
        print(f"Using generation cache_implementation={requested_cache}")
    elif current_cache in {"hybrid", "static"}:
        model.generation_config.cache_implementation = "dynamic"
        print(
            "Using generation cache_implementation=dynamic "
            f"instead of model default {current_cache}"
        )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer, model, torch


def render_prompt(tokenizer, messages: List[Dict[str, str]]) -> str:
    """Render chat messages with the tokenizer template when available."""
    if getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            system_parts = [
                message["content"] for message in messages if message["role"] == "system"
            ]
            non_system_messages = [
                dict(message) for message in messages if message["role"] != "system"
            ]
            if system_parts and non_system_messages and non_system_messages[0]["role"] == "user":
                non_system_messages[0]["content"] = (
                    "\n\n".join(system_parts)
                    + "\n\n"
                    + non_system_messages[0]["content"]
                )
                try:
                    return tokenizer.apply_chat_template(
                        non_system_messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                except Exception:
                    pass

    system_parts = [message["content"] for message in messages if message["role"] == "system"]
    conversation_parts = [
        f"{message['role'].capitalize()}: {message['content']}"
        for message in messages
        if message["role"] != "system"
    ]
    system = "\n\n".join(system_parts)
    conversation = "\n".join(conversation_parts)
    return f"<s>[INST] {system}\n\n{conversation}\nAssistant: [/INST]"


def generate_from_prompt(
    prompt: str,
    tokenizer,
    model,
    torch_module,
    max_new_tokens: int = 512,
) -> str:
    """Generate an answer from a fully rendered prompt."""
    inputs = tokenizer(prompt, return_tensors="pt")
    try:
        first_device = next(model.parameters()).device
        inputs = {key: value.to(first_device) for key, value in inputs.items()}
    except StopIteration:
        pass

    with torch_module.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    prompt_length = inputs["input_ids"].shape[-1]
    generated = output_ids[0][prompt_length:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def generate_chat_reply(
    messages: List[Dict[str, str]],
    tokenizer,
    model,
    torch_module,
    max_new_tokens: int = 512,
) -> str:
    """Generate one assistant reply from direct chat messages."""
    prompt = render_prompt(tokenizer, messages)
    return generate_from_prompt(
        prompt,
        tokenizer,
        model,
        torch_module,
        max_new_tokens=max_new_tokens,
    )


def append_sources(answer: str, chunks: List[Dict], slovenian: bool) -> str:
    """Append deterministic source citations from retrieved chunks."""
    heading = "Viri" if slovenian else "Sources"
    page_label = "stran" if slovenian else "page"
    lines = []
    seen = set()

    for chunk in chunks:
        key = (chunk.get("source"), chunk.get("chunk_id"), chunk.get("page"))
        if key in seen:
            continue
        seen.add(key)

        page = chunk.get("page")
        page_text = f", {page_label} {page}" if page is not None else ""
        lines.append(f"- {chunk.get('source')} ({chunk.get('chunk_id')}{page_text})")

    return f"{answer.strip()}\n\n{heading}:\n" + "\n".join(lines)


def answer_question(
    question: str,
    top_k: int = DEFAULT_TOP_K,
    index_path: Path = DEFAULT_INDEX_PATH,
    chunks_path: Path = DEFAULT_INDEX_CHUNKS,
    model_path: Path = DEFAULT_LOCAL_MODEL_PATH,
    system_prompt_path: Path = DEFAULT_SYSTEM_PROMPT,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    retrieval_mode: str = DEFAULT_RETRIEVAL_MODE,
    candidate_k: int = DEFAULT_CANDIDATE_K,
    lexical_weight: float = DEFAULT_LEXICAL_WEIGHT,
    source_boost: float = DEFAULT_SOURCE_BOOST,
    article_boost: float = DEFAULT_ARTICLE_BOOST,
    title_weight: float = DEFAULT_TITLE_WEIGHT,
    max_new_tokens: int = 512,
) -> str:
    """Retrieve context and generate a grounded answer."""
    chunks = retrieve(
        question,
        top_k=top_k,
        index_path=index_path,
        chunks_path=chunks_path,
        embedding_model=embedding_model,
        retrieval_mode=retrieval_mode,
        candidate_k=candidate_k,
        lexical_weight=lexical_weight,
        source_boost=source_boost,
        article_boost=article_boost,
        title_weight=title_weight,
    )
    if not chunks:
        if looks_slovenian(question):
            return "V indeksu nisem našel relevantnih virov za to vprašanje."
        return "I did not find relevant sources in the index for this question."

    system_prompt = load_system_prompt(system_prompt_path)
    messages = build_messages(question, chunks, system_prompt)
    tokenizer, model, torch_module = load_llm(model_path)
    prompt = render_prompt(tokenizer, messages)
    answer = generate_from_prompt(
        prompt,
        tokenizer,
        model,
        torch_module,
        max_new_tokens=max_new_tokens,
    )
    if not answer.strip():
        answer = (
            "V pridobljenih virih tega podatka ne najdem."
            if looks_slovenian(question)
            else "I did not find this information in the retrieved sources."
        )
    return append_sources(answer, chunks, slovenian=looks_slovenian(question))
