"""Answer generation with local Mistral using retrieved context."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

from .retrieve import DEFAULT_TOP_K, retrieve


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SYSTEM_PROMPT = PROJECT_ROOT / "prompts" / "tax_assistant_system_prompt.txt"
DEFAULT_LOCAL_MODEL_PATH = Path("/d/hpc/projects/onj_fri/models/intent")


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
        blocks.append(
            f"[{chunk['rank']}] source={chunk.get('source')}, "
            f"chunk_id={chunk.get('chunk_id')}{page_text}, score={chunk.get('score'):.4f}\n"
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

Answer using only the retrieved context. If the retrieved context does not contain
the answer, say that the answer was not found in the provided sources. Cite source
filenames and chunk IDs."""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def preferred_torch_dtype(torch_module):
    """Use bfloat16 on supported CUDA devices, then fp16 CUDA, then float32."""
    if torch_module.cuda.is_available() and torch_module.cuda.is_bf16_supported():
        return torch_module.bfloat16
    if torch_module.cuda.is_available():
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
            "On the HPC, this should point to /d/hpc/projects/onj_fri/models/intent."
        )

    dtype = preferred_torch_dtype(torch)
    print(f"Loading local LLM from {model_path} with dtype={dtype}")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
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
            pass

    system = messages[0]["content"]
    user = messages[1]["content"]
    return f"<s>[INST] {system}\n\n{user} [/INST]"


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
    model_path: Path = DEFAULT_LOCAL_MODEL_PATH,
    system_prompt_path: Path = DEFAULT_SYSTEM_PROMPT,
    embedding_model: str | None = None,
    max_new_tokens: int = 512,
) -> str:
    """Retrieve context and generate a grounded answer."""
    retrieve_kwargs = {"top_k": top_k}
    if embedding_model:
        retrieve_kwargs["embedding_model"] = embedding_model

    chunks = retrieve(question, **retrieve_kwargs)
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
