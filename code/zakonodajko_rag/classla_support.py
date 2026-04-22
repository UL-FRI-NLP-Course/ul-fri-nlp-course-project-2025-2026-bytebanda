from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any

from .text_utils import read_jsonl, write_jsonl


def annotate_jsonl(
    input_path: Path,
    output_path: Path,
    classla_python: str | None = None,
) -> None:
    if can_import_classla() and (classla_python is None or Path(classla_python).resolve() == Path(sys.executable).resolve()):
        rows = read_jsonl(input_path)
        write_jsonl(output_path, annotate_records(rows))
        return

    python_bin = classla_python or os.environ.get("CLASSLA_PYTHON_BIN") or sys.executable
    env = os.environ.copy()
    code_dir = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = code_dir + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    command = [
        python_bin,
        "-m",
        "zakonodajko_rag.classla_cli",
        "annotate-jsonl",
        "--input",
        str(input_path),
        "--output",
        str(output_path),
    ]
    subprocess.run(command, check=True, env=env, cwd=Path(__file__).resolve().parents[2])


def lemmatize_query_text(text: str, classla_python: str | None = None) -> list[str]:
    if can_import_classla() and (classla_python is None or Path(classla_python).resolve() == Path(sys.executable).resolve()):
        result = annotate_records([{"unit_id": "__query__", "content_text": text}])[0]
        return lemmas_from_annotation(result)

    python_bin = classla_python or os.environ.get("CLASSLA_PYTHON_BIN") or sys.executable
    env = os.environ.copy()
    code_dir = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = code_dir + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    command = [
        python_bin,
        "-m",
        "zakonodajko_rag.classla_cli",
        "lemmatize-text",
        "--text",
        text,
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True, env=env, cwd=Path(__file__).resolve().parents[2])
    payload = json.loads(completed.stdout)
    return payload["lemmas"]


def lemmatize_query_texts(query_texts: dict[str, str], classla_python: str | None = None) -> dict[str, list[str]]:
    records = [{"unit_id": query_id, "content_text": text} for query_id, text in query_texts.items()]
    if can_import_classla() and (classla_python is None or Path(classla_python).resolve() == Path(sys.executable).resolve()):
        return {row["unit_id"]: lemmas_from_annotation(row) for row in annotate_records(records)}

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        input_path = tmp_path / "query_batch.jsonl"
        output_path = tmp_path / "query_batch_annotations.jsonl"
        write_jsonl(input_path, records)
        annotate_jsonl(input_path, output_path, classla_python=classla_python)
        return {row["unit_id"]: lemmas_from_annotation(row) for row in read_jsonl(output_path)}


def can_import_classla() -> bool:
    try:
        import classla  # noqa: F401
    except Exception:
        return False
    return True


def annotate_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pipeline = load_classla_pipeline()

    annotated_rows = []
    for record in records:
        text = record.get("content_text") or record.get("text") or ""
        doc = pipeline(text)
        sentences = []
        for sentence in doc.sentences:
            tokens = []
            for token in sentence.tokens:
                word = token.words[0]
                tokens.append(
                    {
                        "text": token.text,
                        "lemma": word.lemma,
                        "upos": word.upos,
                        "xpos": word.xpos,
                        "head": word.head,
                        "deprel": word.deprel,
                        "ner": token.ner,
                    }
                )
            sentences.append(
                {
                    "text": sentence.text,
                    "tokens": tokens,
                    "entities": entities_from_tokens(tokens),
                }
            )
        annotated_rows.append(
            {
                **record,
                "sentences": sentences,
            }
        )
    return annotated_rows


@lru_cache(maxsize=1)
def load_classla_pipeline():
    import classla

    classla.download("sl")
    return classla.Pipeline("sl", processors="tokenize,ner,pos,lemma,depparse", use_gpu=False)


def entities_from_tokens(tokens: list[dict[str, Any]]) -> list[dict[str, str]]:
    entities: list[dict[str, str]] = []
    current_type: str | None = None
    current_tokens: list[str] = []
    for token in tokens:
        tag = token.get("ner") or "O"
        text = token.get("text") or ""
        if tag == "O":
            if current_type and current_tokens:
                entities.append({"text": " ".join(current_tokens), "type": current_type})
            current_type = None
            current_tokens = []
            continue
        prefix, _, entity_type = tag.partition("-")
        if prefix in {"B", "S"}:
            if current_type and current_tokens:
                entities.append({"text": " ".join(current_tokens), "type": current_type})
            current_type = entity_type
            current_tokens = [text]
            if prefix == "S":
                entities.append({"text": " ".join(current_tokens), "type": current_type})
                current_type = None
                current_tokens = []
        elif prefix in {"I", "E"} and current_type == entity_type:
            current_tokens.append(text)
            if prefix == "E":
                entities.append({"text": " ".join(current_tokens), "type": current_type})
                current_type = None
                current_tokens = []
        else:
            if current_type and current_tokens:
                entities.append({"text": " ".join(current_tokens), "type": current_type})
            current_type = entity_type
            current_tokens = [text]
    if current_type and current_tokens:
        entities.append({"text": " ".join(current_tokens), "type": current_type})
    return entities


def lemmas_from_annotation(annotation_row: dict[str, Any]) -> list[str]:
    return [
        token.get("lemma", token.get("text", "")).lower()
        for sentence in annotation_row.get("sentences", [])
        for token in sentence.get("tokens", [])
        if token.get("lemma") or token.get("text")
    ]
