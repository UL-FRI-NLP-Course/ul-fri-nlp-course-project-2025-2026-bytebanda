from __future__ import annotations

import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup, NavigableString, Tag

from .constants import SECTION_KEYS
from .regexes import extract_regex_features
from .text_utils import normalize_multiline, normalize_text, slugify, stable_id_fragment


ARTICLE_NUMBER_RE = re.compile(r"^\d+\.(?:[a-zčšž])?\s*člen$", re.IGNORECASE)

BLOCK_TYPE_MAP = {
    "odstavek": "paragraph",
    "stevilcna_tocka": "numbered_item",
    "crkovna_tocka": "letter_item",
    "crkovna_tocka_za_odstavkom": "letter_item",
    "crkovna_tocka_za_stevilcno_tocko": "letter_item",
    "alinea": "dash_item",
    "alinea_za_odstavkom": "dash_item",
    "alinea_za_stevilcno_tocko": "dash_item",
    "alinea_za_crkovno_tocko": "dash_item",
    "alinea_za_podtocko": "dash_item",
    "priloga": "attachment_reference",
    "slika": "image",
    "tabela": "table",
    "napaka": "inline_warning",
}

SECTION_TYPE_MAP = {
    "del": "del",
    "poglavje": "poglavje",
    "oddelek": "oddelek",
    "pododdelek": "pododdelek",
    "odsek": "odsek",
}


def load_download_report(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_all_documents(download_report_path: Path) -> list[dict[str, Any]]:
    records = load_download_report(download_report_path)
    return [parse_document(record, download_report_path.parent.parent.parent) for record in records]


def parse_document(record: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    html_path = repo_root / record["file_path"]
    html = html_path.read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "html.parser")
    main = soup.select_one("div.mainText")
    if main is None:
        raise ValueError(f"Missing div.mainText in {html_path}")

    section_state = {key: None for key in SECTION_KEYS}
    document_notes: list[dict[str, Any]] = []
    preamble_blocks: list[dict[str, Any]] = []
    articles: list[dict[str, Any]] = []
    current_article: dict[str, Any] | None = None
    title_lines: list[str] = []
    npb_label: str | None = None

    for child in main.children:
        if isinstance(child, NavigableString):
            continue
        if not isinstance(child, Tag):
            continue
        classes = set(child.get("class", []))
        text = normalize_text(child.get_text(" ", strip=True))
        if not text and not child.find("table") and not child.find("img"):
            continue
        if "opozorilo" in classes:
            continue
        if "naslov" in classes:
            if text:
                title_lines.append(text)
            continue
        if "npb" in classes:
            npb_label = text
            continue
        if "navezava-npb" in classes:
            note_block = make_note_block(child)
            if current_article is None:
                document_notes.append(note_block)
            else:
                current_article["note_blocks"].append(note_block)
            continue

        section_key = next((key for key in SECTION_TYPE_MAP if key in classes), None)
        if section_key:
            update_section_state(section_state, section_key, text)
            continue

        if "clen" in classes:
            if ARTICLE_NUMBER_RE.match(text):
                if current_article is not None:
                    articles.append(current_article)
                current_article = {
                    "unit_id": f"{record['law_id']}::{slugify(text)}-{stable_id_fragment(text)}",
                    "article_number": text,
                    "article_title": None,
                    "section_path": current_section_path(section_state),
                    "blocks": [],
                    "note_blocks": [],
                }
            else:
                if current_article is not None and current_article["article_title"] is None:
                    current_article["article_title"] = text
                else:
                    target_blocks = preamble_blocks if current_article is None else current_article["blocks"]
                    target_blocks.append(make_generic_block(child, "article_heading"))
            continue

        block = make_content_block(child)
        if block is None:
            continue
        if current_article is None:
            preamble_blocks.append(block)
        else:
            current_article["blocks"].append(block)

    if current_article is not None:
        articles.append(current_article)

    title = " ".join(title_lines) if title_lines else record["name"]
    doc_type = "pravilnik" if record["law_id"].startswith("PRAV") else "zakon"
    effective_date = None
    for note in document_notes:
        effective_date = note["pairs"].get("Datum začetka uporabe") or effective_date

    parsed = {
        "doc_id": record["law_id"],
        "law_id": record["law_id"],
        "title": normalize_text(title),
        "doc_type": doc_type,
        "source_type": "pisrs",
        "source_url": record["source_url"],
        "selected_npb": record["selected_npb"],
        "available_npbs": record["available_npbs"],
        "effective_date": effective_date,
        "html_path": record["file_path"],
        "current_only": True,
        "npb_label": npb_label,
        "document_notes": document_notes,
        "preamble_blocks": preamble_blocks,
        "articles": articles,
    }
    return parsed


def current_section_path(state: dict[str, str | None]) -> list[str]:
    return [value for key, value in state.items() if value]


def update_section_state(state: dict[str, str | None], section_key: str, text: str) -> None:
    state[section_key] = text
    reset = False
    for key in SECTION_KEYS:
        if reset:
            state[key] = None
        if key == section_key:
            reset = True


def make_note_block(tag: Tag) -> dict[str, Any]:
    pairs: dict[str, str] = {}
    current_label: str | None = None
    buffer: list[str] = []
    for child in tag.children:
        if isinstance(child, NavigableString):
            part = normalize_text(str(child))
            if part:
                buffer.append(part)
            continue
        if not isinstance(child, Tag):
            continue
        if child.name == "strong":
            if current_label is not None:
                pairs[current_label] = normalize_text(" ".join(buffer))
            current_label = normalize_text(child.get_text(" ", strip=True)).rstrip(":")
            buffer = []
        elif child.name == "br":
            continue
        else:
            part = normalize_text(child.get_text(" ", strip=True))
            if part:
                buffer.append(part)
    if current_label is not None:
        pairs[current_label] = normalize_text(" ".join(buffer))
    links = [
        {"text": normalize_text(link.get_text(" ", strip=True)), "href": link.get("href")}
        for link in tag.find_all("a")
    ]
    raw_text = normalize_text(tag.get_text(" ", strip=True))
    return {
        "block_type": "note",
        "text": raw_text,
        "pairs": pairs,
        "links": links,
    }


def make_content_block(tag: Tag) -> dict[str, Any] | None:
    classes = tag.get("class", [])
    block_type = next((BLOCK_TYPE_MAP[class_name] for class_name in classes if class_name in BLOCK_TYPE_MAP), "generic")
    if block_type == "table" or tag.find("table"):
        text = render_table_text(tag.find("table") or tag)
    elif block_type == "image" or tag.find("img"):
        image = tag.find("img")
        text = "[SLIKA: inline image omitted]"
        metadata = {
            "has_inline_image": image is not None,
            "image_src_prefix": (image.get("src", "")[:32] if image else None),
        }
        block = make_generic_block(tag, block_type)
        block["text"] = text
        block["metadata"].update(metadata)
        return block
    else:
        text = normalize_multiline(tag.get_text("\n", strip=True))

    if not text:
        return None

    block = make_generic_block(tag, block_type)
    block["text"] = text
    if block_type == "attachment_reference":
        block["metadata"]["deferred_attachment"] = True
    return block


def make_generic_block(tag: Tag, block_type: str) -> dict[str, Any]:
    classes = [class_name for class_name in tag.get("class", []) if class_name]
    links = [
        {"text": normalize_text(link.get_text(" ", strip=True)), "href": link.get("href")}
        for link in tag.find_all("a")
    ]
    return {
        "block_type": block_type,
        "text": normalize_multiline(tag.get_text("\n", strip=True)),
        "raw_classes": classes,
        "links": links,
        "metadata": {},
    }


def render_table_text(tag: Tag) -> str:
    rows: list[list[str]] = []
    for row in tag.find_all("tr"):
        cells = [normalize_text(cell.get_text(" ", strip=True)) for cell in row.find_all(["th", "td"])]
        cells = [cell for cell in cells if cell]
        if cells:
            rows.append(cells)
    if not rows:
        return normalize_multiline(tag.get_text("\n", strip=True))
    return "\n".join(" | ".join(row) for row in rows)


def build_annotation_units(parsed_documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    for document in parsed_documents:
        preamble = build_preamble_unit(document)
        if preamble is not None:
            units.append(preamble)
        for article in document["articles"]:
            units.append(build_article_unit(document, article))
    return units


def build_preamble_unit(document: dict[str, Any]) -> dict[str, Any] | None:
    body_lines = []
    for note in document["document_notes"]:
        body_lines.append(f"[OPOMBA] {note['text']}")
    for block in document["preamble_blocks"]:
        body_lines.append(format_block(block))
    body_lines = [line for line in body_lines if line]
    if not body_lines:
        return None
    header = f"{document['title']} > Preambula"
    content = "\n".join(body_lines)
    return {
        "unit_id": f"{document['doc_id']}::preamble",
        "doc_id": document["doc_id"],
        "law_id": document["law_id"],
        "title": document["title"],
        "section_path": "Preambula",
        "article_number": None,
        "article_title": None,
        "chunk_type": "preamble",
        "header_text": header,
        "content_text": content,
        "note_flags": note_flags(document["document_notes"], document["preamble_blocks"]),
        "source_url": document["source_url"],
        "selected_npb": document["selected_npb"],
        "doc_regex": extract_regex_features(content),
        "source_type": document.get("source_type", "pisrs"),
        "guidance_kind": document.get("guidance_kind"),
        "guidance_year": document.get("guidance_year"),
    }


def build_article_unit(document: dict[str, Any], article: dict[str, Any]) -> dict[str, Any]:
    path_parts = [document["title"], *article["section_path"], article["article_number"]]
    if article["article_title"]:
        path_parts.append(article["article_title"])
    header = " > ".join(part for part in path_parts if part)
    body_lines = []
    for note in article["note_blocks"]:
        body_lines.append(f"[OPOMBA] {note['text']}")
    for block in article["blocks"]:
        body_lines.append(format_block(block))
    content = "\n".join(line for line in body_lines if line).strip()
    if not content and article["article_title"]:
        content = article["article_title"]
    return {
        "unit_id": article["unit_id"],
        "doc_id": document["doc_id"],
        "law_id": document["law_id"],
        "title": document["title"],
        "section_path": " > ".join(article["section_path"]),
        "article_number": article["article_number"],
        "article_title": article["article_title"],
        "chunk_type": "article",
        "header_text": header,
        "content_text": content,
        "note_flags": note_flags(article["note_blocks"], article["blocks"]),
        "source_url": document["source_url"],
        "selected_npb": document["selected_npb"],
        "doc_regex": extract_regex_features(" ".join([header, content])),
        "source_type": document.get("source_type", "pisrs"),
        "guidance_kind": document.get("guidance_kind"),
        "guidance_year": document.get("guidance_year"),
    }


def format_block(block: dict[str, Any]) -> str:
    text = block["text"]
    block_type = block["block_type"]
    if block_type == "dash_item":
        return f"- {text.lstrip('- ')}"
    if block_type == "table":
        return f"Tabela:\n{text}"
    if block_type == "attachment_reference":
        return f"[PRILOGA] {text}"
    if block_type == "inline_warning":
        return f"[OPOMBA] {text}"
    return text


def note_flags(notes: list[dict[str, Any]], blocks: list[dict[str, Any]]) -> dict[str, bool]:
    note_text = " ".join(note["text"] for note in notes)
    has_effective_date = any("Datum začetka uporabe" in note.get("pairs", {}) for note in notes)
    has_expiry_date = any("Datum konca veljavnosti" in note.get("pairs", {}) for note in notes)
    has_note = bool(notes)
    has_linked_reference = any(note.get("links") for note in notes) or any(block.get("links") for block in blocks)
    has_attachment_reference = any(block["block_type"] == "attachment_reference" for block in blocks)
    return {
        "has_effective_date": has_effective_date,
        "has_expiry_date": has_expiry_date,
        "has_note": has_note,
        "has_linked_reference": has_linked_reference,
        "has_attachment_reference": has_attachment_reference,
        "contains_prenehal_veljati": "prenehal veljati" in note_text.lower(),
    }


def chunk_units(
    units: list[dict[str, Any]],
    annotations_by_unit: dict[str, dict[str, Any]],
    split_trigger_chars: int,
    max_chunk_chars: int,
    overlap_chars: int,
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for unit in units:
        annotation = deepcopy(annotations_by_unit.get(unit["unit_id"], {"sentences": []}))
        unit_chunks = split_annotated_unit(unit, annotation, split_trigger_chars, max_chunk_chars, overlap_chars)
        chunks.extend(unit_chunks)
    return chunks


def split_annotated_unit(
    unit: dict[str, Any],
    annotation: dict[str, Any],
    split_trigger_chars: int,
    max_chunk_chars: int,
    overlap_chars: int,
) -> list[dict[str, Any]]:
    sentences = annotation.get("sentences", [])
    total_chars = len(unit["content_text"])
    if not sentences or total_chars <= split_trigger_chars:
        return [build_chunk_record(unit, annotation, 0, 1)]

    sentence_groups: list[list[dict[str, Any]]] = []
    index = 0
    while index < len(sentences):
        current_group: list[dict[str, Any]] = []
        current_chars = 0
        cursor = index
        while cursor < len(sentences):
            sentence = sentences[cursor]
            sentence_len = len(sentence["text"]) + (1 if current_group else 0)
            if current_group and current_chars + sentence_len > max_chunk_chars:
                break
            current_group.append(sentence)
            current_chars += sentence_len
            cursor += 1
            if current_chars >= max_chunk_chars:
                break
        if not current_group:
            current_group.append(sentences[index])
            cursor = index + 1
        sentence_groups.append(current_group)
        if cursor >= len(sentences):
            break
        overlap_count = 0
        overlap_size = 0
        for sentence in reversed(current_group):
            overlap_size += len(sentence["text"]) + 1
            overlap_count += 1
            if overlap_size >= overlap_chars:
                break
        index = max(index + 1, cursor - overlap_count)

    chunk_count = len(sentence_groups)
    return [
        build_chunk_record(unit, {"sentences": group}, group_index, chunk_count)
        for group_index, group in enumerate(sentence_groups)
    ]


def build_chunk_record(
    unit: dict[str, Any],
    annotation: dict[str, Any],
    piece_index: int,
    piece_count: int,
) -> dict[str, Any]:
    sentences = annotation.get("sentences", [])
    content_text = " ".join(sentence["text"] for sentence in sentences).strip() if sentences else unit["content_text"]
    if not content_text:
        content_text = unit["content_text"]
    header_text = unit["header_text"]
    raw_chunk_text = f"{header_text}\n{content_text}".strip()

    sentence_spans = []
    cursor = 0
    lemma_tokens = []
    entities = []
    pos_tags = []
    dependencies = []
    for sentence_index, sentence in enumerate(sentences):
        sentence_text = sentence["text"]
        start = cursor
        end = start + len(sentence_text)
        sentence_spans.append({"text": sentence_text, "start": start, "end": end})
        cursor = end + 1
        tokens = sentence.get("tokens", [])
        lemma_tokens.extend(token.get("lemma", token.get("text", "")).lower() for token in tokens if token.get("lemma") or token.get("text"))
        pos_tags.append(
            {
                "sentence_index": sentence_index,
                "tokens": [
                    {
                        "text": token.get("text"),
                        "lemma": token.get("lemma"),
                        "upos": token.get("upos"),
                        "xpos": token.get("xpos"),
                    }
                    for token in tokens
                ],
            }
        )
        dependencies.append(
            {
                "sentence_index": sentence_index,
                "tokens": [
                    {
                        "text": token.get("text"),
                        "lemma": token.get("lemma"),
                        "head": token.get("head"),
                        "deprel": token.get("deprel"),
                    }
                    for token in tokens
                ],
            }
        )
        entities.extend(sentence.get("entities", []))

    if not lemma_tokens:
        lemma_tokens = [token.lower() for token in re.findall(r"\w+", raw_chunk_text, flags=re.UNICODE)]
    lemma_chunk_text = " ".join(token for token in lemma_tokens if token)
    regex_features = extract_regex_features(raw_chunk_text)
    suffix = "" if piece_count == 1 else f"-part-{piece_index + 1}"
    return {
        "chunk_id": f"{unit['unit_id']}{suffix}",
        "doc_id": unit["doc_id"],
        "law_id": unit["law_id"],
        "title": unit["title"],
        "section_path": unit["section_path"],
        "article_number": unit["article_number"],
        "article_title": unit["article_title"],
        "raw_chunk_text": raw_chunk_text,
        "lemma_chunk_text": lemma_chunk_text,
        "chunk_type": unit["chunk_type"],
        "sentence_spans": sentence_spans,
        "entities": entities,
        "pos_tags": pos_tags,
        "dependencies": dependencies,
        "legal_refs": {
            "law_refs": regex_features["law_refs"],
            "act_ids": regex_features["act_ids"],
            "articles": regex_features["articles"],
            "paragraphs": regex_features["paragraphs"],
            "items": regex_features["items"],
            "deadlines": regex_features["deadlines"],
        },
        "dates": regex_features["dates"],
        "amounts": regex_features["amounts"],
        "percentages": regex_features["percentages"],
        "note_flags": unit["note_flags"],
        "source_url": unit["source_url"],
        "selected_npb": unit["selected_npb"],
        "source_type": unit.get("source_type", "pisrs"),
        "guidance_kind": unit.get("guidance_kind"),
        "guidance_year": unit.get("guidance_year"),
    }
