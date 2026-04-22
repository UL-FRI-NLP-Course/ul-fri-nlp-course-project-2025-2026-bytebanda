from __future__ import annotations

import io
import json
import re
import subprocess
import zipfile
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urljoin, urlparse
from xml.etree import ElementTree as ET

import requests
from bs4 import BeautifulSoup, Tag
from pypdf import PdfReader

from .constants import DEFAULT_FURS_DDV_TOPIC_URL, DEFAULT_FURS_GUIDANCE_URL
from .regexes import extract_regex_features
from .text_utils import ensure_parent_dir, normalize_multiline, normalize_text, slugify, stable_id_fragment


WORD_NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
SUPPORTED_FURS_EXTENSIONS = {".doc", ".docx", ".pdf", ".zip"}
TAX_TITLE_HINTS = (
    "davč",
    "ddv",
    "dohod",
    "zddpo",
    "zdoh",
    "zdavp",
    "odtegljaj",
    "olajš",
    "obračun",
    "dividend",
    "obrest",
    "dobič",
    "nepremičn",
    "bonitet",
    "refundacij",
    "terjatev",
    "stečaj",
    "restavracij",
)
TAX_URL_HINTS = (
    "/davki_in_druge_dajatve/",
    "ddv",
    "zddpo",
    "zdoh",
    "zdavp",
    "davcna_",
    "davčni_",
)
DDV_TECHNICAL_TITLE_HINTS = (
    "evidenc",
    "edavk",
    "xml",
    "csv",
    "mini blagajna",
    "miniblagajna",
    "predizpolnitev",
    "uvoz pripravljenih evidenc",
    "vpogled",
    "preverjanje evidenc",
    "stornacije",
    "88.b",
    "88b",
)
DDV_TECHNICAL_URL_HINTS = (
    "evidenc",
    "edavk",
    "xml",
    "csv",
    "miniblagajna",
    "predizpolnitev",
    "stornacije",
    "88b",
)
FURS_PORTAL_HOSTS = ("edavki.durs.si", "beta.edavki.durs.si")
FURS_PORTAL_PATHS = (
    "/EdavkiPortal/OpenPortal/CommonPages/Opdynp/PageC.aspx",
    "/EdavkiPortal/OpenPortal/CommonPages/Opdynp/PageD.aspx",
)
FURS_PORTAL_TITLE_HINTS = (
    "edavki",
    "evidenc",
    "ddv",
    "oddaja",
    "obračun",
    "poročilo",
    "rekapitulacijsko",
    "prevajalnik",
    "tehnična specifikacija",
    "spletni servis",
    "xml",
    "csv",
    "certifikat",
)
FURS_PORTAL_CATEGORY_HINTS = (
    "ddv",
    "kirkpr",
    "prevajalnik_evidenci_ddv",
    "spletni_servis_za_sprejem_kir_kpr",
    "rekapitulacijsko_porocilo",
    "porocilo_o_dobavah",
    "ddv_obracun",
    "porocanje_blago",
    "pavsalno_nadomestilo",
)
REAL_EVAL_HOLDOUT_URLS = {
    "https://www.fu.gov.si/fileadmin/Internet/Davki_in_druge_dajatve/Podrocja/Davek_na_dodano_vrednost/Opis/Kratka_vprasanja_in_odgovori_Evidenca_obracunanega_DDV_in_Evidenca_odbitka_DDV.doc"
}
SECTION_STYLE_RE = re.compile(r"(heading|naslov|odstavekseznama|podnaslov)", re.IGNORECASE)
NUMBERED_HEADING_RE = re.compile(r"^\d+(?:\.\d+)*[.)]?\s+\S")
ALL_CAPS_LINE_RE = re.compile(r"^[A-ZČŠŽ0-9][A-ZČŠŽ0-9 /().,:-]{4,}$")


def scrape_furs_guidance_index(index_url: str = DEFAULT_FURS_GUIDANCE_URL) -> list[dict[str, Any]]:
    html = requests.get(index_url, timeout=30).text
    return parse_furs_guidance_index_html(html, index_url=index_url)


def parse_furs_guidance_index_html(html: str, index_url: str = DEFAULT_FURS_GUIDANCE_URL) -> list[dict[str, Any]]:
    soup = BeautifulSoup(html, "html.parser")
    accordion_items = soup.select("div#content div.accordion-item")
    entries: list[dict[str, Any]] = []
    seen_urls: set[str] = set()

    for item in accordion_items:
        year_heading = item.find("h2")
        year_label = normalize_text(year_heading.get_text(" ", strip=True)) if year_heading else ""
        year = parse_year_label(year_label)
        guidance_kind = "other"
        for child in item.descendants:
            if not isinstance(child, Tag):
                continue
            if child.name == "h3":
                guidance_kind = normalize_guidance_kind(child.get_text(" ", strip=True))
                continue
            if child.name != "a":
                continue
            href = child.get("href")
            title = normalize_text(child.get_text(" ", strip=True))
            if not href or not title:
                continue
            absolute_url = urljoin(index_url, href)
            extension = Path(urlparse(absolute_url).path).suffix.lower()
            if extension and extension not in SUPPORTED_FURS_EXTENSIONS:
                continue
            if absolute_url in seen_urls:
                continue
            entries.append(
                {
                    "title": title,
                    "source_url": absolute_url,
                    "download_url": absolute_url,
                    "year": year,
                    "year_label": year_label,
                    "guidance_kind": guidance_kind,
                    "extension": extension,
                    "source_type": "furs_guidance",
                }
            )
            seen_urls.add(absolute_url)
    return entries


def fetch_and_parse_furs_guidance(
    download_dir: Path,
    index_url: str = DEFAULT_FURS_GUIDANCE_URL,
    min_year: int | None = None,
) -> list[dict[str, Any]]:
    entries = scrape_furs_guidance_index(index_url=index_url)
    relevant_entries = [
        entry
        for entry in entries
        if is_supported_furs_entry(entry) and is_tax_relevant_furs_entry(entry) and is_within_year_window(entry, min_year)
    ]
    manifest_path = download_dir / "guidance_manifest.json"
    ensure_parent_dir(manifest_path)
    manifest_path.write_text(json.dumps(relevant_entries, ensure_ascii=False, indent=2), encoding="utf-8")

    parsed_documents: list[dict[str, Any]] = []
    download_report: list[dict[str, Any]] = []
    for entry in relevant_entries:
        local_path = download_furs_entry(entry, download_dir)
        parsed_documents.extend(parse_downloaded_furs_entry(entry, local_path, download_dir.parent.parent))
        download_report.append({**entry, "local_path": str(local_path.relative_to(download_dir.parent.parent))})

    report_path = download_dir / "download_report.json"
    report_path.write_text(json.dumps(download_report, ensure_ascii=False, indent=2), encoding="utf-8")
    return parsed_documents


def scrape_furs_topic_resources(topic_url: str = DEFAULT_FURS_DDV_TOPIC_URL) -> list[dict[str, Any]]:
    html = requests.get(topic_url, timeout=30).text
    return parse_furs_topic_resources_html(html, topic_url=topic_url)


def parse_furs_topic_resources_html(html: str, topic_url: str = DEFAULT_FURS_DDV_TOPIC_URL) -> list[dict[str, Any]]:
    soup = BeautifulSoup(html, "html.parser")
    entries: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    for link in soup.find_all("a", href=True):
        title = normalize_text(link.get_text(" ", strip=True))
        href = link.get("href")
        if not href or not title:
            continue
        absolute_url = urljoin(topic_url, href)
        extension = Path(urlparse(absolute_url).path).suffix.lower()
        if extension not in SUPPORTED_FURS_EXTENSIONS:
            continue
        if absolute_url in seen_urls:
            continue
        entries.append(
            {
                "title": title,
                "source_url": absolute_url,
                "download_url": absolute_url,
                "year": None,
                "year_label": "",
                "guidance_kind": "ddv_technical",
                "extension": extension,
                "source_type": "furs_guidance",
            }
        )
        seen_urls.add(absolute_url)
    return entries


def fetch_and_parse_furs_ddv_technical_resources(
    download_dir: Path,
    topic_url: str = DEFAULT_FURS_DDV_TOPIC_URL,
    preserve_real_eval_holdout: bool = True,
) -> list[dict[str, Any]]:
    entries = scrape_furs_topic_resources(topic_url=topic_url)
    relevant_entries = [
        entry
        for entry in entries
        if is_supported_furs_entry(entry)
        and is_ddv_technical_furs_entry(entry)
        and (not preserve_real_eval_holdout or entry["download_url"] not in REAL_EVAL_HOLDOUT_URLS)
    ]
    manifest_path = download_dir / "ddv_technical_manifest.json"
    ensure_parent_dir(manifest_path)
    manifest_path.write_text(json.dumps(relevant_entries, ensure_ascii=False, indent=2), encoding="utf-8")

    parsed_documents: list[dict[str, Any]] = []
    download_report: list[dict[str, Any]] = []
    for entry in relevant_entries:
        local_path = download_furs_entry(entry, download_dir)
        parsed_documents.extend(parse_downloaded_furs_entry(entry, local_path, download_dir.parent.parent))
        download_report.append({**entry, "local_path": str(local_path.relative_to(download_dir.parent.parent))})

    report_path = download_dir / "ddv_technical_download_report.json"
    report_path.write_text(json.dumps(download_report, ensure_ascii=False, indent=2), encoding="utf-8")
    return parsed_documents


def fetch_and_parse_furs_portal_resources(
    download_dir: Path,
    topic_url: str = DEFAULT_FURS_DDV_TOPIC_URL,
    max_pages: int = 24,
) -> list[dict[str, Any]]:
    seed_entries = scrape_furs_portal_seed_entries(topic_url=topic_url)
    entries_by_url = {entry["source_url"]: entry for entry in seed_entries}
    queue = [entry["source_url"] for entry in seed_entries]
    visited: set[str] = set()
    parsed_documents: list[dict[str, Any]] = []
    download_report: list[dict[str, Any]] = []

    while queue and len(visited) < max_pages:
        source_url = queue.pop(0)
        if source_url in visited:
            continue
        visited.add(source_url)
        try:
            response = requests.get(source_url, timeout=60)
            response.raise_for_status()
        except requests.RequestException:
            continue
        html = response.text
        parsed = parse_furs_portal_page_html(html, source_url=source_url)
        entry = {
            **entries_by_url.get(source_url, build_furs_portal_entry(source_url=source_url)),
            "title": parsed["title"] or entries_by_url.get(source_url, {}).get("title") or derive_portal_title(source_url),
        }
        local_path = download_furs_entry(entry, download_dir, content=response.content)
        if parsed["sections"]:
            parsed_documents.append(build_furs_document(entry, local_path, download_dir.parent.parent, parsed["sections"]))
            download_report.append({**entry, "local_path": str(local_path.relative_to(download_dir.parent.parent))})

        for related in parsed["related_links"]:
            if len(entries_by_url) >= max_pages:
                break
            if related["source_url"] in entries_by_url:
                continue
            entries_by_url[related["source_url"]] = related
            queue.append(related["source_url"])

    manifest_path = download_dir / "portal_manifest.json"
    ensure_parent_dir(manifest_path)
    manifest_path.write_text(json.dumps(list(entries_by_url.values()), ensure_ascii=False, indent=2), encoding="utf-8")

    report_path = download_dir / "portal_download_report.json"
    report_path.write_text(json.dumps(download_report, ensure_ascii=False, indent=2), encoding="utf-8")
    return parsed_documents


def parse_year_label(label: str) -> int | None:
    match = re.search(r"(20\d{2})", normalize_text(label))
    return int(match.group(1)) if match else None


def normalize_guidance_kind(label: str) -> str:
    lowered = normalize_text(label).lower()
    if "pojasnil" in lowered:
        return "pojasnila"
    if "navodil" in lowered:
        return "navodila"
    if "smernic" in lowered:
        return "eu_smernice"
    return slugify(lowered)


def is_supported_furs_entry(entry: dict[str, Any]) -> bool:
    return (entry.get("extension") or "").lower() in SUPPORTED_FURS_EXTENSIONS


def is_tax_relevant_furs_entry(entry: dict[str, Any]) -> bool:
    title = normalize_text(entry.get("title", "")).lower()
    url = (entry.get("download_url") or "").lower()
    guidance_kind = entry.get("guidance_kind")
    if guidance_kind == "navodila":
        return False
    if guidance_kind == "eu_smernice":
        return "ddv" in title or "ddv" in url or "e-trgovanje" in title
    return any(hint in title for hint in TAX_TITLE_HINTS) or any(hint in url for hint in TAX_URL_HINTS)


def is_ddv_technical_furs_entry(entry: dict[str, Any]) -> bool:
    title = normalize_text(entry.get("title", "")).lower()
    url = (entry.get("download_url") or "").lower()
    return any(hint in title for hint in DDV_TECHNICAL_TITLE_HINTS) or any(hint in url for hint in DDV_TECHNICAL_URL_HINTS)


def scrape_furs_portal_seed_entries(topic_url: str = DEFAULT_FURS_DDV_TOPIC_URL) -> list[dict[str, Any]]:
    html = requests.get(topic_url, timeout=30).text
    soup = BeautifulSoup(html, "html.parser")
    entries: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    for link in soup.find_all("a", href=True):
        title = normalize_text(link.get_text(" ", strip=True))
        absolute_url = urljoin(topic_url, link["href"])
        if not is_furs_portal_url(absolute_url):
            continue
        if absolute_url in seen_urls:
            continue
        lowered_title = title.lower()
        if not title or not (
            any(hint in lowered_title for hint in FURS_PORTAL_TITLE_HINTS)
            or any(hint in absolute_url.lower() for hint in FURS_PORTAL_CATEGORY_HINTS)
        ):
            continue
        entries.append(build_furs_portal_entry(source_url=absolute_url, title=title))
        seen_urls.add(absolute_url)
    return entries


def build_furs_portal_entry(source_url: str, title: str | None = None) -> dict[str, Any]:
    resolved_title = normalize_text(title or derive_portal_title(source_url))
    filename = f"{slugify(resolved_title or derive_portal_category(source_url) or stable_id_fragment(source_url))}.html"
    return {
        "title": resolved_title,
        "source_url": source_url,
        "download_url": source_url,
        "year": None,
        "year_label": "",
        "guidance_kind": "portal_ddv",
        "extension": ".html",
        "filename": filename,
        "source_type": "furs_guidance",
    }


def is_furs_portal_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme in {"http", "https"} and parsed.netloc in FURS_PORTAL_HOSTS and parsed.path in FURS_PORTAL_PATHS


def derive_portal_category(url: str) -> str:
    parsed = urlparse(url)
    category = parse_qs(parsed.query).get("category", [""])[0]
    return normalize_text(category)


def derive_portal_title(url: str) -> str:
    category = derive_portal_category(url)
    if category:
        return f"eDavki - {category.replace('_', ' ')}"
    return normalize_text(Path(urlparse(url).path).name)


def parse_furs_portal_page_html(html: str, source_url: str) -> dict[str, Any]:
    soup = BeautifulSoup(html, "html.parser")
    title = normalize_text(soup.title.get_text(" ", strip=True) if soup.title else "")
    if title.lower().startswith("edavki - "):
        title = title[9:].strip()
    main = soup.select_one("#main-content")
    if main is None:
        return {"title": title or derive_portal_title(source_url), "sections": [], "related_links": []}

    sections: list[dict[str, Any]] = []
    lead = normalize_text(" ".join(node.get_text(" ", strip=True) for node in main.select("span.h4.eddis")[:1]))
    if lead:
        sections.append({"heading": "Uvod", "text": lead})

    for element in main.select("div.element"):
        heading_node = element.select_one("[data-ext='Header']") or element.select_one(".c_title")
        body_node = element.select_one("[data-ext='Body']") or element.find("p")
        heading = normalize_text(heading_node.get_text(" ", strip=True) if heading_node else "")
        body_text = extract_furs_portal_body_text(body_node, source_url) if body_node is not None else ""
        if not body_text:
            continue
        sections.append({"heading": heading or "Podrobnosti", "text": body_text})

    related_links: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    for link in main.select("a[href]"):
        absolute_url = urljoin(source_url, link["href"])
        if not is_furs_portal_url(absolute_url) or absolute_url in seen_urls:
            continue
        label = normalize_text(link.get_text(" ", strip=True)) or derive_portal_title(absolute_url)
        related_links.append(build_furs_portal_entry(source_url=absolute_url, title=label))
        seen_urls.add(absolute_url)

    return {
        "title": title or derive_portal_title(source_url),
        "sections": dedupe_furs_sections(sections),
        "related_links": related_links,
    }


def extract_furs_portal_body_text(node: Tag, source_url: str) -> str:
    text = normalize_multiline(node.get_text("\n", strip=True))
    links: list[str] = []
    seen: set[str] = set()
    for link in node.select("a[href]"):
        absolute_url = urljoin(source_url, link["href"])
        if absolute_url in seen:
            continue
        seen.add(absolute_url)
        label = normalize_text(link.get_text(" ", strip=True)) or Path(urlparse(absolute_url).path).name or absolute_url
        links.append(f"{label}: {absolute_url}")
    if links:
        link_block = "Povezave:\n" + "\n".join(f"- {item}" for item in links)
        text = f"{text}\n{link_block}" if text else link_block
    return normalize_multiline(text)


def dedupe_furs_sections(sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for section in sections:
        heading = normalize_text(section.get("heading", "")) or "Podrobnosti"
        text = normalize_multiline(section.get("text", ""))
        if not text:
            continue
        key = (heading, text)
        if key in seen:
            continue
        deduped.append({"heading": heading, "text": text})
        seen.add(key)
    return deduped


def is_within_year_window(entry: dict[str, Any], min_year: int | None) -> bool:
    if min_year is None:
        return True
    year = entry.get("year")
    return year is not None and int(year) >= min_year


def download_furs_entry(entry: dict[str, Any], download_dir: Path, content: bytes | None = None) -> Path:
    year_label = str(entry.get("year") or "undated")
    extension = entry.get("extension") or ".bin"
    basename = entry.get("filename") or Path(urlparse(entry["download_url"]).path).name or f"{slugify(entry['title'])}{extension}"
    target_path = download_dir / year_label / entry["guidance_kind"] / basename
    if target_path.exists():
        return target_path
    ensure_parent_dir(target_path)
    if content is None:
        response = requests.get(entry["download_url"], timeout=60)
        response.raise_for_status()
        content = response.content
    target_path.write_bytes(content)
    return target_path


def parse_downloaded_furs_entry(entry: dict[str, Any], local_path: Path, repo_root: Path) -> list[dict[str, Any]]:
    suffix = local_path.suffix.lower()
    if suffix == ".html":
        sections = parse_furs_portal_page_html(local_path.read_text(encoding="utf-8", errors="ignore"), entry["source_url"])[
            "sections"
        ]
        return [build_furs_document(entry, local_path, repo_root, sections)]
    if suffix == ".doc":
        sections = split_furs_sections(parse_doc_file(local_path))
        return [build_furs_document(entry, local_path, repo_root, sections)]
    if suffix == ".docx":
        sections = split_furs_sections(parse_docx_file(local_path))
        return [build_furs_document(entry, local_path, repo_root, sections)]
    if suffix == ".pdf":
        sections = split_furs_sections(parse_pdf_file(local_path))
        return [build_furs_document(entry, local_path, repo_root, sections)]
    if suffix == ".zip":
        return parse_zip_guidance(entry, local_path, repo_root)
    return []


def parse_zip_guidance(entry: dict[str, Any], local_path: Path, repo_root: Path) -> list[dict[str, Any]]:
    parsed_documents: list[dict[str, Any]] = []
    with zipfile.ZipFile(local_path) as archive:
        for member in archive.namelist():
            member_path = Path(member)
            if member_path.suffix.lower() not in {".docx", ".pdf"}:
                continue
            content = archive.read(member)
            if member_path.suffix.lower() == ".docx":
                sections = split_furs_sections(parse_docx_bytes(content))
            else:
                sections = split_furs_sections(parse_pdf_bytes(content))
            derived_entry = {
                **entry,
                "title": f"{entry['title']} [{member_path.name}]",
                "download_url": entry["download_url"],
            }
            parsed_documents.append(build_furs_document(derived_entry, local_path, repo_root, sections, zip_member=member))
    return parsed_documents


def parse_docx_file(path: Path) -> list[dict[str, Any]]:
    return parse_docx_bytes(path.read_bytes())


def parse_doc_file(path: Path) -> list[dict[str, Any]]:
    result = subprocess.run(
        ["textutil", "-convert", "txt", "-stdout", str(path)],
        check=True,
        capture_output=True,
    )
    return parse_text_document(result.stdout.decode("utf-8", errors="ignore"))


def parse_docx_bytes(content: bytes) -> list[dict[str, Any]]:
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        root = ET.fromstring(archive.read("word/document.xml"))
    body = root.find("w:body", WORD_NS)
    elements: list[dict[str, Any]] = []
    if body is None:
        return elements
    for child in body:
        tag_name = child.tag.rsplit("}", 1)[-1]
        if tag_name == "p":
            paragraph = parse_docx_paragraph(child)
            if paragraph:
                elements.append(paragraph)
        elif tag_name == "tbl":
            table = parse_docx_table(child)
            if table:
                elements.append(table)
    return elements


def parse_docx_paragraph(node: ET.Element) -> dict[str, Any] | None:
    text = "".join(text_node.text or "" for text_node in node.findall(".//w:t", WORD_NS))
    text = normalize_text(text)
    if not text:
        return None
    style_node = node.find(".//w:pStyle", WORD_NS)
    style = style_node.attrib.get(f"{{{WORD_NS['w']}}}val") if style_node is not None else None
    return {"kind": "paragraph", "text": text, "style": style}


def parse_docx_table(node: ET.Element) -> dict[str, Any] | None:
    rows: list[str] = []
    for row in node.findall(".//w:tr", WORD_NS):
        cells = []
        for cell in row.findall(".//w:tc", WORD_NS):
            cell_text = "".join(text_node.text or "" for text_node in cell.findall(".//w:t", WORD_NS))
            cell_text = normalize_text(cell_text)
            if cell_text:
                cells.append(cell_text)
        if cells:
            rows.append(" | ".join(cells))
    if not rows:
        return None
    return {"kind": "table", "text": "\n".join(rows), "style": "table"}


def parse_pdf_file(path: Path) -> list[dict[str, Any]]:
    return parse_pdf_bytes(path.read_bytes())


def parse_pdf_bytes(content: bytes) -> list[dict[str, Any]]:
    reader = PdfReader(io.BytesIO(content))
    elements: list[dict[str, Any]] = []
    for page in reader.pages:
        page_text = normalize_multiline(page.extract_text() or "")
        if not page_text:
            continue
        for block in page_text.split("\n"):
            text = normalize_text(block)
            if text:
                elements.append({"kind": "paragraph", "text": text, "style": None})
    return elements


def parse_text_document(text: str) -> list[dict[str, Any]]:
    elements: list[dict[str, Any]] = []
    for block in normalize_multiline(text).split("\n"):
        normalized = normalize_text(block)
        if normalized:
            elements.append({"kind": "paragraph", "text": normalized, "style": None})
    return elements


def split_furs_sections(elements: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not elements:
        return []

    doc_title = first_meaningful_line(elements)
    sections: list[dict[str, Any]] = []
    current_heading = "Uvod"
    current_lines: list[str] = []
    seen_first_title = False

    for element in elements:
        text = normalize_text(element.get("text", ""))
        if not text:
            continue
        style = element.get("style")
        if doc_title and text == doc_title and not seen_first_title:
            seen_first_title = True
            continue
        if element.get("kind") == "table":
            current_lines.append(f"Tabela:\n{text}")
            continue
        if is_furs_section_heading(text, style):
            if current_lines:
                sections.append({"heading": current_heading, "text": "\n".join(current_lines).strip()})
            current_heading = text
            current_lines = []
            continue
        current_lines.append(text)

    if current_lines:
        sections.append({"heading": current_heading, "text": "\n".join(current_lines).strip()})

    if not sections:
        full_text = "\n".join(element["text"] for element in elements if normalize_text(element.get("text", ""))).strip()
        sections.append({"heading": "Celotno pojasnilo", "text": full_text})

    return [section for section in sections if normalize_text(section.get("text", ""))]


def first_meaningful_line(elements: list[dict[str, Any]]) -> str:
    for element in elements:
        text = normalize_text(element.get("text", ""))
        if text:
            return text
    return ""


def is_furs_section_heading(text: str, style: str | None) -> bool:
    normalized = normalize_text(text)
    if not normalized:
        return False
    lowered_style = (style or "").lower()
    if lowered_style and SECTION_STYLE_RE.search(lowered_style):
        return len(normalized) <= 180
    if len(normalized) > 140:
        return False
    if NUMBERED_HEADING_RE.match(normalized):
        return True
    if normalized.endswith(":") and len(normalized.split()) <= 12:
        return True
    if ALL_CAPS_LINE_RE.match(normalized):
        return True
    return False


def build_furs_document(
    entry: dict[str, Any],
    local_path: Path,
    repo_root: Path,
    sections: list[dict[str, Any]],
    zip_member: str | None = None,
) -> dict[str, Any]:
    title = normalize_text(entry["title"])
    base_identifier = f"{title}|{entry['download_url']}|{zip_member or ''}"
    doc_id = f"FURS::{slugify(title)}-{stable_id_fragment(base_identifier)}"
    source_path = str(local_path.relative_to(repo_root))
    if zip_member:
        source_path = f"{source_path}::{zip_member}"

    return {
        "doc_id": doc_id,
        "law_id": doc_id,
        "title": title,
        "doc_type": f"furs_{entry['guidance_kind']}",
        "source_type": entry.get("source_type", "furs_guidance"),
        "guidance_kind": entry["guidance_kind"],
        "guidance_year": entry.get("year"),
        "source_url": entry["source_url"],
        "selected_npb": 0,
        "available_npbs": [],
        "effective_date": None,
        "html_path": source_path,
        "current_only": True,
        "npb_label": None,
        "document_notes": [],
        "preamble_blocks": [],
        "articles": [],
        "sections": sections,
        "document_regex": extract_regex_features("\n".join(section["text"] for section in sections)),
    }


def build_furs_annotation_units(parsed_documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    for document in parsed_documents:
        sections = document.get("sections") or []
        for index, section in enumerate(sections):
            header_parts = ["FURS", document["title"]]
            if section.get("heading"):
                header_parts.append(section["heading"])
            header = " > ".join(part for part in header_parts if part)
            content = normalize_multiline(section.get("text", ""))
            if not content:
                continue
            combined_text = f"{header}\n{content}".strip()
            units.append(
                {
                    "unit_id": f"{document['doc_id']}::section-{index + 1}",
                    "doc_id": document["doc_id"],
                    "law_id": document["law_id"],
                    "title": document["title"],
                    "section_path": section.get("heading") or "Celotno pojasnilo",
                    "article_number": None,
                    "article_title": None,
                    "chunk_type": "guidance_section",
                    "header_text": header,
                    "content_text": content,
                    "note_flags": {
                        "has_effective_date": False,
                        "has_expiry_date": False,
                        "has_note": False,
                        "has_linked_reference": False,
                        "has_attachment_reference": False,
                        "contains_prenehal_veljati": False,
                    },
                    "source_url": document["source_url"],
                    "selected_npb": 0,
                    "doc_regex": extract_regex_features(combined_text),
                    "source_type": document.get("source_type", "furs_guidance"),
                    "guidance_kind": document.get("guidance_kind"),
                    "guidance_year": document.get("guidance_year"),
                }
            )
    return units
