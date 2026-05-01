"""Document ingestion and text extraction for local tax documents."""

from __future__ import annotations

import html
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".html", ".htm"}
TEXT_ENCODINGS = ("utf-8", "utf-8-sig", "cp1250", "latin-1")


def clean_text(text: str) -> str:
    """Normalize extracted text while keeping paragraph boundaries readable."""
    text = html.unescape(text).replace("\u00a0", " ")
    lines = []
    for line in text.splitlines():
        normalized = re.sub(r"[ \t\r\f\v]+", " ", line).strip()
        if normalized:
            lines.append(normalized)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def read_text_file(path: Path) -> str:
    """Read text with common encodings used by Slovenian legal documents."""
    for encoding in TEXT_ENCODINGS:
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="replace")


def relative_raw_path(path: Path, raw_dir: Path) -> str:
    """Return a reproducible raw-data-relative path for metadata."""
    try:
        return path.relative_to(raw_dir).as_posix()
    except ValueError:
        return path.name


def base_metadata(path: Path, raw_dir: Path) -> Dict:
    """Build common metadata for extracted documents."""
    return {
        "raw_path": relative_raw_path(path, raw_dir),
        "document_type": path.suffix.lower().lstrip("."),
    }


def extract_text_document(path: Path, raw_dir: Path) -> List[Dict]:
    """Extract a text-like document as a single record."""
    text = clean_text(read_text_file(path))
    if not text:
        return []

    return [
        {
            "text": text,
            "source": path.name,
            "page": None,
            "document_type": path.suffix.lower().lstrip("."),
            "metadata": base_metadata(path, raw_dir),
        }
    ]


def extract_html_document(path: Path, raw_dir: Path) -> List[Dict]:
    """Extract visible text from an HTML document."""
    try:
        from bs4 import BeautifulSoup
    except ImportError as exc:
        raise RuntimeError(
            "HTML ingestion requires beautifulsoup4. Install requirements.txt first."
        ) from exc

    soup = BeautifulSoup(read_text_file(path), "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    metadata = base_metadata(path, raw_dir)
    if soup.title and soup.title.string:
        metadata["title"] = soup.title.string.strip()

    text = clean_text(soup.get_text(separator="\n"))
    if not text:
        return []

    return [
        {
            "text": text,
            "source": path.name,
            "page": None,
            "document_type": "html",
            "metadata": metadata,
        }
    ]


def extract_pdf_document(path: Path, raw_dir: Path) -> List[Dict]:
    """Extract PDF text page by page using pypdf."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("PDF ingestion requires pypdf. Install requirements.txt first.") from exc

    reader = PdfReader(str(path))
    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception as exc:
            raise RuntimeError(f"Could not decrypt PDF {path.name}") from exc

    documents: List[Dict] = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = clean_text(page.extract_text() or "")
        if not text:
            continue

        metadata = base_metadata(path, raw_dir)
        metadata["page"] = page_number

        documents.append(
            {
                "text": text,
                "source": path.name,
                "page": page_number,
                "document_type": "pdf",
                "metadata": metadata,
            }
        )

    return documents


def iter_supported_files(raw_dir: Path) -> Iterable[Path]:
    """Yield supported raw documents recursively in a deterministic order."""
    return sorted(
        path
        for path in raw_dir.rglob("*")
        if path.is_file()
        and not path.name.startswith(".")
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def extract_document(path: Path, raw_dir: Path) -> List[Dict]:
    """Dispatch extraction based on file extension."""
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return extract_text_document(path, raw_dir)
    if suffix in {".html", ".htm"}:
        return extract_html_document(path, raw_dir)
    if suffix == ".pdf":
        return extract_pdf_document(path, raw_dir)
    return []


def load_documents(raw_dir: Optional[Path] = None) -> List[Dict]:
    """Load all supported raw documents from data/raw/."""
    raw_dir = Path(raw_dir or DEFAULT_RAW_DIR)
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"Raw data directory not found: {raw_dir}. Put tax documents into data/raw/."
        )

    files = list(iter_supported_files(raw_dir))
    if not files:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise FileNotFoundError(
            f"No supported files found in {raw_dir}. Add raw documents with one of: {supported}."
        )

    documents: List[Dict] = []
    for path in files:
        try:
            extracted = extract_document(path, raw_dir)
            documents.extend(extracted)
            print(f"Ingested {path.name}: {len(extracted)} document record(s)")
        except Exception as exc:
            print(f"Warning: skipped {path.name}: {exc}", file=sys.stderr)

    if not documents:
        raise ValueError("No text could be extracted from the files in data/raw/.")

    return documents
