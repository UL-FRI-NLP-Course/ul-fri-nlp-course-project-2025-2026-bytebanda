from pathlib import Path

from zakonodajko_rag.pisrs import build_annotation_units, parse_all_documents
from zakonodajko_rag.regexes import extract_regex_features


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_parse_all_documents_finds_seven_documents():
    docs = parse_all_documents(REPO_ROOT / "downloads" / "pisrs" / "download_report.json")
    assert len(docs) == 7
    assert all(doc["articles"] for doc in docs)


def test_slovene_characters_are_preserved_in_parsed_text():
    docs = parse_all_documents(REPO_ROOT / "downloads" / "pisrs" / "download_report.json")
    zdavp = next(doc for doc in docs if doc["law_id"] == "ZAKO4703")
    unit = next(unit for unit in build_annotation_units([zdavp]) if unit["article_number"] == "1. člen")
    assert "davčnem postopku" in unit["header_text"].lower()
    assert "č" in unit["content_text"]


def test_attachment_references_are_marked_as_deferred():
    docs = parse_all_documents(REPO_ROOT / "downloads" / "pisrs" / "download_report.json")
    pravilnik = next(doc for doc in docs if doc["law_id"] == "PRAV7927")
    assert any(block["block_type"] == "attachment_reference" for article in pravilnik["articles"] for block in article["blocks"])


def test_regex_extraction_captures_articles_dates_amounts_and_percentages():
    sample = "395. člen ZDavP-2 določa rok do 31. januarja 2026 in globo 400 eurov ter stopnjo 16%."
    extracted = extract_regex_features(sample)
    assert "395. člen" in extracted["articles"]
    assert "31. januarja 2026" in extracted["dates"]
    assert "400 eurov" in extracted["amounts"]
    assert "16%" in extracted["percentages"]
