from zakonodajko_rag.real_eval import extract_furs_qa_pairs, keyword_recall, score_real_eval_row


def test_extract_furs_qa_pairs_from_numbered_document():
    sample = """
    1.1. Kdaj je treba predložiti obračun DDV? (13. 2. 2025)

    ODGOVOR:
    Obračun DDV je treba predložiti najpozneje zadnji delovni dan naslednjega meseca.

    1.2. Ali lahko zavezanec ponovno odda evidenci?

    ODGOVOR:
    Da, če rok še ni potekel.
    """

    rows = extract_furs_qa_pairs(
        sample,
        "DDV kratka vprašanja in odgovori",
        "https://www.fu.gov.si/example.doc",
        "furs_ddv_qna",
    )

    assert len(rows) == 2
    assert rows[0]["query"] == "Kdaj je treba predložiti obračun DDV?"
    assert "najpozneje zadnji delovni dan" in rows[0]["reference_answer"]


def test_extract_furs_qa_pairs_from_vprasanje_document():
    sample = """
    Vprašanje 1: Kaj je poročanje po državah in koga zadeva?

    Poročanje po državah zadeva mednarodne skupine podjetij.

    Vprašanje 2: Kako je CbC poročanje urejeno v slovenski zakonodaji?

    Urejeno je v ZDavP-2 v členih 248.b, 255.i in 255.j.
    """
    rows = extract_furs_qa_pairs(
        sample,
        "CbCR vprašanja in odgovori",
        "https://www.fu.gov.si/cbcr.docx",
        "furs_cbcr_qna",
    )

    assert len(rows) == 2
    assert rows[1]["expected_law_refs"] == ["ZDavP-2"]


def test_keyword_recall_detects_overlap():
    score = keyword_recall(
        "Obračun DDV je treba predložiti najpozneje zadnji delovni dan naslednjega meseca.",
        "Po FURS je obračun DDV treba oddati zadnji delovni dan naslednjega meseca.",
    )

    assert score > 0.4


def test_score_real_eval_row_uses_top_citation_and_reference_overlap():
    row = {
        "expected_source_type": "furs_guidance",
        "expected_source_title": "DDV kratka vprašanja in odgovori",
        "expected_law_refs": ["ZDDV-1"],
        "expected_articles": [],
        "reference_answer": "Obračun DDV je treba predložiti najpozneje zadnji delovni dan naslednjega meseca.",
    }
    payload = {
        "answer": "Po FURS je obračun DDV treba predložiti najpozneje zadnji delovni dan naslednjega meseca.",
        "citations": [
            {
                "source_type": "furs_guidance",
                "title": "DDV kratka vprašanja in odgovori",
                "law_ref": "FURS",
                "article_number": None,
                "article_title": None,
            }
        ],
    }

    metrics = score_real_eval_row(row, payload)

    assert metrics["top_citation_source_type_match"] is True
    assert metrics["top_citation_title_match"] is True
    assert metrics["heuristic_useful"] is True
