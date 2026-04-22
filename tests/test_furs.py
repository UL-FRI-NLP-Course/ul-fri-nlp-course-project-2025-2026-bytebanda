from zakonodajko_rag.furs import (
    is_ddv_technical_furs_entry,
    is_furs_portal_url,
    parse_furs_guidance_index_html,
    parse_furs_portal_page_html,
    parse_furs_topic_resources_html,
    scrape_furs_portal_seed_entries,
    split_furs_sections,
)
from zakonodajko_rag.retrieval import query_source_preference_bonus, should_blend_sparse_without_article_citation


def test_parse_furs_guidance_index_html_extracts_year_kind_and_deduplicates():
    html = """
    <div id="content">
      <div class="accordion">
        <div class="accordion-item">
          <h2>Leto 2025</h2>
          <h3>Pojasnila</h3>
          <ul>
            <li><a href="/fileadmin/ddv_pojasnilo.docx">0920-11069/2025-1: DDV obravnava restavracijskih storitev</a></li>
            <li><a href="/fileadmin/ddv_pojasnilo.docx">0920-11069/2025-1: DDV obravnava restavracijskih storitev</a></li>
          </ul>
        </div>
      </div>
    </div>
    """

    entries = parse_furs_guidance_index_html(html, index_url="https://www.fu.gov.si/navodila_pojasnila_in_smernice")

    assert len(entries) == 1
    assert entries[0]["year"] == 2025
    assert entries[0]["guidance_kind"] == "pojasnila"
    assert entries[0]["download_url"] == "https://www.fu.gov.si/fileadmin/ddv_pojasnilo.docx"


def test_parse_furs_topic_resources_html_extracts_supported_documents_and_deduplicates():
    html = """
    <div id="content">
      <a href="/fileadmin/Opis/Uvoz_pripravljenih_evidenc_preko_portala_eDavki.docx">Uvoz pripravljenih evidenc preko portala eDavki</a>
      <a href="/fileadmin/Opis/Uvoz_pripravljenih_evidenc_preko_portala_eDavki.docx">Uvoz pripravljenih evidenc preko portala eDavki</a>
      <a href="/fileadmin/Opis/Primeri_izpolnjevanja_evidenc.zip">Primeri izpolnjevanja evidenc obračunanega DDV in odbitka DDV</a>
      <a href="/OpenPortal/CommonPages/PageD.aspx?category=kirkpr">Evidenca obračunanega DDV</a>
    </div>
    """

    entries = parse_furs_topic_resources_html(
        html,
        topic_url="https://www.fu.gov.si/davki_in_druge_dajatve/podrocja/davek_na_dodano_vrednost_ddv/",
    )

    assert len(entries) == 2
    assert entries[0]["guidance_kind"] == "ddv_technical"
    assert entries[0]["download_url"].endswith("Uvoz_pripravljenih_evidenc_preko_portala_eDavki.docx")


def test_scrape_furs_portal_seed_entries_extracts_open_portal_pages(monkeypatch):
    html = """
    <div id="content">
      <a href="https://edavki.durs.si/EdavkiPortal/OpenPortal/CommonPages/Opdynp/PageD.aspx?category=kirkpr">Evidenca obračunanega DDV in evidenca odbitka DDV</a>
      <a href="https://edavki.durs.si/EdavkiPortal/OpenPortal/CommonPages/Opdynp/PageC.aspx?category=izpolnjevanje_obveznosti_zavezancev_identificiranih_za_ddv_podjetja">Izpolnjevanje obveznosti davčnih zavezancev</a>
      <a href="https://www.example.com/other">Drugo</a>
    </div>
    """

    class FakeResponse:
        text = html

    monkeypatch.setattr("zakonodajko_rag.furs.requests.get", lambda *args, **kwargs: FakeResponse())
    entries = scrape_furs_portal_seed_entries("https://www.fu.gov.si/davki_in_druge_dajatve/podrocja/davek_na_dodano_vrednost_ddv/")

    assert len(entries) == 2
    assert entries[0]["guidance_kind"] == "portal_ddv"
    assert entries[0]["extension"] == ".html"
    assert entries[0]["source_url"].endswith("category=kirkpr")


def test_parse_furs_portal_page_html_extracts_sections_and_related_links():
    html = """
    <html>
      <head><title>eDavki - Oddaja evidenc obračunanega DDV in odbitka DDV</title></head>
      <body>
        <div id="main-content">
          <span class="h4 eddis">Oddaja evidenc za davčne zavezance.</span>
          <div class="element">
            <span class="c_title" data-ext="Header">Kje in kako</span>
            <p><span data-ext="Body">Oddaja poteka prek eDavkov.<br/>Možen je uvoz ZIP datoteke.</span></p>
          </div>
          <div class="element">
            <span class="c_title" data-ext="Header">Obrazci</span>
            <p><span data-ext="Body">Tehnične specifikacije <a href="PageD.aspx?category=spletni_servis_za_sprejem_kir_kpr">odpri</a></span></p>
          </div>
        </div>
      </body>
    </html>
    """

    parsed = parse_furs_portal_page_html(
        html,
        source_url="https://edavki.durs.si/EdavkiPortal/OpenPortal/CommonPages/Opdynp/PageD.aspx?category=kirkpr",
    )

    assert parsed["title"] == "Oddaja evidenc obračunanega DDV in odbitka DDV"
    assert parsed["sections"][0]["heading"] == "Uvod"
    assert "Oddaja poteka prek eDavkov." in parsed["sections"][1]["text"]
    assert any(link["source_url"].endswith("category=spletni_servis_za_sprejem_kir_kpr") for link in parsed["related_links"])


def test_is_furs_portal_url_accepts_open_portal_pages():
    assert is_furs_portal_url(
        "https://edavki.durs.si/EdavkiPortal/OpenPortal/CommonPages/Opdynp/PageD.aspx?category=kirkpr"
    )
    assert not is_furs_portal_url("https://www.fu.gov.si/fileadmin/Opis/Uvoz_pripravljenih_evidenc_preko_portala_eDavki.docx")


def test_is_ddv_technical_furs_entry_prefers_evidence_and_edavki_documents():
    assert is_ddv_technical_furs_entry(
        {
            "title": "Navodila za vpogled in preverjanje evidenc obračunanega DDV in odbitka DDV v eDavkih",
            "download_url": "https://www.fu.gov.si/fileadmin/Opis/Navodila_za_vpogled_in_preverjanje_evidenc_v_eDavkih.docx",
        }
    )
    assert not is_ddv_technical_furs_entry(
        {
            "title": "DDV obravnava restavracijskih storitev",
            "download_url": "https://www.fu.gov.si/fileadmin/Opis/DDV_obravnava_restavracijskih_storitev.docx",
        }
    )


def test_split_furs_sections_uses_heading_styles():
    elements = [
        {"kind": "paragraph", "text": "DDV obravnava restavracijskih storitev", "style": None},
        {"kind": "paragraph", "text": "Uvodni odstavek dokumenta.", "style": None},
        {"kind": "paragraph", "text": "Obdavčitev jedi, ki se odnesejo s seboj", "style": "Odstavekseznama"},
        {"kind": "paragraph", "text": "Pri dobavi jedi, ki se odnesejo s seboj, gre za dobavo blaga.", "style": None},
        {"kind": "paragraph", "text": "Meniji s pijačo", "style": "Heading2"},
        {"kind": "paragraph", "text": "Pri menijih s pijačo se presoja enotna dobava.", "style": None},
    ]

    sections = split_furs_sections(elements)

    assert len(sections) == 3
    assert sections[0]["heading"] == "Uvod"
    assert sections[1]["heading"] == "Obdavčitev jedi, ki se odnesejo s seboj"
    assert "Pri menijih s pijačo" in sections[2]["text"]


def test_source_aware_query_uses_sparse_blending_for_furs_queries():
    assert should_blend_sparse_without_article_citation("Kako FURS obravnava restavracijske storitve z DDV?") is True
    assert should_blend_sparse_without_article_citation("Katere so stopnje dohodnine?") is False


def test_query_source_preference_bonus_prefers_furs_when_query_explicitly_mentions_furs():
    assert query_source_preference_bonus("Kako FURS obravnava DDV?", {"source_type": "furs_guidance"}) > 0
    assert query_source_preference_bonus("Kako FURS obravnava DDV?", {"source_type": "pisrs"}) < 0
