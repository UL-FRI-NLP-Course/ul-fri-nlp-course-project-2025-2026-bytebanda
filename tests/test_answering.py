from zakonodajko_rag.answering import compose_answer_from_results, format_short_citation
from zakonodajko_rag.router import QueryRoute, build_query_profile, route_query


def make_chunk(
    *,
    chunk_id: str,
    law_id: str,
    title: str,
    article_number: str | None,
    article_title: str | None,
    section_path: str,
    body_text: str,
    legal_law_refs: list[str],
    percentages: list[str] | None = None,
    source_type: str = "pisrs",
) -> dict:
    return {
        "chunk_id": chunk_id,
        "doc_id": law_id,
        "law_id": law_id,
        "title": title,
        "section_path": section_path,
        "article_number": article_number,
        "article_title": article_title,
        "chunk_type": "article" if article_number else "preamble",
        "raw_chunk_text": f"{title} > {section_path}\n{body_text}",
        "sentence_spans": [{"text": body_text, "start": 0, "end": len(body_text)}],
        "legal_refs": {
            "law_refs": legal_law_refs,
            "act_ids": [],
            "articles": [article_number] if article_number else [],
            "paragraphs": [],
            "items": [],
            "deadlines": [],
        },
        "percentages": percentages or [],
        "amounts": [],
        "dates": [],
        "source_url": "http://example.invalid",
        "source_type": source_type,
    }


def test_extractive_answer_uses_matching_article_and_citation():
    chunk = make_chunk(
        chunk_id="ZAKO4703::395-clen",
        law_id="ZAKO4703",
        title="ZAKON o davčnem postopku (ZDavP-2)",
        article_number="395. člen",
        article_title="(hujši davčni prekrški posameznikov)",
        section_path="ŠESTI DEL KAZENSKE DOLOČBE",
        body_text="Z globo od 400 do 5.000 eurov se kaznuje za prekršek posameznik, če navede neresnične podatke.",
        legal_law_refs=["ZDavP-2"],
    )
    payload = compose_answer_from_results(
        "Kaj določa 395. člen ZDavP-2?",
        ["kaj", "določati", "395", "člen", "zdavp-2"],
        [{"chunk": chunk, "score": 7.5}],
    )

    assert payload["insufficient_evidence"] is False
    assert "395. člen" in payload["answer"]
    assert "Z globo od 400 do 5.000 eurov" in payload["answer"]
    assert payload["citations"][0]["law_ref"] == "ZDavP-2"
    assert payload["citations"][0]["article_number"] == "395. člen"


def test_insufficient_evidence_is_reported_for_weak_support():
    chunk = make_chunk(
        chunk_id="ZAKO4703::1-clen",
        law_id="ZAKO4703",
        title="ZAKON o davčnem postopku (ZDavP-2)",
        article_number="1. člen",
        article_title="(vsebina zakona)",
        section_path="I. SPLOŠNE DOLOČBE",
        body_text="Ta zakon ureja pobiranje davkov in pristojnosti davčnega organa.",
        legal_law_refs=["ZDavP-2"],
    )
    payload = compose_answer_from_results(
        "Kdaj mora zavezanec oddati napoved?",
        ["kdaj", "morati", "zavezanec", "oddati", "napoved"],
        [{"chunk": chunk, "score": 0.2}],
    )

    assert payload["insufficient_evidence"] is True
    assert "nisem našel dovolj neposredne pravne podlage" in payload["answer"].lower()


def test_short_citation_prefers_law_reference_and_article_title():
    chunk = make_chunk(
        chunk_id="ZAKO4697::122-clen",
        law_id="ZAKO4697",
        title="ZAKON o dohodnini (ZDoh-2)",
        article_number="122. člen",
        article_title="(letna davčna osnova)",
        section_path="III. poglavje",
        body_text="Letna davčna osnova je seštevek davčnih osnov.",
        legal_law_refs=["ZDoh-2"],
    )

    assert format_short_citation(chunk) == "ZDoh-2, 122. člen (letna davčna osnova)"


def test_percentage_question_uses_structured_percentages_from_top_chunk():
    top_chunk = make_chunk(
        chunk_id="ZAKO4697::122-clen",
        law_id="ZAKO4697",
        title="ZAKON o dohodnini (ZDoh-2)",
        article_number="122. člen",
        article_title="(stopnje dohodnine)",
        section_path="V. STOPNJE DOHODNINE",
        body_text="(1) Stopnje dohodnine za davčno leto so: tabela z razredi.",
        legal_law_refs=["ZDoh-2"],
        percentages=["16%", "26%", "33%", "39%", "50 %"],
    )
    secondary_chunk = make_chunk(
        chunk_id="ZAKO4697::120-clen",
        law_id="ZAKO4697",
        title="ZAKON o dohodnini (ZDoh-2)",
        article_number="120. člen",
        article_title="(povprečenje)",
        section_path="IV. LETNA DAVČNA OSNOVA",
        body_text="Povprečna stopnja se izračuna ob upoštevanju stopenj dohodnine iz 122. člena tega zakona.",
        legal_law_refs=["ZDoh-2"],
    )

    payload = compose_answer_from_results(
        "Katere so stopnje dohodnine?",
        ["kateri", "stopnja", "dohodnina"],
        [
            {"chunk": top_chunk, "score": 0.016393},
            {"chunk": secondary_chunk, "score": 0.015873},
        ],
    )

    assert payload["insufficient_evidence"] is False
    assert "16%, 26%, 33%, 39% in 50%" in payload["answer"]
    assert payload["citations"][0]["article_number"] == "122. člen"


def test_highest_dohodnina_rate_question_returns_top_bracket_threshold():
    top_chunk = make_chunk(
        chunk_id="ZAKO4697::122-clen",
        law_id="ZAKO4697",
        title="ZAKON o dohodnini (ZDoh-2)",
        article_number="122. člen",
        article_title="(stopnje dohodnine)",
        section_path="V. STOPNJE DOHODNINE",
        body_text=(
            "(1) Stopnje dohodnine za davčno leto so: "
            "Tabela: Če znaša neto letna osnova v eurih | znaša dohodnina v eurih nad | do 8.500,00 | 16% "
            "8.500,00 | 25.000,00 | 1.360,00 | + | 26% | nad | 8.500,00 25.000,00 | 50.000,00 | "
            "5.650,00 | + | 33% | nad | 25.000,00 50.000,00 | 72.000,00 | 13.900,00 | + | 39% | nad | "
            "50.000,00 72.000,00 | 22.480,00 | + | 50% | nad | 72.000,00"
        ),
        legal_law_refs=["ZDoh-2"],
        percentages=["16%", "26%", "33%", "39%", "50 %"],
    )

    payload = compose_answer_from_results(
        "Katera od teh stopenj dohodnin je najvišja in kdaj pademo v njo?",
        ["kateri", "stopnja", "dohodnina", "najvišji", "pasti"],
        [{"chunk": top_chunk, "score": 0.5}],
    )

    assert payload["insufficient_evidence"] is False
    assert "50%" in payload["answer"]
    assert "nad 72.000,00 EUR" in payload["answer"]


def test_highest_dohodnina_rate_query_is_not_routed_as_deadline():
    profile = build_query_profile(
        "Katera od teh stopenj dohodnin je najvišja in kdaj pademo v njo?",
        ["kateri", "stopnja", "dohodnina", "najvišji", "pasti"],
    )
    route = route_query(profile)

    assert profile.expects_deadline is False
    assert profile.asks_bracket_threshold is True
    assert route.intent == "amount_percentage"


def test_definition_question_prefers_definition_excerpt_from_matching_article():
    chunk = make_chunk(
        chunk_id="ZAKO4703::15-clen",
        law_id="ZAKO4703",
        title="ZAKON o davčnem postopku (ZDavP-2)",
        article_number="15. člen",
        article_title="(davčna tajnost)",
        section_path="V. poglavje",
        body_text=(
            "Davčni organ mora kot zaupne varovati podatke, ki jih zavezanec za davek v davčnem postopku posreduje davčnemu organu. "
            "Ne glede na prvi odstavek tega člena se za davčno tajnost ne šteje davčna številka poslovnih subjektov."
        ),
        legal_law_refs=["ZDavP-2"],
    )

    payload = compose_answer_from_results(
        "Kaj pomeni davčna tajnost?",
        ["kaj", "pomeniti", "davčen", "tajnost"],
        [{"chunk": chunk, "score": 0.5}],
    )

    assert payload["insufficient_evidence"] is False
    assert "Davčni organ mora kot zaupne varovati podatke" in payload["answer"]
    assert payload["citations"][0]["article_number"] == "15. člen"


def test_definition_question_prefers_thematic_article_over_generic_glossary():
    glossary_chunk = make_chunk(
        chunk_id="ZAKO4703::7-clen",
        law_id="ZAKO4703",
        title="ZAKON o davčnem postopku (ZDavP-2)",
        article_number="7. člen",
        article_title="(pomen izrazov)",
        section_path="I. SPLOŠNE DOLOČBE",
        body_text=(
            "V tem delu uporabljeni izrazi imajo naslednji pomen: "
            "1. podatki, ki so davčna tajnost: podatki, ki so kot taki opredeljeni v 15. členu ZDavP-2."
        ),
        legal_law_refs=["ZDavP-2"],
    )
    thematic_chunk = make_chunk(
        chunk_id="ZAKO4703::15-clen",
        law_id="ZAKO4703",
        title="ZAKON o davčnem postopku (ZDavP-2)",
        article_number="15. člen",
        article_title="(davčna tajnost)",
        section_path="V. poglavje",
        body_text=(
            "Davčni organ mora kot zaupne varovati podatke, ki jih zavezanec za davek v davčnem postopku "
            "posreduje davčnemu organu, ter druge podatke v zvezi z davčno obveznostjo zavezancev za davek."
        ),
        legal_law_refs=["ZDavP-2"],
    )

    payload = compose_answer_from_results(
        "Kaj pomeni davčna tajnost?",
        ["kaj", "pomeniti", "davčen", "tajnost"],
        [
            {"chunk": glossary_chunk, "score": 0.92},
            {"chunk": thematic_chunk, "score": 0.88},
        ],
    )

    assert payload["citations"][0]["article_number"] == "15. člen"
    assert "Davčni organ mora kot zaupne varovati podatke" in payload["answer"]


def test_explicit_article_summary_covers_multiple_numbered_items():
    chunk = make_chunk(
        chunk_id="ZAKO4703::395-clen",
        law_id="ZAKO4703",
        title="ZAKON o davčnem postopku (ZDavP-2)",
        article_number="395. člen",
        article_title="(hujši davčni prekrški posameznikov)",
        section_path="ŠESTI DEL KAZENSKE DOLOČBE",
        body_text=(
            "Z globo od 400 do 5.000 eurov se kaznuje za prekršek posameznik, če: "
            "1. v davčni napovedi navede neresnične ali nepopolne podatke; "
            "2. v nasprotju z zakonom razkrije podatke, ki so davčna tajnost; "
            "3. ne predloži obračuna v predpisanem roku."
        ),
        legal_law_refs=["ZDavP-2"],
    )

    payload = compose_answer_from_results(
        "Kaj določa 395. člen ZDavP-2?",
        ["kaj", "določati", "395", "člen", "zdavp-2"],
        [{"chunk": chunk, "score": 7.5}],
    )

    assert payload["insufficient_evidence"] is False
    assert "Tema člena: hujši davčni prekrški posameznikov." in payload["answer"]
    assert "davčna tajnost" in payload["answer"]


def test_short_citation_for_furs_guidance_uses_furs_and_document_title():
    chunk = make_chunk(
        chunk_id="FURS::ddv-restavracije",
        law_id="FURS::ddv-restavracije",
        title="DDV obravnava restavracijskih storitev",
        article_number=None,
        article_title=None,
        section_path="Obdavčitev jedi, ki se odnesejo s seboj (take-away meniji)",
        body_text="Pri dobavi jedi, ki se odnesejo s seboj, gre za dobavo blaga.",
        legal_law_refs=["ZDDV-1"],
        source_type="furs_guidance",
    )

    assert format_short_citation(chunk) == (
        "FURS, DDV obravnava restavracijskih storitev "
        "(Obdavčitev jedi, ki se odnesejo s seboj (take-away meniji))"
    )


def test_practical_guidance_prefers_furs_citation_and_section_order():
    legal_chunk = make_chunk(
        chunk_id="ZAKO4701::85-b-clen",
        law_id="ZAKO4701",
        title="ZAKON o davku na dodano vrednost (ZDDV-1)",
        article_number="85.b člen",
        article_title="(obveznost predložitve evidence obračunanega DDV in evidence odbitka DDV)",
        section_path="X. OBVEZNOSTI DAVČNIH ZAVEZANCEV",
        body_text="Davčni zavezanec mora davčnemu organu predložiti evidenci DDV.",
        legal_law_refs=["ZDDV-1"],
    )
    furs_chunk = make_chunk(
        chunk_id="FURS::evidenci-ddv",
        law_id="FURS::evidenci-ddv",
        title="Oddaja evidenc DDV prek eDavkov",
        article_number=None,
        article_title=None,
        section_path="Vprašanja in odgovori",
        body_text="FURS pojasnjuje, da se evidenci oddajata prek spletnega servisa eDavki v predpisani XML strukturi.",
        legal_law_refs=["ZDDV-1"],
        source_type="furs_guidance",
    )

    payload = compose_answer_from_results(
        "V povezavi z navodili za spletni servis za sprejem evidenc me zanima, kako se evidenci oddajata v eDavkih?",
        ["navodilo", "spletni", "servis", "sprejem", "evidenca", "edavki", "oddajati", "xml"],
        [
            {"chunk": legal_chunk, "score": 0.92},
            {"chunk": furs_chunk, "score": 0.9},
        ],
        route=QueryRoute(
            intent="practical_guidance",
            source_policy="furs_preferred",
            answer_style="guided_explanation",
            allow_generation=True,
            follow_referrals=True,
        ),
    )

    assert payload["citations"][0]["source_type"] == "furs_guidance"
    labels = [section["label"] for section in payload["answer_sections"]]
    assert labels[:3] == ["Kratek odgovor", "Pojasnilo FURS", "Pravna podlaga"]
    assert "eDavki" in payload["answer"]


def test_deadline_question_prefers_substantive_deadline_over_cross_reference():
    direct_chunk = make_chunk(
        chunk_id="ZAKO4703::284-clen",
        law_id="ZAKO4703",
        title="ZAKON o davčnem postopku (ZDavP-2)",
        article_number="284. člen",
        article_title="(rok predložitve obračuna davčnega odtegljaja in določenih podatkov)",
        section_path="PETI DEL",
        body_text=(
            "Plačnik davka mora predložiti obračun davčnih odtegljajev davčnemu organu najpozneje na dan "
            "izplačila dohodka. V istem roku mora podatke predložiti tudi davčnemu zavezancu."
        ),
        legal_law_refs=["ZDavP-2"],
    )
    referral_chunk = make_chunk(
        chunk_id="ZAKO4703::336-clen",
        law_id="ZAKO4703",
        title="ZAKON o davčnem postopku (ZDavP-2)",
        article_number="336. člen",
        article_title="(rok predložitve obračuna davčnega odtegljaja in določenih podatkov)",
        section_path="PETI DEL",
        body_text="Glede roka predložitve obračuna davčnega odtegljaja in določenih podatkov davčnemu zavezancu, velja 284. člen tega zakona.",
        legal_law_refs=["ZDavP-2"],
    )

    payload = compose_answer_from_results(
        "Kdaj je rok predložitve obračuna davčnega odtegljaja?",
        ["kdaj", "rok", "predložitev", "obračun", "davčni", "odtegljaj"],
        [
            {"chunk": direct_chunk, "score": 0.5},
            {"chunk": referral_chunk, "score": 0.49},
        ],
    )

    assert payload["insufficient_evidence"] is False
    assert "najpozneje na dan izplačila dohodka" in payload["answer"]
    assert payload["citations"][0]["article_number"] == "284. člen"
