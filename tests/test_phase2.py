from zakonodajko_rag.answering import compose_answer_from_results
from zakonodajko_rag.embeddings import prepare_query_for_embedding, resolve_embedding_profile
from zakonodajko_rag.resolver import resolve_referral_results
from zakonodajko_rag.router import build_query_profile, route_query


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
        "percentages": [],
        "amounts": [],
        "dates": [],
        "source_url": "http://example.invalid",
        "source_type": source_type,
    }


def test_router_prefers_furs_for_practical_guidance_query():
    profile = build_query_profile(
        "Kako FURS razlaga DDV pri restavracijskih storitvah?",
        ["kako", "furs", "razlagati", "ddv", "restavracijski", "storitev"],
    )
    route = route_query(profile)

    assert route.intent == "practical_guidance"
    assert route.source_policy == "furs_preferred"
    assert route.allow_generation is True


def test_router_treats_technical_edavki_question_as_practical_guidance():
    profile = build_query_profile(
        "V povezavi z navodili za spletni servis za sprejem evidenc ter enkratno prijavo s protokolom OAuth me zanima, kako pridobim identifikator klienta?",
        ["navodilo", "spletni", "servis", "sprejem", "evidenca", "oauth", "pridobiti", "identifikator", "klient"],
    )
    route = route_query(profile)

    assert route.intent == "practical_guidance"
    assert route.source_policy == "furs_preferred"


def test_referral_resolver_promotes_target_article_for_deadline_query():
    referral_chunk = make_chunk(
        chunk_id="ZAKO4703::336-clen",
        law_id="ZAKO4703",
        title="ZAKON o davčnem postopku (ZDavP-2)",
        article_number="336. člen",
        article_title="(rok predložitve obračuna davčnega odtegljaja in določenih podatkov)",
        section_path="PETI DEL",
        body_text="Glede roka predložitve obračuna davčnega odtegljaja velja 284. člen tega zakona.",
        legal_law_refs=["ZDavP-2"],
    )
    target_chunk = make_chunk(
        chunk_id="ZAKO4703::284-clen",
        law_id="ZAKO4703",
        title="ZAKON o davčnem postopku (ZDavP-2)",
        article_number="284. člen",
        article_title="(rok predložitve obračuna davčnega odtegljaja in določenih podatkov)",
        section_path="PETI DEL",
        body_text="Plačnik davka mora obračun predložiti najpozneje na dan izplačila dohodka.",
        legal_law_refs=["ZDavP-2"],
    )
    chunk_map = {
        referral_chunk["chunk_id"]: referral_chunk,
        target_chunk["chunk_id"]: target_chunk,
    }
    route = route_query(build_query_profile("Kdaj je rok predložitve obračuna davčnega odtegljaja?", ["kdaj", "rok"]))

    resolved_results, chains = resolve_referral_results(
        [{"chunk": referral_chunk, "score": 0.5}],
        chunk_map,
        route,
    )

    assert resolved_results[0]["chunk"]["article_number"] == "284. člen"
    assert chains[0]["from_citation"] == "ZDavP-2, 336. člen (rok predložitve obračuna davčnega odtegljaja in določenih podatkov)"
    assert chains[0]["to_citation"] == "ZDavP-2, 284. člen (rok predložitve obračuna davčnega odtegljaja in določenih podatkov)"


def test_structured_answer_sections_include_resolution_chain():
    chunk = make_chunk(
        chunk_id="ZAKO4703::284-clen",
        law_id="ZAKO4703",
        title="ZAKON o davčnem postopku (ZDavP-2)",
        article_number="284. člen",
        article_title="(rok predložitve obračuna davčnega odtegljaja in določenih podatkov)",
        section_path="PETI DEL",
        body_text="Plačnik davka mora predložiti obračun davčnih odtegljajev najpozneje na dan izplačila dohodka.",
        legal_law_refs=["ZDavP-2"],
    )
    route = route_query(build_query_profile("Kdaj je rok predložitve obračuna davčnega odtegljaja?", ["kdaj", "rok"]))
    payload = compose_answer_from_results(
        "Kdaj je rok predložitve obračuna davčnega odtegljaja?",
        ["kdaj", "rok", "predložitev", "obračun", "davčni", "odtegljaj"],
        [{"chunk": chunk, "score": 0.9}],
        route=route,
        resolution_chain=[
            {
                "from_citation": "ZDavP-2, 336. člen (rok predložitve obračuna davčnega odtegljaja in določenih podatkov)",
                "to_citation": "ZDavP-2, 284. člen (rok predložitve obračuna davčnega odtegljaja in določenih podatkov)",
            }
        ],
    )

    labels = [section["label"] for section in payload["answer_sections"]]
    assert "Kratek odgovor" in labels
    assert "Pravna podlaga" in labels
    assert "Uporabljena napotitev" in labels
    assert "Kratek odgovor:" in payload["answer"]
    assert "Uporabljena napotitev:" in payload["answer"]


def test_e5_embedding_profile_formats_instruction_query():
    profile = resolve_embedding_profile("e5_large_instruct")
    prepared = prepare_query_for_embedding("Katere so stopnje dohodnine?", profile)

    assert prepared.startswith("Instruct:")
    assert "\nQuery: Katere so stopnje dohodnine?" in prepared
