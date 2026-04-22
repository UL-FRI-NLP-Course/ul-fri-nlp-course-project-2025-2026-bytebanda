from pathlib import Path

from fastapi.testclient import TestClient

from zakonodajko_rag.answering import QueryUnderstanding
from zakonodajko_rag.planner import LegalAgentPlan
from zakonodajko_rag.webapp import WebAppSettings, contextualize_chat_query, create_app


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_contextualize_chat_query_uses_previous_user_turn_for_follow_up():
    query = contextualize_chat_query(
        "Kaj pa za pravno osebo?",
        [
            {"role": "user", "content": "Kakšna je globa za hujši davčni prekršek posameznika?"},
            {
                "role": "assistant",
                "content": "Po 395. členu ...",
                "memory_topic": "ZDavP-2, 395. člen (hujši davčni prekrški posameznikov)",
            },
        ],
    )

    assert "Prejšnje vprašanje:" in query
    assert "Kakšna je globa za hujši davčni prekršek posameznika?" in query
    assert "Zadnja pravna tema: ZDavP-2, 395. člen (hujši davčni prekrški posameznikov)" in query
    assert "Kaj pa za pravno osebo?" in query


def test_contextualize_chat_query_keeps_standalone_question_independent():
    query = contextualize_chat_query(
        "Katere so stopnje dohodnine?",
        [
            {"role": "user", "content": "Kaj določa 395. člen ZDavP-2?"},
            {"role": "assistant", "content": "Po 395. členu ..."},
        ],
    )

    assert query == "Katere so stopnje dohodnine?"


def test_contextualize_chat_query_rewrites_explanatory_follow_up_to_previous_topic():
    query = contextualize_chat_query(
        "Razloži mi prejšnje vprašanje bolj podrobno",
        [
            {"role": "user", "content": "Kaj pomeni davčna tajnost?"},
            {
                "role": "assistant",
                "content": "Po 15. členu ...",
                "memory_topic": "ZDavP-2, 15. člen (davčna tajnost)",
            },
        ],
    )

    assert "Podrobneje razloži vprašanje: Kaj pomeni davčna tajnost?" in query
    assert "Pravna tema: ZDavP-2, 15. člen (davčna tajnost)" in query
    assert "Navodilo uporabnika: Razloži mi prejšnje vprašanje bolj podrobno" in query


def test_contextualize_chat_query_recognizes_short_expansion_request():
    query = contextualize_chat_query(
        "Daj mi daljši odgovor",
        [
            {"role": "user", "content": "Kaj pomeni davčna tajnost?"},
            {
                "role": "assistant",
                "content": "Po 15. členu ...",
                "memory_topic": "ZDavP-2, 15. člen (davčna tajnost)",
            },
        ],
    )

    assert "Podrobneje razloži vprašanje: Kaj pomeni davčna tajnost?" in query
    assert "Pravna tema: ZDavP-2, 15. člen (davčna tajnost)" in query
    assert "Navodilo uporabnika: Daj mi daljši odgovor" in query


def test_contextualize_chat_query_recognizes_pronoun_based_tax_bracket_follow_up():
    query = contextualize_chat_query(
        "Katera od teh stopenj dohodnin je najvišja in kdaj pademo v njo?",
        [
            {"role": "user", "content": "Katere so stopnje dohodnine?"},
            {
                "role": "assistant",
                "content": "Po 122. členu ...",
                "memory_topic": "ZDoh-2, 122. člen (stopnje dohodnine)",
            },
        ],
    )

    assert "Prejšnje vprašanje: Katere so stopnje dohodnine?" in query
    assert "Zadnja pravna tema: ZDoh-2, 122. člen (stopnje dohodnine)" in query
    assert "Trenutno vprašanje: Katera od teh stopenj dohodnin je najvišja in kdaj pademo v njo?" in query


def test_chat_endpoint_returns_answer_payload_without_real_models():
    def fake_lemmatize(text: str, _classla_python: str | None = None) -> list[str]:
        return text.lower().split()

    def fake_answer(*args, **kwargs):
        return {
            "answer": "To je testni odgovor.",
            "citations": [
                {
                    "law_id": "ZAKO4703",
                    "law_ref": "ZDavP-2",
                    "title": "ZAKON o davčnem postopku (ZDavP-2)",
                    "section_path": "I. DEL",
                    "article_number": "1. člen",
                    "article_title": "(vsebina zakona)",
                    "source_url": "http://pisrs.si/Pis.web/pregledPredpisa?id=ZAKO4703",
                    "chunk_id": "ZAKO4703::1-clen",
                }
            ],
            "used_chunks": [],
            "supporting_sentences": [],
            "insufficient_evidence": False,
            "backend": "extractive",
        }

    app = create_app(
        WebAppSettings(
            chunks_path=REPO_ROOT / "artifacts" / "retrieval" / "chunks.jsonl",
            chroma_dir=REPO_ROOT / "artifacts" / "retrieval" / "chroma",
            bm25_path=REPO_ROOT / "artifacts" / "retrieval" / "bm25_corpus.json",
        ),
        answer_pipeline=fake_answer,
        lemmatize_fn=fake_lemmatize,
    )
    client = TestClient(app)
    response = client.post(
        "/api/chat",
        json={"message": "Kaj določa 1. člen ZDavP-2?", "history": []},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["message"] == "To je testni odgovor."
    assert payload["citations"][0]["law_ref"] == "ZDavP-2"
    assert payload["contextualized"] is False
    assert payload["memory_topic"] == "ZDavP-2, 1. člen (vsebina zakona)"


def test_chat_endpoint_uses_query_understanding_rewrite_when_confident():
    captured: dict[str, str] = {}

    def fake_lemmatize(text: str, _classla_python: str | None = None) -> list[str]:
        captured["lemmatized_query"] = text
        return text.lower().split()

    def fake_answer(query: str, *args, **kwargs):
        captured["answer_query"] = query
        return {
            "answer": "Prepisan odgovor.",
            "citations": [],
            "used_chunks": [],
            "supporting_sentences": [],
            "insufficient_evidence": False,
            "backend": "extractive",
        }

    def fake_query_understanding(message: str, history: list[dict], _generator_model: str | None, _max_new_tokens: int):
        return QueryUnderstanding(
            standalone_query="Po ZDoh-2, 122. členu: katera je najvišja stopnja dohodnine in od katere neto letne osnove velja?",
            intent="amount_percentage",
            use_context=True,
            confidence=0.92,
            reason="Follow-up na predhodne stopnje dohodnine.",
        )

    def fake_planner(query: str, history: list[dict], _generator_model: str | None, _max_new_tokens: int):
        return LegalAgentPlan(
            query=query,
            intent="amount_percentage",
            source_policy="pisrs_first",
            answer_style="structured_rule",
            follow_referrals=True,
            preserve_origin_article=False,
            actions=("retrieve_chunks", "resolve_referrals", "prefer_pisrs", "compose_structured_answer", "verify_primary_citation"),
            confidence=0.9,
            reason="Query o stopnji dohodnine.",
            backend="local_model",
        )

    app = create_app(
        WebAppSettings(
            chunks_path=REPO_ROOT / "artifacts" / "retrieval" / "chunks.jsonl",
            chroma_dir=REPO_ROOT / "artifacts" / "retrieval" / "chroma",
            bm25_path=REPO_ROOT / "artifacts" / "retrieval" / "bm25_corpus.json",
            generator_model="fake-local-model",
        ),
        answer_pipeline=fake_answer,
        lemmatize_fn=fake_lemmatize,
        query_understanding_fn=fake_query_understanding,
        agent_planner_fn=fake_planner,
    )
    client = TestClient(app)
    response = client.post(
        "/api/chat",
        json={
            "message": "Katera od teh stopenj dohodnin je najvišja in kdaj pademo v njo?",
            "history": [
                {"role": "user", "content": "Katere so stopnje dohodnine?"},
                {"role": "assistant", "content": "Po 122. členu ...", "memory_topic": "ZDoh-2, 122. člen (stopnje dohodnine)"},
            ],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert captured["answer_query"].startswith("Po ZDoh-2, 122. členu")
    assert payload["retrieval_query"].startswith("Po ZDoh-2, 122. členu")
    assert payload["query_understanding"]["intent"] == "amount_percentage"


def test_chat_endpoint_falls_back_to_heuristic_context_when_query_understanding_is_low_confidence():
    captured: dict[str, str] = {}

    def fake_lemmatize(text: str, _classla_python: str | None = None) -> list[str]:
        captured["lemmatized_query"] = text
        return text.lower().split()

    def fake_answer(query: str, *args, **kwargs):
        captured["answer_query"] = query
        return {
            "answer": "Heuristični odgovor.",
            "citations": [],
            "used_chunks": [],
            "supporting_sentences": [],
            "insufficient_evidence": False,
            "backend": "extractive",
        }

    def fake_query_understanding(message: str, history: list[dict], _generator_model: str | None, _max_new_tokens: int):
        return QueryUnderstanding(
            standalone_query="Zelo nejasen rewrite",
            intent="general",
            use_context=True,
            confidence=0.31,
            reason="Nizka zanesljivost.",
        )

    def fake_planner(query: str, history: list[dict], _generator_model: str | None, _max_new_tokens: int):
        return LegalAgentPlan(
            query=query,
            intent="follow_up",
            source_policy="pisrs_first",
            answer_style="guided_explanation",
            follow_referrals=True,
            preserve_origin_article=False,
            actions=("retrieve_chunks", "resolve_referrals", "prefer_pisrs", "compose_guided_explanation", "verify_primary_citation"),
            confidence=0.78,
            reason="Follow-up na prejšnje vprašanje.",
            backend="local_model",
        )

    app = create_app(
        WebAppSettings(
            chunks_path=REPO_ROOT / "artifacts" / "retrieval" / "chunks.jsonl",
            chroma_dir=REPO_ROOT / "artifacts" / "retrieval" / "chroma",
            bm25_path=REPO_ROOT / "artifacts" / "retrieval" / "bm25_corpus.json",
            generator_model="fake-local-model",
        ),
        answer_pipeline=fake_answer,
        lemmatize_fn=fake_lemmatize,
        query_understanding_fn=fake_query_understanding,
        agent_planner_fn=fake_planner,
    )
    client = TestClient(app)
    response = client.post(
        "/api/chat",
        json={
            "message": "Kaj pa za pravno osebo?",
            "history": [
                {"role": "user", "content": "Kakšna je globa za hujši davčni prekršek posameznika?"},
                {
                    "role": "assistant",
                    "content": "Po 395. členu ...",
                    "memory_topic": "ZDavP-2, 395. člen (hujši davčni prekrški posameznikov)",
                },
            ],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert "Prejšnje vprašanje:" in captured["answer_query"]
    assert payload["retrieval_query"] == captured["answer_query"]


def test_chat_endpoint_passes_agent_plan_and_returns_verification_payload():
    captured: dict[str, object] = {}

    def fake_lemmatize(text: str, _classla_python: str | None = None) -> list[str]:
        return text.lower().split()

    def fake_answer(query: str, *args, **kwargs):
        captured["query"] = query
        captured["agent_plan"] = kwargs.get("agent_plan")
        return {
            "answer": "Planiran odgovor.",
            "citations": [
                {
                    "law_id": "ZAKO4703",
                    "law_ref": "ZDavP-2",
                    "title": "ZAKON o davčnem postopku (ZDavP-2)",
                    "section_path": "I. DEL",
                    "article_number": "15. člen",
                    "article_title": "(davčna tajnost)",
                    "source_url": "http://pisrs.si/Pis.web/pregledPredpisa?id=ZAKO4703",
                    "chunk_id": "ZAKO4703::15-clen",
                    "source_type": "pisrs",
                }
            ],
            "used_chunks": [],
            "supporting_sentences": [],
            "insufficient_evidence": False,
            "backend": "extractive",
            "agent_plan": kwargs.get("agent_plan").as_dict(),
            "citation_verification": {"status": "verified", "score": 1.0, "checks": {"has_citation": True}},
        }

    def fake_planner(query: str, history: list[dict], _generator_model: str | None, _max_new_tokens: int):
        captured["planned_query"] = query
        captured["planned_history"] = history
        return LegalAgentPlan(
            query=query,
            intent="definition",
            source_policy="pisrs_first",
            answer_style="structured_rule",
            follow_referrals=False,
            preserve_origin_article=False,
            actions=("retrieve_chunks", "prefer_pisrs", "compose_structured_answer", "verify_primary_citation"),
            confidence=0.88,
            reason="Gre za definicijsko vprašanje.",
            backend="local_model",
        )

    def fake_query_understanding(message: str, history: list[dict], _generator_model: str | None, _max_new_tokens: int):
        return None

    app = create_app(
        WebAppSettings(
            chunks_path=REPO_ROOT / "artifacts" / "retrieval" / "chunks.jsonl",
            chroma_dir=REPO_ROOT / "artifacts" / "retrieval" / "chroma",
            bm25_path=REPO_ROOT / "artifacts" / "retrieval" / "bm25_corpus.json",
        ),
        answer_pipeline=fake_answer,
        lemmatize_fn=fake_lemmatize,
        query_understanding_fn=fake_query_understanding,
        agent_planner_fn=fake_planner,
    )
    client = TestClient(app)
    response = client.post(
        "/api/chat",
        json={
            "message": "Kaj pomeni davčna tajnost?",
            "history": [{"role": "user", "content": "Kaj pomeni davčna tajnost?"}],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert isinstance(captured["agent_plan"], LegalAgentPlan)
    assert payload["agent_plan"]["backend"] == "local_model"
    assert payload["citation_verification"]["status"] == "verified"
    assert [step["label"] for step in payload["processing_trace"]] == [
        "Razumevanje vprašanja",
        "Načrt odgovora",
        "Iskanje virov",
        "Sestava odgovora",
        "Preverjanje citata",
    ]


def test_chat_endpoint_handles_vat_calculation():
    app = create_app(
        WebAppSettings(
            chunks_path=REPO_ROOT / "artifacts" / "retrieval" / "chunks.jsonl",
            chroma_dir=REPO_ROOT / "artifacts" / "retrieval" / "chroma",
            bm25_path=REPO_ROOT / "artifacts" / "retrieval" / "bm25_corpus.json",
            generator_model=None,
        ),
    )
    client = TestClient(app)
    response = client.post(
        "/api/chat",
        json={"message": "Izračunaj DDV na neto znesek 1.200 EUR po 22%.", "history": []},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["backend"] == "calculator"
    assert payload["calculator_result"]["calculator"] == "vat"
    assert payload["calculator_result"]["breakdown"][1]["value"] == "264,00 EUR"
    assert payload["citations"][0]["law_ref"] == "ZDDV-1"
    assert any(step["label"] == "Uporaba kalkulatorja" for step in payload["processing_trace"])


def test_chat_endpoint_handles_pending_calculator_follow_up():
    app = create_app(
        WebAppSettings(
            chunks_path=REPO_ROOT / "artifacts" / "retrieval" / "chunks.jsonl",
            chroma_dir=REPO_ROOT / "artifacts" / "retrieval" / "chroma",
            bm25_path=REPO_ROOT / "artifacts" / "retrieval" / "bm25_corpus.json",
            generator_model=None,
        ),
    )
    client = TestClient(app)
    first = client.post(
        "/api/chat",
        json={"message": "Izračunaj DDV za neto znesek 1.200 EUR.", "history": []},
    )

    assert first.status_code == 200
    first_payload = first.json()
    assert first_payload["backend"] == "calculator_clarification"
    assert first_payload["calculator_context"]["missing_params"] == ["vat_rate"]

    second = client.post(
        "/api/chat",
        json={
            "message": "Uporabi 22%.",
            "history": [
                {"role": "user", "content": "Izračunaj DDV za neto znesek 1.200 EUR."},
                {
                    "role": "assistant",
                    "content": first_payload["message"],
                    "calculator_context": first_payload["calculator_context"],
                    "memory_topic": first_payload["memory_topic"],
                },
            ],
        },
    )

    assert second.status_code == 200
    second_payload = second.json()
    assert second_payload["backend"] == "calculator"
    assert second_payload["calculator_result"]["calculator"] == "vat"
    assert second_payload["calculator_result"]["breakdown"][2]["value"] == "1.464,00 EUR"
    assert any(step["label"] == "Uporaba kalkulatorja" for step in second_payload["processing_trace"])


def test_chat_endpoint_handles_income_tax_calculation():
    app = create_app(
        WebAppSettings(
            chunks_path=REPO_ROOT / "artifacts" / "retrieval" / "chunks.jsonl",
            chroma_dir=REPO_ROOT / "artifacts" / "retrieval" / "chroma",
            bm25_path=REPO_ROOT / "artifacts" / "retrieval" / "bm25_corpus.json",
            generator_model=None,
        ),
    )
    client = TestClient(app)
    response = client.post(
        "/api/chat",
        json={"message": "Izračunaj dohodnino za neto letno davčno osnovo 45.000 EUR.", "history": []},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["backend"] == "calculator"
    assert payload["calculator_result"]["calculator"] == "income_tax_brackets"
    assert payload["citations"][0]["law_ref"] == "ZDoh-2"
    assert "dohodnina po lestvici" in payload["message"].lower()


def test_chat_endpoint_handles_income_tax_bracket_question():
    app = create_app(
        WebAppSettings(
            chunks_path=REPO_ROOT / "artifacts" / "retrieval" / "chunks.jsonl",
            chroma_dir=REPO_ROOT / "artifacts" / "retrieval" / "chroma",
            bm25_path=REPO_ROOT / "artifacts" / "retrieval" / "bm25_corpus.json",
            generator_model=None,
        ),
    )
    client = TestClient(app)
    response = client.post(
        "/api/chat",
        json={"message": "V kateri dohodninski razred padem pri neto letni davčni osnovi 30.000 EUR?", "history": []},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["backend"] == "calculator"
    assert payload["calculator_result"]["calculator"] == "income_tax_brackets"
    assert "padeš v razred" in payload["calculator_result"]["result_summary"].lower()


def test_chat_endpoint_keeps_completed_income_tax_calculator_context_for_total_tax_follow_up():
    app = create_app(
        WebAppSettings(
            chunks_path=REPO_ROOT / "artifacts" / "retrieval" / "chunks.jsonl",
            chroma_dir=REPO_ROOT / "artifacts" / "retrieval" / "chroma",
            bm25_path=REPO_ROOT / "artifacts" / "retrieval" / "bm25_corpus.json",
            generator_model=None,
        ),
    )
    client = TestClient(app)
    first = client.post(
        "/api/chat",
        json={"message": "V kateri dohodninski razred padem pri neto letni davčni osnovi 30.000 EUR?", "history": []},
    )

    assert first.status_code == 200
    first_payload = first.json()
    assert first_payload["backend"] == "calculator"
    assert first_payload["calculator_result"]["calculator"] == "income_tax_brackets"
    assert first_payload["calculator_context"]["status"] == "completed"

    second = client.post(
        "/api/chat",
        json={
            "message": "Koliko pa moram v tem primeru skupno plačati dohodnine potem?",
            "history": [
                {
                    "role": "user",
                    "content": "V kateri dohodninski razred padem pri neto letni davčni osnovi 30.000 EUR?",
                },
                {
                    "role": "assistant",
                    "content": first_payload["message"],
                    "calculator_context": first_payload["calculator_context"],
                    "memory_topic": first_payload["memory_topic"],
                    "citations": first_payload["citations"],
                },
            ],
        },
    )

    assert second.status_code == 200
    second_payload = second.json()
    assert second_payload["backend"] == "calculator"
    assert second_payload["calculator_result"]["calculator"] == "income_tax_brackets"
    assert "7.300,00 EUR" in second_payload["message"]
    assert "dohodnina po lestvici" in second_payload["message"].lower()
