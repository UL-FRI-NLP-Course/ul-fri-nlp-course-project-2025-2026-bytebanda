from zakonodajko_rag.planner import LegalAgentPlan, build_heuristic_legal_agent_plan, merge_plans


def test_heuristic_plan_for_explicit_article_prefers_pisrs_and_preserves_origin():
    plan = build_heuristic_legal_agent_plan("Kaj določa 395. člen ZDavP-2?")

    assert plan.intent == "explicit_article"
    assert plan.source_policy == "pisrs_only"
    assert plan.preserve_origin_article is True
    assert "retrieve_chunks" in plan.actions
    assert "verify_primary_citation" in plan.actions
    assert "prefer_pisrs" in plan.actions


def test_merge_plans_keeps_explicit_article_safety_even_if_model_plan_prefers_furs():
    heuristic = build_heuristic_legal_agent_plan("Kaj določa 395. člen ZDavP-2?")
    model_plan = LegalAgentPlan(
        query=heuristic.query,
        intent="practical_guidance",
        source_policy="furs_preferred",
        answer_style="guided_explanation",
        follow_referrals=True,
        preserve_origin_article=False,
        actions=("retrieve_chunks", "prefer_furs_guidance", "compose_guided_explanation", "verify_primary_citation"),
        confidence=0.93,
        reason="Model misli, da želi uporabnik pojasnilo FURS.",
        backend="local_model",
    )

    merged = merge_plans(heuristic, model_plan)

    assert merged.intent == "explicit_article"
    assert merged.source_policy == "pisrs_only"
    assert merged.preserve_origin_article is True
    assert "prefer_pisrs" in merged.actions


def test_heuristic_plan_detects_calculation_queries():
    plan = build_heuristic_legal_agent_plan("Izračunaj DDV na neto znesek 1.200 EUR po 22%.")

    assert plan.intent == "calculation"
    assert "run_calculator" in plan.actions
    assert "prefer_pisrs" in plan.actions
