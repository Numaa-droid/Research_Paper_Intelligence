"""Planning, the state graph, retry policy and progress events."""

from __future__ import annotations

import pytest

from paperpilot.domain.enums import FindingStatus
from paperpilot.domain.models import AgentResult, Finding, VerificationResult
from paperpilot.domain.state import ResearchState, merge_state, new_state
from paperpilot.orchestration.engine import END, StateGraph
from paperpilot.orchestration.events import EventRecorder
from paperpilot.orchestration.planner import create_plan, heuristic_plan, paper_structure
from paperpilot.orchestration.retry import decide_retry, merge_retry_findings, weak_ratio

# --- planning --------------------------------------------------------------


def test_plan_selects_agents_the_paper_has_content_for(index, settings):
    plan = heuristic_plan(index, settings)
    assert plan.run_summary and plan.run_methodology and plan.run_experiments
    assert plan.run_math is True  # three equations were detected
    assert plan.run_table_analysis is True


def test_plan_explains_every_decision(index, settings):
    plan = heuristic_plan(index, settings)
    for key in ("run_summary", "run_methodology", "run_experiments", "run_math"):
        assert plan.rationale.get(key), f"no rationale recorded for {key}"


def test_plan_skips_math_when_the_threshold_is_not_met(index, settings):
    settings.math_min_equations = 99
    plan = heuristic_plan(index, settings)
    assert plan.run_math is False
    assert "below the threshold" in plan.rationale["run_math"]


def test_plan_excludes_references_from_priority_sections(index, settings):
    plan = heuristic_plan(index, settings)
    assert not any("reference" in s.lower() for s in plan.priority_sections)
    assert not any("front matter" in s.lower() for s in plan.priority_sections)


def test_plan_enabled_agents_matches_the_flags(index, settings):
    plan = heuristic_plan(index, settings)
    assert set(plan.enabled_agents()) == {"summary", "methodology", "experiments", "math"}


def test_paper_structure_exposes_only_structural_facts(index, settings):
    structure = paper_structure(index, settings)
    assert structure["equation_count"] == 3
    assert structure["page_count"] == 6
    assert "full_text" not in structure, "the planner must not receive the paper body"


def test_model_cannot_enable_an_agent_for_content_that_does_not_exist(index, settings, llm):
    """The planner may veto an agent, but structure is a fact it cannot override."""
    settings.math_min_equations = 99
    plan = create_plan(index, settings, llm)
    assert plan.run_math is False
    assert "declined" in plan.rationale["run_math"] or "below the threshold" in plan.rationale["run_math"]


def test_planner_falls_back_to_the_heuristic_when_the_model_fails(index, settings):
    from paperpilot.errors import ProviderError
    from paperpilot.providers.llm import LLMProvider

    class Broken(LLMProvider):
        name = "broken"

        def generate(self, request):
            raise ProviderError("down")

    plan = create_plan(index, settings, Broken())
    assert plan.created_by == "heuristic"
    assert plan.run_summary is True


def test_plan_without_a_model_is_purely_heuristic(index, settings, llm):
    assert create_plan(index, settings, llm, use_model=False).created_by == "heuristic"


# --- state merging ---------------------------------------------------------


def test_new_state_initialises_every_collection():
    state = new_state("a1", "d1", "/tmp/x.pdf")
    assert state["findings"] == [] and state["agent_results"] == {}
    assert state["retry_count"] == 0


def test_append_keys_accumulate():
    state = new_state("a1", "d1", "x")
    merge_state(state, ResearchState(errors=["first"]))
    merge_state(state, ResearchState(errors=["second"]))
    assert state["errors"] == ["first", "second"]


def test_dict_keys_merge_rather_than_replace():
    state = new_state("a1", "d1", "x")
    merge_state(state, ResearchState(agent_results={"a": AgentResult(agent="a")}))
    merge_state(state, ResearchState(agent_results={"b": AgentResult(agent="b")}))
    assert set(state["agent_results"]) == {"a", "b"}


def test_scalar_keys_are_replaced():
    state = new_state("a1", "d1", "x")
    merge_state(state, ResearchState(retry_count=1))
    merge_state(state, ResearchState(retry_count=2))
    assert state["retry_count"] == 2


def test_merging_nothing_is_a_no_op():
    state = new_state("a1", "d1", "x")
    assert merge_state(state, None) is state


# --- the graph engine ------------------------------------------------------


def test_graph_runs_nodes_in_order():
    order: list[str] = []
    graph = StateGraph()
    graph.add_node("a", lambda s: order.append("a"))
    graph.add_node("b", lambda s: order.append("b"))
    graph.set_entry("a").add_edge("a", "b").add_edge("b", END)
    graph.run(new_state("a1", "d1", "x"))
    assert order == ["a", "b"]


def test_conditional_edges_choose_the_branch():
    graph = StateGraph()
    graph.add_node("start", lambda s: ResearchState(index_status="ready"))
    graph.add_node("ready", lambda s: ResearchState(final_report="ready path"))
    graph.add_node("other", lambda s: ResearchState(final_report="other path"))
    graph.set_entry("start")
    graph.add_conditional_edge(
        "start", lambda s: "ready" if s.get("index_status") == "ready" else "other"
    )
    graph.add_edge("ready", END).add_edge("other", END)
    assert graph.run(new_state("a1", "d1", "x"))["final_report"] == "ready path"


def test_a_crashing_node_is_recorded_and_the_run_continues():
    def boom(state):
        raise RuntimeError("node exploded")

    graph = StateGraph()
    graph.add_node("boom", boom)
    graph.add_node("after", lambda s: ResearchState(final_report="still ran"))
    graph.set_entry("boom").add_edge("boom", "after").add_edge("after", END)
    final = graph.run(new_state("a1", "d1", "x"))
    assert final["final_report"] == "still ran"
    assert any("RuntimeError" in e for e in final["errors"])


def test_parallel_nodes_all_contribute():
    graph = StateGraph()
    for name in ("x", "y", "z"):
        graph.add_node(name, lambda s, n=name: ResearchState(agent_results={n: AgentResult(agent=n)}))
    graph.add_parallel_node("group", ["x", "y", "z"])
    graph.set_entry("group").add_edge("group", END)
    final = graph.run(new_state("a1", "d1", "x"))
    assert set(final["agent_results"]) == {"x", "y", "z"}


def test_parallel_merge_order_is_deterministic():
    graph = StateGraph()
    for name in ("x", "y"):
        graph.add_node(name, lambda s, n=name: ResearchState(errors=[n]))
    graph.add_parallel_node("group", ["x", "y"])
    graph.set_entry("group").add_edge("group", END)
    runs = [graph.run(new_state("a1", "d1", "x"))["errors"] for _ in range(5)]
    assert all(r == runs[0] for r in runs)


def test_step_limit_stops_an_infinite_loop():
    graph = StateGraph(max_steps=5)
    graph.add_node("loop", lambda s: None)
    graph.set_entry("loop").add_edge("loop", "loop")
    final = graph.run(new_state("a1", "d1", "x"))
    assert any("exceeded" in e for e in final["errors"])


def test_duplicate_node_names_are_rejected():
    graph = StateGraph()
    graph.add_node("a", lambda s: None)
    with pytest.raises(ValueError, match="duplicate node"):
        graph.add_node("a", lambda s: None)


def test_parallel_group_rejects_unknown_members():
    graph = StateGraph()
    with pytest.raises(ValueError, match="unknown nodes"):
        graph.add_parallel_node("group", ["missing"])


def test_graph_without_an_entry_point_is_rejected():
    with pytest.raises(ValueError, match="no entry node"):
        StateGraph().run(new_state("a1", "d1", "x"))


# --- retry policy ----------------------------------------------------------


def _findings(agent: str, statuses: list[FindingStatus]) -> list[Finding]:
    return [Finding(claim=f"{agent} claim {i}", agent=agent, status=s) for i, s in enumerate(statuses)]


def test_weak_ratio():
    assert weak_ratio([]) == 0.0
    assert weak_ratio(_findings("a", [FindingStatus.SUPPORTED] * 4)) == 0.0
    assert weak_ratio(
        _findings("a", [FindingStatus.SUPPORTED, FindingStatus.UNSUPPORTED])
    ) == 0.5


def test_a_poorly_grounded_agent_is_retried():
    decision = decide_retry(
        agent_results={"summary": AgentResult(agent="summary")},
        findings=_findings("summary", [FindingStatus.UNSUPPORTED] * 3),
        verification_results=[],
        retry_count=0,
        max_retries=1,
        threshold=0.85,
    )
    assert decision.should_retry
    assert decision.agents == ["summary"]
    assert "could be grounded" in decision.feedback_for("summary")


def test_a_well_grounded_agent_is_not_retried():
    decision = decide_retry(
        agent_results={"summary": AgentResult(agent="summary")},
        findings=_findings("summary", [FindingStatus.SUPPORTED] * 4),
        verification_results=[],
        retry_count=0,
        max_retries=1,
        threshold=0.85,
    )
    assert not decision.should_retry
    assert "threshold" in decision.stop_reason


def test_retry_feedback_names_the_rejected_claims():
    decision = decide_retry(
        agent_results={"summary": AgentResult(agent="summary")},
        findings=_findings("summary", [FindingStatus.UNSUPPORTED]),
        verification_results=[],
        retry_count=0,
        max_retries=1,
        threshold=0.85,
    )
    assert "summary claim 0" in decision.feedback_for("summary")


def test_retries_stop_at_the_budget():
    decision = decide_retry(
        agent_results={"summary": AgentResult(agent="summary")},
        findings=_findings("summary", [FindingStatus.UNSUPPORTED] * 3),
        verification_results=[],
        retry_count=1,
        max_retries=1,
        threshold=0.85,
    )
    assert not decision.should_retry
    assert "budget exhausted" in decision.stop_reason


def test_retries_disabled_means_no_retry():
    decision = decide_retry(
        agent_results={"summary": AgentResult(agent="summary")},
        findings=_findings("summary", [FindingStatus.UNSUPPORTED] * 3),
        verification_results=[],
        retry_count=0,
        max_retries=0,
        threshold=0.85,
    )
    assert not decision.should_retry
    assert "disabled" in decision.stop_reason


def test_an_agent_is_never_retried_twice():
    decision = decide_retry(
        agent_results={"summary": AgentResult(agent="summary")},
        findings=_findings("summary", [FindingStatus.UNSUPPORTED] * 3),
        verification_results=[],
        retry_count=0,
        max_retries=2,
        threshold=0.85,
        already_retried=["summary"],
    )
    assert not decision.should_retry


def test_a_skipped_agent_is_never_retried():
    decision = decide_retry(
        agent_results={"math": AgentResult(agent="math", status="skipped", skip_reason="no maths")},
        findings=[],
        verification_results=[],
        retry_count=0,
        max_retries=1,
        threshold=0.85,
    )
    assert not decision.should_retry


def test_a_failed_agent_is_retried_with_its_error():
    decision = decide_retry(
        agent_results={"summary": AgentResult(agent="summary", status="failed", error="timeout")},
        findings=[],
        verification_results=[],
        retry_count=0,
        max_retries=1,
        threshold=0.85,
    )
    assert decision.should_retry
    assert "timeout" in decision.feedback_for("summary")


def test_an_agent_producing_nothing_is_retried():
    decision = decide_retry(
        agent_results={"summary": AgentResult(agent="summary")},
        findings=[],
        verification_results=[],
        retry_count=0,
        max_retries=1,
        threshold=0.85,
    )
    assert decision.should_retry
    assert "no findings" in decision.feedback_for("summary")


def test_verification_failures_feed_the_retry_decision():
    findings = _findings("experiments", [FindingStatus.UNSUPPORTED])
    results = [
        VerificationResult(
            claim=findings[0].claim,
            source_agent="experiments",
            status=FindingStatus.UNSUPPORTED,
        )
    ]
    decision = decide_retry(
        agent_results={"experiments": AgentResult(agent="experiments")},
        findings=findings,
        verification_results=results,
        retry_count=0,
        max_retries=1,
        threshold=0.85,
    )
    assert "experiments" in decision.agents


# --- retry merging ---------------------------------------------------------


def test_retried_findings_replace_only_that_agents_findings():
    original = _findings("summary", [FindingStatus.UNSUPPORTED]) + _findings(
        "experiments", [FindingStatus.SUPPORTED]
    )
    retried = [Finding(claim="better summary claim", agent="summary", status=FindingStatus.SUPPORTED)]
    merged = merge_retry_findings(original, retried, ["summary"])
    claims = {f.claim for f in merged}
    assert "better summary claim" in claims
    assert "summary claim 0" not in claims
    assert "experiments claim 0" in claims


def test_a_retry_that_produced_nothing_keeps_the_original_findings():
    original = _findings("summary", [FindingStatus.UNSUPPORTED])
    merged = merge_retry_findings(original, [], ["summary"])
    assert merged == original


# --- progress events -------------------------------------------------------


def test_recorder_forwards_events_to_the_sink():
    seen = []
    recorder = EventRecorder(sink=seen.append)
    recorder.started("ingestion", "reading")
    recorder.completed("ingestion", "done", pages=6)
    assert [e.status for e in seen] == ["started", "completed"]
    assert seen[1].detail["pages"] == 6


def test_drain_empties_the_buffer():
    recorder = EventRecorder()
    recorder.started("x")
    assert len(recorder.drain()) == 1
    assert recorder.drain() == []
