"""The visual summary: an analysis shaped for charts rather than for reading.

The web UI draws the paper's story, its method pipeline, its reported numbers
and its verification verdicts as pictures. This module picks those out of the
agents' validated payloads, by the same rule as the brief: a reported number is
charted only when its evidence was re-found in the paper and is the paper's own
(not a cited work's). Like the brief it makes no model call and adds nothing;
it only selects and shortens.
"""

from __future__ import annotations

import re
from typing import Any

from ..domain.enums import AgentName
from ..domain.models import AgentResult, AnalysisRecord, Evidence, PaperMetadata
from .brief import clip
from .synthesizer import _stated

#: Most results the chart view is given; a paper's full table stays in the report.
MAX_RESULTS = 40
MAX_STEPS = 8

_NUMBER = re.compile(r"[-+−]?\d[\d,]*(?:\.\d+)?")


def parse_number(value: str | None) -> tuple[float | None, str]:
    """The first number in a reported value and its unit (``%``, ``×`` or ``""``).

    ``"81.4 ± 0.3"`` → ``(81.4, "")``; ``"3.1x"`` → ``(3.1, "×")``. A value with
    no number, such as ``"n/a"``, gives ``None`` and is shown, not charted.
    """
    text = str(value or "")
    match = _NUMBER.search(text)
    if not match:
        return None, ""
    try:
        number = float(match.group().replace(",", "").replace("−", "-"))
    except ValueError:
        return None, ""
    rest = text[match.end() :].lstrip()
    if rest.startswith("%"):
        unit = "%"
    elif rest[:1] in ("x", "×", "X") and not rest[1:2].isalpha():
        unit = "×"
    else:
        unit = ""
    return number, unit


def _payload(agent_results: dict[str, AgentResult], agent: AgentName) -> dict:
    result = agent_results.get(agent.value)
    return result.payload if result and result.status == "ok" else {}


def _first_page(evidence: list[Evidence]) -> int | None:
    verified = [e for e in evidence if e.is_verified]
    return verified[0].page if verified else None


def _short(text: Any, sentences: int = 2, max_chars: int = 300) -> str:
    return clip(_stated(str(text or "")), sentences=sentences, max_chars=max_chars)


def _story(summary: dict) -> dict[str, str]:
    return {
        "problem": _short(summary.get("research_problem")),
        "approach": _short(summary.get("proposed_method") or summary.get("main_contribution")),
        "result": _short(summary.get("main_results")),
        "limitations": _short(summary.get("limitations")),
    }


def _pipeline(methodology: dict) -> list[dict[str, Any]]:
    steps = sorted(methodology.get("pipeline_steps") or [], key=lambda s: s.get("order", 0))
    out = []
    for step in steps[:MAX_STEPS]:
        evidence = [Evidence(**e) for e in step.get("evidence", [])]
        out.append(
            {
                "name": str(step.get("name") or "").strip(),
                "description": clip(step.get("description"), sentences=1, max_chars=180),
                "page": _first_page(evidence),
                "inferred": bool(step.get("is_inferred")),
            }
        )
    return [s for s in out if s["name"]]


def _results(experiments: dict) -> list[dict[str, Any]]:
    rows = []
    for item in experiments.get("reported_results") or []:
        evidence = [Evidence(**e) for e in item.get("evidence", [])]
        verified = [e for e in evidence if e.is_verified]
        # Only the paper's own, re-found numbers are drawn.
        if not verified or all(e.attribution == "cited" for e in verified):
            continue
        number, unit = parse_number(item.get("value"))
        rows.append(
            {
                "system": item.get("system") or None,
                "dataset": item.get("dataset") or None,
                "split": item.get("split") or None,
                "metric": str(item.get("metric") or "").strip() or "Result",
                "value": str(item.get("value") or "").strip(),
                "number": number,
                "unit": unit,
                "is_best": bool(item.get("is_best")),
                "page": verified[0].page,
            }
        )
    return rows[:MAX_RESULTS]


def build_visuals(record: AnalysisRecord, metadata: PaperMetadata | None) -> dict[str, Any]:
    """Everything the visual summary draws, for one completed analysis."""
    results = record.agent_results
    summary = _payload(results, AgentName.SUMMARY)
    methodology = _payload(results, AgentName.METHODOLOGY)
    experiments = _payload(results, AgentName.EXPERIMENTS)
    math = _payload(results, AgentName.MATH)

    conclusions = experiments.get("conclusions_supported")
    return {
        "analysis_id": record.analysis_id,
        "paper": {
            "title": (metadata.title or metadata.filename) if metadata else None,
            "page_count": metadata.page_count if metadata else 0,
        },
        "story": _story(summary),
        "pipeline": _pipeline(methodology),
        "results": _results(experiments),
        "setup": {
            "datasets": _stated(experiments.get("datasets") or [])[:12],
            "baselines": _stated(experiments.get("baselines") or [])[:12],
            "metrics": _stated(experiments.get("metrics") or [])[:12],
        },
        "conclusions": {
            "status": conclusions if conclusions and conclusions != "unknown" else None,
            "assessment": _short(experiments.get("conclusions_assessment"), sentences=1, max_chars=260),
        },
        "equation_count": len(math.get("equations") or []),
        "verification": [
            {
                "claim": clip(v.claim, sentences=2, max_chars=260),
                "status": v.status.value,
                "confidence": v.confidence,
                "agent": v.source_agent,
                "pages": v.pages[:4],
            }
            for v in record.verification_results
        ],
    }
