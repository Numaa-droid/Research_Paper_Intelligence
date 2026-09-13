"""Bounded retry policy.

Retries are targeted, not global: only the agents whose findings failed to ground
are re-run, they are told exactly what was rejected, and the whole thing stops
after ``max_retries`` regardless of whether the second attempt was any better.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..domain.enums import FindingStatus
from ..domain.models import AgentResult, Finding, VerificationResult
from ..observability import info

#: Statuses that mark a finding as failing to ground.
WEAK = frozenset({FindingStatus.UNSUPPORTED, FindingStatus.CONTRADICTED, FindingStatus.UNKNOWN})


@dataclass(slots=True)
class RetryDecision:
    """Whether to retry, which agents, and why."""

    should_retry: bool
    agents: list[str]
    reasons: dict[str, str]
    stop_reason: str | None = None

    def feedback_for(self, agent: str) -> str | None:
        return self.reasons.get(agent)


def weak_ratio(findings: list[Finding]) -> float:
    """Share of findings that failed to ground. 0.0 when there are none."""
    if not findings:
        return 0.0
    return round(sum(1 for f in findings if f.status in WEAK) / len(findings), 4)


def decide_retry(
    *,
    agent_results: dict[str, AgentResult],
    findings: list[Finding],
    verification_results: list[VerificationResult],
    retry_count: int,
    max_retries: int,
    threshold: float,
    already_retried: list[str] | None = None,
) -> RetryDecision:
    """Decide which agents to re-run after verification.

    An agent is retried when it *failed outright*, or when the share of its
    findings that grounded falls below ``threshold``. An agent is never retried
    twice, and no retry happens at all once the budget is spent.
    """
    already_retried = already_retried or []

    if retry_count >= max_retries:
        return RetryDecision(
            should_retry=False,
            agents=[],
            reasons={},
            stop_reason=(
                f"retry budget exhausted ({retry_count}/{max_retries})"
                if max_retries
                else "retries are disabled (max_retries=0)"
            ),
        )

    by_agent: dict[str, list[Finding]] = {}
    for finding in findings:
        by_agent.setdefault(finding.agent or "unknown", []).append(finding)

    weak_by_agent: dict[str, list[VerificationResult]] = {}
    for result in verification_results:
        if result.is_weak and result.source_agent:
            weak_by_agent.setdefault(result.source_agent, []).append(result)

    agents: list[str] = []
    reasons: dict[str, str] = {}

    for name, result in agent_results.items():
        if name in already_retried or result.status == "skipped":
            continue

        if result.status == "failed":
            agents.append(name)
            reasons[name] = f"The previous attempt failed: {result.error or 'unknown error'}"
            continue

        agent_findings = by_agent.get(name, [])
        if not agent_findings:
            agents.append(name)
            reasons[name] = "The previous attempt produced no findings at all."
            continue

        grounded = 1.0 - weak_ratio(agent_findings)
        if grounded < threshold:
            rejected = [f.claim for f in agent_findings if f.status in WEAK][:5]
            detail = "\n".join(f"- {claim}" for claim in rejected)
            reasons[name] = (
                f"Only {grounded:.0%} of your claims could be grounded in the paper "
                f"(required: {threshold:.0%}). These claims were rejected because their "
                f"quotes could not be found in the paper or their numbers did not match:\n"
                f"{detail}"
            )
            agents.append(name)

    if not agents:
        return RetryDecision(
            should_retry=False,
            agents=[],
            reasons={},
            stop_reason=f"all agents met the grounding threshold of {threshold:.0%}",
        )

    info("retry.decided", agents=agents, retry_count=retry_count, max_retries=max_retries)
    return RetryDecision(should_retry=True, agents=agents, reasons=reasons)


def merge_retry_findings(
    original: list[Finding], retried: list[Finding], agents: list[str]
) -> list[Finding]:
    """Replace a retried agent's findings, keeping everyone else's untouched.

    If the retry produced nothing, the original findings are kept: a failed retry
    must not silently delete the first attempt's work.
    """
    retried_by_agent: dict[str, list[Finding]] = {}
    for finding in retried:
        retried_by_agent.setdefault(finding.agent or "unknown", []).append(finding)

    out: list[Finding] = []
    for finding in original:
        agent = finding.agent or "unknown"
        if agent in agents and retried_by_agent.get(agent):
            continue  # superseded below
        out.append(finding)

    for agent in agents:
        out.extend(retried_by_agent.get(agent, []))
    return out
