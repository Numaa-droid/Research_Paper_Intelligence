"""The per-agent YAML reference guides stay in step with the prompts they guide."""

from __future__ import annotations

import re

import pytest

from paperpilot.agents.prompts.registry import get_prompt
from paperpilot.agents.specs import PAPER_KINDS, agent_guidance, load_spec, spec_names

AGENTS = ("summary", "methodology", "experiments", "math")


def test_every_specialist_has_a_guide():
    assert set(AGENTS) <= set(spec_names())


@pytest.mark.parametrize("agent", AGENTS)
def test_guide_headings_are_the_prompts_headings(agent):
    spec = load_spec(agent)
    template_headings = set(re.findall(r"^## (.+)$", get_prompt(agent).user_template, re.MULTILINE))
    assert set(spec["headings"]) <= template_headings
    assert set(spec["paper_kinds"]) == set(PAPER_KINDS)


@pytest.mark.parametrize("agent", AGENTS)
@pytest.mark.parametrize("kind", PAPER_KINDS)
def test_guide_renders_for_every_paper_kind(agent, kind):
    text = agent_guidance(agent, kind, signals=["a signal"])
    assert text.startswith(f"REFERENCE GUIDE — {agent} agent")
    assert f"for a {kind.upper()} paper" in text
    assert "a signal" in text
    assert "REPORTS CITED WORK" in text


def test_a_paper_kind_overrides_the_default_heading_guidance():
    empirical = agent_guidance("methodology", "empirical")
    review = agent_guidance("methodology", "review")
    assert "The literature searched" in review
    assert "The literature searched" not in empirical


def test_an_agent_without_a_guide_renders_nothing():
    assert agent_guidance("coverage", "review") == ""


def test_the_guide_is_placed_above_the_paper_context():
    user = get_prompt("summary").render(guidance="GUIDE\n\n", metadata="m", context="c")
    assert user.index("GUIDE") < user.index("PAPER METADATA") < user.index("RETRIEVED PAPER CONTEXT")
    # Callers that pass no guide still render.
    assert get_prompt("summary").render(metadata="m", context="c").startswith("PAPER METADATA")
