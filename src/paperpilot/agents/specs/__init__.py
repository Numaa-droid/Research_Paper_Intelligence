"""Per-agent reference guides, kept as YAML next to this module.

Each ``<agent>.yaml`` says what a good answer to every heading of that agent's
prompt looks like, with overrides per paper kind (empirical, review,
theoretical, conceptual). The guide for the detected kind is rendered into the
prompt above the paper context. It guides the model; it decides nothing. What
reaches the report is still settled by the deterministic checks.

The files are validated when loaded: a missing key, an unknown paper kind or a
heading the prompt does not have is an error, not a silently ignored line.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import Any

import yaml

SPEC_DIR = Path(__file__).parent
PAPER_KINDS = ("empirical", "review", "theoretical", "conceptual")
_REQUIRED = ("agent", "version", "role", "rules", "headings", "paper_kinds")


class SpecError(ValueError):
    """A reference guide file is malformed."""


def spec_names() -> list[str]:
    return sorted(p.stem for p in SPEC_DIR.glob("*.yaml"))


@cache
def load_spec(agent: str) -> dict[str, Any] | None:
    """The validated guide for ``agent``, or ``None`` if it has none."""
    path = SPEC_DIR / f"{agent}.yaml"
    if not path.exists():
        return None
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SpecError(f"{path.name}: expected a mapping at the top level")
    missing = [key for key in _REQUIRED if key not in data]
    if missing:
        raise SpecError(f"{path.name}: missing key(s) {', '.join(missing)}")
    if data["agent"] != agent:
        raise SpecError(f"{path.name}: 'agent' is {data['agent']!r}, expected {agent!r}")
    unknown = sorted(set(data["paper_kinds"]) - set(PAPER_KINDS))
    if unknown:
        raise SpecError(f"{path.name}: unknown paper kind(s) {', '.join(unknown)}")
    for kind, section in data["paper_kinds"].items():
        extra = sorted(set((section or {}).get("headings") or {}) - set(data["headings"]))
        if extra:
            raise SpecError(
                f"{path.name}: paper kind {kind!r} overrides unknown heading(s) {', '.join(extra)}"
            )
    return data


def _lines(items: list[str] | None, bullet: str = "- ") -> list[str]:
    return [f"{bullet}{' '.join(str(item).split())}" for item in items or []]


def agent_guidance(agent: str, kind: str, *, signals: list[str] | None = None) -> str:
    """The rendered guide for one agent and paper kind; ``""`` if there is none."""
    spec = load_spec(agent)
    if spec is None:
        return ""
    kind = kind if kind in PAPER_KINDS else "empirical"
    section = (spec["paper_kinds"].get(kind) or {})
    headings = {**spec["headings"], **(section.get("headings") or {})}

    lines = [
        f"REFERENCE GUIDE — {agent} agent (guide v{spec['version']}) for a {kind.upper()} paper",
    ]
    if signals:
        lines.append("This paper was classified as " + kind + " because: " + "; ".join(signals) + ".")
    lines.append("")
    lines.append("Role: " + " ".join(str(spec["role"]).split()))
    if section.get("focus"):
        lines.append("For this kind of paper: " + " ".join(str(section["focus"]).split()))
    lines.append("")
    lines.append("Rules:")
    lines.extend(_lines(spec["rules"]))
    lines.extend(_lines(section.get("rules")))
    if spec.get("avoid"):
        lines.append("")
        lines.append("Do not:")
        lines.extend(_lines(spec["avoid"]))
    lines.append("")
    lines.append("How to fill each heading:")
    lines.extend(f"- {name}: {' '.join(str(text).split())}" for name, text in headings.items())
    if spec.get("findings"):
        lines.append("")
        lines.append("Findings: " + " ".join(str(spec["findings"]).split()))
    if spec.get("self_check"):
        lines.append("")
        lines.append("Before answering, check:")
        lines.extend(_lines(spec["self_check"]))
    lines.append("")
    lines.append(
        "(End of guide. The grounding rules and the answer template below take precedence.)"
    )
    return "\n".join(lines) + "\n\n"
