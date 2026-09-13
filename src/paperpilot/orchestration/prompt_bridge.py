"""Adapter between the prompt registry and the planner.

Kept separate so :mod:`paperpilot.orchestration.planner` does not import the
agent package (which would make the dependency direction orchestration -> agents
-> orchestration).
"""

from __future__ import annotations

import json
from typing import Any

from ..agents.prompts.registry import PLANNER


def planner_prompt(structure: dict[str, Any]) -> tuple[str, str]:
    """Render the planner prompt for a structural summary."""
    return PLANNER.system, PLANNER.render(structure=json.dumps(structure, indent=2, default=str))
