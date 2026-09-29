"""Checks on the paper's own structure that need no model.

A numbered heading sequence is a checkable fact: "2.1, 2.3, 2.1" repeats a
number and skips one. Such slips are worth reporting, because a reader citing
"§2.1" of this paper would be ambiguous.
"""

from __future__ import annotations

import re

from ..domain.enums import ElementType
from ..retrieval.index import PaperIndex

#: Elements a layout places wherever they fit: their position says nothing about
#: where their heading sits in the text.
_FLOATING = (ElementType.TABLE, ElementType.FIGURE)

_NUMBERED = re.compile(r"^\s*(\d+(?:\.\d+)*)\.?\s+\S")


def heading_numbering_issues(headings: list[str]) -> list[str]:
    """Repeated, skipped and out-of-order heading numbers, in plain words."""
    numbered: list[tuple[tuple[int, ...], str]] = []
    for heading in headings:
        match = _NUMBERED.match(heading)
        if match:
            numbered.append((tuple(int(p) for p in match.group(1).split(".")), heading))

    issues: list[str] = []
    seen: dict[tuple[int, ...], str] = {}
    for number, heading in numbered:
        label = ".".join(map(str, number))
        if number in seen:
            issues.append(f"heading number {label} is used twice: “{seen[number]}” and “{heading}”")
        else:
            seen[number] = heading

    previous: tuple[int, ...] | None = None
    earlier: set[tuple[int, ...]] = set()
    for number, heading in numbered:
        if previous is not None and number < previous and number not in earlier:
            issues.append(
                f"heading “{heading}” comes after {'.'.join(map(str, previous))}, out of order"
            )
        # Skips: a sibling number jumps by more than one under the same parent.
        if (
            previous is not None
            and len(number) == len(previous)
            and number[:-1] == previous[:-1]
            and number[-1] > previous[-1] + 1
        ):
            missing = ".".join(map(str, (*number[:-1], previous[-1] + 1)))
            issues.append(f"heading number {missing} is skipped before “{heading}”")
        earlier.add(number)
        previous = number
    return issues


def heading_sequence(index: PaperIndex) -> list[str]:
    """Headings in the order the running text reaches them.

    Tables and figures float to the top of a page, so a table under §4 can
    precede the end of §3.5 in reading order; only body chunks count.
    """
    seen: list[str] = []
    for chunk in index.chunks:
        if chunk.element_type in _FLOATING:
            continue
        for label in (chunk.section, chunk.subsection):
            if label and label not in seen:
                seen.append(label)
    return seen
