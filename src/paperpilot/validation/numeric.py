"""Numeric validation.

Numbers are the part of a paper summary readers trust most and models get wrong
most, so every number an agent reports is re-checked against the retrieved text
by string comparison — never by asking a model whether the number looks right.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..domain.models import Evidence, NumericCheck

#: Matches integers, decimals, percentages, multipliers ("3.1x") and exponents
#: ("3e-5"), optionally signed and thousands-separated.
NUMBER_RE = re.compile(
    r"(?<![\w.])"
    r"([-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?|[-+]?\d*\.\d+|[-+]?\d+)"
    r"(\s?%|x|e[-+]?\d+)?"
    r"(?![\w])",
    re.IGNORECASE,
)

#: Tokens that are numeric but are not claims about results.
_IGNORED = {"1", "2", "0"}


@dataclass(frozen=True, slots=True)
class ParsedNumber:
    raw: str
    value: float
    suffix: str = ""

    @property
    def is_percentage(self) -> bool:
        return "%" in self.suffix


def parse_number(raw: str, suffix: str = "") -> ParsedNumber | None:
    cleaned = raw.replace(",", "").strip()
    suffix = (suffix or "").strip().lower()
    try:
        if suffix.startswith("e"):
            value = float(cleaned + suffix)
            suffix = ""
        else:
            value = float(cleaned)
    except ValueError:
        return None
    return ParsedNumber(raw=raw.strip(), value=value, suffix=suffix)


def extract_numbers(text: str, *, skip_trivial: bool = True) -> list[ParsedNumber]:
    """Pull every number out of ``text``, in order of appearance."""
    numbers: list[ParsedNumber] = []
    for match in NUMBER_RE.finditer(text or ""):
        parsed = parse_number(match.group(1), match.group(2) or "")
        if parsed is None:
            continue
        if skip_trivial and parsed.raw in _IGNORED and not parsed.suffix:
            continue
        numbers.append(parsed)
    return numbers


def numbers_equal(a: ParsedNumber, b: ParsedNumber, *, rel_tol: float = 1e-9) -> bool:
    """Compare two parsed numbers, tolerating trailing-zero formatting.

    ``92.4`` and ``92.40`` are the same value; ``92.4`` and ``92.4%`` are not
    treated as different, because papers routinely drop the percent sign in
    running text while keeping it in the table header.
    """
    if a.value == b.value:
        return True
    if b.value == 0:
        return abs(a.value) <= rel_tol
    return abs(a.value - b.value) / max(abs(b.value), 1e-12) <= rel_tol


def check_numbers_against_text(
    claim: str, source_text: str, *, page: int | None = None
) -> list[NumericCheck]:
    """Verify every number in ``claim`` appears in ``source_text``."""
    source_numbers = extract_numbers(source_text, skip_trivial=False)
    checks: list[NumericCheck] = []

    for number in extract_numbers(claim):
        match = next((s for s in source_numbers if numbers_equal(number, s)), None)
        if match is not None:
            checks.append(
                NumericCheck(
                    value=number.raw + number.suffix,
                    found_in_evidence=True,
                    matched_text=match.raw + match.suffix,
                    page=page,
                )
            )
        else:
            checks.append(
                NumericCheck(
                    value=number.raw + number.suffix,
                    found_in_evidence=False,
                    page=page,
                    note="value does not appear in the retrieved evidence",
                )
            )
    return checks


def check_claim_numbers(claim: str, evidence: list[Evidence]) -> list[NumericCheck]:
    """Check a claim's numbers against the union of its evidence quotes.

    A number counts as supported if *any* piece of evidence contains it; the
    page recorded is the page of the evidence that matched.
    """
    if not evidence:
        return [
            NumericCheck(
                value=n.raw + n.suffix,
                found_in_evidence=False,
                note="claim carries no evidence to check against",
            )
            for n in extract_numbers(claim)
        ]

    results: list[NumericCheck] = []
    for number in extract_numbers(claim):
        matched: NumericCheck | None = None
        for item in evidence:
            for source in extract_numbers(item.quote, skip_trivial=False):
                if numbers_equal(number, source):
                    matched = NumericCheck(
                        value=number.raw + number.suffix,
                        found_in_evidence=True,
                        matched_text=source.raw + source.suffix,
                        page=item.page,
                    )
                    break
            if matched:
                break
        results.append(
            matched
            or NumericCheck(
                value=number.raw + number.suffix,
                found_in_evidence=False,
                note="value does not appear in any cited evidence",
            )
        )
    return results


def numeric_accuracy(checks: list[NumericCheck]) -> float:
    """Share of checked numbers that were found. Vacuously 1.0 when none."""
    if not checks:
        return 1.0
    return round(sum(1 for c in checks if c.found_in_evidence) / len(checks), 4)
