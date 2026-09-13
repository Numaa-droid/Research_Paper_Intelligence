#!/usr/bin/env python3
"""Run the PaperPilot evaluation harness over a directory of papers.

    python scripts/evaluate.py --papers ./fixtures/benchmark --output ./eval_results

Provider-independent: the same metrics are produced for any configured model, so
runs can be compared across providers. Supply ``expectations.json`` in the papers
directory (or via ``--expectations``) to enable claim coverage; without it that
metric reports ``n/a`` rather than a made-up number.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from paperpilot.config import settings_from_env  # noqa: E402
from paperpilot.evaluation import EvaluationHarness, load_expectations  # noqa: E402
from paperpilot.observability import configure_logging  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate PaperPilot on a benchmark set.")
    parser.add_argument(
        "--papers", default="./fixtures/benchmark", help="directory containing PDFs"
    )
    parser.add_argument("--output", default="./eval_results", help="where to write JSON results")
    parser.add_argument(
        "--expectations",
        default=None,
        help="ground-truth JSON (defaults to <papers>/expectations.json)",
    )
    parser.add_argument("--provider", choices=["mock", "openai_compatible"])
    parser.add_argument("--model")
    parser.add_argument("--limit", type=int, help="only evaluate the first N papers")
    parser.add_argument("--max-retries", type=int)
    parser.add_argument("--data-dir", default="./eval_results/.work")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    papers_dir = Path(args.papers).expanduser()
    if not papers_dir.exists():
        print(f"error: papers directory not found: {papers_dir}", file=sys.stderr)
        return 2

    overrides: dict = {"data_dir": Path(args.data_dir)}
    overrides["database_url"] = f"sqlite:///{Path(args.data_dir) / 'eval.sqlite'}"
    if args.provider:
        overrides["llm_provider"] = args.provider
    if args.model:
        overrides["llm_model"] = args.model
    if args.max_retries is not None:
        overrides["max_retries"] = args.max_retries
    overrides["log_level"] = "DEBUG" if args.verbose else "WARNING"

    settings = settings_from_env(**overrides)
    configure_logging(settings.log_level, "text")

    expectations_path = (
        Path(args.expectations) if args.expectations else papers_dir / "expectations.json"
    )
    expectations = load_expectations(expectations_path)
    if not expectations:
        print(
            f"note: no ground truth at {expectations_path}; "
            "claim coverage will report n/a.",
            file=sys.stderr,
        )

    harness = EvaluationHarness(settings, expectations=expectations)
    try:
        results = harness.evaluate(papers_dir, limit=args.limit)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    paths = harness.save(results, Path(args.output))
    print(harness.format_summary(results))
    print("")
    for name, path in paths.items():
        print(f"{name:<10} -> {path}")

    return 0 if results["aggregate"]["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
