"""Command-line interface.

    python -m paperpilot analyze ./papers/example.pdf --output ./reports/example.md

Progress is printed per stage as it happens, and the exit code is non-zero on
failure so the CLI composes with shell scripts and CI.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from . import __version__
from .config import Settings, settings_from_env
from .domain.enums import AnalysisStatus, FindingStatus
from .domain.models import AnalysisRecord, ProgressEvent
from .errors import PaperPilotError
from .observability import configure_logging
from .service import PaperPilotService

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_BAD_INPUT = 2
EXIT_INTERRUPTED = 130

_STAGE_ICONS = {
    "started": "▸",   # ▸
    "completed": "✓", # ✓
    "skipped": "○",   # ○
    "scheduled": "↺", # ↺
    "failed": "✗",    # ✗
}


class ProgressPrinter:
    """Prints progress events to stderr, leaving stdout clean for output."""

    def __init__(self, *, quiet: bool = False, stream=None) -> None:
        self.quiet = quiet
        self.stream = stream or sys.stderr
        self.started = time.perf_counter()

    def __call__(self, event: ProgressEvent) -> None:
        if self.quiet:
            return
        icon = _STAGE_ICONS.get(event.status, "·")
        elapsed = time.perf_counter() - self.started
        line = f"  {icon} [{elapsed:6.1f}s] {event.stage:<16} {event.message}"
        print(line, file=self.stream, flush=True)


def _summarise(record: AnalysisRecord, stream) -> None:
    stats = record.stats()
    counts = stats["status_counts"]
    print("", file=stream)
    print(f"  Status          : {record.status.value}", file=stream)
    print(f"  Duration        : {record.duration_seconds}s", file=stream)
    print(f"  Agents run      : {', '.join(stats['agents_run']) or 'none'}", file=stream)
    if stats["agents_skipped"]:
        print(f"  Agents skipped  : {', '.join(stats['agents_skipped'])}", file=stream)
    print(f"  Findings        : {stats['findings']}", file=stream)
    for status in (
        FindingStatus.SUPPORTED,
        FindingStatus.PARTIALLY_SUPPORTED,
        FindingStatus.INFERRED,
        FindingStatus.UNSUPPORTED,
        FindingStatus.CONTRADICTED,
        FindingStatus.UNKNOWN,
    ):
        count = counts.get(status.value, 0)
        if count:
            print(f"    - {status.value:<22}{count}", file=stream)
    print(f"  Claims verified : {stats['verified_claims']}", file=stream)
    print(f"  Retries         : {stats['retry_count']}", file=stream)
    print(f"  Tokens          : {stats['total_tokens']}", file=stream)
    if record.errors:
        print(f"  Errors          : {len(record.errors)}", file=stream)
        for error in record.errors[:5]:
            print(f"    ! {error}", file=stream)


def _build_settings(args: argparse.Namespace) -> Settings:
    overrides: dict = {}
    if getattr(args, "provider", None):
        overrides["llm_provider"] = args.provider
    if getattr(args, "model", None):
        overrides["llm_model"] = args.model
    if getattr(args, "no_ocr", False):
        overrides["ocr_enabled"] = False
    if getattr(args, "max_retries", None) is not None:
        overrides["max_retries"] = args.max_retries
    if getattr(args, "sequential", False):
        overrides["parallel_agents"] = False
    if getattr(args, "data_dir", None):
        overrides["data_dir"] = Path(args.data_dir)
        overrides["database_url"] = f"sqlite:///{Path(args.data_dir) / 'paperpilot.sqlite'}"
    if getattr(args, "verbose", False):
        overrides["log_level"] = "DEBUG"
    elif getattr(args, "quiet", False):
        overrides["log_level"] = "ERROR"
    return settings_from_env(**overrides)


def cmd_analyze(args: argparse.Namespace) -> int:
    settings = _build_settings(args)
    configure_logging(settings.log_level, "text" if not args.json_logs else "json")

    path = Path(args.pdf).expanduser()
    if not path.exists():
        print(f"error: file not found: {path}", file=sys.stderr)
        return EXIT_BAD_INPUT

    if settings.llm_provider == "mock" and not args.quiet:
        print(
            "  note: using the deterministic mock provider (no credentials configured).\n"
            "        The pipeline runs end to end, but the analysis is not a real one.\n"
            "        Set PAPERPILOT_LLM_PROVIDER=openai_compatible and PAPERPILOT_LLM_API_KEY\n"
            "        for a genuine analysis.\n",
            file=sys.stderr,
        )

    print(f"Analysing {path.name}", file=sys.stderr)
    service = PaperPilotService(settings)
    printer = ProgressPrinter(quiet=args.quiet)

    try:
        record = service.analyze_file(path, event_sink=printer)
    except PaperPilotError as exc:
        print(f"error [{exc.code}]: {exc.message}", file=sys.stderr)
        return EXIT_BAD_INPUT if exc.http_status < 500 else EXIT_FAILED
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return EXIT_INTERRUPTED

    if not args.quiet:
        _summarise(record, sys.stderr)

    if record.status is not AnalysisStatus.COMPLETED:
        print("\nanalysis did not complete", file=sys.stderr)
        return EXIT_FAILED

    report = record.report_markdown or ""
    if args.output:
        output = Path(args.output).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report, encoding="utf-8")
        print(f"\nReport written to {output}", file=sys.stderr)
    else:
        print(report)

    if args.json_out:
        target = Path(args.json_out).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(record.model_dump_json(indent=2), encoding="utf-8")
        print(f"Structured result written to {target}", file=sys.stderr)

    return EXIT_OK


def cmd_ingest(args: argparse.Namespace) -> int:
    """Parse and index a PDF without running any agent. Useful for debugging."""
    settings = _build_settings(args)
    configure_logging(settings.log_level, "text")

    path = Path(args.pdf).expanduser()
    if not path.exists():
        print(f"error: file not found: {path}", file=sys.stderr)
        return EXIT_BAD_INPUT

    from .ingestion.service import IngestionService
    from .providers.embeddings import build_embedding_provider

    try:
        result = IngestionService(
            settings, embedder=build_embedding_provider(settings)
        ).ingest(path)
    except PaperPilotError as exc:
        print(f"error [{exc.code}]: {exc.message}", file=sys.stderr)
        return EXIT_BAD_INPUT

    metadata = result.metadata
    if args.json_out or args.format == "json":
        payload = {
            "metadata": metadata.model_dump(mode="json"),
            "chunks": [c.model_dump(mode="json") for c in result.chunks],
        }
        text = json.dumps(payload, indent=2, default=str)
        if args.json_out:
            Path(args.json_out).write_text(text, encoding="utf-8")
        else:
            print(text)
        return EXIT_OK

    print(f"document_id : {metadata.document_id}")
    print(f"title       : {metadata.title}")
    print(f"pages       : {metadata.page_count}")
    print(f"chunks      : {metadata.chunk_count}")
    print(f"ocr pages   : {metadata.ocr_page_count}")
    print(f"elements    : {metadata.element_counts}")
    print(f"sections    : {len(metadata.sections)}")
    for section in metadata.sections:
        print(f"  - {section}")
    return EXIT_OK


def cmd_serve(args: argparse.Namespace) -> int:
    settings = _build_settings(args)
    try:
        import uvicorn
    except ImportError:
        print("error: uvicorn is not installed", file=sys.stderr)
        return EXIT_FAILED

    print(f"PaperPilot API on http://{args.host}:{args.port} (UI at /)", file=sys.stderr)
    uvicorn.run(
        "paperpilot.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=settings.log_level.lower(),
    )
    return EXIT_OK


def cmd_report(args: argparse.Namespace) -> int:
    settings = _build_settings(args)
    service = PaperPilotService(settings)
    try:
        print(service.get_report(args.analysis_id))
    except PaperPilotError as exc:
        print(f"error [{exc.code}]: {exc.message}", file=sys.stderr)
        return EXIT_BAD_INPUT
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="paperpilot",
        description=(
            "PaperPilot: citation-grounded multi-agent analysis of a single research paper."
        ),
    )
    parser.add_argument("--version", action="version", version=f"paperpilot {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--provider", choices=["mock", "openai_compatible"], help="LLM provider")
        p.add_argument("--model", help="model name to use")
        p.add_argument("--data-dir", help="directory for the database, uploads and reports")
        p.add_argument("--no-ocr", action="store_true", help="disable the OCR fallback")
        p.add_argument("-v", "--verbose", action="store_true", help="debug logging")
        p.add_argument("-q", "--quiet", action="store_true", help="suppress progress output")
        p.add_argument("--json-logs", action="store_true", help="emit logs as JSON")

    analyze = sub.add_parser("analyze", help="analyse a PDF end to end")
    analyze.add_argument("pdf", help="path to the research-paper PDF")
    analyze.add_argument("-o", "--output", help="write the Markdown report here")
    analyze.add_argument("--json-out", help="write the full structured result here")
    analyze.add_argument("--max-retries", type=int, help="override the retry budget")
    analyze.add_argument(
        "--sequential", action="store_true", help="run specialist agents one at a time"
    )
    common(analyze)
    analyze.set_defaults(func=cmd_analyze)

    ingest = sub.add_parser("ingest", help="parse and index a PDF without running agents")
    ingest.add_argument("pdf", help="path to the research-paper PDF")
    ingest.add_argument("--format", choices=["text", "json"], default="text")
    ingest.add_argument("--json-out", help="write the parsed structure here")
    common(ingest)
    ingest.set_defaults(func=cmd_ingest)

    serve = sub.add_parser("serve", help="run the HTTP API and web UI")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true")
    common(serve)
    serve.set_defaults(func=cmd_serve)

    report = sub.add_parser("report", help="print a stored report")
    report.add_argument("analysis_id")
    common(report)
    report.set_defaults(func=cmd_report)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return EXIT_INTERRUPTED


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
