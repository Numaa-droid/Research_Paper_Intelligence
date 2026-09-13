"""Benchmark harness: run a directory of papers and save JSON results.

Raw outputs, normalised findings, verification results and aggregate metrics are
all written to disk so a run can be re-scored later without re-running the
models — which matters when a model call is the expensive part.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import Settings
from ..domain.models import AnalysisRecord
from ..ingestion.service import IngestionService
from ..observability import info, warning
from ..providers.embeddings import build_embedding_provider
from ..providers.llm import build_llm_provider
from ..retrieval.index import PaperIndex
from ..service import PaperPilotService
from .metrics import EvaluationMetrics, aggregate, evaluate_record


@dataclass(slots=True)
class PaperExpectation:
    """Human-written ground truth for one benchmark paper."""

    filename: str
    title: str | None = None
    expected_claims: list[str] = field(default_factory=list)
    key_numbers: list[str] = field(default_factory=list)
    notes: str = ""


def load_expectations(path: Path | None) -> dict[str, PaperExpectation]:
    """Load ``expectations.json``, keyed by filename.

    Missing or malformed expectations are not fatal: the harness still runs and
    reports the metrics that need no ground truth.
    """
    if path is None or not Path(path).exists():
        return {}
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        warning("evaluation.expectations_unreadable", path=str(path), error=str(exc))
        return {}

    entries = payload.get("papers", payload) if isinstance(payload, dict) else payload
    expectations: dict[str, PaperExpectation] = {}
    for entry in entries if isinstance(entries, list) else []:
        if not isinstance(entry, dict) or not entry.get("filename"):
            continue
        expectations[str(entry["filename"])] = PaperExpectation(
            filename=str(entry["filename"]),
            title=entry.get("title"),
            expected_claims=[str(c) for c in entry.get("expected_claims", [])],
            key_numbers=[str(n) for n in entry.get("key_numbers", [])],
            notes=str(entry.get("notes", "")),
        )
    return expectations


class EvaluationHarness:
    """Runs PaperPilot over a benchmark directory and scores every run."""

    def __init__(
        self,
        settings: Settings,
        *,
        service: PaperPilotService | None = None,
        expectations: dict[str, PaperExpectation] | None = None,
    ) -> None:
        self.settings = settings
        self.expectations = expectations or {}
        self.embedder = build_embedding_provider(settings)
        self.service = service or PaperPilotService(
            settings,
            llm_provider=build_llm_provider(settings),
            embedding_provider=self.embedder,
        )

    # --- running ----------------------------------------------------------

    def discover(self, directory: Path) -> list[Path]:
        return sorted(p for p in Path(directory).glob("*.pdf") if p.is_file())

    def run_paper(self, pdf: Path) -> tuple[AnalysisRecord, PaperIndex | None]:
        """Analyse one paper and rebuild its index for independent scoring.

        The index is rebuilt from the PDF rather than reused from the run so the
        scoring is not measuring the run against its own intermediate state.
        """
        record = self.service.analyze_file(pdf)
        try:
            index = IngestionService(self.settings, embedder=self.embedder).ingest(
                pdf, document_id=record.document_id
            ).index
        except Exception as exc:  # noqa: BLE001 - scoring must survive a bad paper
            warning("evaluation.index_rebuild_failed", paper=pdf.name, error=str(exc)[:200])
            index = None
        return record, index

    def evaluate(self, directory: Path, *, limit: int | None = None) -> dict[str, Any]:
        """Run and score every PDF in ``directory``."""
        papers = self.discover(directory)
        if limit:
            papers = papers[:limit]
        if not papers:
            raise FileNotFoundError(f"no PDFs found in {directory}")

        started = time.perf_counter()
        results: list[EvaluationMetrics] = []
        raw: list[dict[str, Any]] = []

        for pdf in papers:
            info("evaluation.paper_started", paper=pdf.name)
            paper_started = time.perf_counter()
            try:
                record, index = self.run_paper(pdf)
            except Exception as exc:  # noqa: BLE001 - one bad paper must not end the run
                warning("evaluation.paper_failed", paper=pdf.name, error=str(exc)[:300])
                results.append(
                    EvaluationMetrics(
                        document_id="unknown",
                        paper=pdf.name,
                        status="failed",
                        errors=[str(exc)],
                        latency_seconds=round(time.perf_counter() - paper_started, 3),
                    )
                )
                continue

            expectation = self.expectations.get(pdf.name)
            if index is None:
                results.append(
                    EvaluationMetrics(
                        document_id=record.document_id,
                        paper=pdf.name,
                        status=record.status.value,
                        errors=[*record.errors, "index could not be rebuilt for scoring"],
                        latency_seconds=record.duration_seconds,
                    )
                )
                continue

            metrics = evaluate_record(
                record,
                index,
                expected_claims=expectation.expected_claims if expectation else [],
                paper_name=pdf.name,
            )
            results.append(metrics)
            raw.append(
                {
                    "paper": pdf.name,
                    "analysis_id": record.analysis_id,
                    "document_id": record.document_id,
                    "status": record.status.value,
                    "plan": record.plan.model_dump(mode="json") if record.plan else None,
                    "findings": [f.model_dump(mode="json") for f in record.findings],
                    "verification_results": [
                        v.model_dump(mode="json") for v in record.verification_results
                    ],
                    "usage": [u.model_dump(mode="json") for u in record.usage],
                    "events": [e.model_dump(mode="json") for e in record.events],
                    "report_markdown": record.report_markdown,
                }
            )
            info(
                "evaluation.paper_completed",
                paper=pdf.name,
                citation_precision=metrics.citation_precision,
                citation_recall=metrics.citation_recall,
            )

        return {
            "run": {
                "papers": len(papers),
                "duration_seconds": round(time.perf_counter() - started, 3),
                "llm_provider": self.settings.llm_provider,
                "llm_model": self.settings.llm_model
                if self.settings.llm_provider != "mock"
                else "mock-deterministic-v1",
                "embedding_provider": self.settings.embedding_provider,
                "max_retries": self.settings.max_retries,
                "verification_threshold": self.settings.verification_threshold,
                "expectations_supplied": bool(self.expectations),
            },
            "aggregate": aggregate(results),
            "per_paper": [r.to_dict() for r in results],
            "raw": raw,
        }

    # --- output -----------------------------------------------------------

    @staticmethod
    def save(results: dict[str, Any], output_dir: Path) -> dict[str, Path]:
        """Write metrics, per-paper scores and raw outputs as separate files."""
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")

        paths = {
            "metrics": output / f"metrics-{stamp}.json",
            "per_paper": output / f"per-paper-{stamp}.json",
            "raw": output / f"raw-{stamp}.json",
        }
        paths["metrics"].write_text(
            json.dumps({"run": results["run"], "aggregate": results["aggregate"]}, indent=2),
            encoding="utf-8",
        )
        paths["per_paper"].write_text(
            json.dumps(results["per_paper"], indent=2, default=str), encoding="utf-8"
        )
        paths["raw"].write_text(
            json.dumps(results["raw"], indent=2, default=str), encoding="utf-8"
        )
        return paths

    @staticmethod
    def format_summary(results: dict[str, Any]) -> str:
        """A readable table of the aggregate metrics."""
        run = results["run"]
        agg = results["aggregate"]

        def fmt(key: str) -> str:
            value = agg.get(f"mean_{key}")
            measured = agg.get(f"measured_{key}", 0)
            if value is None:
                return "n/a (not measurable)"
            return f"{value:.3f}  (over {measured} paper(s))"

        lines = [
            "PaperPilot evaluation",
            "=" * 58,
            f"papers              : {agg['papers']}  "
            f"({agg['completed']} completed, {agg['failed']} failed)",
            f"provider            : {run['llm_provider']} / {run['llm_model']}",
            f"embeddings          : {run['embedding_provider']}",
            f"ground truth        : {'supplied' if run['expectations_supplied'] else 'none'}",
            "-" * 58,
            f"claim coverage      : {fmt('claim_coverage')}",
            f"citation precision  : {fmt('citation_precision')}",
            f"citation recall     : {fmt('citation_recall')}",
            f"numeric accuracy    : {fmt('numeric_accuracy')}",
            f"unsupported rate    : {fmt('unsupported_claim_rate')}",
            f"verification acc.   : {fmt('verification_accuracy')}",
            f"latency (s/paper)   : {fmt('latency_seconds')}",
            "-" * 58,
            f"total tokens        : {agg['total_tokens']}",
            "estimated cost      : "
            + (
                f"${agg['total_estimated_cost_usd']:.4f}"
                if agg.get("total_estimated_cost_usd") is not None
                else "n/a (provider reports no pricing)"
            ),
        ]
        if run["llm_provider"] == "mock":
            lines += [
                "",
                "NOTE: this run used the deterministic mock provider. The scores measure",
                "      the pipeline's grounding machinery, not a model's analysis quality.",
            ]
        return "\n".join(lines)
