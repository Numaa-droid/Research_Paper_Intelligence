"""Document layout analysis, isolated behind an interface.

PyMuPDF reports the blocks a PDF's content stream happens to declare. On many
journal templates that is one block per *column run*, with a section heading, its
paragraph, and sometimes a figure caption fused into a single object. Geometry
recovers a great deal of the real structure from those blocks (see
:func:`paperpilot.ingestion.elements.split_embedded_headings`), but it is reading
typographic tea leaves: what the page actually contains — a title here, a table
there, a running header along the top — is a question about the *image* of the
page, and a layout model answers it directly.

This module defines that question as an interface and supplies a PP-Structure
implementation. Nothing in the pipeline imports ``paddleocr``: the engine is
resolved at construction time, and a missing dependency degrades to "no layout
analysis" with a logged warning rather than a crash, exactly as OCR does.

Determinism
-----------
A parse of the same PDF must produce the same report every time, so the engine
is pinned on every axis that could drift:

* one CPU thread, fixed seeds, inference mode — no nondeterministic kernel
  selection, no GPU reduction order;
* pinned model names, so a new PaddleOCR release cannot silently change what a
  page is found to contain;
* a fixed rasterisation DPI, because detections are computed in pixels;
* coordinates rounded to a fixed precision before use, so a float that differs
  in its last bit cannot reorder two regions;
* a total order over regions that never depends on detection order;
* an on-disk cache keyed by the page's content hash *and* the full engine
  configuration, so a second run replays the first run's regions rather than
  re-inferring them.

The cache is an optimisation for repeated runs. It is not what makes the engine
deterministic: with the cache disabled, the same page still yields the same
regions.
"""

from __future__ import annotations

import hashlib
import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from ..observability import debug, info, warning

#: Coordinates are rounded to this many decimal places before anything compares,
#: sorts or hashes them. Two runs that differ only in float noise then agree.
COORD_PRECISION = 2
#: Region kinds PP-Structure reports, mapped onto the vocabulary the parser uses.
#: Anything unrecognised becomes ``text``, which is the harmless default: it
#: neither drops the region nor promotes it to a heading.
LAYOUT_KINDS = frozenset(
    {
        "title",
        "doc_title",
        "text",
        "table",
        "figure",
        "header",
        "footer",
        "reference",
        "equation",
        "caption",
    }
)


@dataclass(frozen=True, slots=True)
class LayoutRegion:
    """One region of a page, as a layout model sees it."""

    kind: str
    bbox: tuple[float, float, float, float]
    score: float = 0.0
    #: Position in the model's reading order, after the deterministic sort.
    order: int = 0

    @property
    def is_furniture(self) -> bool:
        """Page furniture: repeated margin matter that is not document content."""
        return self.kind in ("header", "footer")

    @property
    def is_heading(self) -> bool:
        return self.kind == "title"


@dataclass(slots=True)
class PageLayout:
    """Every region found on one page, in reading order."""

    page: int
    regions: list[LayoutRegion] = field(default_factory=list)
    engine: str = "none"

    def of_kind(self, *kinds: str) -> list[LayoutRegion]:
        return [r for r in self.regions if r.kind in kinds]


class LayoutEngine(ABC):
    """Interface for page-level layout analysis."""

    name = "abstract"

    @abstractmethod
    def analyze_page(self, page, *, page_number: int) -> PageLayout | None:
        """Return the layout of a PyMuPDF page, or ``None`` when unavailable."""

    @property
    def available(self) -> bool:
        return True


class NullLayoutEngine(LayoutEngine):
    """Explicit 'layout analysis is off', so callers never special-case ``None``."""

    name = "null"

    def analyze_page(self, page, *, page_number: int) -> PageLayout | None:
        return None

    @property
    def available(self) -> bool:
        return False


#: PP-Structure's own label vocabulary varies by model version and by language
#: pack. Every label it is known to emit is mapped explicitly; an unknown label
#: is logged once and treated as ``text`` rather than guessed at.
_PP_LABELS = {
    # Section headings — the reason this engine exists.
    "paragraph_title": "title",
    "title": "title",
    # The paper's own title is front matter, not a section that owns the text
    # beneath it, so it is kept apart from the headings above.
    "doc_title": "doc_title",
    # Captions name a table or figure; they are not sections.
    "figure_title": "caption",
    "table_title": "caption",
    "chart_title": "caption",
    "figure_caption": "caption",
    "table_caption": "caption",
    # Running matter. "number" is the folio PP-DocLayout reports separately.
    "header": "header",
    "page_header": "header",
    "header_image": "header",
    "footer": "footer",
    "page_footer": "footer",
    "footer_image": "footer",
    "footnote": "footer",
    "number": "footer",
    "page_number": "footer",
    # Body matter.
    "text": "text",
    "plain text": "text",
    "abstract": "text",
    "content": "text",
    "list": "text",
    "algorithm": "text",
    "aside_text": "text",
    "seal": "text",
    "reference": "reference",
    "reference_content": "reference",
    "table": "table",
    "figure": "figure",
    "image": "figure",
    "chart": "figure",
    "formula": "equation",
    "equation": "equation",
    "formula_number": "equation",
}

#: Environment variables that make CPU inference reproducible. They are set
#: before paddle is imported, because the thread pools are sized at import.
_DETERMINISM_ENV = {
    "FLAGS_cudnn_deterministic": "1",
    "FLAGS_cpu_deterministic": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "PYTHONHASHSEED": "0",
    # No network probe on start-up: availability must not depend on reaching a
    # model host, and the models are already on disk after the first fetch.
    "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK": "True",
}


def _quantise(value: float) -> float:
    return round(float(value), COORD_PRECISION)


def _region_sort_key(region: LayoutRegion) -> tuple:
    """A total order over regions that cannot depend on detection order.

    Top-to-bottom, then left-to-right, then by the remaining coordinates and the
    kind, so that two regions sharing an origin still order the same way on
    every run. Every component is already quantised.
    """
    x0, y0, x1, y1 = region.bbox
    return (y0, x0, y1, x1, region.kind)


class PPStructureLayoutEngine(LayoutEngine):
    """Page layout from PaddleOCR's PP-Structure document-analysis pipeline.

    The page is rasterised by PyMuPDF at a fixed DPI, the layout model is run
    over the bitmap, and the regions it reports are mapped back onto PDF points
    so they can be compared with the text blocks PyMuPDF extracted.
    """

    name = "pp_structure"

    def __init__(
        self,
        *,
        dpi: int = 200,
        layout_model: str = "PP-DocLayout_plus-L",
        lang: str = "en",
        min_score: float = 0.5,
        cache_dir: Path | None = None,
        device: str = "cpu",
    ) -> None:
        self.dpi = int(dpi)
        self.layout_model = layout_model
        self.lang = lang
        self.min_score = float(min_score)
        self.cache_dir = cache_dir
        self.device = device
        self._pipeline = None
        self._checked = False
        self._ok = False
        self._unknown_labels: set[str] = set()

    # --- configuration identity ------------------------------------------

    def fingerprint(self) -> str:
        """Everything that could change this engine's output, as one string.

        Part of every cache key: a cache entry written under one model or DPI
        must never be replayed for another.
        """
        payload = {
            "engine": self.name,
            "dpi": self.dpi,
            "layout_model": self.layout_model,
            "lang": self.lang,
            "min_score": self.min_score,
            "device": self.device,
            "precision": COORD_PRECISION,
            "schema": 1,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:16]

    # --- availability -----------------------------------------------------

    @property
    def available(self) -> bool:
        if not self._checked:
            self._checked = True
            self._ok = self._build()
        return self._ok

    def _build(self) -> bool:
        for key, value in _DETERMINISM_ENV.items():
            os.environ.setdefault(key, value)
        try:
            import paddle
            from paddleocr import LayoutDetection

            paddle.seed(0)
            paddle.set_device(self.device)
            try:
                paddle.set_flags({"FLAGS_cpu_deterministic": True})
            except Exception:  # noqa: BLE001 - flag name varies across releases
                debug("layout.flag_unavailable", flag="FLAGS_cpu_deterministic")

            self._pipeline = LayoutDetection(model_name=self.layout_model, device=self.device)
        except Exception as exc:  # noqa: BLE001 - optional dependency
            warning("layout.unavailable", engine=self.name, error=str(exc)[:300])
            return False
        info("layout.ready", engine=self.name, model=self.layout_model, dpi=self.dpi)
        return True

    # --- analysis ---------------------------------------------------------

    def analyze_page(self, page, *, page_number: int) -> PageLayout | None:
        if not self.available:
            return None

        try:
            pixmap = page.get_pixmap(dpi=self.dpi)
            image_bytes = pixmap.tobytes("png")
        except Exception as exc:  # noqa: BLE001 - a damaged page must not kill the run
            warning("layout.render_failed", page=page_number, error=str(exc)[:200])
            return None

        cached = self._cache_read(image_bytes)
        if cached is not None:
            debug("layout.cache_hit", page=page_number, regions=len(cached))
            return PageLayout(page=page_number, regions=cached, engine=self.name)

        raw = self._detect(image_bytes, page_number=page_number)
        if raw is None:
            return None

        # Pixels back to PDF points: the detection ran on a bitmap rendered at
        # ``dpi``, and every downstream comparison is in points.
        scale = 72.0 / self.dpi
        regions = [
            LayoutRegion(
                kind=kind,
                bbox=(
                    _quantise(box[0] * scale),
                    _quantise(box[1] * scale),
                    _quantise(box[2] * scale),
                    _quantise(box[3] * scale),
                ),
                score=round(float(score), 4),
            )
            for kind, box, score in raw
        ]
        regions.sort(key=_region_sort_key)
        regions = [
            LayoutRegion(kind=r.kind, bbox=r.bbox, score=r.score, order=position)
            for position, r in enumerate(regions)
        ]

        self._cache_write(image_bytes, regions)
        debug("layout.page_analysed", page=page_number, regions=len(regions))
        return PageLayout(page=page_number, regions=regions, engine=self.name)

    def _detect(self, image_bytes: bytes, *, page_number: int):
        """Run the model and return ``(kind, bbox, score)`` triples, or ``None``."""
        try:
            import io

            import numpy as np
            from PIL import Image

            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            outputs = self._pipeline.predict(np.asarray(image), batch_size=1)
        except Exception as exc:  # noqa: BLE001 - a failed page must not kill the run
            warning("layout.page_failed", page=page_number, error=str(exc)[:300])
            return None

        triples = []
        for result in outputs or []:
            boxes = result.get("boxes") if isinstance(result, dict) else None
            for box in boxes or []:
                label = str(box.get("label", "")).strip().lower()
                score = float(box.get("score", 0.0))
                if score < self.min_score:
                    continue
                kind = _PP_LABELS.get(label)
                if kind is None:
                    if label not in self._unknown_labels:
                        self._unknown_labels.add(label)
                        debug("layout.unknown_label", label=label, engine=self.name)
                    kind = "text"
                coords = box.get("coordinate") or box.get("bbox") or []
                if len(coords) != 4:
                    continue
                x0, y0, x1, y1 = (float(v) for v in coords)
                triples.append((kind, (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)), score))
        return triples

    # --- cache ------------------------------------------------------------

    def _cache_path(self, image_bytes: bytes) -> Path | None:
        if self.cache_dir is None:
            return None
        digest = hashlib.sha256(image_bytes).hexdigest()
        return Path(self.cache_dir) / f"{self.fingerprint()}-{digest}.json"

    def _cache_read(self, image_bytes: bytes) -> list[LayoutRegion] | None:
        path = self._cache_path(image_bytes)
        if path is None or not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text())
            return [
                LayoutRegion(
                    kind=str(item["kind"]),
                    bbox=tuple(_quantise(v) for v in item["bbox"]),  # type: ignore[arg-type]
                    score=float(item.get("score", 0.0)),
                    order=int(item.get("order", position)),
                )
                for position, item in enumerate(payload["regions"])
            ]
        except Exception as exc:  # noqa: BLE001 - a bad cache entry is not fatal
            debug("layout.cache_unreadable", path=str(path), error=str(exc)[:200])
            return None

    def _cache_write(self, image_bytes: bytes, regions: list[LayoutRegion]) -> None:
        path = self._cache_path(image_bytes)
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "fingerprint": self.fingerprint(),
                "regions": [
                    {"kind": r.kind, "bbox": list(r.bbox), "score": r.score, "order": r.order}
                    for r in regions
                ],
            }
            # Written whole then moved, so a killed run cannot leave a half file
            # that a later run would read back as a page with no regions.
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))
            temporary.replace(path)
        except Exception as exc:  # noqa: BLE001 - the cache is an optimisation
            debug("layout.cache_unwritable", path=str(path), error=str(exc)[:200])


def build_layout_engine(
    *,
    enabled: bool = False,
    engine: str = "pp_structure",
    dpi: int = 200,
    layout_model: str = "PP-DocLayout_plus-L",
    lang: str = "en",
    min_score: float = 0.5,
    cache_dir: str | Path | None = None,
    device: str = "cpu",
) -> LayoutEngine:
    """Resolve the configured layout engine, or a null engine when off."""
    if not enabled:
        return NullLayoutEngine()
    if engine != "pp_structure":
        warning("layout.unknown_engine", engine=engine)
        return NullLayoutEngine()
    return PPStructureLayoutEngine(
        dpi=dpi,
        layout_model=layout_model,
        lang=lang,
        min_score=min_score,
        cache_dir=Path(cache_dir) if cache_dir else None,
        device=device,
    )
