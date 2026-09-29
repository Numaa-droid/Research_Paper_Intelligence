"""OCR fallback, isolated behind an interface.

Tesseract is optional. Nothing in the pipeline imports ``pytesseract`` directly:
the engine is resolved at construction time and a missing binary degrades to
"no OCR" with a logged warning instead of a crash.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

from ..observability import debug, info, warning


@dataclass(slots=True)
class OCRResult:
    text: str
    confidence: float = 0.6
    engine: str = "unknown"


class OCREngine(ABC):
    """Interface for page-level OCR."""

    name = "abstract"

    #: Whether :meth:`extract_figure` / :meth:`extract_table` can read charts,
    #: diagrams and tables on pages that otherwise have a native text layer.
    supports_figures = False
    supports_tables = False

    @abstractmethod
    def extract_page(self, page, *, page_number: int) -> OCRResult | None:
        """Return OCR text for a PyMuPDF page, or ``None`` when unavailable."""

    def extract_figure(
        self, page, rect, *, page_number: int, printed_text: str = ""
    ) -> OCRResult | None:
        """Return the content of the chart or diagram inside ``rect``.

        ``printed_text`` is any native text found inside the region (axis labels
        of a vector chart), used to ground numbers the model reports.
        """
        return None

    def extract_table(
        self, page, rect, *, page_number: int, printed_text: str = ""
    ) -> OCRResult | None:
        """Return the table inside ``rect`` as Markdown, or ``None`` to keep native text."""
        return None

    @property
    def available(self) -> bool:
        return True


class NullOCREngine(OCREngine):
    """Explicit 'OCR is off' engine, so callers never special-case ``None``."""

    name = "null"

    def extract_page(self, page, *, page_number: int) -> OCRResult | None:
        return None

    @property
    def available(self) -> bool:
        return False


class TesseractOCREngine(OCREngine):
    """Rasterise a page with PyMuPDF and run Tesseract over the bitmap."""

    name = "tesseract"

    def __init__(self, *, language: str = "eng", dpi: int = 300, confidence: float = 0.6) -> None:
        self.language = language
        self.dpi = dpi
        self.confidence = confidence
        self._checked = False
        self._ok = False

    @property
    def available(self) -> bool:
        if not self._checked:
            self._checked = True
            try:
                import pytesseract  # noqa: F401
                import pytesseract as _pt
                from PIL import Image  # noqa: F401

                _pt.get_tesseract_version()
                self._ok = True
            except Exception as exc:  # noqa: BLE001 - optional dependency
                warning("ocr.unavailable", engine=self.name, error=str(exc)[:200])
                self._ok = False
        return self._ok

    def extract_page(self, page, *, page_number: int) -> OCRResult | None:
        if not self.available:
            return None
        try:
            import io

            import pytesseract
            from PIL import Image

            pixmap = page.get_pixmap(dpi=self.dpi)
            image = Image.open(io.BytesIO(pixmap.tobytes("png")))
            text = pytesseract.image_to_string(image, lang=self.language)
        except Exception as exc:  # noqa: BLE001 - a failed page must not kill the run
            warning("ocr.page_failed", page=page_number, error=str(exc)[:200])
            return None

        debug("ocr.page_extracted", page=page_number, chars=len(text))
        return OCRResult(text=text, confidence=self.confidence, engine=self.name)


class PaddleOCRVLEngine(OCREngine):
    """Rasterise a page with PyMuPDF and send it to PaddleOCR-VL over HTTP.

    The model (``PaddlePaddle/PaddleOCR-VL-1.6-GGUF``) runs in ``llama-server``,
    which exposes an OpenAI-compatible chat-completions API. The server may be
    started after PaperPilot, so a failed availability probe is retried on the
    next ingestion rather than remembered.
    """

    name = "paddleocr_vl"
    supports_figures = True
    supports_tables = True

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:8080/v1",
        model: str = "PaddleOCR-VL-1.6",
        dpi: int = 200,
        timeout: float = 300.0,
        max_tokens: int = 4096,
        confidence: float = 0.6,
        transport=None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.dpi = dpi
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.confidence = confidence
        self._transport = transport
        self._ok = False
        self._warned = False

    @property
    def available(self) -> bool:
        if self._ok:
            return True
        import httpx

        try:
            with httpx.Client(timeout=3.0, transport=self._transport) as client:
                client.get(f"{self.base_url}/models").raise_for_status()
            self._ok = True
        except httpx.HTTPError as exc:
            if not self._warned:
                self._warned = True
                warning(
                    "ocr.unavailable",
                    engine=self.name,
                    url=self.base_url,
                    error=str(exc)[:200] or type(exc).__name__,
                )
        return self._ok

    def _recognise(
        self, png: bytes, prompt: str, *, page_number: int, max_tokens: int | None = None
    ) -> str | None:
        """One PaddleOCR-VL task prompt ("OCR:", "Chart Recognition:") over an image."""
        import base64

        import httpx

        payload = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": max_tokens or self.max_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/png;base64," + base64.b64encode(png).decode()
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        }
        try:
            with httpx.Client(timeout=self.timeout, transport=self._transport) as client:
                response = client.post(f"{self.base_url}/chat/completions", json=payload)
                response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"] or ""
        except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError) as exc:
            # A failed page must not kill the run; the native text layer is kept.
            warning(
                "ocr.page_failed",
                engine=self.name,
                page=page_number,
                prompt=prompt,
                error=str(exc)[:200],
            )
            return None

    def extract_page(self, page, *, page_number: int) -> OCRResult | None:
        png = page.get_pixmap(dpi=self.dpi).tobytes("png")
        text = self._recognise(png, "OCR:", page_number=page_number)
        if text is None:
            return None
        debug("ocr.page_extracted", engine=self.name, page=page_number, chars=len(text))
        return OCRResult(text=text, confidence=self.confidence, engine=self.name)

    def extract_figure(
        self, page, rect, *, page_number: int, printed_text: str = ""
    ) -> OCRResult | None:
        """Read a chart or diagram with two task prompts.

        ``OCR:`` returns the labels printed in the figure and is kept as-is.
        ``Chart Recognition:`` returns a data table, but it cannot tell a chart
        from a diagram: on a diagram it invents numbers, and on an unlabelled
        chart it estimates values from bar heights. Only rows whose numbers are
        all printed in the figure are indexed, so neither reaches the verifier
        as if the paper had stated it.
        """
        png = page.get_pixmap(dpi=self.dpi, clip=rect).tobytes("png")
        labels = (self._recognise(png, "OCR:", page_number=page_number) or "").strip()
        chart = (
            self._recognise(
                png, "Chart Recognition:", page_number=page_number, max_tokens=_CHART_MAX_TOKENS
            )
            or ""
        )
        rows, dropped = grounded_chart_rows(chart, printed=f"{labels}\n{printed_text}")

        parts = []
        if labels:
            parts.append(f"Figure content read by PaddleOCR-VL:\n{labels}")
        if rows:
            parts.append("Chart data (values printed in the figure):\n" + "\n".join(rows))
        info(
            "ingestion.figure_ocr",
            page=page_number,
            label_chars=len(labels),
            chart_rows=max(len(rows) - 1, 0),
            chart_rows_dropped=dropped,
        )
        if not parts:
            return None
        return OCRResult(text="\n\n".join(parts), confidence=self.confidence, engine=self.name)

    def extract_table(
        self, page, rect, *, page_number: int, printed_text: str = ""
    ) -> OCRResult | None:
        """Read a table with ``Table Recognition:``, verified against the text layer.

        PyMuPDF has every character of a native table but loses its grid: merged
        headers split, cells reorder, superscripts flatten. The model recovers the
        grid but can misread a digit. The model's table is used only when every
        numeric cell's digits occur in ``printed_text`` and its words match;
        otherwise ``None`` is returned and the caller keeps the native text.
        """
        png = page.get_pixmap(dpi=self.dpi, clip=rect).tobytes("png")
        raw = self._recognise(png, "Table Recognition:", page_number=page_number)
        grid, spanned = otsl_to_grid(raw or "")
        check = verify_table(grid, printed_text)
        info(
            "ingestion.table_ocr",
            page=page_number,
            rows=len(grid),
            columns=len(grid[0]) if grid else 0,
            accepted=check.ok,
            reason=check.reason,
        )
        if not check.ok:
            return None
        return OCRResult(
            text=grid_to_markdown(grid, spanned),
            confidence=_VERIFIED_TABLE_CONFIDENCE,
            engine=self.name,
        )


#: Every numeric cell of an accepted table was checked against the native text
#: layer, so its chunks are trusted above unverified OCR output.
_VERIFIED_TABLE_CONFIDENCE = 0.9
#: Share of words that must agree between the model's table and the text layer.
_TABLE_WORD_AGREEMENT = 0.8

_OTSL_TOKEN = re.compile(r"<(fcel|ecel|lcel|ucel|xcel|ched|rhed|srow)>")
_WORD_RE = re.compile(r"[a-z]{3,}")


def _plain_math(text: str) -> str:
    """``\\(1.0 \\cdot 10^{20}\\)`` -> ``1.0 · 10^20``: readable, quotable cell text."""
    text = re.sub(r"\\[()\[\]]", "", text)
    text = text.replace("\\cdot", "·").replace("\\times", "×").replace("\\pm", "±")
    text = re.sub(r"([_^])\{([^{}]*)\}", r"\1\2", text)
    text = re.sub(r"\\(log|exp|max|min|sin|cos)(?![A-Za-z])", r"\1", text)
    text = re.sub(
        r"\\(epsilon|varepsilon|alpha|beta|gamma|delta|lambda|mu|sigma|theta|pi|tau|rho)(?![A-Za-z])",
        lambda m: _GREEK[m.group(1)],
        text,
    )
    return re.sub(r"\s+", " ", text).strip()


_GREEK = {
    "epsilon": "ϵ", "varepsilon": "ε", "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ",
    "lambda": "λ", "mu": "μ", "sigma": "σ", "theta": "θ", "pi": "π", "tau": "τ", "rho": "ρ",
}


def otsl_to_grid(otsl: str) -> tuple[list[list[str]], set[tuple[int, int]]]:
    """Parse PaddleOCR-VL's OTSL table markup into a rectangular grid.

    Merged cells (``<lcel>`` spans left, ``<ucel>`` spans up, ``<xcel>`` both)
    repeat the value they merge with, so every column keeps its header and
    every row its group label. Returns the grid and the positions of cells
    merged horizontally, which a body row must not show twice.
    """
    rows: list[list[tuple[str, str]]] = []
    for line in (otsl or "").split("<nl>"):
        parts = _OTSL_TOKEN.split(line)
        # re.split with one group: [before, tag, text, tag, text, ...]
        cells = [(parts[i], _plain_math(parts[i + 1])) for i in range(1, len(parts) - 1, 2)]
        if cells:
            rows.append(cells)
    if not rows:
        return [], set()

    width = max(len(r) for r in rows)
    grid: list[list[str]] = []
    spanned: set[tuple[int, int]] = set()
    for r, cells in enumerate(rows):
        values: list[str] = []
        for c in range(width):
            tag, text = cells[c] if c < len(cells) else ("ecel", "")
            if tag in ("lcel", "xcel") and c > 0:
                spanned.add((r, c))
            if tag == "lcel" and c > 0:
                values.append(values[c - 1])
            elif tag in ("ucel", "xcel") and r > 0:
                values.append(grid[r - 1][c])
            elif tag == "ecel":
                values.append("")
            else:
                values.append(text)
        grid.append(values)
    return grid, spanned


def _header_rows(otsl_grid: list[list[str]]) -> int:
    """Leading rows that form the header: a row repeating the row above in its
    first column is a second header level (``<ucel>`` under "Model")."""
    count = 1
    while (
        count < len(otsl_grid)
        and otsl_grid[count][0]
        and otsl_grid[count][0] == otsl_grid[count - 1][0]
        and not any(ch.isdigit() for ch in "".join(otsl_grid[count]))
    ):
        count += 1
    return count


def grid_to_markdown(grid: list[list[str]], spanned: set[tuple[int, int]] = frozenset()) -> str:
    """Render the grid as Markdown, folding multi-row headers into column names.

    A value spanning several body columns ("3.3 · 10^18" across EN-DE and EN-FR)
    is shown once, in its first column, rather than read as separate values.
    """
    header_count = _header_rows(grid)
    columns = []
    for c in range(len(grid[0])):
        names: list[str] = []
        for r in range(header_count):
            if grid[r][c] and grid[r][c] not in names:
                names.append(grid[r][c])
        columns.append(" ".join(names))

    def line(cells):
        return "| " + " | ".join(cell.replace("|", "\\|") for cell in cells) + " |"

    out = [line(columns), line(["---"] * len(columns))]
    out += [
        line(["" if (r, c) in spanned else cell for c, cell in enumerate(grid[r])])
        for r in range(header_count, len(grid))
    ]
    return "\n".join(out)


@dataclass(slots=True)
class TableCheck:
    ok: bool
    reason: str


def verify_table(grid: list[list[str]], printed_text: str) -> TableCheck:
    """Accept a recognised table only if it agrees with the native text layer."""
    if len(grid) < 2 or len(grid[0]) < 2:
        return TableCheck(False, "not a table (fewer than 2 rows or columns)")
    printed = printed_text or ""
    printed_digits = re.sub(r"\D", "", printed)
    for row in grid:
        for cell in row:
            digits = re.sub(r"\D", "", cell)
            if digits and digits not in printed_digits:
                return TableCheck(False, f"cell {cell!r} is not in the text layer")

    model_words = set(_WORD_RE.findall(" ".join(" ".join(r) for r in grid).lower()))
    native_words = set(_WORD_RE.findall(printed.lower()))
    if not native_words or not model_words:
        return TableCheck(False, "no words to compare with the text layer")
    recall = len(native_words & model_words) / len(native_words)
    precision = len(native_words & model_words) / len(model_words)
    if min(recall, precision) < _TABLE_WORD_AGREEMENT:
        return TableCheck(False, f"words disagree with the text layer ({recall:.0%}/{precision:.0%})")
    return TableCheck(True, "verified against the text layer")


#: Real chart tables are short. On a diagram the model can instead emit hundreds
#: of invented rows (all later dropped), so the cap bounds that wasted time.
_CHART_MAX_TOKENS = 1024

_NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _numbers(text: str) -> set[str]:
    found = set()
    for token in _NUMBER_RE.findall(text or ""):
        try:
            value = Decimal(token.replace(",", ""))
        except InvalidOperation:
            continue
        found.add(format(value.normalize(), "f"))
    return found


def grounded_chart_rows(chart_text: str, *, printed: str) -> tuple[list[str], int]:
    """Keep the header plus every data row whose numbers are all printed.

    Returns ``(rows, dropped)``. A data row with no number carries no chart value
    and is dropped too. When no data row survives the header is dropped with it.
    """
    lines = [line.strip() for line in (chart_text or "").splitlines() if "|" in line]
    if len(lines) < 2:
        return [], 0
    header, data = lines[0], lines[1:]
    printed_numbers = _numbers(printed)
    kept = [row for row in data if _numbers(row) and _numbers(row) <= printed_numbers]
    dropped = len(data) - len(kept)
    return ([header, *kept] if kept else []), dropped


def build_ocr_engine(
    *,
    enabled: bool,
    language: str = "eng",
    engine: str = "tesseract",
    base_url: str = "http://127.0.0.1:8080/v1",
    model: str = "PaddleOCR-VL-1.6",
    timeout: float = 300.0,
) -> OCREngine:
    """Resolve the OCR engine for the current configuration."""
    if not enabled:
        return NullOCREngine()
    if engine == "paddleocr_vl":
        # Returned even while the server is down: availability is re-probed per
        # ingestion, so starting llama-server later needs no PaperPilot restart.
        return PaddleOCRVLEngine(base_url=base_url, model=model, timeout=timeout)
    engine = TesseractOCREngine(language=language)
    if not engine.available:
        return NullOCREngine()
    return engine
