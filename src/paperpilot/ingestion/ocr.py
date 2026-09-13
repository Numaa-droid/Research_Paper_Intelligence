"""OCR fallback, isolated behind an interface.

Tesseract is optional. Nothing in the pipeline imports ``pytesseract`` directly:
the engine is resolved at construction time and a missing binary degrades to
"no OCR" with a logged warning instead of a crash.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..observability import debug, warning


@dataclass(slots=True)
class OCRResult:
    text: str
    confidence: float = 0.6
    engine: str = "unknown"


class OCREngine(ABC):
    """Interface for page-level OCR."""

    name = "abstract"

    @abstractmethod
    def extract_page(self, page, *, page_number: int) -> OCRResult | None:
        """Return OCR text for a PyMuPDF page, or ``None`` when unavailable."""

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


def build_ocr_engine(*, enabled: bool, language: str = "eng") -> OCREngine:
    """Resolve the OCR engine for the current configuration."""
    if not enabled:
        return NullOCREngine()
    engine = TesseractOCREngine(language=language)
    if not engine.available:
        return NullOCREngine()
    return engine
