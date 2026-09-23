"""PDF/image extraction using the optional Surya OCR 2 runtime."""

from __future__ import annotations

import html
import os
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class OCRResult:
    text: str
    confidence: float
    bbox: list[Any]
    polygon: list[Any] = field(default_factory=list)
    label: str = "Text"
    html: str = ""


@dataclass
class PageResult:
    page_number: int
    text: str
    regions: list[OCRResult] = field(default_factory=list)
    processing_time: float = 0.0
    error: str | None = None


@dataclass
class ExtractionResult:
    input_file: str
    total_pages: int
    processed_pages: int
    pages: list[PageResult] = field(default_factory=list)
    total_time: float = 0.0
    errors: list[str] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        if self.total_pages == 0:
            return 0.0
        return self.processed_pages / self.total_pages * 100


def _plain_text(value: str) -> str:
    text = re.sub(r"<[^>]+>", " ", value or "")
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


class BanglaOCRExtractor:
    """Extract page-aware Bengali text while loading OCR dependencies lazily."""

    def __init__(
        self,
        confidence_threshold: float = 0.2,
        dpi: int = 300,
        keep_server: bool = False,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.dpi = dpi
        self.keep_server = keep_server
        self._manager = None
        self._predictor = None

    def _get_predictor(self):
        if self._predictor is None:
            try:
                from surya.inference import SuryaInferenceManager
                from surya.recognition import RecognitionPredictor
            except ImportError as exc:
                raise RuntimeError(
                    "Surya OCR is not installed. Install requirements-ocr.txt first."
                ) from exc

            print("Loading Surya OCR 2...", file=sys.stderr)
            if self.keep_server:
                os.environ["SURYA_INFERENCE_KEEP_ALIVE"] = "1"
            self._manager = SuryaInferenceManager()
            self._predictor = RecognitionPredictor(self._manager)
        return self._predictor

    @staticmethod
    def _open_pdf(pdf_path: str):
        try:
            import pypdfium2 as pdfium
        except ImportError as exc:
            raise RuntimeError(
                "PDF support is not installed. Install requirements-ocr.txt first."
            ) from exc
        return pdfium.PdfDocument(pdf_path)

    def _render_page(self, pdf_page):
        image = pdf_page.render(scale=self.dpi / 72.0).to_pil()
        return image if image.mode == "RGB" else image.convert("RGB")

    def _parse_page(self, surya_page, page_number: int) -> PageResult:
        regions: list[OCRResult] = []
        for block in getattr(surya_page, "blocks", []):
            if getattr(block, "skipped", False) or getattr(block, "error", False):
                continue
            confidence = getattr(block, "confidence", 0.0) or 0.0
            text = _plain_text(getattr(block, "html", "") or "")
            if confidence < self.confidence_threshold or not text:
                continue
            regions.append(
                OCRResult(
                    text=text,
                    confidence=confidence,
                    bbox=getattr(block, "bbox", None) or [0, 0, 0, 0],
                    polygon=getattr(block, "polygon", None) or [],
                    label=getattr(block, "label", None) or "Text",
                    html=getattr(block, "html", "") or "",
                )
            )
        return PageResult(
            page_number=page_number,
            text="\n".join(region.text for region in regions),
            regions=regions,
        )

    def extract(
        self,
        pdf_path: str,
        page_range: tuple[int, int] | None = None,
        callback: Callable[[int, int, PageResult], None] | None = None,
    ) -> ExtractionResult:
        if not os.path.isfile(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        try:
            from tqdm import tqdm
        except ImportError as exc:
            raise RuntimeError(
                "OCR progress support is not installed. Install requirements-ocr.txt first."
            ) from exc

        started = time.time()
        pdf = self._open_pdf(pdf_path)
        try:
            pdf_page_count = len(pdf)
            start, end = page_range or (1, pdf_page_count)
            if start < 1 or end < start or start > pdf_page_count:
                raise ValueError(
                    f"Invalid page range {start}-{end} for a {pdf_page_count}-page PDF."
                )
            end = min(end, pdf_page_count)
            page_numbers = range(start, end + 1)
            result = ExtractionResult(
                input_file=os.path.abspath(pdf_path),
                total_pages=end - start + 1,
                processed_pages=0,
            )
            predictor = self._get_predictor()

            with tqdm(
                page_numbers,
                total=result.total_pages,
                desc="OCR progress",
                unit="page",
            ) as progress:
                for page_number in progress:
                    page_started = time.time()
                    try:
                        image = self._render_page(pdf[page_number - 1])
                        predictions = predictor([image])
                        if not predictions:
                            raise RuntimeError("Surya returned no page result")
                        page = self._parse_page(predictions[0], page_number)
                    except Exception as exc:  # preserve other pages on one-page failure
                        page = PageResult(page_number, "", error=str(exc))
                        result.errors.append(f"Page {page_number}: {exc}")
                    page.processing_time = time.time() - page_started
                    result.pages.append(page)
                    if page.error is None:
                        result.processed_pages += 1
                    progress.set_postfix(page=page_number)
                    if callback:
                        callback(page_number, result.total_pages, page)
        finally:
            pdf.close()

        result.total_time = time.time() - started
        return result

    def extract_from_image(self, image_path: str) -> PageResult:
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError(
                "Image support is not installed. Install requirements-ocr.txt first."
            ) from exc

        started = time.time()
        with Image.open(image_path) as source:
            image = source.convert("RGB")
            predictions = self._get_predictor()([image])
        if not predictions:
            return PageResult(
                page_number=1,
                text="",
                processing_time=time.time() - started,
                error="Surya returned no result",
            )
        page = self._parse_page(predictions[0], 1)
        page.processing_time = time.time() - started
        return page
