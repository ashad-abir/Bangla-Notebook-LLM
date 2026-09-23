"""Command-line interface for PDF/image to Markdown or JSON conversion."""

from __future__ import annotations

import argparse
import os

from bangla_ocr.extractor import BanglaOCRExtractor, ExtractionResult
from bangla_ocr.output_formatter import OutputFormatter


def parse_page_range(value: str) -> tuple[int, int]:
    try:
        start_text, separator, end_text = value.partition("-")
        start = int(start_text.strip())
        end = int(end_text.strip()) if separator else start
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Pages must look like 1 or 1-50.") from exc
    if start < 1 or end < start:
        raise argparse.ArgumentTypeError("Pages must be positive and in ascending order.")
    return start, end


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bangla_ocr",
        description="Convert a Bengali textbook PDF or image with Surya OCR 2.",
    )
    parser.add_argument("input", help="PDF or image file")
    parser.add_argument("-o", "--output", required=True, help="Output .md or .json file")
    parser.add_argument("--pages", type=parse_page_range, help="1-indexed page or range")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--confidence", type=float, default=0.2)
    parser.add_argument("--keep-server", action="store_true")
    parser.add_argument("--show-confidence", action="store_true")
    parser.add_argument("--no-bboxes", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    input_path = os.path.abspath(args.input)
    output_path = os.path.abspath(args.output)
    if not os.path.isfile(input_path):
        raise SystemExit(f"Input file not found: {input_path}")

    extension = os.path.splitext(input_path)[1].lower()
    output_extension = os.path.splitext(output_path)[1].lower()
    if output_extension not in {".md", ".json"}:
        raise SystemExit("Output must end in .md or .json.")

    extractor = BanglaOCRExtractor(args.confidence, args.dpi, args.keep_server)
    if extension == ".pdf":
        result = extractor.extract(input_path, args.pages)
    elif extension in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}:
        page = extractor.extract_from_image(input_path)
        result = ExtractionResult(
            input_file=input_path,
            total_pages=1,
            processed_pages=0 if page.error else 1,
            pages=[page],
            total_time=page.processing_time,
            errors=[page.error] if page.error else [],
        )
    else:
        raise SystemExit(f"Unsupported input type: {extension}")

    if output_extension == ".md":
        saved = OutputFormatter.save(
            result,
            output_path,
            "markdown",
            include_confidence=args.show_confidence,
        )
    else:
        saved = OutputFormatter.save(
            result,
            output_path,
            "json",
            include_bboxes=not args.no_bboxes,
        )
    print(f"OCR output saved to: {saved}")

