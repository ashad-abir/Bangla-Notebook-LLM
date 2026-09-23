"""Markdown and JSON output for OCR extraction results."""

from __future__ import annotations

import json
import os
from datetime import datetime

from bangla_ocr.extractor import ExtractionResult


class OutputFormatter:
    @staticmethod
    def to_markdown(result: ExtractionResult, include_confidence: bool = False) -> str:
        lines = [
            f"# {os.path.basename(result.input_file)}",
            "",
            f"> OCR pages: {result.processed_pages}/{result.total_pages}",
            "",
        ]
        for page in result.pages:
            lines.extend([f"## পৃষ্ঠা {page.page_number}", ""])
            if page.error:
                lines.append(f"> OCR error: {page.error}")
            elif include_confidence:
                lines.extend(
                    f"{region.text} *({region.confidence:.0%}, {region.label})*"
                    for region in page.regions
                )
            else:
                lines.append(page.text or "*এই পৃষ্ঠায় কোনো লেখা পাওয়া যায়নি।*")
            lines.extend(["", "---", ""])
        return "\n".join(lines)

    @staticmethod
    def to_json(result: ExtractionResult, include_bboxes: bool = True) -> str:
        pages = []
        for page in result.pages:
            item = {
                "page_number": page.page_number,
                "text": page.text,
                "processing_time_seconds": round(page.processing_time, 2),
            }
            if page.error:
                item["error"] = page.error
            if include_bboxes:
                item["regions"] = [
                    {
                        "text": region.text,
                        "confidence": round(region.confidence, 4),
                        "bounding_box": region.bbox,
                        "polygon": region.polygon,
                        "label": region.label,
                        "html": region.html,
                    }
                    for region in page.regions
                ]
            pages.append(item)
        return json.dumps(
            {
                "metadata": {
                    "input_file": result.input_file,
                    "extraction_date": datetime.now().isoformat(),
                    "total_pages": result.total_pages,
                    "processed_pages": result.processed_pages,
                    "success_rate": round(result.success_rate, 2),
                    "total_time_seconds": round(result.total_time, 2),
                    "errors": result.errors,
                },
                "pages": pages,
            },
            ensure_ascii=False,
            indent=2,
        )

    @classmethod
    def save(
        cls,
        result: ExtractionResult,
        output_path: str,
        output_format: str,
        **kwargs,
    ) -> str:
        if output_format == "markdown":
            content = cls.to_markdown(result, **kwargs)
        elif output_format == "json":
            content = cls.to_json(result, **kwargs)
        else:
            raise ValueError("Output format must be 'markdown' or 'json'.")
        parent = os.path.dirname(os.path.abspath(output_path))
        os.makedirs(parent, exist_ok=True)
        with open(output_path, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
        return os.path.abspath(output_path)

