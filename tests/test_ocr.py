import argparse
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from bangla_ocr.cli import parse_page_range
from bangla_ocr.extractor import BanglaOCRExtractor, ExtractionResult, PageResult
from bangla_ocr.output_formatter import OutputFormatter


class FakePdf:
    def __init__(self, page_count):
        self.pages = [object() for _ in range(page_count)]
        self.closed = False

    def __len__(self):
        return len(self.pages)

    def __getitem__(self, index):
        return self.pages[index]

    def close(self):
        self.closed = True


class OCRTests(unittest.TestCase):
    def test_page_range_validation(self):
        self.assertEqual(parse_page_range("3"), (3, 3))
        self.assertEqual(parse_page_range("2-7"), (2, 7))
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_page_range("7-2")

    def test_markdown_uses_parser_compatible_page_headings(self):
        result = ExtractionResult("book.pdf", 1, 1, [PageResult(4, "বাংলা লেখা")])
        markdown = OutputFormatter.to_markdown(result)
        self.assertIn("## পৃষ্ঠা 4", markdown)
        self.assertIn("বাংলা লেখা", markdown)

    def test_json_keeps_unicode_and_page_metadata(self):
        result = ExtractionResult("book.pdf", 1, 1, [PageResult(1, "পদার্থবিজ্ঞান")])
        data = json.loads(OutputFormatter.to_json(result))
        self.assertEqual(data["pages"][0]["text"], "পদার্থবিজ্ঞান")
        self.assertEqual(data["metadata"]["processed_pages"], 1)

    def test_extractor_processes_one_page_per_predictor_call(self):
        fake_pdf = FakePdf(2)
        predictor = Mock(
            side_effect=[
                [SimpleNamespace(blocks=[])],
                [SimpleNamespace(blocks=[])],
            ]
        )
        extractor = BanglaOCRExtractor(dpi=96)
        with tempfile.TemporaryDirectory() as temp_dir:
            pdf_path = Path(temp_dir) / "book.pdf"
            pdf_path.write_bytes(b"fake")
            with (
                patch.object(extractor, "_open_pdf", return_value=fake_pdf),
                patch.object(extractor, "_get_predictor", return_value=predictor),
                patch.object(extractor, "_render_page", return_value=object()),
            ):
                result = extractor.extract(str(pdf_path))

        self.assertEqual([len(call.args[0]) for call in predictor.call_args_list], [1, 1])
        self.assertEqual(result.processed_pages, 2)
        self.assertTrue(fake_pdf.closed)


if __name__ == "__main__":
    unittest.main()

