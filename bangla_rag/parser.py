import re
import unicodedata
from pathlib import Path

from bangla_rag.models import Book, Chapter, Page


PAGE_HEADING = re.compile(r"^##\s+পৃষ্ঠা\s+([0-9]+)\s*$")
PROCESSING_TIME = re.compile(r"^\*⏱\s*Processing time:.*\*\s*$")
OCR_ERROR = re.compile(r"^>\s*⚠️?\s*\*\*Error:\*\*\s*OCR processing failed for this page\s*$")


class BookParseError(ValueError):
    pass


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip()


def parse_book(book: Book) -> list[Page]:
    try:
        content = book.markdown_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise BookParseError(f"Could not read {book.markdown_path}: {exc}") from exc

    pages: list[Page] = []
    current_number: int | None = None
    current_lines: list[str] = []

    def finish() -> None:
        if current_number is None:
            return
        cleaned = [
            line.strip()
            for line in current_lines
            if line.strip() != "---"
            and not PROCESSING_TIME.fullmatch(line.strip())
            and not OCR_ERROR.fullmatch(line.strip())
        ]
        pages.append(Page(book.id, current_number, normalize("\n".join(cleaned))))

    for line in content.splitlines():
        match = PAGE_HEADING.fullmatch(line.strip())
        if match:
            finish()
            number = int(match.group(1))
            if pages and number <= pages[-1].number:
                raise BookParseError("Page numbers must be strictly increasing")
            current_number = number
            current_lines = []
        elif current_number is not None:
            current_lines.append(line)
    finish()

    if not pages:
        raise BookParseError("No '## পৃষ্ঠা N' headings were found")
    return pages


def load_books(repo_root: Path) -> list[Book]:
    import json

    path = repo_root / "config" / "books.json"
    records = json.loads(path.read_text(encoding="utf-8"))
    books = []
    for record in records:
        source = (repo_root / record["markdown_path"]).resolve()
        chapters = tuple(
            Chapter(
                str(chapter["id"]),
                chapter["title_bn"],
                int(chapter["page_start"]),
                int(chapter["page_end"]),
            )
            for chapter in record.get("chapters", [])
        )
        for previous, current in zip(chapters, chapters[1:]):
            if previous.page_end >= current.page_start:
                raise BookParseError(f"Overlapping chapter ranges in {record['id']}")
        books.append(Book(record["id"], record["title"], source, chapters))
    return books
