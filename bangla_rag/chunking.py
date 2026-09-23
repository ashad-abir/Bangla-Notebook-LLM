import hashlib
import re

from bangla_rag.models import Book, Chunk, Page


WORD = re.compile(r"\S+")


def _chunk_id(book_id: str, page: int, ordinal: int, text: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:10]
    return f"{book_id}-p{page}-c{ordinal}-{digest}"


def chunk_pages(
    book: Book,
    pages: list[Page],
    max_words: int = 220,
    overlap_words: int = 35,
) -> list[Chunk]:
    if max_words <= 0 or overlap_words < 0 or overlap_words >= max_words:
        raise ValueError("Invalid chunk size or overlap")
    chunks: list[Chunk] = []
    for page in pages:
        words = WORD.findall(page.text)
        start = 0
        ordinal = 0
        while start < len(words):
            end = min(start + max_words, len(words))
            if end < len(words):
                minimum = start + max_words // 2
                for position in range(end - 1, minimum, -1):
                    if words[position - 1].endswith(("।", "?", "!", ".")):
                        end = position
                        break
            text = " ".join(words[start:end]).strip()
            if text:
                chunks.append(
                    Chunk(
                        _chunk_id(book.id, page.number, ordinal, text),
                        book.id,
                        book.title,
                        page.number,
                        ordinal,
                        text,
                    )
                )
                ordinal += 1
            if end >= len(words):
                break
            start = max(start + 1, end - overlap_words)
    return chunks

