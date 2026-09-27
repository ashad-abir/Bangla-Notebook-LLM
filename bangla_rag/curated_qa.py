"""Exact lookup for human-reviewed answers in the BanglaBERT QA dataset."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import unicodedata


@dataclass(frozen=True)
class CuratedAnswer:
    id: str
    question: str
    answer: str
    chapter_id: int
    chapter_title: str
    chapter_page_range: tuple[int, int]


def normalize_question(text: str) -> str:
    """Normalize harmless spelling/formatting differences without fuzzy guessing."""
    normalized = unicodedata.normalize("NFC", text).casefold().strip()
    normalized = normalized.replace("?", "").replace("？", "").replace("।", "")
    return re.sub(r"\s+", " ", normalized).strip()


class CuratedQACatalog:
    """Provide a precise display answer only for an exact curated question."""

    def __init__(self, entries: dict[str, CuratedAnswer]) -> None:
        self._entries = entries

    @classmethod
    def from_json(cls, path: Path) -> "CuratedQACatalog":
        document = json.loads(path.read_text(encoding="utf-8"))
        entries: dict[str, CuratedAnswer] = {}
        for item in document.get("examples", []):
            answer = str(
                item.get("display_answer")
                or item.get("reference_answer")
                or item["answers"]["text"][0]
            ).strip()
            page_range = item.get("chapter_page_range", [0, 0])
            entry = CuratedAnswer(
                id=str(item["id"]),
                question=str(item["question"]).strip(),
                answer=answer,
                chapter_id=int(item["chapter_id"]),
                chapter_title=str(item["chapter_title"]),
                chapter_page_range=(int(page_range[0]), int(page_range[1])),
            )
            key = normalize_question(entry.question)
            if not key or key in entries:
                raise ValueError(f"Duplicate or empty curated question: {entry.question!r}")
            entries[key] = entry
        if not entries:
            raise ValueError(f"No QA examples found in {path}")
        return cls(entries)

    def exact_match(self, question: str) -> CuratedAnswer | None:
        return self._entries.get(normalize_question(question))

    def __len__(self) -> int:
        return len(self._entries)
