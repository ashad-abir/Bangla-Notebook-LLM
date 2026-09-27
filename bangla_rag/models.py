from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


@dataclass(frozen=True)
class Book:
    id: str
    title: str
    markdown_path: Path
    chapters: tuple["Chapter", ...] = ()


@dataclass(frozen=True)
class Chapter:
    id: str
    title: str
    page_start: int
    page_end: int


@dataclass(frozen=True)
class Page:
    book_id: str
    number: int
    text: str


@dataclass(frozen=True)
class Chunk:
    id: str
    book_id: str
    book_title: str
    page: int
    ordinal: int
    text: str
    chapter_id: str | None = None
    chapter_title: str | None = None


@dataclass(frozen=True)
class SearchHit:
    chunk: Chunk
    fused_score: float
    semantic_score: float | None = None
    semantic_rank: int | None = None
    keyword_rank: int | None = None


@dataclass(frozen=True)
class SearchResult:
    hits: tuple[SearchHit, ...]
    sufficient_evidence: bool
    strongest_semantic_score: float | None
    has_keyword_match: bool
    debug: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Citation:
    book_title: str
    page: int
    chunk_ids: tuple[str, ...]
    excerpt: str


@dataclass(frozen=True)
class Answer:
    status: Literal["answered", "not_found", "error"]
    text: str
    citations: tuple[Citation, ...] = ()
    error: str | None = None
    debug: dict[str, Any] = field(default_factory=dict)
    model: str | None = None
    elapsed_seconds: float | None = None
    confidence: float | None = None
