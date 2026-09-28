#!/usr/bin/env python3
"""Validate the question-only, chapter-balanced BanglaBERT dataset."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset" / "physics_questions_234.json"
FORBIDDEN_FIELDS = {
    "answer",
    "answers",
    "answer_start",
    "context",
    "direct_answer",
    "display_answer",
    "explanation",
    "passage",
    "reference_answer",
}


def validate(path: Path = DATASET) -> dict:
    raw = path.read_bytes()
    document = json.loads(raw.decode("utf-8"))
    rows = document.get("questions", [])

    if document.get("task") != "physics-chapter-classification":
        raise ValueError("Dataset task must be physics-chapter-classification")
    if len(rows) != 234 or document.get("count") != 234:
        raise ValueError("Dataset must contain exactly 234 questions")
    if len({row.get("id") for row in rows}) != len(rows):
        raise ValueError("Question IDs must be unique")
    if len({row.get("question") for row in rows}) != len(rows):
        raise ValueError("Question text must be unique")

    chapter_counts = Counter(int(row["chapter_id"]) for row in rows)
    expected_counts = Counter({chapter_id: 18 for chapter_id in range(1, 14)})
    if chapter_counts != expected_counts:
        raise ValueError(f"Expected 18 questions per chapter; got {dict(chapter_counts)}")

    for row in rows:
        forbidden = FORBIDDEN_FIELDS.intersection(row)
        if forbidden:
            raise ValueError(f"{row['id']} contains answer-bearing fields: {sorted(forbidden)}")
        if not isinstance(row.get("question"), str) or not row["question"].strip():
            raise ValueError(f"{row.get('id')} has an empty question")
        if int(row["label"]) != int(row["chapter_id"]) - 1:
            raise ValueError(f"{row['id']} has an invalid label")

    return {
        "path": str(path),
        "questions": len(rows),
        "chapters": len(chapter_counts),
        "questions_per_chapter": 18,
        "contains_answers": False,
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def main() -> None:
    print(json.dumps(validate(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
