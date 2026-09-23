"""Catalog metadata for class, subject, and book selection."""

import json
from pathlib import Path
from typing import Any


def load_catalog(config_path: Path, chunk_counts: dict[str, int] | None = None) -> dict[str, Any]:
    records = json.loads(config_path.read_text(encoding="utf-8"))
    counts = chunk_counts or {}
    classes: dict[str, dict[str, str]] = {}
    subjects: dict[str, dict[str, Any]] = {}
    books = []

    for record in records:
        book_classes = record.get("classes") or []
        subject = record.get("subject") or {}
        if not book_classes or not subject.get("id"):
            raise ValueError(
                f"Book {record.get('id', '<unknown>')!r} needs classes and subject metadata"
            )
        class_ids = []
        for class_record in book_classes:
            class_id = str(class_record["id"])
            class_ids.append(class_id)
            classes.setdefault(
                class_id,
                {
                    "id": class_id,
                    "label_bn": class_record["label_bn"],
                    "label_en": class_record["label_en"],
                },
            )

        subject_id = subject["id"]
        subject_entry = subjects.setdefault(
            subject_id,
            {
                "id": subject_id,
                "label_bn": subject["label_bn"],
                "label_en": subject["label_en"],
                "icon": subject.get("icon", "•"),
                "class_ids": [],
                "book_ids": [],
            },
        )
        for class_id in class_ids:
            if class_id not in subject_entry["class_ids"]:
                subject_entry["class_ids"].append(class_id)
        subject_entry["book_ids"].append(record["id"])

        books.append(
            {
                "id": record["id"],
                "title_bn": record["title"],
                "title_en": record.get("title_en", record["title"]),
                "class_ids": class_ids,
                "subject_id": subject_id,
                "chunk_count": counts.get(record["id"], 0),
                "chapters": record.get("chapters", []),
            }
        )

    def class_order(item: dict[str, str]):
        first_grade = item["id"].split("-", 1)[0]
        return (0, int(first_grade)) if first_grade.isdigit() else (1, item["id"])

    return {
        "classes": sorted(classes.values(), key=class_order),
        "subjects": sorted(subjects.values(), key=lambda item: item["id"]),
        "books": books,
    }
