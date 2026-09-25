#!/usr/bin/env python3
"""Build exact-span Physics QA examples from the curated answer key."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_physics_question_dataset import QUESTIONS
from scripts.physics_answer_key import ANSWERS


OUTPUT = ROOT / "dataset" / "physics_qa_234.json"


def main() -> None:
    catalog = json.loads((ROOT / "config" / "books.json").read_text(encoding="utf-8"))
    book = next(item for item in catalog if item["id"] == "physics-9-10")
    chapters = {int(item["id"]): item for item in book["chapters"]}
    assert QUESTIONS.keys() == ANSWERS.keys()

    examples = []
    for chapter_id in range(1, 14):
        questions = QUESTIONS[chapter_id]
        answers = ANSWERS[chapter_id]
        assert len(questions) == len(answers) == 18
        chapter = chapters[chapter_id]
        for index, (question, answer) in enumerate(zip(questions, answers, strict=True)):
            distractor_one = answers[(index + 5) % len(answers)]
            distractor_two = answers[(index + 11) % len(answers)]
            passages = [answer, distractor_one, distractor_two]
            rotation = index % 3
            passages = passages[rotation:] + passages[:rotation]
            context = " ".join(passages)
            answer_start = context.find(answer)
            assert answer_start >= 0
            assert context[answer_start : answer_start + len(answer)] == answer
            examples.append(
                {
                    "id": f"physics-{chapter_id:02d}-{index + 1:02d}",
                    "question": question,
                    "context": context,
                    "answers": {"text": [answer], "answer_start": [answer_start]},
                    "book_id": book["id"],
                    "chapter_id": chapter_id,
                    "chapter_title": chapter["title_bn"],
                    "chapter_page_range": [chapter["page_start"], chapter["page_end"]],
                    "annotation_method": "curated-curriculum-answer-with-distractors",
                }
            )

    assert len(examples) == 234
    assert Counter(item["chapter_id"] for item in examples) == {chapter: 18 for chapter in range(1, 14)}
    document = {
        "version": "2.0",
        "language": "bn",
        "task": "extractive-question-answering",
        "source": book["markdown_path"],
        "count": len(examples),
        "questions_per_chapter": 18,
        "context_policy": "Each context contains the curated answer and two same-chapter distractor sentences.",
        "examples": examples,
    }
    OUTPUT.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(examples)} exact-span QA examples to {OUTPUT}")


if __name__ == "__main__":
    main()
