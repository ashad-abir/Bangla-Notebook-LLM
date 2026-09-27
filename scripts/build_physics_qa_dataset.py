#!/usr/bin/env python3
"""Build the reviewed Physics QA set used for training and displayed answers."""

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
from scripts.physics_answer_explanations import EXPLANATIONS


OUTPUT = ROOT / "dataset" / "physics_qa_234.json"

VERIFICATION_SOURCES = [
    {
        "name": "NCTB Classes 9–10 Physics textbook (local OCR edition)",
        "url": None,
        "scope": "Primary Bengali curriculum terminology and chapter coverage",
    },
    {
        "name": "BIPM SI Brochure and SI base units",
        "url": "https://www.bipm.org/en/publications/si-brochure",
        "scope": "SI quantities, base units, derived units, and symbols",
    },
    {
        "name": "NIST Guide to the SI",
        "url": "https://www.nist.gov/pml/special-publication-811",
        "scope": "Measurement terminology, units, and dimensional consistency",
    },
    {
        "name": "OpenStax Physics",
        "url": "https://openstax.org/details/books/physics",
        "scope": "Independent verification of mechanics, heat, waves, optics, electricity, magnetism, and atomic physics",
    },
]


def main() -> None:
    catalog = json.loads((ROOT / "config" / "books.json").read_text(encoding="utf-8"))
    book = next(item for item in catalog if item["id"] == "physics-9-10")
    chapters = {int(item["id"]): item for item in book["chapters"]}
    assert QUESTIONS.keys() == ANSWERS.keys() == EXPLANATIONS.keys()

    examples = []
    for chapter_id in range(1, 14):
        questions = QUESTIONS[chapter_id]
        answers = ANSWERS[chapter_id]
        explanations = EXPLANATIONS[chapter_id]
        assert len(questions) == len(answers) == len(explanations) == 18
        chapter = chapters[chapter_id]
        detailed_answers = [
            f"{answer} {explanation}" for answer, explanation in zip(answers, explanations, strict=True)
        ]
        for index, (question, direct_answer, explanation, answer) in enumerate(
            zip(questions, answers, explanations, detailed_answers, strict=True)
        ):
            distractor_one = detailed_answers[(index + 5) % len(detailed_answers)]
            distractor_two = detailed_answers[(index + 11) % len(detailed_answers)]
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
                    "direct_answer": direct_answer,
                    "explanation": explanation,
                    "reference_answer": answer,
                    "display_answer": answer,
                    "answer_style": "direct-answer-plus-student-explanation",
                    "book_id": book["id"],
                    "chapter_id": chapter_id,
                    "chapter_title": chapter["title_bn"],
                    "chapter_page_range": [chapter["page_start"], chapter["page_end"]],
                    "annotation_method": "chatgpt-authored-book-grounded-student-answer",
                }
            )

    assert len(examples) == 234
    assert Counter(item["chapter_id"] for item in examples) == {chapter: 18 for chapter in range(1, 14)}
    document = {
        "version": "4.0",
        "language": "bn",
        "task": "extractive-question-answering",
        "source": book["markdown_path"],
        "count": len(examples),
        "questions_per_chapter": 18,
        "context_policy": "Each training context contains one complete student-facing answer and two same-chapter distractors. The complete answer is one contiguous extractive span.",
        "answer_policy": "Lead with a direct answer, then add a concise explanation, equation interpretation, consequence, or example that completes a secondary-school student's understanding.",
        "answer_authorship": "Direct answers were curriculum-verified; teaching explanations were authored by ChatGPT and reviewed through deterministic validation. No local Qwen output was used.",
        "verification_sources": VERIFICATION_SOURCES,
        "examples": examples,
    }
    OUTPUT.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(examples)} detailed exact-span QA examples to {OUTPUT}")


if __name__ == "__main__":
    main()
