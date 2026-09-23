import json
import re
import tempfile
import unittest
from pathlib import Path

import numpy as np

from bangla_rag.chunking import chunk_pages
from bangla_rag.index_store import build_index, load_index
from bangla_rag.models import Book, Chunk, SearchHit, SearchResult
from bangla_rag.parser import load_books, parse_book
from bangla_rag.retriever import HybridRetriever
from bangla_rag.service import FALLBACK_TEXT, QAService, QuizService


class FakeEmbeddings:
    def __init__(self, vector):
        self.vector = np.asarray(vector, dtype=np.float32)

    def query(self, _text):
        return self.vector


class FakeLLM:
    def __init__(self, response):
        self.response = response
        self.calls = 0

    def complete_json(self, *_args, **_kwargs):
        self.calls += 1
        return self.response


class BatchQuizLLM:
    def __init__(self):
        self.calls = 0

    def complete_json(self, system, user, **_kwargs):
        self.calls += 1
        count = int(re.search(r"COUNT: (\d+)", user).group(1))
        source_ids = re.findall(r"SOURCE_ID: ([^\n]+)", user)
        questions = []
        for offset in range(count):
            number = (self.calls - 1) * 5 + offset + 1
            if "multiple-choice" in system:
                questions.append({
                    "question": f"এমসিকিউ প্রশ্ন {number}?",
                    "options": {"A": "এক", "B": "দুই", "C": "তিন", "D": "চার"},
                    "correct_answer": "A",
                    "explanation": "সংক্ষিপ্ত ব্যাখ্যা",
                    "difficulty": "সহজ",
                    "source_ids": [source_ids[offset % len(source_ids)]],
                })
            else:
                questions.append({
                    "question": f"জ্ঞানমূলক প্রশ্ন {number}?",
                    "answer": "সংক্ষিপ্ত উত্তর।",
                    "difficulty": "সহজ",
                    "source_ids": [source_ids[offset % len(source_ids)]],
                })
        return {"questions": questions}


class CoreTests(unittest.TestCase):
    def test_real_book_has_366_pages(self):
        source = Path(__file__).resolve().parents[1] / "dataset" / "raw" / "physics.md"
        pages = parse_book(Book("physics-9-10", "পদার্থবিজ্ঞান", source))
        self.assertEqual([page.number for page in pages], list(range(1, 367)))

    def test_catalog_books_are_parseable(self):
        root = Path(__file__).resolve().parents[1]
        books = {book.id: book for book in load_books(root)}

        self.assertEqual(set(books), {"bgs-8", "physics-9-10"})
        self.assertEqual(len(parse_book(books["bgs-8"])), 149)

    def test_chunks_never_cross_pages(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "book.md"
            source.write_text("## পৃষ্ঠা 1\nএক দুই তিন চার\n## পৃষ্ঠা 2\nপাঁচ ছয়", encoding="utf-8")
            book = Book("book", "বই", source)
            chunks = chunk_pages(book, parse_book(book), max_words=3, overlap_words=1)
            self.assertEqual({chunk.page for chunk in chunks}, {1, 2})
            self.assertTrue(all("পাঁচ" not in chunk.text for chunk in chunks if chunk.page == 1))

    def test_low_evidence_refuses_without_calling_llm(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            chunk = Chunk("c1", "book", "বই", 1, 0, "আলোর প্রতিফলন")
            build_index(root, [chunk], np.asarray([[0.0, 1.0]], dtype=np.float32), "test")
            retriever = HybridRetriever(load_index(root), FakeEmbeddings([1.0, 0.0]))
            llm = FakeLLM({})
            answer = QAService(retriever, llm).ask("বিরিয়ানির রেসিপি")
            self.assertEqual(answer.text, FALLBACK_TEXT)
            self.assertEqual(llm.calls, 0)

    def test_invalid_source_id_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            chunk = Chunk("c1", "book", "বই", 1, 0, "বল এবং নিউটনের সূত্র")
            build_index(root, [chunk], np.asarray([[1.0, 0.0]], dtype=np.float32), "test")
            retriever = HybridRetriever(load_index(root), FakeEmbeddings([1.0, 0.0]))
            llm = FakeLLM({"answerable": True, "answer": "উত্তর", "source_ids": ["invented"]})
            answer = QAService(retriever, llm).ask("নিউটনের সূত্র কী")
            self.assertEqual(answer.text, FALLBACK_TEXT)

    def test_retrieval_is_limited_to_selected_books(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            chunks = [
                Chunk("physics", "physics", "পদার্থবিজ্ঞান", 1, 0, "বল এবং নিউটনের সূত্র"),
                Chunk("chemistry", "chemistry", "রসায়ন", 1, 0, "পরমাণুর গঠন"),
            ]
            build_index(
                root,
                chunks,
                np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
                "test",
            )
            retriever = HybridRetriever(
                load_index(root),
                FakeEmbeddings([1.0, 0.0]),
                semantic_threshold=0.0,
            )

            result = retriever.search("নিউটনের সূত্র", book_ids={"chemistry"})

            self.assertTrue(result.hits)
            self.assertTrue(all(hit.chunk.book_id == "chemistry" for hit in result.hits))

    def test_quiz_service_batches_mcq_and_builds_short_answer_quiz(self):
        chunk = Chunk("c1", "physics", "পদার্থবিজ্ঞান", 10, 0, "বল ও নিউটনের সূত্র")
        hit = SearchHit(chunk, 1.0, semantic_score=0.9)

        class QuizRetriever:
            by_id = {"c1": chunk}

            @staticmethod
            def search(*_args, **_kwargs):
                return SearchResult((hit,), True, 0.9, True)

        llm = BatchQuizLLM()
        service = QuizService(QuizRetriever(), llm)

        mcq = service.generate("বল", 10, "mcq", {"physics"})
        knowledge = service.generate("বল", 5, "knowledge", {"physics"})

        self.assertEqual(mcq["status"], "ok")
        self.assertEqual(len(mcq["questions"]), 10)
        self.assertEqual(mcq["questions"][0]["correct_answer"], "A")
        self.assertEqual(knowledge["status"], "ok")
        self.assertEqual(len(knowledge["questions"]), 5)
        self.assertEqual(knowledge["questions"][0]["answer"], "সংক্ষিপ্ত উত্তর।")

    def test_quiz_service_enforces_type_specific_count_ranges(self):
        service = QuizService(None, None)
        with self.assertRaises(ValueError):
            service.generate("বল", 9, "mcq")
        with self.assertRaises(ValueError):
            service.generate("বল", 11, "knowledge")

    def test_quiz_service_uses_diverse_chunks_from_chapter_page_range(self):
        chunks = tuple(
            Chunk(f"c{page}", "physics", "পদার্থবিজ্ঞান", page, 0, f"পৃষ্ঠা {page} এর বিষয়")
            for page in range(60, 111)
        )

        class ChapterRetriever:
            pass

        retriever = ChapterRetriever()
        retriever.index = type("Index", (), {"chunks": chunks})()
        retriever.by_id = {chunk.id: chunk for chunk in chunks}

        llm = BatchQuizLLM()
        result = QuizService(retriever, llm).generate(
            "বল", 10, "mcq", {"physics"}, page_range=(67, 102)
        )
        pages = {source["page"] for item in result["questions"] for source in item["sources"]}

        self.assertEqual(result["accepted_count"], 10)
        self.assertGreater(len(pages), 3)
        self.assertTrue(all(67 <= page <= 102 for page in pages))


if __name__ == "__main__":
    unittest.main()
