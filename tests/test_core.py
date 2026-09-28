import json
import re
import tempfile
import unittest
from pathlib import Path

import numpy as np

from bangla_rag.chunking import chunk_pages
from bangla_rag.chapter_classifier import ChapterPrediction, LocalChapterClassifier
from bangla_rag.index_store import build_index, load_index
from bangla_rag.models import Book, Chapter, Chunk, SearchHit, SearchResult
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


class FakeChapterClassifier:
    def __init__(self, chapter_id="3", confidence=0.9):
        self.chapter_id = chapter_id
        self.confidence = confidence
        self.calls = 0

    def predict(self, _question, top_k=3):
        self.calls += 1
        predictions = [
            ChapterPrediction(self.chapter_id, f"chapter_{int(self.chapter_id):02d}", "বল", self.confidence),
            ChapterPrediction("2", "chapter_02", "গতি", 0.05),
            ChapterPrediction("4", "chapter_04", "কাজ, ক্ষমতা ও শক্তি", 0.05),
        ]
        return tuple(predictions[:top_k])

    @staticmethod
    def metadata():
        return {"test_metrics": {"test_accuracy": 0.8}}


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
    def test_chapter_classifier_health_requires_exported_model_files(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory)
            classifier = LocalChapterClassifier(model_path)
            healthy, detail = classifier.health()
            self.assertFalse(healthy)
            self.assertIn("config.json", detail)

            (model_path / "config.json").write_text("{}", encoding="utf-8")
            (model_path / "model.safetensors").write_bytes(b"placeholder")
            (model_path / "vocab.txt").write_text("[PAD]\n", encoding="utf-8")
            (model_path / "training_metadata.json").write_text(
                json.dumps({
                    "task": "physics-chapter-classification",
                    "contains_answers": False,
                }),
                encoding="utf-8",
            )
            healthy, _ = classifier.health()
            self.assertTrue(healthy)

            (model_path / "training_metadata.json").write_text(
                json.dumps({
                    "task": "physics-chapter-classification",
                    "contains_answers": True,
                }),
                encoding="utf-8",
            )
            healthy, detail = classifier.health()
            self.assertFalse(healthy)
            self.assertIn("answer-free", detail)

    def test_real_book_has_366_pages(self):
        source = Path(__file__).resolve().parents[1] / "dataset" / "raw" / "physics.md"
        pages = parse_book(Book("physics-9-10", "পদার্থবিজ্ঞান", source))
        self.assertEqual([page.number for page in pages], list(range(1, 367)))

    def test_catalog_books_are_parseable(self):
        root = Path(__file__).resolve().parents[1]
        books = {book.id: book for book in load_books(root)}

        self.assertEqual(set(books), {"physics-9-10"})
        self.assertEqual(len(parse_book(books["physics-9-10"])), 366)

    def test_chunks_never_cross_pages(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "book.md"
            source.write_text("## পৃষ্ঠা 1\nএক দুই তিন চার\n## পৃষ্ঠা 2\nপাঁচ ছয়", encoding="utf-8")
            book = Book("book", "বই", source)
            chunks = chunk_pages(book, parse_book(book), max_words=3, overlap_words=1)
            self.assertEqual({chunk.page for chunk in chunks}, {1, 2})
            self.assertTrue(all("পাঁচ" not in chunk.text for chunk in chunks if chunk.page == 1))

    def test_chunks_preserve_chapter_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "book.md"
            source.write_text("## পৃষ্ঠা 1\nগতি\n## পৃষ্ঠা 2\nবল", encoding="utf-8")
            book = Book(
                "book",
                "বই",
                source,
                (Chapter("1", "গতি", 1, 1), Chapter("2", "বল", 2, 2)),
            )

            chunks = chunk_pages(book, parse_book(book))

            self.assertEqual([chunk.chapter_id for chunk in chunks], ["1", "2"])
            self.assertTrue(chunks[0].id.startswith("book-ch1-"))

    def test_low_evidence_is_sent_to_model_before_refusal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            chunk = Chunk("c1", "book", "বই", 1, 0, "আলোর প্রতিফলন")
            build_index(root, [chunk], np.asarray([[0.0, 1.0]], dtype=np.float32), "test")
            retriever = HybridRetriever(load_index(root), FakeEmbeddings([1.0, 0.0]))
            llm = FakeLLM({})
            answer = QAService(retriever, llm).ask("বিরিয়ানির রেসিপি", debug=True)
            self.assertEqual(answer.text, FALLBACK_TEXT)
            self.assertEqual(llm.calls, 1)
            self.assertFalse(answer.debug["retrieval_gate_used"])
            self.assertEqual(answer.debug["refusal_source"], "model_context_review")

    def test_invalid_source_id_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            chunk = Chunk("c1", "book", "বই", 1, 0, "বল এবং নিউটনের সূত্র")
            build_index(root, [chunk], np.asarray([[1.0, 0.0]], dtype=np.float32), "test")
            retriever = HybridRetriever(load_index(root), FakeEmbeddings([1.0, 0.0]))
            llm = FakeLLM({"answerable": True, "answer": "উত্তর", "source_ids": ["invented"]})
            answer = QAService(retriever, llm).ask("নিউটনের সূত্র কী")
            self.assertEqual(answer.text, FALLBACK_TEXT)

    def test_grounded_generator_is_used_without_dataset_answer_lookup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            chunk = Chunk("c1", "book", "পদার্থবিজ্ঞান", 42, 0, "ঘনত্ব হলো একক আয়তনের ভর।", "5", "চাপ")
            build_index(root, [chunk], None, None)
            retriever = HybridRetriever(
                load_index(root),
                semantic_threshold=0.0,
                keyword_coverage_threshold=0.0,
            )
            llm = FakeLLM({
                "answerable": True,
                "answer": "একক আয়তনের ভরকে ঘনত্ব বলে।",
                "source_ids": ["c1"],
                "confidence": 0.82,
            })
            classifier = FakeChapterClassifier("3")

            answer = QAService(
                retriever,
                llm,
                llm_name="Qwen3-4B",
                chapter_classifier=classifier,
                classifier_book_id="book",
            ).ask(
                "ঘনত্ব কাকে বলে?",
                book_ids={"book"},
                chapter_refs={("book", "5")},
            )

            self.assertEqual(answer.status, "answered")
            self.assertEqual(answer.model, "Qwen3-4B")
            self.assertEqual(answer.confidence, 0.82)
            self.assertIsNotNone(answer.elapsed_seconds)
            self.assertEqual(answer.citations[0].chunk_ids, ("c1",))
            self.assertEqual(llm.calls, 1)
            self.assertEqual(classifier.calls, 1)
            self.assertEqual(answer.chapter_prediction["predicted_chapter_id"], "3")
            self.assertFalse(answer.chapter_prediction["matches_selected_chapter"])
            self.assertTrue(answer.chapter_prediction["advisory_only"])

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

    def test_retrieval_is_hard_limited_to_selected_chapter(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            chunks = [
                Chunk("motion", "physics", "পদার্থবিজ্ঞান", 40, 0, "বেগ ও গতি", "2", "গতি"),
                Chunk("force", "physics", "পদার্থবিজ্ঞান", 75, 0, "বেগ থেকে ভরবেগ", "3", "বল"),
            ]
            build_index(root, chunks, None, None)
            retriever = HybridRetriever(
                load_index(root),
                semantic_threshold=0.0,
                keyword_coverage_threshold=0.0,
            )

            result = retriever.search(
                "বেগ",
                chapter_refs={("physics", "3")},
            )

            self.assertEqual([hit.chunk.id for hit in result.hits], ["force"])
            self.assertTrue(all(hit.chunk.chapter_id == "3" for hit in result.hits))

    def test_no_match_still_returns_selected_chapter_context_for_model_review(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            chunks = [
                Chunk("c1", "physics", "পদার্থবিজ্ঞান", 67, 0, "জড়তা ও বল", "3", "বল"),
                Chunk("c2", "physics", "পদার্থবিজ্ঞান", 80, 0, "ভরবেগের সূত্র", "3", "বল"),
            ]
            build_index(root, chunks, None, None)
            retriever = HybridRetriever(load_index(root))

            result = retriever.search(
                "How do plants make food?",
                debug=True,
                chapter_refs={("physics", "3")},
            )

            self.assertEqual({hit.chunk.id for hit in result.hits}, {"c1", "c2"})
            self.assertFalse(result.sufficient_evidence)
            self.assertEqual(result.debug["reason"], "chapter_context_fallback")

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
