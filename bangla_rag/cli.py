import argparse
import json
import os
import sys
from pathlib import Path

from bangla_rag.chapter_classifier import ChapterClassifierError, LocalChapterClassifier
from bangla_rag.chunking import chunk_pages
from bangla_rag.embeddings import EmbeddingProvider
from bangla_rag.index_store import build_index
from bangla_rag.llm import LLMError
from bangla_rag.parser import load_books, parse_book
from bangla_rag.runtime import get_runtime, load_settings, repo_root
from bangla_rag.service import QAService


def root() -> Path:
    return repo_root()


def settings() -> dict:
    return load_settings()


def runtime():
    bundle = get_runtime()
    return bundle.retriever, bundle.llm


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m bangla_rag")
    commands = parser.add_subparsers(dest="command", required=True)
    ingest = commands.add_parser("ingest", help="Build the local book index")
    ingest.add_argument("--lexical-only", action="store_true", help="Skip embedding download")
    ask = commands.add_parser("ask", help="Ask one grounded question")
    ask.add_argument("question", nargs="?")
    ask.add_argument("--debug", action="store_true")
    ask.add_argument("--book", required=True, help="Catalog book ID")
    ask.add_argument("--chapter", required=True, help="Chapter ID within the book")
    search = commands.add_parser("search", help="Inspect retrieved passages")
    search.add_argument("question")
    search.add_argument("--limit", type=int, default=5)
    search.add_argument("--debug", action="store_true")
    search.add_argument("--book", required=True, help="Catalog book ID")
    search.add_argument("--chapter", required=True, help="Chapter ID within the book")
    quiz = commands.add_parser("quiz", help="Generate a grounded Bengali MCQ quiz")
    quiz.add_argument("topic")
    quiz.add_argument("--type", dest="quiz_type", choices=("mcq", "knowledge"), default="mcq")
    quiz.add_argument("--count", type=int, default=10)
    quiz.add_argument("--output", type=Path)
    quiz.add_argument("--book", required=True, help="Catalog book ID")
    quiz.add_argument("--chapter", required=True, help="Chapter ID within the book")
    evaluate = commands.add_parser("evaluate", help="Measure retrieval/refusal gates")
    evaluate.add_argument("dataset", type=Path, nargs="?", default=root() / "evaluation" / "starter.json")
    classify = commands.add_parser(
        "classify",
        help="Predict the Physics chapter with the question-only BanglaBERT model",
    )
    classify.add_argument("question")
    classify.add_argument("--top-k", type=int, default=3)
    commands.add_parser("doctor", help="Check index, packages, and model server")
    return parser


def ingest(lexical_only: bool) -> int:
    os.environ.setdefault("HF_HOME", str(root() / ".runtime" / "huggingface"))
    config = settings()
    chunks = []
    page_count = 0
    for book in load_books(root()):
        pages = parse_book(book)
        page_count += len(pages)
        chunks.extend(chunk_pages(book, pages, config["max_chunk_words"], config["overlap_words"]))
    vectors = None
    model_name = None
    if not lexical_only:
        model_name = config["embedding_model"]
        print(f"Creating multilingual embeddings for {len(chunks)} chunks...")
        vectors = EmbeddingProvider(
            model_name,
            base_url=config.get("embedding_base_url"),
            timeout=config["request_timeout_seconds"],
        ).passages([chunk.text for chunk in chunks])
    build_index(root() / "dataset" / "index", chunks, vectors, model_name)
    mode = "hybrid" if vectors is not None else "lexical-only"
    print(f"Built {mode} index: {len(load_books(root()))} book, {page_count} pages, {len(chunks)} chunks.")
    return 0


def catalog_scope(book_id: str, chapter_id: str) -> tuple[set[str], set[tuple[str, str]], tuple[int, int]]:
    records = json.loads((root() / "config" / "books.json").read_text(encoding="utf-8"))
    for record in records:
        if record["id"] != book_id:
            continue
        for chapter in record.get("chapters", []):
            if str(chapter["id"]) == chapter_id:
                return (
                    {book_id},
                    {(book_id, chapter_id)},
                    (int(chapter["page_start"]), int(chapter["page_end"])),
                )
    raise ValueError("Unknown chapter for the selected book")


def print_answer(
    service: QAService,
    question: str,
    debug: bool,
    book_ids: set[str],
    chapter_refs: set[tuple[str, str]],
) -> int:
    answer = service.ask(
        question,
        debug,
        book_ids=book_ids,
        chapter_refs=chapter_refs,
    )
    print(answer.text)
    for citation in answer.citations:
        print(f"উৎস: {citation.book_title}, পৃষ্ঠা {citation.page}")
    if debug:
        print(json.dumps(answer.debug, ensure_ascii=False, indent=2), file=sys.stderr)
    if answer.error:
        print(answer.error, file=sys.stderr)
    return 1 if answer.status == "error" else 0


def evaluate(path: Path) -> int:
    retriever, _ = runtime()
    cases = json.loads(path.read_text(encoding="utf-8"))
    correct = 0
    details = []
    for case in cases:
        result = retriever.search(case["question"], debug=True)
        predicted = result.sufficient_evidence
        expected = bool(case["answerable"])
        passed = predicted == expected
        correct += int(passed)
        details.append({
            "question": case["question"], "expected": expected, "predicted": predicted,
            "passed": passed, "top_score": result.strongest_semantic_score,
            "keyword": result.has_keyword_match,
        })
    print(json.dumps(details, ensure_ascii=False, indent=2))
    print(f"Evidence-gate accuracy: {correct}/{len(cases)} ({correct / len(cases):.1%})")
    return 0 if correct == len(cases) else 1


def chapter_classifier() -> LocalChapterClassifier:
    configured = Path(
        settings().get(
            "banglabert_model_path",
            ".runtime/models/pathshongi-banglabert-physics-chapters",
        )
    )
    model_path = configured if configured.is_absolute() else root() / configured
    return LocalChapterClassifier(model_path)


def main(argv: list[str] | None = None) -> int:
    # Windows PowerShell may inherit a legacy code page even when the source and
    # data are UTF-8. Keep Bengali output usable without requiring global changes.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args(argv)
    try:
        if args.command == "ingest":
            return ingest(args.lexical_only)
        if args.command == "evaluate":
            return evaluate(args.dataset)
        if args.command == "classify":
            predictions = chapter_classifier().predict(args.question, args.top_k)
            print(json.dumps([
                {
                    "chapter_id": prediction.chapter_id,
                    "label": prediction.label,
                    "confidence": round(prediction.confidence, 6),
                }
                for prediction in predictions
            ], ensure_ascii=False, indent=2))
            return 0
        retriever, llm = runtime()
        if args.command == "search":
            book_ids, chapter_refs, _ = catalog_scope(args.book, args.chapter)
            result = retriever.search(
                args.question,
                args.limit,
                args.debug,
                book_ids=book_ids,
                chapter_refs=chapter_refs,
            )
            for number, hit in enumerate(result.hits, 1):
                score = "n/a" if hit.semantic_score is None else f"{hit.semantic_score:.4f}"
                print(f"{number}. পৃষ্ঠা {hit.chunk.page} | semantic={score} | {hit.chunk.id}")
                print(hit.chunk.text + "\n")
            print(f"Sufficient evidence: {result.sufficient_evidence}")
            if args.debug:
                print(json.dumps(result.debug, ensure_ascii=False, indent=2), file=sys.stderr)
            return 0
        if args.command == "ask":
            service = get_runtime().qa
            book_ids, chapter_refs, _ = catalog_scope(args.book, args.chapter)
            if args.question:
                return print_answer(service, args.question, args.debug, book_ids, chapter_refs)
            print("প্রশ্ন লিখুন; বন্ধ করতে exit লিখুন।")
            while True:
                question = input("\nআপনি: ").strip()
                if question.lower() in {"exit", "quit", "q"}:
                    return 0
                print_answer(service, question, args.debug, book_ids, chapter_refs)
        if args.command == "quiz":
            book_ids, chapter_refs, page_range = catalog_scope(args.book, args.chapter)
            result = get_runtime().quiz.generate(
                args.topic,
                args.count,
                args.quiz_type,
                book_ids=book_ids,
                page_range=page_range,
                chapter_refs=chapter_refs,
            )
            rendered = json.dumps(result, ensure_ascii=False, indent=2)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(rendered + "\n", encoding="utf-8")
                print(f"Saved quiz to {args.output}")
            else:
                print(rendered)
            return 0 if result.get("status") == "ok" else 1
        if args.command == "doctor":
            bundle = get_runtime()
            answer_healthy, answer_detail = llm.health()
            checks = [
                ("index", True, f"{len(retriever.index.chunks)} chunks"),
                ("semantic", retriever.index.vectors is not None, "enabled" if retriever.index.vectors is not None else "lexical-only"),
                (f"answers ({bundle.answer_backend})", answer_healthy, answer_detail),
            ]
            healthy, detail = llm.health()
            checks.append(("grounded answer and quiz model", healthy, detail))
            classifier_ok, classifier_detail = chapter_classifier().health()
            checks.append(("BanglaBERT chapter classifier (optional)", classifier_ok, classifier_detail))
            for name, passed, detail in checks:
                print(f"{'PASS' if passed else 'FAIL'} {name}: {detail}")
            required = checks[:3]
            return 0 if all(item[1] for item in required) else 1
    except (RuntimeError, ValueError, OSError, LLMError, ChapterClassifierError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 2
