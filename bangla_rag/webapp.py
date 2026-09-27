"""FastAPI application for the local Bengali textbook assistant."""

from collections import Counter

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from typing import Literal

from pydantic import BaseModel, Field

from bangla_rag.catalog import load_catalog
from bangla_rag.runtime import RuntimeBundle, get_runtime, repo_root


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    book_id: str = Field(min_length=1, max_length=100)
    chapter_id: str = Field(min_length=1, max_length=50)


class QuizRequest(BaseModel):
    chapter_id: str = Field(min_length=1, max_length=50)
    quiz_type: Literal["mcq", "knowledge"]
    count: int = Field(ge=5, le=25)
    book_id: str = Field(min_length=1, max_length=100)


def selected_chapter(book_id: str, chapter_id: str, bundle: RuntimeBundle) -> dict:
    import json

    records = json.loads(
        (repo_root() / "config" / "books.json").read_text(encoding="utf-8")
    )
    matches = [
        chapter
        for record in records
        if record["id"] == book_id
        for chapter in record.get("chapters", [])
        if str(chapter["id"]) == chapter_id
    ]
    if len(matches) != 1:
        raise HTTPException(status_code=400, detail="Unknown chapter for the selected book")
    chapter = matches[0]
    indexed = any(
        chunk.book_id == book_id and chunk.chapter_id == chapter_id
        for chunk in bundle.index.chunks
    )
    if not indexed:
        raise HTTPException(
            status_code=400,
            detail="The selected chapter is not present in the current index; rebuild the index",
        )
    return chapter


def answer_payload(answer) -> dict:
    return {
        "status": answer.status,
        "answer": answer.text,
        "error": answer.error,
        "model": answer.model,
        "response_time_seconds": answer.elapsed_seconds,
        "confidence": answer.confidence,
        "citations": [
            {
                "book": citation.book_title,
                "page": citation.page,
                "chunk_ids": list(citation.chunk_ids),
                "excerpt": citation.excerpt,
            }
            for citation in answer.citations
        ],
    }


def create_app() -> FastAPI:
    app = FastAPI(
        title="পাঠসঙ্গী — Bengali Textbook Assistant",
        version="0.1.0",
        docs_url="/api/docs",
        redoc_url=None,
    )
    static_dir = repo_root() / "web" / "static"
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/", include_in_schema=False)
    def home() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    @app.get("/api/health")
    def health(bundle: RuntimeBundle = Depends(get_runtime)) -> dict:
        if bundle.reader is not None:
            model_ok, detail = bundle.reader.health()
        else:
            model_ok, detail = bundle.llm.health()
        quiz_ok, quiz_detail = bundle.llm.health()
        return {
            "status": "ready" if model_ok else "limited",
            "model_ready": model_ok,
            "model_detail": detail,
            "answer_backend": bundle.answer_backend,
            "quiz_model_ready": quiz_ok,
            "quiz_model_detail": quiz_detail,
            "semantic_search": bundle.index.vectors is not None,
            "chunks": len(bundle.index.chunks),
            "books": len({chunk.book_id for chunk in bundle.index.chunks}),
            "pages": len({(chunk.book_id, chunk.page) for chunk in bundle.index.chunks}),
        }

    @app.get("/api/catalog")
    def catalog(bundle: RuntimeBundle = Depends(get_runtime)) -> dict:
        counts = Counter(chunk.book_id for chunk in bundle.index.chunks)
        chapter_counts = Counter(
            (chunk.book_id, chunk.chapter_id)
            for chunk in bundle.index.chunks
            if chunk.chapter_id is not None
        )
        return load_catalog(
            repo_root() / "config" / "books.json",
            dict(counts),
            dict(chapter_counts),
        )

    @app.post("/api/ask")
    def ask(
        request: AskRequest,
        bundle: RuntimeBundle = Depends(get_runtime),
    ) -> dict:
        selected_chapter(request.book_id, request.chapter_id, bundle)
        book_ids = {request.book_id}
        chapter_refs = {(request.book_id, request.chapter_id)}
        try:
            with bundle.generation_lock:
                return answer_payload(
                    bundle.qa.ask(
                        request.question,
                        book_ids=book_ids,
                        chapter_refs=chapter_refs,
                    )
                )
        except (OSError, RuntimeError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post("/api/quiz")
    def quiz(
        request: QuizRequest,
        bundle: RuntimeBundle = Depends(get_runtime),
    ) -> dict:
        quiz_ready, quiz_detail = bundle.llm.health()
        if not quiz_ready:
            raise HTTPException(
                status_code=503,
                detail=f"Quiz model is unavailable: {quiz_detail}",
            )
        chapter = selected_chapter(request.book_id, request.chapter_id, bundle)
        book_ids = {request.book_id}
        chapter_refs = {(request.book_id, request.chapter_id)}
        valid_range = (10, 25) if request.quiz_type == "mcq" else (5, 10)
        if not valid_range[0] <= request.count <= valid_range[1]:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"{request.quiz_type} quiz count must be between "
                    f"{valid_range[0]} and {valid_range[1]}"
                ),
            )
        try:
            with bundle.generation_lock:
                result = bundle.quiz.generate(
                    chapter.get("query_bn", chapter["title_bn"]),
                    request.count,
                    quiz_type=request.quiz_type,
                    book_ids=book_ids,
                    page_range=(chapter["page_start"], chapter["page_end"]),
                    chapter_refs=chapter_refs,
                )
                result["chapter"] = chapter
                return result
        except (OSError, RuntimeError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    return app


app = create_app()
