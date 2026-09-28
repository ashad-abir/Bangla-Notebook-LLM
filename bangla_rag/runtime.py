"""Shared local runtime used by both the CLI and web application."""

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from threading import Lock

from bangla_rag.chapter_classifier import LocalChapterClassifier
from bangla_rag.embeddings import EmbeddingProvider
from bangla_rag.index_store import LoadedIndex, load_index
from bangla_rag.llm import LocalLLM
from bangla_rag.retriever import HybridRetriever
from bangla_rag.service import QAService, QuizService


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_settings() -> dict:
    return json.loads(
        (repo_root() / "config" / "settings.json").read_text(encoding="utf-8")
    )


@dataclass(frozen=True)
class RuntimeBundle:
    index: LoadedIndex
    retriever: HybridRetriever
    llm: LocalLLM
    chapter_classifier: LocalChapterClassifier
    answer_backend: str
    qa: QAService
    quiz: QuizService
    generation_lock: Lock


@lru_cache(maxsize=1)
def get_runtime() -> RuntimeBundle:
    root = repo_root()
    os.environ.setdefault("HF_HOME", str(root / ".runtime" / "huggingface"))
    config = load_settings()
    index = load_index(root / "dataset" / "index")
    embeddings = (
        EmbeddingProvider(
            config["embedding_model"],
            local_files_only=True,
            base_url=config.get("embedding_base_url"),
            timeout=config["request_timeout_seconds"],
        )
        if index.vectors is not None
        else None
    )
    retriever = HybridRetriever(
        index,
        embeddings,
        config["semantic_threshold"],
        config["semantic_with_keyword_threshold"],
        config["keyword_coverage_threshold"],
    )
    llm = LocalLLM(
        config["llm_base_url"],
        config["llm_model"],
        timeout=config["request_timeout_seconds"],
    )
    answer_backend = "chapter_rag"
    configured_classifier_path = Path(
        config.get(
            "banglabert_model_path",
            ".runtime/models/pathshongi-banglabert-physics-chapters",
        )
    )
    classifier_path = (
        configured_classifier_path
        if configured_classifier_path.is_absolute()
        else root / configured_classifier_path
    )
    chapter_classifier = LocalChapterClassifier(classifier_path)
    return RuntimeBundle(
        index=index,
        retriever=retriever,
        llm=llm,
        chapter_classifier=chapter_classifier,
        answer_backend=answer_backend,
        qa=QAService(
            retriever,
            llm,
            config["answer_passages"],
            llm_name=config.get("llm_display_name", "Qwen"),
            qwen_passage_limit=int(config.get("qwen_answer_passages", 1)),
            chapter_classifier=chapter_classifier,
            classifier_book_id=config.get("banglabert_book_id", "physics-9-10"),
        ),
        quiz=QuizService(retriever, llm),
        generation_lock=Lock(),
    )
