"""Shared local runtime used by both the CLI and web application."""

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from threading import Lock

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
    return RuntimeBundle(
        index=index,
        retriever=retriever,
        llm=llm,
        qa=QAService(retriever, llm, config["answer_passages"]),
        quiz=QuizService(retriever, llm),
        generation_lock=Lock(),
    )
