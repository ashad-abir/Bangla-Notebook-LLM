import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from bangla_rag.models import Chunk


@dataclass(frozen=True)
class LoadedIndex:
    chunks: tuple[Chunk, ...]
    vectors: np.ndarray | None
    sqlite_path: Path
    manifest: dict


def build_index(
    index_dir: Path,
    chunks: list[Chunk],
    vectors: np.ndarray | None,
    model_name: str | None,
) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    (index_dir / "chunks.jsonl").write_text(
        "\n".join(json.dumps(asdict(chunk), ensure_ascii=False) for chunk in chunks) + "\n",
        encoding="utf-8",
    )
    vector_path = index_dir / "vectors.npy"
    if vectors is not None:
        if len(vectors) != len(chunks):
            raise ValueError("Chunk/vector count mismatch")
        np.save(vector_path, vectors)
    elif vector_path.exists():
        vector_path.unlink()

    database = index_dir / "search.sqlite3"
    if database.exists():
        database.unlink()
    connection = sqlite3.connect(database)
    try:
        connection.execute(
            "CREATE VIRTUAL TABLE passages USING fts5(chunk_id UNINDEXED, text, tokenize='unicode61')"
        )
        connection.executemany(
            "INSERT INTO passages(chunk_id, text) VALUES (?, ?)",
            [(chunk.id, chunk.text) for chunk in chunks],
        )
        connection.commit()
    finally:
        connection.close()

    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "chunks": len(chunks),
        "embedding_model": model_name,
        "semantic_enabled": vectors is not None,
    }
    (index_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_index(index_dir: Path) -> LoadedIndex:
    required = ["chunks.jsonl", "search.sqlite3", "manifest.json"]
    missing = [name for name in required if not (index_dir / name).exists()]
    if missing:
        raise RuntimeError(f"Index is missing {', '.join(missing)}; run ingest first")
    chunks = tuple(
        Chunk(**json.loads(line))
        for line in (index_dir / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
    vector_path = index_dir / "vectors.npy"
    vectors = np.load(vector_path) if vector_path.exists() else None
    if vectors is not None and len(vectors) != len(chunks):
        raise RuntimeError("Index is corrupt: chunk/vector count mismatch")
    return LoadedIndex(chunks, vectors, index_dir / "search.sqlite3", manifest)

