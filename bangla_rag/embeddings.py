from collections.abc import Sequence
import json
from typing import Any
from urllib import error, request

import numpy as np


class EmbeddingProvider:
    """Embedding adapter backed by either llama.cpp HTTP or sentence-transformers."""

    def __init__(
        self,
        model_name: str,
        local_files_only: bool = False,
        base_url: str | None = None,
        timeout: int = 120,
    ) -> None:
        self.model_name = model_name
        self.local_files_only = local_files_only
        self.base_url = base_url.rstrip("/") if base_url else None
        self.timeout = timeout
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "sentence-transformers is not installed; install requirements.txt"
                ) from exc
            self._model = SentenceTransformer(
                self.model_name, local_files_only=self.local_files_only
            )
        return self._model

    def _remote(self, texts: Sequence[str]) -> np.ndarray:
        if not self.base_url:
            raise RuntimeError("The embedding service URL is not configured")
        body = json.dumps(
            {"model": self.model_name, "input": list(texts)},
            ensure_ascii=False,
        ).encode("utf-8")
        http_request = request.Request(
            f"{self.base_url}/embeddings",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(http_request, timeout=self.timeout) as response:
                payload = json.load(response)
        except (error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Could not use the local embedding service at {self.base_url}: {exc}"
            ) from exc
        rows = sorted(payload.get("data", []), key=lambda item: item.get("index", 0))
        if len(rows) != len(texts) or any("embedding" not in row for row in rows):
            raise RuntimeError("The local embedding service returned an invalid response")
        return np.asarray([row["embedding"] for row in rows], dtype=np.float32)

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        if self.base_url:
            # Keep requests small enough for the local llama.cpp context and
            # return vectors in the same order as the supplied texts.
            batches = [texts[offset : offset + 8] for offset in range(0, len(texts), 8)]
            return np.concatenate([self._remote(batch) for batch in batches], axis=0)
        return np.asarray(
            self._load().encode(
                list(texts),
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )

    def passages(self, texts: Sequence[str], progress: bool = True) -> np.ndarray:
        values = [f"passage: {text}" for text in texts]
        return self._encode(values)

    def query(self, text: str) -> np.ndarray:
        result = self._encode([f"query: {text}"])
        return np.asarray(result[0], dtype=np.float32)
