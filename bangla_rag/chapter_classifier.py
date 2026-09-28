"""Local inference for the question-only BanglaBERT chapter classifier."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


class ChapterClassifierError(RuntimeError):
    pass


@dataclass(frozen=True)
class ChapterPrediction:
    chapter_id: str
    label: str
    chapter_title: str
    confidence: float


class LocalChapterClassifier:
    """Predict a textbook chapter without generating or retrieving an answer."""

    def __init__(self, model_path: Path, max_length: int = 128) -> None:
        self.model_path = model_path
        self.max_length = max_length
        self._tokenizer: Any = None
        self._model: Any = None
        self._torch: Any = None
        self._metadata: dict[str, Any] | None = None

    def metadata(self) -> dict[str, Any]:
        if self._metadata is None:
            metadata_path = self.model_path / "training_metadata.json"
            try:
                self._metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ChapterClassifierError(f"Could not read model metadata: {exc}") from exc
        return self._metadata

    def health(self) -> tuple[bool, str]:
        if not self.model_path.is_dir():
            return False, f"Model directory not found: {self.model_path}"
        if not (self.model_path / "config.json").is_file():
            return False, "config.json is missing"
        weights = (
            self.model_path / "model.safetensors",
            self.model_path / "pytorch_model.bin",
        )
        if not any(path.is_file() for path in weights):
            return False, "Model weights are missing"
        if not any((self.model_path / name).is_file() for name in ("tokenizer.json", "vocab.txt")):
            return False, "Tokenizer files are missing"
        metadata_path = self.model_path / "training_metadata.json"
        if not metadata_path.is_file():
            return False, "training_metadata.json is missing"
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self._metadata = metadata
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            return False, f"training_metadata.json is invalid: {exc}"
        if metadata.get("task") != "physics-chapter-classification":
            return False, "Model metadata has the wrong task"
        if metadata.get("contains_answers") is not False:
            return False, "Model metadata does not confirm answer-free training"
        return True, str(self.model_path)

    def _load(self) -> None:
        if self._model is not None:
            return
        healthy, detail = self.health()
        if not healthy:
            raise ChapterClassifierError(detail)
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                local_files_only=True,
                use_fast=True,
            )
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_path,
                local_files_only=True,
            )
            self._model.eval()
            self._torch = torch
        except (OSError, RuntimeError, ValueError) as exc:
            raise ChapterClassifierError(
                f"Could not load chapter classifier from {self.model_path}: {exc}"
            ) from exc

    def predict(self, question: str, top_k: int = 3) -> tuple[ChapterPrediction, ...]:
        question = question.strip()
        if not question:
            raise ValueError("Question cannot be empty")
        if top_k < 1:
            raise ValueError("top_k must be positive")
        self._load()
        encoded = self._tokenizer(
            question,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        try:
            with self._torch.inference_mode():
                probabilities = self._torch.softmax(self._model(**encoded).logits[0], dim=-1)
        except RuntimeError as exc:
            raise ChapterClassifierError(f"Chapter classification failed: {exc}") from exc
        count = min(top_k, int(probabilities.shape[0]))
        scores, indices = self._torch.topk(probabilities, count)
        id2label = self._model.config.id2label
        chapters = self.metadata().get("chapters", {})
        predictions = []
        for score, index in zip(scores.tolist(), indices.tolist(), strict=True):
            chapter_id = str(index + 1)
            label = str(id2label.get(index, id2label.get(str(index), f"chapter_{index + 1}")))
            predictions.append(
                ChapterPrediction(
                    chapter_id,
                    label,
                    str(chapters.get(chapter_id, label)),
                    float(score),
                )
            )
        return tuple(predictions)
