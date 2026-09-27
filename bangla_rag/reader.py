"""Local extractive question-answering with the fine-tuned BanglaBERT model."""

from dataclasses import dataclass
from pathlib import Path
import re
from threading import Lock
from typing import Any, Sequence

from bangla_rag.models import SearchHit


class ReaderError(RuntimeError):
    """Raised when the local extractive reader cannot be used."""


@dataclass(frozen=True)
class ReaderPrediction:
    text: str
    score: float
    hit: SearchHit
    start: int
    end: int
    ranking_score: float | None = None


class LocalQAReader:
    """Lazy, thread-safe adapter for an extractive Transformers QA checkpoint."""

    def __init__(
        self,
        model_path: Path,
        max_length: int = 384,
        doc_stride: int = 96,
        max_answer_tokens: int = 128,
        minimum_score: float = 0.0,
    ) -> None:
        self.model_path = model_path
        self.max_length = max_length
        self.doc_stride = doc_stride
        self.max_answer_tokens = max_answer_tokens
        self.minimum_score = minimum_score
        self._tokenizer: Any = None
        self._model: Any = None
        self._device = "cpu"
        self._load_lock = Lock()

    def health(self) -> tuple[bool, str]:
        required = ("config.json", "tokenizer_config.json")
        missing = [name for name in required if not (self.model_path / name).is_file()]
        weights = any(
            (self.model_path / name).is_file()
            for name in ("model.safetensors", "pytorch_model.bin")
        )
        if not weights:
            missing.append("model.safetensors or pytorch_model.bin")
        if missing:
            return False, f"Missing from {self.model_path}: {', '.join(missing)}"
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except ImportError:
            return False, "Install the BanglaBERT runtime with: pip install -r requirements.txt"
        return True, str(self.model_path)

    def _load(self) -> tuple[Any, Any, str]:
        if self._model is not None and self._tokenizer is not None:
            return self._tokenizer, self._model, self._device
        with self._load_lock:
            if self._model is not None and self._tokenizer is not None:
                return self._tokenizer, self._model, self._device
            healthy, detail = self.health()
            if not healthy:
                raise ReaderError(detail)
            try:
                import torch
                from transformers import AutoModelForQuestionAnswering, AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path,
                    local_files_only=True,
                    use_fast=True,
                )
                if not tokenizer.is_fast:
                    raise ReaderError("The BanglaBERT reader requires its fast tokenizer")
                model = AutoModelForQuestionAnswering.from_pretrained(
                    self.model_path,
                    local_files_only=True,
                )
                device = "cuda" if torch.cuda.is_available() else "cpu"
                model.to(device)
                model.eval()
            except ReaderError:
                raise
            except Exception as exc:
                raise ReaderError(f"Could not load BanglaBERT from {self.model_path}: {exc}") from exc
            self._tokenizer = tokenizer
            self._model = model
            self._device = device
        return self._tokenizer, self._model, self._device

    def predict(
        self,
        question: str,
        hits: Sequence[SearchHit],
    ) -> ReaderPrediction | None:
        if not question.strip() or not hits:
            return None
        tokenizer, model, device = self._load()
        try:
            import torch

            contexts = [hit.chunk.text for hit in hits]
            encoded = tokenizer(
                [question.strip()] * len(contexts),
                contexts,
                truncation="only_second",
                max_length=self.max_length,
                stride=self.doc_stride,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding=True,
                return_tensors="pt",
            )
            sample_map = encoded.pop("overflow_to_sample_mapping").tolist()
            offsets = encoded.pop("offset_mapping").tolist()
            sequence_ids = [encoded.sequence_ids(index) for index in range(len(sample_map))]
            model_inputs = {name: value.to(device) for name, value in encoded.items()}
            with torch.inference_mode():
                output = model(**model_inputs)
            start_logits = output.start_logits.detach().cpu()
            end_logits = output.end_logits.detach().cpu()
        except Exception as exc:
            raise ReaderError(f"BanglaBERT inference failed: {exc}") from exc

        best: ReaderPrediction | None = None
        for feature_index, sample_index in enumerate(sample_map):
            context_positions = [
                index for index, sequence_id in enumerate(sequence_ids[feature_index])
                if sequence_id == 1 and offsets[feature_index][index][1] > offsets[feature_index][index][0]
            ]
            if not context_positions:
                continue
            start_values = start_logits[feature_index, context_positions]
            end_values = end_logits[feature_index, context_positions]
            start_probabilities = torch.softmax(start_values, dim=0)
            end_probabilities = torch.softmax(end_values, dim=0)
            candidate_count = min(20, len(context_positions))
            top_starts = torch.topk(start_probabilities, candidate_count).indices.tolist()
            top_ends = torch.topk(end_probabilities, candidate_count).indices.tolist()
            context = contexts[sample_index]
            for start_rank in top_starts:
                token_start = context_positions[start_rank]
                for end_rank in top_ends:
                    token_end = context_positions[end_rank]
                    if token_end < token_start or token_end - token_start + 1 > self.max_answer_tokens:
                        continue
                    char_start = offsets[feature_index][token_start][0]
                    char_end = offsets[feature_index][token_end][1]
                    raw_text = context[char_start:char_end]
                    text = raw_text.strip()
                    if not text or not re.search(r"[\w\u0980-\u09FF]", text):
                        continue
                    leading = len(raw_text) - len(raw_text.lstrip())
                    actual_start = char_start + leading
                    actual_end = actual_start + len(text)
                    model_score = float(
                        start_probabilities[start_rank] * end_probabilities[end_rank]
                    )
                    # QA logits are not calibrated across unrelated passages.
                    # Preserve the retriever's evidence ordering while still
                    # allowing a strong span to win among nearby candidates.
                    ranking_score = model_score / ((sample_index + 1) ** 2)
                    prediction = ReaderPrediction(
                        text,
                        model_score,
                        hits[sample_index],
                        actual_start,
                        actual_end,
                        ranking_score,
                    )
                    if best is None or ranking_score > (best.ranking_score or best.score):
                        best = prediction
        if best is None or best.score < self.minimum_score:
            return None
        unit = self._prefer_unit(question, best)
        if unit is not None:
            return unit
        return self._prefer_explicit_definition(question, best)

    @staticmethod
    def _prefer_unit(
        question: str,
        prediction: ReaderPrediction,
    ) -> ReaderPrediction | None:
        if "একক" not in question:
            return None
        units = (
            "কিলোগ্রাম", "অ্যাম্পিয়ার", "ক্যান্ডেলা", "প্যাসকেল", "সেকেন্ড",
            "কেলভিন", "নিউটন", "মিটার", "হার্টজ", "কুলম্ব", "ভোল্ট",
            "ওয়াট", "জুল", "ওহম", "মোল",
        )
        context = prediction.hit.chunk.text
        unit_positions = [
            (match.start(), match.group(0))
            for unit in units
            for match in re.finditer(re.escape(unit), context)
        ]
        anchor_positions = [match.start() for match in re.finditer("একক", context)]
        if not unit_positions or not anchor_positions:
            return None
        start, text = min(
            unit_positions,
            key=lambda item: min(abs(item[0] - anchor) for anchor in anchor_positions),
        )
        if min(abs(start - anchor) for anchor in anchor_positions) > 180:
            return None
        return ReaderPrediction(
            text,
            prediction.score,
            prediction.hit,
            start,
            start + len(text),
            prediction.ranking_score,
        )

    @staticmethod
    def _prefer_explicit_definition(
        question: str,
        prediction: ReaderPrediction,
    ) -> ReaderPrediction:
        """Prefer a labelled textbook definition in the selected evidence.

        OCR textbook prose often introduces a definition with ``label:``.
        Fine-tuned span logits can include the lead-in instead of the concise
        definition, so this deterministic cleanup is applied only when the
        question itself names that label and the answer remains an exact span.
        """
        context = prediction.hit.chunk.text
        label = re.sub(r"[?？।\s]+$", "", question.strip())
        label = re.sub(r"\s+(?:কি|কী)$", "", label).strip()
        labels = [label] if label else []
        ordinal = re.search(r"(?:প্রথম|দ্বিতীয়|তৃতীয়)\s+সূত্র", label)
        if ordinal and ordinal.group(0) not in labels:
            labels.append(ordinal.group(0))
        for candidate in labels:
            match = re.search(
                rf"{re.escape(candidate)}\s*:\s*([^।\n]+(?:।|$))",
                context,
            )
            if not match:
                continue
            text = match.group(1).strip()
            text = re.sub(rf"^{re.escape(candidate)}\s*:\s*", "", text).strip()
            if not text:
                continue
            start = context.find(text, match.start(1), match.end(1))
            return ReaderPrediction(
                text,
                prediction.score,
                prediction.hit,
                start,
                start + len(text),
                prediction.ranking_score,
            )
        return prediction
