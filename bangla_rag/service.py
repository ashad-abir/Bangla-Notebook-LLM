import re
from collections import defaultdict
from math import ceil
from time import perf_counter
from typing import Any

from bangla_rag.llm import LLMError, LocalLLM
from bangla_rag.models import Answer, Citation, SearchHit
from bangla_rag.retriever import HybridRetriever


FALLBACK_TEXT = "এই অধ্যায়ে এই প্রশ্নের উত্তর পাওয়া যায়নি।"


def _sources(hits: tuple[SearchHit, ...]) -> str:
    return "\n\n".join(
        f"SOURCE_ID: {hit.chunk.id}\nBOOK: {hit.chunk.book_title}\nPAGE: {hit.chunk.page}\nTEXT: {hit.chunk.text}"
        for hit in hits
    )


class QAService:
    def __init__(
        self,
        retriever: HybridRetriever,
        llm: LocalLLM | None,
        passage_limit: int = 5,
        llm_name: str = "Qwen",
        qwen_passage_limit: int = 1,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.passage_limit = passage_limit
        self.llm_name = llm_name
        self.qwen_passage_limit = max(1, qwen_passage_limit)

    @staticmethod
    def _evidence_confidence(result) -> float:
        if result.strongest_semantic_score is not None:
            return max(0.0, min(1.0, result.strongest_semantic_score))
        scores = result.debug.get("keyword_scores", {})
        return max(0.0, min(1.0, max(scores.values(), default=0.0)))

    @staticmethod
    def _select_qwen_hits(question: str, hits: tuple[SearchHit, ...], limit: int) -> tuple[SearchHit, ...]:
        """Keep the top passage, then greedily add passages covering missing terms."""
        if len(hits) <= limit:
            return hits
        terms = set(re.findall(r"[\u0980-\u09FF]+|[A-Za-z]+", question.casefold()))
        for term in tuple(terms):
            if term.startswith("কিলো") and len(term) > 4:
                terms.add(term[4:])

        def covered(hit: SearchHit) -> set[str]:
            text = hit.chunk.text.casefold()
            return {term for term in terms if term in text}

        selected = [hits[0]]
        remaining = list(hits[1:])
        seen = covered(hits[0])
        while remaining and len(selected) < limit:
            best = max(
                remaining,
                key=lambda hit: (
                    len(covered(hit) - seen),
                    len(covered(hit)),
                    hit.semantic_score or 0.0,
                ),
            )
            selected.append(best)
            seen.update(covered(best))
            remaining.remove(best)
        return tuple(selected)

    def ask(
        self,
        question: str,
        debug: bool = False,
        book_ids: set[str] | None = None,
        chapter_refs: set[tuple[str, str]] | None = None,
    ) -> Answer:
        started = perf_counter()

        def finish(
            status: str,
            text: str,
            citations: tuple[Citation, ...] = (),
            *,
            error: str | None = None,
            details: dict[str, Any] | None = None,
            model: str | None = None,
            confidence: float | None = None,
        ) -> Answer:
            return Answer(
                status,
                text,
                citations,
                error,
                details or {},
                model,
                round(perf_counter() - started, 3),
                None if confidence is None else round(max(0.0, min(1.0, confidence)), 6),
            )

        if not question.strip() or len(question) > 2000:
            return finish("error", "প্রশ্নটি খালি অথবা অতিরিক্ত দীর্ঘ।")
        result = self.retriever.search(
            question,
            self.passage_limit,
            True,
            book_ids=book_ids,
            chapter_refs=chapter_refs,
        )
        details = dict(result.debug) if debug else {}
        evidence_confidence = self._evidence_confidence(result)
        if not result.hits:
            return finish(
                "not_found",
                FALLBACK_TEXT,
                details=details,
                confidence=evidence_confidence,
            )
        if not result.sufficient_evidence:
            return finish(
                "not_found",
                FALLBACK_TEXT,
                details=details,
                confidence=evidence_confidence,
            )
        if self.llm is None:
            return finish(
                "error",
                "স্থানীয় উত্তর মডেলটি কনফিগার করা হয়নি।",
                error="No question-answering backend is configured",
                details=details,
            )
        system = (
            "/no_think\nYou are a Bengali textbook QA verifier. Use only the supplied sources, never prior knowledge. "
            "Answer in Bengali even when the question is English. If the sources do not explicitly support "
            "the answer, set answerable=false. Return JSON only: "
            '{"answerable":true|false,"answer":"...","source_ids":["..."],"confidence":0.0}. '
            "Confidence must be between 0 and 1 and reflect only how directly the cited sources support the answer. "
            "Every factual claim must be supported by the listed source IDs. Keep the answer concise. "
            "For a numerical answer, show the defining relation, substitute units explicitly, verify the arithmetic, "
            "and prefer scientific notation or full digits over ambiguous lakh/crore wording. If the supplied sources "
            "do not contain every quantity or relation needed for the calculation, set answerable=false."
        )
        qwen_hits = self._select_qwen_hits(question, result.hits, self.qwen_passage_limit)
        user = f"QUESTION:\n{question}\n\nSOURCES:\n{_sources(qwen_hits)}"
        try:
            generated = self.llm.complete_json(system, user, max_tokens=300)
        except LLMError as exc:
            return finish(
                "error",
                "স্থানীয় উত্তর মডেলগুলো ব্যবহার করা যায়নি।",
                error=str(exc),
                details=details,
                model=self.llm_name,
            )
        valid = {hit.chunk.id: hit.chunk for hit in qwen_hits}
        source_ids = [item for item in generated.get("source_ids", []) if item in valid]
        answer_text = str(generated.get("answer", "")).strip()
        answer_text = (
            answer_text.replace("\b", r"\b")
            .replace("\f", r"\f")
            .replace("\r", r"\r")
            .replace("\t", r"\t")
        )
        try:
            qwen_confidence = float(generated.get("confidence", evidence_confidence))
        except (TypeError, ValueError):
            qwen_confidence = evidence_confidence
        if generated.get("answerable") is not True or not answer_text or not source_ids:
            return finish(
                "not_found",
                FALLBACK_TEXT,
                details=details,
                model=self.llm_name,
                confidence=qwen_confidence,
            )

        grouped: dict[tuple[str, int], list[str]] = defaultdict(list)
        for chunk_id in source_ids:
            chunk = valid[chunk_id]
            grouped[(chunk.book_title, chunk.page)].append(chunk_id)
        citations = tuple(
            Citation(title, page, tuple(ids), valid[ids[0]].text[:240])
            for (title, page), ids in grouped.items()
        )
        if debug:
            details["answer_backend"] = "qwen"
        return finish(
            "answered",
            answer_text,
            citations,
            details=details,
            model=self.llm_name,
            confidence=qwen_confidence,
        )


class QuizService:
    def __init__(self, retriever: HybridRetriever, llm: LocalLLM, passage_limit: int = 12) -> None:
        self.retriever = retriever
        self.llm = llm
        self.passage_limit = passage_limit

    @staticmethod
    def _count_range(quiz_type: str) -> tuple[int, int]:
        if quiz_type == "mcq":
            return 10, 25
        if quiz_type == "knowledge":
            return 5, 10
        raise ValueError("Quiz type must be 'mcq' or 'knowledge'")

    @staticmethod
    def _normalize_options(item: dict[str, Any]) -> dict[str, str] | None:
        options = item.get("options")
        if isinstance(options, list) and len(options) == 4:
            options = dict(zip(("A", "B", "C", "D"), options))
        elif isinstance(options, dict) and len(options) == 4 and set(options) != {"A", "B", "C", "D"}:
            old_keys = list(options)
            answer = item.get("correct_answer")
            options = dict(zip(("A", "B", "C", "D"), options.values()))
            if answer in old_keys:
                item["correct_answer"] = ("A", "B", "C", "D")[old_keys.index(answer)]
        if not isinstance(options, dict) or set(options) != {"A", "B", "C", "D"}:
            return None
        cleaned = {
            key: re.sub(
                rf"^\s*{re.escape(key)}\s*[.)।:\-]\s*",
                "",
                str(value),
                flags=re.IGNORECASE,
            ).strip()
            for key, value in options.items()
        }
        item["options"] = cleaned
        answer = item.get("correct_answer")
        if isinstance(answer, str):
            key_match = re.match(r"^\s*([A-D])(?:\s*[.)।:\-])?", answer, flags=re.IGNORECASE)
            if key_match:
                answer = key_match.group(1).upper()
                item["correct_answer"] = answer
        if answer not in cleaned:
            matching = [key for key, value in cleaned.items() if value == answer]
            if len(matching) == 1:
                item["correct_answer"] = matching[0]
        return cleaned

    def _validate_item(
        self,
        item: Any,
        quiz_type: str,
        valid_ids: set[str],
    ) -> tuple[dict[str, Any] | None, list[str]]:
        if not isinstance(item, dict):
            return None, ["question item was not an object"]
        reasons = []
        question = str(item.get("question", "")).strip()
        ids = item.get("source_ids")
        if not question:
            reasons.append("question was empty")
        if not isinstance(ids, list) or not ids:
            reasons.append("source_ids must be a non-empty list")
        elif not all(source_id in valid_ids for source_id in ids):
            reasons.append("source_ids contained an untrusted ID")

        if quiz_type == "mcq":
            options = self._normalize_options(item)
            answer = item.get("correct_answer")
            if options is None:
                reasons.append("options must contain A/B/C/D")
            elif answer not in options:
                reasons.append("correct_answer must identify one option")
            elif len({value.casefold() for value in options.values()}) != 4:
                reasons.append("options must be distinct")
        elif not str(item.get("answer", "")).strip():
            reasons.append("answer was empty")

        if reasons:
            return None, reasons
        item["question"] = question
        item["sources"] = [
            {
                "book": self.retriever.by_id[source_id].book_title,
                "page": self.retriever.by_id[source_id].page,
                "chunk_id": source_id,
            }
            for source_id in ids
        ]
        return item, []

    def generate(
        self,
        topic: str,
        count: int = 10,
        quiz_type: str = "mcq",
        book_ids: set[str] | None = None,
        page_range: tuple[int, int] | None = None,
        chapter_refs: set[tuple[str, str]] | None = None,
    ) -> dict[str, Any]:
        minimum, maximum = self._count_range(quiz_type)
        if not minimum <= count <= maximum:
            raise ValueError(f"{quiz_type} quiz count must be between {minimum} and {maximum}")
        if page_range is not None:
            page_start, page_end = page_range
            candidates = [
                chunk
                for chunk in self.retriever.index.chunks
                if (book_ids is None or chunk.book_id in book_ids)
                and (
                    chapter_refs is None
                    or (chunk.book_id, chunk.chapter_id) in chapter_refs
                )
                and page_start <= chunk.page <= page_end
            ]
            desired = min(len(candidates), max(self.passage_limit, count))
            if desired:
                positions = (
                    [0]
                    if desired == 1
                    else [round(index * (len(candidates) - 1) / (desired - 1)) for index in range(desired)]
                )
                hits = tuple(SearchHit(candidates[position], 0.0) for position in positions)
            else:
                hits = ()
        else:
            result = self.retriever.search(
                topic,
                self.passage_limit,
                book_ids=book_ids,
                chapter_refs=chapter_refs,
            )
            hits = result.hits if result.sufficient_evidence else ()
        if not hits:
            return {
                "status": "not_found",
                "quiz_type": quiz_type,
                "message": FALLBACK_TEXT,
                "questions": [],
            }

        if quiz_type == "mcq":
            system = (
                "/no_think\nCreate concise Bengali multiple-choice questions using only the supplied textbook sources. "
                "Return JSON only with a questions array. Every item must contain question, options with exactly "
                "A/B/C/D, correct_answer, explanation, difficulty, and source_ids. Exactly one option must be correct. "
                "Use only supplied SOURCE_ID values. Test distinct concepts and do not repeat or paraphrase prior questions."
                " Create one question from each supplied source passage and cite that passage only."
                " Keep every option under ten words and every explanation to one short sentence."
            )
        else:
            system = (
                "/no_think\nCreate concise Bengali short-answer knowledge questions using only the supplied textbook sources. "
                "Return JSON only with a questions array. Every item must contain question, a brief answer of one to three "
                "sentences, difficulty, and source_ids. Use only supplied SOURCE_ID values. Test distinct concepts and do "
                "not repeat or paraphrase prior questions."
            )

        valid_ids = {hit.chunk.id for hit in hits}
        accepted: list[dict[str, Any]] = []
        seen: set[str] = set()
        validation_errors: list[str] = []
        batch_limit = 2
        maximum_attempts = ceil(count / batch_limit) + 5
        for attempt in range(maximum_attempts):
            remaining = count - len(accepted)
            if remaining <= 0:
                break
            batch_count = min(batch_limit, remaining)
            source_count = min(batch_count, len(hits))
            start = (attempt * batch_limit) % len(hits)
            batch_hits = tuple(hits[(start + offset) % len(hits)] for offset in range(source_count))
            prior = "\n".join(f"- {item['question']}" for item in accepted[-12:]) or "- none"
            user = (
                f"CHAPTER: {topic}\nCOUNT: {batch_count}\n"
                f"QUESTIONS ALREADY USED (do not repeat):\n{prior}\n\n"
                f"SOURCES:\n{_sources(batch_hits)}"
            )
            try:
                data = self.llm.complete_json(
                    system,
                    user,
                    temperature=0.3,
                    max_tokens=900 if quiz_type == "mcq" else 600,
                )
            except LLMError as exc:
                validation_errors.append(str(exc))
                continue
            for item in data.get("questions", []):
                validated, reasons = self._validate_item(item, quiz_type, valid_ids)
                if not validated:
                    validation_errors.extend(reasons)
                    continue
                identity = re.sub(r"\W+", "", validated["question"].casefold())
                if not identity or identity in seen:
                    validation_errors.append("duplicate question")
                    continue
                seen.add(identity)
                accepted.append(validated)
                if len(accepted) == count:
                    break

        if not accepted:
            return {
                "status": "error",
                "quiz_type": quiz_type,
                "topic": topic,
                "message": "স্থানীয় মডেলের তৈরি কোনো প্রশ্ন যাচাইয়ে উত্তীর্ণ হয়নি।",
                "questions": [],
                "validation_errors": validation_errors,
            }
        complete = len(accepted) >= count
        return {
            "status": "ok" if complete else "partial",
            "quiz_type": quiz_type,
            "topic": topic,
            "requested_count": count,
            "accepted_count": min(len(accepted), count),
            "questions": accepted[:count],
            "message": None if complete else "অনুরোধের চেয়ে কম সংখ্যক যাচাইযোগ্য প্রশ্ন তৈরি হয়েছে।",
            "validation_errors": validation_errors,
        }
