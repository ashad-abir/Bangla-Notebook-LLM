import re
from collections import defaultdict
from math import ceil
from typing import Any

from bangla_rag.llm import LLMError, LocalLLM
from bangla_rag.models import Answer, Citation, SearchHit
from bangla_rag.retriever import HybridRetriever


FALLBACK_TEXT = "এই বইয়ে এই প্রশ্নের উত্তর পাওয়া যায়নি।"


def _sources(hits: tuple[SearchHit, ...]) -> str:
    return "\n\n".join(
        f"SOURCE_ID: {hit.chunk.id}\nBOOK: {hit.chunk.book_title}\nPAGE: {hit.chunk.page}\nTEXT: {hit.chunk.text}"
        for hit in hits
    )


class QAService:
    def __init__(self, retriever: HybridRetriever, llm: LocalLLM, passage_limit: int = 5) -> None:
        self.retriever = retriever
        self.llm = llm
        self.passage_limit = passage_limit

    def ask(
        self,
        question: str,
        debug: bool = False,
        book_ids: set[str] | None = None,
    ) -> Answer:
        if not question.strip() or len(question) > 2000:
            return Answer("error", "প্রশ্নটি খালি অথবা অতিরিক্ত দীর্ঘ।")
        result = self.retriever.search(
            question,
            self.passage_limit,
            debug,
            book_ids=book_ids,
        )
        if not result.sufficient_evidence:
            return Answer("not_found", FALLBACK_TEXT, debug=result.debug)
        system = (
            "/no_think\nYou are a Bengali textbook QA verifier. Use only the supplied sources, never prior knowledge. "
            "Answer in Bengali even when the question is English. If the sources do not explicitly support "
            "the answer, set answerable=false. Return JSON only: "
            '{"answerable":true|false,"answer":"...","source_ids":["..."]}. '
            "Every factual claim must be supported by the listed source IDs. Keep the answer concise."
        )
        user = f"QUESTION:\n{question}\n\nSOURCES:\n{_sources(result.hits)}"
        try:
            generated = self.llm.complete_json(system, user)
        except LLMError as exc:
            return Answer("error", "স্থানীয় উত্তর মডেলটি ব্যবহার করা যায়নি।", error=str(exc), debug=result.debug)
        valid = {hit.chunk.id: hit.chunk for hit in result.hits}
        source_ids = [item for item in generated.get("source_ids", []) if item in valid]
        answer_text = str(generated.get("answer", "")).strip()
        if generated.get("answerable") is not True or not answer_text or not source_ids:
            return Answer("not_found", FALLBACK_TEXT, debug=result.debug)

        grouped: dict[tuple[str, int], list[str]] = defaultdict(list)
        for chunk_id in source_ids:
            chunk = valid[chunk_id]
            grouped[(chunk.book_title, chunk.page)].append(chunk_id)
        citations = tuple(
            Citation(title, page, tuple(ids), valid[ids[0]].text[:240])
            for (title, page), ids in grouped.items()
        )
        return Answer("answered", answer_text, citations, debug=result.debug)


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
            result = self.retriever.search(topic, self.passage_limit, book_ids=book_ids)
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
