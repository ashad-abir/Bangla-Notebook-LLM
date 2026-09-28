import re
from collections import defaultdict
from math import log

import numpy as np

from bangla_rag.index_store import LoadedIndex
from bangla_rag.models import SearchHit, SearchResult


TOKEN = re.compile(r"[\w\u0980-\u09FF]+", re.UNICODE)
STOPWORDS = {
    "কি", "কী", "কে", "কাকে", "কেন", "কিভাবে", "কীভাবে", "কেমন", "কোন", "কোনটি",
    "কত", "এবং", "ও", "এর", "এই", "একটি", "হয়", "হলো", "বল", "বলে", "দাও",
    "লেখ", "লিখ", "ব্যাখ্যা", "কর", "করো", "আলাদা", "the", "a", "an", "is", "what",
    "এসআই", "why", "how", "which", "of", "and", "in", "to", "explain", "write",
}


def query_terms(text: str) -> list[str]:
    return [token.lower() for token in TOKEN.findall(text) if len(token) > 1 and token.lower() not in STOPWORDS]


class HybridRetriever:
    def __init__(
        self,
        index: LoadedIndex,
        embeddings=None,
        semantic_threshold: float = 0.72,
        semantic_with_keyword_threshold: float = 0.60,
        keyword_coverage_threshold: float = 0.61,
        rrf_k: int = 60,
    ) -> None:
        self.index = index
        self.embeddings = embeddings
        self.semantic_threshold = semantic_threshold
        self.semantic_with_keyword_threshold = semantic_with_keyword_threshold
        self.keyword_coverage_threshold = keyword_coverage_threshold
        self.rrf_k = rrf_k
        self.by_id = {chunk.id: chunk for chunk in index.chunks}
        self.position = {chunk.id: i for i, chunk in enumerate(index.chunks)}
        self._lower_text = {chunk.id: chunk.text.lower() for chunk in index.chunks}

    def _keyword(
        self,
        question: str,
        limit: int = 20,
        book_ids: set[str] | None = None,
        chapter_refs: set[tuple[str, str]] | None = None,
    ) -> tuple[list[str], dict[str, float]]:
        """Rank exact Bengali/English terms with an IDF-weighted coverage gate.

        SQLite unicode61 tokenization is retained in the on-disk index for
        inspection, but it can split Bengali combining characters poorly on
        some Windows builds. For a textbook-sized corpus, this deterministic
        scan is fast and provides a useful lexical relevance signal.
        """
        terms = query_terms(question)
        if not terms:
            return [], {}
        terms = list(dict.fromkeys(terms[:12]))
        allowed_text = {
            chunk_id: text
            for chunk_id, text in self._lower_text.items()
            if book_ids is None or self.by_id[chunk_id].book_id in book_ids
            if chapter_refs is None
            or (self.by_id[chunk_id].book_id, self.by_id[chunk_id].chapter_id) in chapter_refs
        }
        if not allowed_text:
            return [], {}
        document_count = len(allowed_text)
        frequencies = {
            term: sum(term in text for text in allowed_text.values()) for term in terms
        }
        weights = {
            term: log((document_count + 1) / (frequencies[term] + 1)) + 1.0
            for term in terms
        }
        total_weight = sum(weights.values())
        scores: dict[str, float] = {}
        phrase = " ".join(terms)
        for chunk_id, text in allowed_text.items():
            matched = [term for term in terms if term in text]
            if not matched:
                continue
            weighted_coverage = sum(weights[term] for term in matched) / total_weight
            minimum_matches = 1 if len(terms) == 1 else 2
            if (
                len(matched) >= minimum_matches
                and weighted_coverage >= self.keyword_coverage_threshold
            ):
                # Coverage remains the evidence gate. Repeated terms and the
                # full content phrase are only tie-breakers, which prevents a
                # chapter outline from outranking the actual definition.
                repetition = sum(min(text.count(term), 5) for term in matched)
                repetition_bonus = 0.1 * repetition / (5 * len(terms))
                phrase_bonus = 0.2 if len(terms) > 1 and phrase in text else 0.0
                definition_bonus = 0.3 if len(terms) > 1 and f"{phrase}:" in text else 0.0
                scores[chunk_id] = (
                    weighted_coverage + repetition_bonus + phrase_bonus + definition_bonus
                )
        ranked = sorted(scores, key=lambda item: (-scores[item], item))[:limit]
        return ranked, scores

    def search(
        self,
        question: str,
        limit: int = 5,
        debug: bool = False,
        book_ids: set[str] | None = None,
        chapter_refs: set[tuple[str, str]] | None = None,
    ) -> SearchResult:
        if not question.strip():
            raise ValueError("Question cannot be empty")
        keyword_ids, keyword_scores = self._keyword(
            question,
            book_ids=book_ids,
            chapter_refs=chapter_refs,
        )
        semantic_ids: list[str] = []
        semantic_scores: dict[str, float] = {}
        semantic_error: str | None = None
        if self.index.vectors is not None and self.embeddings is not None:
            try:
                vector = self.embeddings.query(question)
                allowed_positions = [
                    position
                    for position, chunk in enumerate(self.index.chunks)
                    if book_ids is None or chunk.book_id in book_ids
                    if chapter_refs is None
                    or (chunk.book_id, chunk.chapter_id) in chapter_refs
                ]
                scoped_vectors = self.index.vectors[allowed_positions]
                scoped_scores = scoped_vectors @ vector
                scoped_order = np.argsort(-scoped_scores)[:20]
                for scoped_position in scoped_order:
                    position = allowed_positions[int(scoped_position)]
                    chunk_id = self.index.chunks[position].id
                    semantic_ids.append(chunk_id)
                    semantic_scores[chunk_id] = float(scoped_scores[int(scoped_position)])
            except RuntimeError as exc:
                # The in-process reader can still answer exact-term questions
                # when the optional semantic service is temporarily offline.
                semantic_error = str(exc)

        fused: dict[str, float] = defaultdict(float)
        for rank, chunk_id in enumerate(semantic_ids, 1):
            fused[chunk_id] += 1.0 / (self.rrf_k + rank)
        for rank, chunk_id in enumerate(keyword_ids, 1):
            fused[chunk_id] += 1.0 / (self.rrf_k + rank)
        if not fused:
            scoped_chunks = [
                chunk
                for chunk in self.index.chunks
                if book_ids is None or chunk.book_id in book_ids
                if chapter_refs is None
                or (chunk.book_id, chunk.chapter_id) in chapter_refs
            ]
            # Never make the answer/refusal decision in retrieval. If lexical
            # and semantic ranking produce nothing, provide a small,
            # deterministic cross-section of the selected chapter so the
            # grounded model can inspect actual textbook context first.
            desired = min(limit, len(scoped_chunks))
            positions = (
                []
                if desired == 0
                else [0]
                if desired == 1
                else [round(index * (len(scoped_chunks) - 1) / (desired - 1)) for index in range(desired)]
            )
            fallback_hits = tuple(
                SearchHit(scoped_chunks[position], 0.0)
                for position in positions
            )
            details = {
                "reason": "chapter_context_fallback" if fallback_hits else "no_scoped_chunks",
                "book_ids": sorted(book_ids) if book_ids is not None else None,
                "chapter_refs": sorted(chapter_refs) if chapter_refs is not None else None,
                "semantic_error": semantic_error,
            }
            return SearchResult(fallback_hits, False, None, False, details if debug else {})

        ranked = sorted(fused, key=lambda item: (-fused[item], item))[:limit]
        hits = []
        for chunk_id in ranked:
            hits.append(
                SearchHit(
                    self.by_id[chunk_id],
                    fused[chunk_id],
                    semantic_scores.get(chunk_id),
                    semantic_ids.index(chunk_id) + 1 if chunk_id in semantic_ids else None,
                    keyword_ids.index(chunk_id) + 1 if chunk_id in keyword_ids else None,
                )
            )
        strongest = max(semantic_scores.values()) if semantic_scores else None
        has_keyword = bool(keyword_ids)
        sufficient = has_keyword if strongest is None else (
            strongest >= self.semantic_threshold
            or (has_keyword and strongest >= self.semantic_with_keyword_threshold)
        )
        details = {}
        if debug:
            details = {
                "query_terms": query_terms(question),
                "semantic_top_score": strongest,
                "keyword_candidates": keyword_ids[:10],
                "keyword_scores": {
                    chunk_id: round(keyword_scores[chunk_id], 4)
                    for chunk_id in keyword_ids[:10]
                },
                "threshold": self.semantic_threshold,
                "threshold_with_keyword": self.semantic_with_keyword_threshold,
                "keyword_coverage_threshold": self.keyword_coverage_threshold,
                "book_ids": sorted(book_ids) if book_ids is not None else None,
                "chapter_refs": sorted(chapter_refs) if chapter_refs is not None else None,
                "semantic_error": semantic_error,
            }
        return SearchResult(tuple(hits), sufficient, strongest, has_keyword, details)
