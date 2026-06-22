"""Re-rankers.

``CohereReranker`` uses Cohere's cross-encoder rerank endpoint; ``IdentityReranker``
preserves fusion order. ``get_reranker`` picks Cohere when a key is configured,
otherwise falls back to identity so the pipeline always runs.
"""

from __future__ import annotations

import logging

from documind.config import Settings
from documind.models import RetrievedChunk

logger = logging.getLogger(__name__)


class IdentityReranker:
    name = "identity"

    def rerank(self, query: str, candidates: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        return candidates[:top_k]


class CohereReranker:
    name = "cohere"

    def __init__(self, settings: Settings) -> None:
        import cohere

        self._client = cohere.Client(api_key=settings.cohere_api_key)
        self._model = settings.cohere_rerank_model

    def rerank(self, query: str, candidates: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        if not candidates:
            return []
        docs = [c.chunk.text for c in candidates]
        try:
            result = self._client.rerank(
                query=query, documents=docs, top_n=min(top_k, len(docs)), model=self._model
            )
        except Exception as exc:  # pragma: no cover - network
            logger.warning("Cohere rerank failed (%s); using fusion order.", exc)
            return candidates[:top_k]
        ranked: list[RetrievedChunk] = []
        for item in result.results:
            cand = candidates[item.index]
            ranked.append(
                RetrievedChunk(chunk=cand.chunk, score=float(item.relevance_score), retriever="rerank")
            )
        return ranked


def get_reranker(settings: Settings):
    if settings.cohere_rerank_available:
        try:
            return CohereReranker(settings)
        except Exception as exc:  # pragma: no cover - optional dependency
            logger.warning("Cohere unavailable (%s); using identity reranker.", exc)
    return IdentityReranker()
