"""Hybrid retrieval: vector + keyword search fused with Reciprocal Rank Fusion,
then re-ranked.

RRF combines rankings without needing to calibrate scores across retrievers:
    score(d) = Σ_retriever 1 / (k + rank_retriever(d))
"""

from __future__ import annotations

from documind.config import Settings
from documind.models import Chunk, RetrievedChunk
from documind.reranker import get_reranker

RRF_K = 60  # standard RRF constant


class HybridRetriever:
    def __init__(self, store, settings: Settings, reranker=None) -> None:
        self.store = store
        self.settings = settings
        self.reranker = reranker or get_reranker(settings)

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        top_k = top_k or self.settings.top_k
        pool = self.settings.candidate_pool

        vector_hits = self.store.vector_search(query, pool)
        keyword_hits = self.store.keyword_search(query, pool)

        fused = _reciprocal_rank_fusion(vector_hits, keyword_hits)
        candidates = [
            RetrievedChunk(chunk=chunk, score=score, retriever="fused")
            for chunk, score in fused
        ]
        return self.reranker.rerank(query, candidates, top_k)


def _reciprocal_rank_fusion(
    vector_hits: list[tuple[Chunk, float]],
    keyword_hits: list[tuple[Chunk, float]],
) -> list[tuple[Chunk, float]]:
    scores: dict[str, float] = {}
    chunks: dict[str, Chunk] = {}
    for ranking in (vector_hits, keyword_hits):
        for rank, (chunk, _score) in enumerate(ranking):
            scores[chunk.id] = scores.get(chunk.id, 0.0) + 1.0 / (RRF_K + rank + 1)
            chunks[chunk.id] = chunk
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [(chunks[cid], score) for cid, score in ordered]
