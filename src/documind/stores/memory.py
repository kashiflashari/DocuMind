"""In-memory vector store (pure Python).

Combines cosine similarity over embeddings with a BM25 keyword index. Used by
default and for tests; no external services required. Optionally persists to a
JSON file so the CLI can ingest and query across separate invocations.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from documind.keyword import BM25Index
from documind.models import Chunk


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class InMemoryVectorStore:
    def __init__(self, embedder, persist_path: str | None = None) -> None:
        self._embedder = embedder
        self._chunks: dict[str, Chunk] = {}
        self._bm25 = BM25Index()
        self._persist_path = Path(persist_path) if persist_path else None
        if self._persist_path and self._persist_path.exists():
            self._load()

    def add(self, chunks: list[Chunk]) -> int:
        to_embed = [c for c in chunks if c.embedding is None]
        if to_embed:
            vectors = self._embedder.embed_documents([c.text for c in to_embed])
            for chunk, vec in zip(to_embed, vectors):
                chunk.embedding = vec
        for chunk in chunks:
            self._chunks[chunk.id] = chunk
            self._bm25.add(chunk.id, chunk.text)
        if self._persist_path:
            self._save()
        return len(chunks)

    def vector_search(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        if not self._chunks:
            return []
        qvec = self._embedder.embed_query(query)
        scored = [(c, _cosine(qvec, c.embedding or [])) for c in self._chunks.values()]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    def keyword_search(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        hits = self._bm25.search(query, top_k=k)
        return [(self._chunks[cid], score) for cid, score in hits if cid in self._chunks]

    def count(self) -> int:
        return len(self._chunks)

    # --- persistence -----------------------------------------------------
    def _save(self) -> None:
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        payload = [
            {
                "id": c.id, "text": c.text, "doc_id": c.doc_id, "source": c.source,
                "ordinal": c.ordinal, "embedding": c.embedding, "metadata": c.metadata,
            }
            for c in self._chunks.values()
        ]
        self._persist_path.write_text(json.dumps(payload), encoding="utf-8")

    def _load(self) -> None:
        data = json.loads(self._persist_path.read_text(encoding="utf-8"))
        for d in data:
            chunk = Chunk(
                id=d["id"], text=d["text"], doc_id=d["doc_id"], source=d["source"],
                ordinal=d.get("ordinal", 0), embedding=d.get("embedding"),
                metadata=d.get("metadata", {}),
            )
            self._chunks[chunk.id] = chunk
            self._bm25.add(chunk.id, chunk.text)
