"""Vector store interface.

A store owns both the dense vectors and the lexical (keyword) index so it can
serve the two halves of hybrid search behind one API.
"""

from __future__ import annotations

from typing import Protocol

from documind.models import Chunk


class VectorStore(Protocol):
    def add(self, chunks: list[Chunk]) -> int:
        """Embed (if needed) and persist chunks. Returns the count added."""
        ...

    def vector_search(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        """Dense similarity search. Returns (chunk, score) pairs, score descending."""
        ...

    def keyword_search(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        """Lexical search. Returns (chunk, score) pairs, score descending."""
        ...

    def count(self) -> int:
        ...
