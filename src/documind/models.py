"""Core data structures shared across the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Chunk:
    """A unit of indexed text plus its provenance."""

    id: str
    text: str
    doc_id: str
    source: str  # filename or URL
    ordinal: int = 0  # position within the source document
    embedding: list[float] | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class RetrievedChunk:
    """A chunk returned by retrieval, with its relevance score and provenance."""

    chunk: Chunk
    score: float
    retriever: str  # "vector" | "keyword" | "fused" | "rerank"


@dataclass
class Citation:
    """A numbered citation backing part of an answer."""

    n: int
    chunk_id: str
    source: str
    snippet: str


@dataclass
class Answer:
    """A grounded answer with the citations that support it."""

    question: str
    text: str
    citations: list[Citation] = field(default_factory=list)
    contexts: list[RetrievedChunk] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.text,
            "citations": [
                {"n": c.n, "chunk_id": c.chunk_id, "source": c.source, "snippet": c.snippet}
                for c in self.citations
            ],
        }
