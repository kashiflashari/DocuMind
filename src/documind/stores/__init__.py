"""Vector store backends."""

from documind.stores.base import VectorStore
from documind.stores.memory import InMemoryVectorStore

__all__ = ["VectorStore", "InMemoryVectorStore", "get_store"]


def get_store(settings, embedder):
    """Return a vector store for the configured backend."""
    if settings.vector_store.lower() == "pgvector":
        from documind.stores.pgvector import PgVectorStore

        return PgVectorStore(settings, embedder)
    return InMemoryVectorStore(embedder, persist_path=settings.memory_persist_path)
