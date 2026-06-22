"""DocuMind configuration, loaded from environment / ``.env``.

Defaults are fully offline (hash embeddings, in-memory store, stub LLM,
identity reranker) so the system runs end-to-end with no external services.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", case_sensitive=False
    )

    # --- Embeddings: hash | openai | cohere -----------------------------
    embedding_provider: str = "hash"
    embedding_dim: int = 256  # used by the offline hash embedder
    openai_api_key: str | None = None
    openai_embedding_model: str = "text-embedding-3-small"
    cohere_api_key: str | None = None
    cohere_embedding_model: str = "embed-english-v3.0"

    # --- Re-ranker: auto (cohere if key, else identity) -----------------
    rerank_enabled: bool = True
    cohere_rerank_model: str = "rerank-english-v3.0"

    # --- Answer LLM: stub | anthropic | openai --------------------------
    llm_provider: str = "stub"
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-sonnet-4-6"
    openai_model: str = "gpt-4o"
    temperature: float = 0.1
    max_tokens: int = 1024

    # --- Vector store: memory | pgvector --------------------------------
    vector_store: str = "memory"
    database_url: str | None = None  # postgresql://user:pass@host:5432/db
    table_name: str = "documind_chunks"
    # Optional JSON persistence for the in-memory store so the CLI can ingest
    # and query across separate invocations (None = in-process only).
    memory_persist_path: str | None = None

    # --- Chunking -------------------------------------------------------
    chunk_size: int = 800
    chunk_overlap: int = 120

    # --- Retrieval ------------------------------------------------------
    candidate_pool: int = 20  # candidates fetched per retriever before fusion
    top_k: int = 5  # final chunks passed to the LLM

    @property
    def cohere_rerank_available(self) -> bool:
        return self.rerank_enabled and bool(self.cohere_api_key)


@lru_cache
def get_settings() -> Settings:
    return Settings()
