"""The high-level knowledge-assistant facade.

Wires together the embedder, vector store, hybrid retriever, re-ranker, and
answer LLM. This is the main entry point used by the API, CLI, and tests.
"""

from __future__ import annotations

from documind.answer import generate_answer
from documind.config import Settings, get_settings
from documind.embeddings import get_embedder
from documind.ingest import ingest_files, ingest_texts
from documind.llm import get_chat_model
from documind.models import Answer
from documind.retrieval import HybridRetriever
from documind.stores import get_store


class KnowledgeAssistant:
    def __init__(self, settings: Settings | None = None, store=None) -> None:
        self.settings = settings or get_settings()
        self.embedder = get_embedder(self.settings)
        self.store = store or get_store(self.settings, self.embedder)
        self.retriever = HybridRetriever(self.store, self.settings)
        self.model = get_chat_model(self.settings)

    # --- ingestion -------------------------------------------------------
    def ingest_texts(self, items: list[dict]) -> int:
        return ingest_texts(items, self.store, self.settings)

    def ingest_files(self, paths: list[str]) -> int:
        return ingest_files(paths, self.store, self.settings)

    # --- query -----------------------------------------------------------
    def query(self, question: str, top_k: int | None = None) -> Answer:
        contexts = self.retriever.retrieve(question, top_k)
        return generate_answer(question, contexts, self.settings, model=self.model)

    @property
    def size(self) -> int:
        return self.store.count()
