import pytest

from documind.config import Settings
from documind.pipeline import KnowledgeAssistant

CORPUS = [
    {
        "text": "Cohere rerank reorders candidate passages using a cross encoder to "
        "score relevance between the query and each passage.",
        "source": "rerank.md",
    },
    {
        "text": "pgvector stores embeddings inside PostgreSQL and supports cosine "
        "similarity search over high dimensional vectors.",
        "source": "pgvector.md",
    },
    {
        "text": "Reciprocal rank fusion merges vector and keyword rankings by summing "
        "the reciprocal of each rank position.",
        "source": "fusion.md",
    },
]


@pytest.fixture
def stub_settings() -> Settings:
    return Settings(
        _env_file=None,
        embedding_provider="hash",
        embedding_dim=256,
        vector_store="memory",
        llm_provider="stub",
        rerank_enabled=True,  # no Cohere key -> falls back to identity
        top_k=5,
    )


@pytest.fixture
def assistant(stub_settings) -> KnowledgeAssistant:
    ka = KnowledgeAssistant(stub_settings)
    ka.ingest_texts(CORPUS)
    return ka
