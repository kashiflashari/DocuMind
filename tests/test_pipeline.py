"""End-to-end retrieval + answer tests (offline: hash embeddings + stub LLM)."""

import pytest


def test_ingest_counts_chunks(assistant):
    assert assistant.size == 3


@pytest.mark.parametrize(
    "question,expected_source",
    [
        ("How does rerank improve relevance?", "rerank.md"),
        ("What does pgvector store and how does it search?", "pgvector.md"),
        ("What is reciprocal rank fusion?", "fusion.md"),
    ],
)
def test_hybrid_retrieval_ranks_relevant_chunk_first(assistant, question, expected_source):
    contexts = assistant.retriever.retrieve(question, top_k=3)
    assert contexts
    assert contexts[0].chunk.source == expected_source


def test_query_returns_grounded_answer_with_valid_citations(assistant):
    answer = assistant.query("How does rerank improve relevance?")
    assert answer.text
    # Every citation must resolve to a retrieved passage.
    assert answer.citations
    for c in answer.citations:
        assert 1 <= c.n <= len(answer.contexts)
        assert c.source.endswith(".md")


def test_query_on_empty_store_is_honest(stub_settings):
    from documind.pipeline import KnowledgeAssistant

    empty = KnowledgeAssistant(stub_settings)
    answer = empty.query("anything at all?")
    assert answer.citations == []
    assert "could not find" in answer.text.lower()
