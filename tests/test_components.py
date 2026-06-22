from documind.chunking import chunk_text
from documind.embeddings import HashEmbedder
from documind.keyword import BM25Index
from documind.stores.memory import _cosine


def test_chunking_splits_long_text_with_overlap():
    text = "word " * 1000  # ~5000 chars
    chunks = chunk_text(text, chunk_size=400, chunk_overlap=80)
    assert len(chunks) > 1
    assert all(len(c) <= 600 for c in chunks)  # splitter may exceed slightly


def test_chunking_empty():
    assert chunk_text("") == []


def test_hash_embedder_dim_and_determinism():
    emb = HashEmbedder(dim=128)
    v1 = emb.embed_query("hybrid search")
    v2 = emb.embed_query("hybrid search")
    assert len(v1) == 128
    assert v1 == v2  # deterministic


def test_hash_embedder_similarity_reflects_shared_terms():
    emb = HashEmbedder(dim=256)
    q = emb.embed_query("cohere rerank cross encoder")
    near = emb.embed_query("rerank cross encoder relevance")
    far = emb.embed_query("postgresql cosine vectors")
    assert _cosine(q, near) > _cosine(q, far)


def test_bm25_ranks_term_match_first():
    idx = BM25Index()
    idx.add("a", "reciprocal rank fusion merges rankings")
    idx.add("b", "cohere rerank cross encoder relevance")
    hits = idx.search("rerank cross encoder", top_k=2)
    assert hits[0][0] == "b"
