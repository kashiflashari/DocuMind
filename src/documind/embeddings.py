"""Embedding providers.

``HashEmbedder`` is a deterministic, dependency-free embedder (hashed bag of
tokens, L2-normalised) used offline so retrieval is meaningful without any API
— documents sharing terms with the query score higher. Real providers
(OpenAI, Cohere) are imported lazily.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

from documind.config import Settings

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class Embedder(Protocol):
    dim: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class HashEmbedder:
    """Deterministic offline embedder via the hashing trick."""

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in _tokenize(text):
            h = int(hashlib.md5(token.encode()).hexdigest(), 16)
            idx = h % self.dim
            sign = 1.0 if (h >> 8) % 2 == 0 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def get_embedder(settings: Settings) -> Embedder:
    provider = settings.embedding_provider.lower()
    if provider == "openai" and settings.openai_api_key:
        from langchain_openai import OpenAIEmbeddings

        client = OpenAIEmbeddings(
            model=settings.openai_embedding_model, api_key=settings.openai_api_key
        )
        return _LangChainEmbedder(client)
    if provider == "cohere" and settings.cohere_api_key:
        from langchain_cohere import CohereEmbeddings

        client = CohereEmbeddings(
            model=settings.cohere_embedding_model, cohere_api_key=settings.cohere_api_key
        )
        return _LangChainEmbedder(client)
    return HashEmbedder(dim=settings.embedding_dim)


class _LangChainEmbedder:
    """Adapt a LangChain embeddings object to the :class:`Embedder` protocol."""

    def __init__(self, client) -> None:
        self._client = client
        self.dim = len(client.embed_query("dimension probe"))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._client.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._client.embed_query(text)
