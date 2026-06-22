"""A small, dependency-free BM25 keyword index.

Used as the lexical half of hybrid search for the in-memory store. (The
pgvector store uses PostgreSQL full-text search instead.)
"""

from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.ids: list[str] = []
        self.docs: list[list[str]] = []
        self.doc_freq: Counter[str] = Counter()
        self.avg_len: float = 0.0

    def add(self, doc_id: str, text: str) -> None:
        tokens = tokenize(text)
        self.ids.append(doc_id)
        self.docs.append(tokens)
        for term in set(tokens):
            self.doc_freq[term] += 1
        total = sum(len(d) for d in self.docs)
        self.avg_len = total / len(self.docs) if self.docs else 0.0

    def _idf(self, term: str) -> float:
        n = len(self.docs)
        df = self.doc_freq.get(term, 0)
        # BM25+ style idf, always positive.
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def search(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        if not self.docs:
            return []
        q_terms = tokenize(query)
        scores: list[tuple[str, float]] = []
        for doc_id, tokens in zip(self.ids, self.docs):
            if not tokens:
                continue
            tf = Counter(tokens)
            dl = len(tokens)
            score = 0.0
            for term in q_terms:
                if term not in tf:
                    continue
                idf = self._idf(term)
                freq = tf[term]
                denom = freq + self.k1 * (1 - self.b + self.b * dl / (self.avg_len or 1))
                score += idf * (freq * (self.k1 + 1)) / (denom or 1)
            if score > 0:
                scores.append((doc_id, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
