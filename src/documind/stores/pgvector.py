"""PostgreSQL + pgvector store with hybrid (vector + full-text) search.

Requires ``psycopg`` and a Postgres instance with the ``pgvector`` extension.
Dense search uses the cosine-distance operator ``<=>``; lexical search uses
PostgreSQL full-text (``to_tsvector`` / ``plainto_tsquery``).
"""

from __future__ import annotations

import json

from documind.config import Settings
from documind.models import Chunk


class PgVectorStore:
    def __init__(self, settings: Settings, embedder) -> None:
        if not settings.database_url:
            raise ValueError("vector_store=pgvector requires DATABASE_URL")
        self.settings = settings
        self._embedder = embedder
        self._dim = embedder.dim
        self._connect()
        self._ensure_schema()

    def _connect(self) -> None:
        import psycopg

        self._conn = psycopg.connect(self.settings.database_url, autocommit=True)

    def _ensure_schema(self) -> None:
        table = self.settings.table_name
        with self._conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {table} (
                    id        TEXT PRIMARY KEY,
                    doc_id    TEXT,
                    source    TEXT,
                    ordinal   INT,
                    text      TEXT,
                    metadata  JSONB,
                    embedding vector({self._dim}),
                    tsv       tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED
                )
                """
            )
            cur.execute(f"CREATE INDEX IF NOT EXISTS {table}_tsv_idx ON {table} USING GIN (tsv)")
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS {table}_vec_idx ON {table} "
                f"USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)"
            )

    def add(self, chunks: list[Chunk]) -> int:
        to_embed = [c for c in chunks if c.embedding is None]
        if to_embed:
            vectors = self._embedder.embed_documents([c.text for c in to_embed])
            for chunk, vec in zip(to_embed, vectors):
                chunk.embedding = vec
        table = self.settings.table_name
        with self._conn.cursor() as cur:
            for c in chunks:
                cur.execute(
                    f"""INSERT INTO {table} (id, doc_id, source, ordinal, text, metadata, embedding)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (id) DO UPDATE SET text = EXCLUDED.text,
                            embedding = EXCLUDED.embedding""",
                    (c.id, c.doc_id, c.source, c.ordinal, c.text, json.dumps(c.metadata),
                     _vec_literal(c.embedding)),
                )
        return len(chunks)

    def vector_search(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        qvec = _vec_literal(self._embedder.embed_query(query))
        table = self.settings.table_name
        with self._conn.cursor() as cur:
            cur.execute(
                f"""SELECT id, doc_id, source, ordinal, text, metadata,
                           1 - (embedding <=> %s) AS score
                    FROM {table} ORDER BY embedding <=> %s LIMIT %s""",
                (qvec, qvec, k),
            )
            return [_row_to_scored(r) for r in cur.fetchall()]

    def keyword_search(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        table = self.settings.table_name
        with self._conn.cursor() as cur:
            cur.execute(
                f"""SELECT id, doc_id, source, ordinal, text, metadata,
                           ts_rank(tsv, plainto_tsquery('english', %s)) AS score
                    FROM {table}
                    WHERE tsv @@ plainto_tsquery('english', %s)
                    ORDER BY score DESC LIMIT %s""",
                (query, query, k),
            )
            return [_row_to_scored(r) for r in cur.fetchall()]

    def count(self) -> int:
        with self._conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {self.settings.table_name}")
            return int(cur.fetchone()[0])


def _vec_literal(vec: list[float]) -> str:
    return "[" + ",".join(f"{x:.8f}" for x in vec) + "]"


def _row_to_scored(row) -> tuple[Chunk, float]:
    id_, doc_id, source, ordinal, text, metadata, score = row
    chunk = Chunk(
        id=id_, text=text, doc_id=doc_id, source=source, ordinal=ordinal,
        metadata=metadata or {},
    )
    return chunk, float(score)
