"""DocuMind HTTP API.

    uvicorn documind.api.main:app --reload
"""

from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from documind.config import get_settings
from documind.pipeline import KnowledgeAssistant

app = FastAPI(title="DocuMind", description="Self-hosted RAG knowledge assistant", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache
def get_assistant() -> KnowledgeAssistant:
    return KnowledgeAssistant(get_settings())


class IngestText(BaseModel):
    text: str
    source: str = "inline"


class IngestRequest(BaseModel):
    documents: list[IngestText]


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=2)
    top_k: int | None = None


class CitationModel(BaseModel):
    n: int
    chunk_id: str
    source: str
    snippet: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    citations: list[CitationModel]


@app.get("/health")
def health() -> dict:
    settings = get_settings()
    return {
        "status": "ok",
        "store": settings.vector_store,
        "embeddings": settings.embedding_provider,
        "llm_provider": settings.llm_provider,
        "rerank": "cohere" if settings.cohere_rerank_available else "identity",
        "chunks_indexed": get_assistant().size,
    }


@app.post("/ingest")
def ingest(req: IngestRequest) -> dict:
    n = get_assistant().ingest_texts([d.model_dump() for d in req.documents])
    return {"ingested_chunks": n, "chunks_indexed": get_assistant().size}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    answer = get_assistant().query(req.question, req.top_k)
    return QueryResponse(
        question=answer.question,
        answer=answer.text,
        citations=[CitationModel(**c.__dict__) for c in answer.citations],
    )
