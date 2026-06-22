# DocuMind

**A self-hosted RAG knowledge assistant** that ingests your PDFs and documents into a hybrid-search knowledge base with re-ranking and inline source citations — serving grounded answers with sub-~2s latency.

> **Status:** 🚧 In active development. Architecture and roadmap below; implementation in progress.

---

## Overview

DocuMind turns a pile of documents into a queryable, citation-backed knowledge base you can host yourself. It combines dense vector search with keyword search (hybrid retrieval), re-ranks the candidates with a cross-encoder, and answers questions with inline citations pointing back to the source passages. A versioned evaluation harness tracks retrieval precision and answer faithfulness across releases so quality is measurable, not vibes.

## How it works

```
   PDFs / docs ──▶ Ingestion ──▶ chunk + embed ──▶ pgvector
                                                      │
   question ──▶ Hybrid search (vector + keyword) ◀────┘
                          │
                          ▼
                  Cohere re-ranker  (re-orders top candidates)
                          │
                          ▼
                  LLM answer + inline citations
                          │
                          ▼
            Eval harness: retrieval precision + faithfulness
```

- **Ingestion** — parses PDFs/docs, chunks them, and stores embeddings in pgvector.
- **Hybrid search** — combines semantic vector search with keyword search for better recall.
- **Re-ranking** — a Cohere cross-encoder re-orders the top candidates for precision.
- **Grounded answers** — responses cite the exact source passages inline.
- **Eval harness** — versioned tracking of retrieval precision and answer faithfulness across releases.

## Tech stack

| Layer            | Technology            |
| ---------------- | --------------------- |
| RAG framework    | LangChain             |
| Vector store     | pgvector (PostgreSQL) |
| Re-ranking       | Cohere re-ranker      |
| API              | FastAPI               |
| Frontend         | React                 |
| Packaging        | Docker                |

## Key features

- Self-hosted ingestion of PDFs and documents into a hybrid-search knowledge base.
- Cross-encoder re-ranking for high-precision retrieval.
- Grounded answers with inline source citations.
- Sub-~2s answer latency target.
- Versioned eval harness tracking retrieval precision and answer faithfulness.

## Roadmap

- [ ] Document ingestion + chunking pipeline
- [ ] pgvector storage + hybrid (vector + keyword) search
- [ ] Cohere re-ranking stage
- [ ] Grounded answering with inline citations
- [ ] React chat UI
- [ ] Versioned evaluation harness
- [ ] FastAPI service + Dockerfile

## Getting started

> Setup instructions will be added as the implementation lands.

```bash
# Coming soon
git clone https://github.com/kashiflashari/DocuMind.git
cd DocuMind
```

## License

[MIT](./LICENSE)
