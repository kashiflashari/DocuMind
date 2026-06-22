# DocuMind

**A self-hosted RAG knowledge assistant** that ingests your PDFs and documents into a hybrid-search knowledge base with re-ranking and inline source citations — serving grounded answers with sub-~2s latency.

Built on **LangChain**, with **hybrid retrieval** (dense + keyword via Reciprocal Rank Fusion), a **Cohere cross-encoder re-ranker**, a **pgvector** backend, a **FastAPI** API, a **React** chat UI, and a **versioned evaluation harness**.

> Runs **fully offline out of the box** (hash embeddings, in-memory store, stub LLM, identity re-ranker) — no API keys or database needed to try it. Plug in OpenAI/Cohere + pgvector for production.

---

## How it works

```
  PDFs / docs ─▶ Ingestion ─▶ chunk ─▶ embed ─┐
                                              ▼
                                      Vector store (pgvector / in-memory)
                                              │
   question ─▶ ┌── vector search ──┐          │
               │                   ├─▶ Reciprocal Rank Fusion ─▶ Cohere re-rank ─▶ top-k
               └── keyword (BM25) ──┘                                                │
                                                                                     ▼
                                          LLM answer with inline [n] citations ◀─ context
                                                                                     │
                                                                                     ▼
                                          Eval harness: precision@k · recall ·
                                          citation-validity · groundedness
```

- **Ingestion** ([ingest.py](src/documind/ingest.py)) — loads PDFs (`pypdf`) and text/markdown, chunks with LangChain's `RecursiveCharacterTextSplitter`.
- **Hybrid retrieval** ([retrieval.py](src/documind/retrieval.py)) — fuses dense vector search and BM25 keyword search with [Reciprocal Rank Fusion](src/documind/retrieval.py).
- **Re-ranking** ([reranker.py](src/documind/reranker.py)) — Cohere cross-encoder, with an identity fallback when no key is set.
- **Grounded answers** ([answer.py](src/documind/answer.py)) — the LLM answers only from retrieved context and cites `[n]`; cited numbers are resolved back to sources, hallucinated ones are dropped.
- **Vector stores** ([stores/](src/documind/stores/)) — `pgvector` (Postgres `<=>` + full-text) or a pure-Python in-memory store (with optional JSON persistence).
- **Eval harness** ([evaluation/harness.py](src/documind/evaluation/harness.py)) — versioned metrics written to `eval_results/<version>.json` and compared across releases.

## Quickstart (offline, no keys)

```bash
python -m venv .venv && source .venv/Scripts/activate   # Windows Git Bash
pip install -e ".[dev]"

python examples/quickstart.py        # ingest sample docs, query, run eval

# or the CLI (set MEMORY_PERSIST_PATH in .env to share data across calls):
documind ingest examples/sample_docs/*.md
documind query "How does re-ranking improve the final answer?"
```

## Real deployment

```bash
cp .env.example .env   # set providers + DATABASE_URL
pip install -e ".[openai,cohere,pgvector,api]"
```

```dotenv
EMBEDDING_PROVIDER=openai
OPENAI_API_KEY=sk-...
COHERE_API_KEY=...                 # enables cross-encoder re-ranking
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...
VECTOR_STORE=pgvector
DATABASE_URL=postgresql://documind:documind@localhost:5432/documind
```

## HTTP API

```bash
uvicorn documind.api.main:app --reload
```

| Method | Path      | Description                                  |
| ------ | --------- | -------------------------------------------- |
| GET    | `/health` | Backend status + chunks indexed              |
| POST   | `/ingest` | `{ "documents": [{ "text", "source" }] }`    |
| POST   | `/query`  | `{ "question", "top_k" }` → answer + citations |

## Web UI

A React + Vite chat interface lives in [web/](web/): ask questions, expand inline citations, and paste documents to index on the fly.

```bash
cd web && npm install && npm run dev   # http://localhost:5173
```

## Evaluation harness

Track retrieval and answer quality across releases:

```bash
documind eval datasets/sample.json --version v0.1.0 --ingest examples/sample_docs/*.md
```

Reports `precision@k`, `recall`, `citation_validity` (no hallucinated citations), `groundedness`, and `answer_coverage`, writes `eval_results/v0.1.0.json`, and prints a comparison table across all recorded versions.

## Docker

```bash
cp .env.example .env
docker compose up --build
# Postgres+pgvector on :5432, API on :8000, web UI on :8080
```

## Testing

```bash
pytest -q     # 13 tests, fully offline
```

## Project layout

```
src/documind/
├── config.py          # pydantic-settings configuration
├── models.py          # Chunk / RetrievedChunk / Citation / Answer
├── chunking.py        # LangChain splitter (+ fallback)
├── embeddings.py      # hash (offline) / OpenAI / Cohere
├── keyword.py         # BM25 lexical index
├── stores/            # in-memory + pgvector backends
├── reranker.py        # Cohere cross-encoder (+ identity fallback)
├── retrieval.py       # hybrid RRF retriever
├── llm.py             # answer model (stub / Anthropic / OpenAI)
├── answer.py          # grounded answer + citation resolution
├── ingest.py          # PDF/text loaders → chunks
├── pipeline.py        # KnowledgeAssistant facade
├── evaluation/        # versioned eval harness
├── api/               # FastAPI app
└── cli.py             # `documind` command
web/                   # React + Vite chat UI
tests/                 # offline pytest suite
```

## Tech stack

| Layer        | Technology               |
| ------------ | ------------------------ |
| RAG framework| LangChain                |
| Vector store | pgvector (PostgreSQL)    |
| Re-ranking   | Cohere cross-encoder     |
| LLM          | Anthropic / OpenAI       |
| API          | FastAPI                  |
| Frontend     | React + Vite             |
| Packaging    | Docker                   |

## Design notes

- **Graceful degradation** — missing keys/DB fall back to offline defaults rather than failing, so the system always runs.
- **Honest citations** — only `[n]` markers that resolve to a retrieved passage become citations; the eval harness explicitly scores citation validity.
- **Swap-able everything** — embedder, vector store, re-ranker, and LLM are all behind small interfaces.

## Roadmap

- [x] Ingestion + chunking (PDF/text) via LangChain
- [x] Hybrid vector + keyword retrieval with RRF
- [x] Cohere cross-encoder re-ranking (+ fallback)
- [x] Grounded answers with resolved inline citations
- [x] pgvector + in-memory backends
- [x] React chat UI
- [x] Versioned evaluation harness
- [x] FastAPI + Docker Compose, offline test suite
- [ ] Streaming responses
- [ ] Incremental re-indexing & document deletion
- [ ] Auth + multi-tenant collections

## License

[MIT](./LICENSE)
