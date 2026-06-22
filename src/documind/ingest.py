"""Document ingestion: load files → chunk → build Chunk objects.

Supports PDFs (via ``pypdf``) and plain-text / markdown files. Embedding and
persistence are handled by the vector store.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from documind.chunking import chunk_text
from documind.config import Settings
from documind.models import Chunk

logger = logging.getLogger(__name__)

TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".rst", ".text"}


def load_file(path: str | Path) -> str:
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        return _load_pdf(path)
    if path.suffix.lower() in TEXT_SUFFIXES:
        return path.read_text(encoding="utf-8", errors="ignore")
    raise ValueError(f"Unsupported file type: {path.suffix} ({path})")


def _load_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)


def _doc_id(source: str) -> str:
    return hashlib.sha1(source.encode()).hexdigest()[:12]


def build_chunks(text: str, source: str, settings: Settings, metadata: dict | None = None) -> list[Chunk]:
    doc_id = _doc_id(source)
    pieces = chunk_text(text, settings.chunk_size, settings.chunk_overlap)
    return [
        Chunk(
            id=f"{doc_id}:{i}",
            text=piece,
            doc_id=doc_id,
            source=source,
            ordinal=i,
            metadata=metadata or {},
        )
        for i, piece in enumerate(pieces)
    ]


def ingest_texts(items: list[dict], store, settings: Settings) -> int:
    """Ingest ``[{"text": ..., "source": ...}]`` items. Returns chunks added."""
    chunks: list[Chunk] = []
    for item in items:
        chunks.extend(build_chunks(item["text"], item.get("source", "inline"), settings,
                                   item.get("metadata")))
    added = store.add(chunks)
    logger.info("Ingested %d chunks from %d documents.", added, len(items))
    return added


def ingest_files(paths: list[str | Path], store, settings: Settings) -> int:
    items = []
    for p in paths:
        try:
            items.append({"text": load_file(p), "source": str(p)})
        except Exception as exc:
            logger.warning("Skipping %s: %s", p, exc)
    return ingest_texts(items, store, settings)
