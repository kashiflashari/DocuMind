"""Text chunking.

Uses LangChain's ``RecursiveCharacterTextSplitter`` when available (the
documented path), with a dependency-free fallback so chunking always works.
"""

from __future__ import annotations


def chunk_text(text: str, chunk_size: int = 800, chunk_overlap: int = 120) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        return [c for c in splitter.split_text(text) if c.strip()]
    except Exception:
        return _fallback_chunk(text, chunk_size, chunk_overlap)


def _fallback_chunk(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Paragraph-aware sliding window, used if LangChain isn't installed."""
    if chunk_overlap >= chunk_size:
        chunk_overlap = chunk_size // 4
    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        # Prefer to break on a paragraph/sentence boundary near the end.
        if end < n:
            for sep in ("\n\n", "\n", ". ", " "):
                idx = text.rfind(sep, start + chunk_size // 2, end)
                if idx != -1:
                    end = idx + len(sep)
                    break
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start = max(end - chunk_overlap, start + 1)
    return chunks
