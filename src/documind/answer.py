"""Grounded answer generation with inline citations.

Numbers the retrieved chunks ``[1]…[k]``, asks the LLM to answer using only that
context and to cite with ``[n]``, then resolves the cited numbers back to source
chunks. Only citations actually used in the answer are returned.
"""

from __future__ import annotations

import re

from documind.config import Settings
from documind.llm import get_chat_model, message_text
from documind.models import Answer, Citation, RetrievedChunk

from langchain_core.messages import HumanMessage, SystemMessage

_SYSTEM = (
    "You are a precise knowledge-base assistant. Answer the question using ONLY the "
    "numbered context passages below. Cite every claim with the passage number in "
    "square brackets, e.g. [1]. If the context does not contain the answer, say so "
    "plainly — do not use outside knowledge."
)


def generate_answer(
    question: str,
    contexts: list[RetrievedChunk],
    settings: Settings,
    model=None,
) -> Answer:
    model = model or get_chat_model(settings)
    context_block = _format_contexts(contexts)
    messages = [
        SystemMessage(content=_SYSTEM),
        HumanMessage(content=f"CONTEXT:\n{context_block}\n\nQUESTION: {question}"),
    ]
    text = message_text(model.invoke(messages)).strip()

    citations = _resolve_citations(text, contexts)
    return Answer(question=question, text=text, citations=citations, contexts=contexts)


def _format_contexts(contexts: list[RetrievedChunk]) -> str:
    if not contexts:
        return "(no relevant passages found)"
    lines = []
    for i, rc in enumerate(contexts, start=1):
        lines.append(f"[{i}] (source: {rc.chunk.source}) {rc.chunk.text}")
    return "\n\n".join(lines)


def _resolve_citations(text: str, contexts: list[RetrievedChunk]) -> list[Citation]:
    used = sorted({int(n) for n in re.findall(r"\[(\d+)\]", text)})
    citations: list[Citation] = []
    for n in used:
        if 1 <= n <= len(contexts):
            chunk = contexts[n - 1].chunk
            citations.append(
                Citation(
                    n=n,
                    chunk_id=chunk.id,
                    source=chunk.source,
                    snippet=chunk.text[:200],
                )
            )
    return citations
