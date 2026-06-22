"""Answer-generation LLM factory.

Anthropic / OpenAI via LangChain, plus a dependency-free ``StubChatModel`` that
produces a grounded, citation-bearing answer offline (for demos and tests).
"""

from __future__ import annotations

import re
from typing import Any, Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from documind.config import Settings


def get_chat_model(settings: Settings) -> BaseChatModel:
    provider = settings.llm_provider.lower()
    if provider == "stub":
        return StubChatModel()
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=settings.anthropic_model,
            api_key=settings.anthropic_api_key,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
        )
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
        )
    raise ValueError(f"Unknown LLM_PROVIDER: {settings.llm_provider!r}")


def message_text(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            b if isinstance(b, str) else str(b.get("text", "")) for b in content
        )
    return str(content)


class StubChatModel(BaseChatModel):
    """Offline answer model: grounds its reply in whatever context numbers it sees."""

    @property
    def _llm_type(self) -> str:
        return "documind-stub"

    def _generate(
        self,
        messages: Sequence[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        prompt = "\n".join(message_text(m) for m in messages)
        numbers = re.findall(r"^\s*\[(\d+)\]", prompt, re.MULTILINE)
        question = ""
        m = re.search(r"QUESTION:\s*(.+)", prompt)
        if m:
            question = m.group(1).strip()

        if numbers:
            cites = "".join(f"[{n}]" for n in numbers[:2])
            text = (
                f"Based on the retrieved knowledge base, here is a grounded answer to "
                f"\"{question}\": the supporting passages {cites} address the question "
                "directly. (Offline stub answer — set LLM_PROVIDER to anthropic/openai "
                "for a fully reasoned response.)"
            )
        else:
            text = (
                "I could not find anything relevant in the knowledge base to answer "
                f"\"{question}\". (Offline stub answer.)"
            )
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])
