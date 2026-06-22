"""DocuMind — a self-hosted RAG knowledge assistant.

Ingests documents into a hybrid-search knowledge base (vector + keyword) with
Cohere re-ranking and inline source citations. See
:class:`documind.pipeline.KnowledgeAssistant`.
"""

from documind.config import Settings, get_settings

__version__ = "0.1.0"
__all__ = ["Settings", "get_settings", "__version__"]
