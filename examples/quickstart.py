"""Quickstart: ingest the sample docs, ask a question, and run the eval harness.

    python examples/quickstart.py

Runs fully offline (hash embeddings + stub LLM). Configure `.env` for real
embeddings/LLM/Cohere re-ranking and a pgvector backend.
"""

from pathlib import Path

from documind.config import Settings
from documind.evaluation import evaluate, load_dataset
from documind.pipeline import KnowledgeAssistant

HERE = Path(__file__).parent


def main() -> None:
    settings = Settings(_env_file=None)  # offline defaults
    assistant = KnowledgeAssistant(settings)

    docs = sorted((HERE / "sample_docs").glob("*.md"))
    assistant.ingest_files([str(p) for p in docs])
    print(f"Indexed {assistant.size} chunks from {len(docs)} documents.\n")

    answer = assistant.query("How does re-ranking improve the final answer?")
    print("Q:", answer.question)
    print("A:", answer.text)
    for c in answer.citations:
        print(f"   [{c.n}] {Path(c.source).name}: {c.snippet[:80]}…")

    print("\nRunning eval harness …")
    report = evaluate(
        assistant,
        load_dataset(HERE.parent / "datasets" / "sample.json"),
        version="quickstart",
        results_dir=HERE.parent / "eval_results",
    )
    print("Metrics:", report["metrics"])


if __name__ == "__main__":
    main()
