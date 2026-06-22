"""DocuMind CLI:  ingest documents, query, or run the eval harness.

    documind ingest docs/*.pdf
    documind query "How does re-ranking work?"
    documind eval datasets/sample.json --version v0.1.0
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from documind.config import get_settings
from documind.pipeline import KnowledgeAssistant


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="documind", description="Self-hosted RAG knowledge assistant.")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Ingest files into the knowledge base.")
    p_ingest.add_argument("paths", nargs="+")

    p_query = sub.add_parser("query", help="Ask a question.")
    p_query.add_argument("question")
    p_query.add_argument("-k", "--top-k", type=int, default=None)

    p_eval = sub.add_parser("eval", help="Run the versioned eval harness.")
    p_eval.add_argument("dataset")
    p_eval.add_argument("--version", required=True)
    p_eval.add_argument("--ingest", nargs="*", default=[], help="Files to ingest before evaluating.")

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    assistant = KnowledgeAssistant(get_settings())

    if args.command == "ingest":
        n = assistant.ingest_files(args.paths)
        print(f"Ingested {n} chunks. Knowledge base now holds {assistant.size} chunks.")
        return 0

    if args.command == "query":
        answer = assistant.query(args.question, args.top_k)
        print(answer.text + "\n")
        if answer.citations:
            print("Sources:")
            for c in answer.citations:
                print(f"  [{c.n}] {c.source}: {c.snippet[:100]}…")
        return 0

    if args.command == "eval":
        from documind.evaluation import evaluate, load_dataset

        if args.ingest:
            assistant.ingest_files(args.ingest)
        dataset = load_dataset(args.dataset)
        report = evaluate(assistant, dataset, version=args.version)
        print(json.dumps(report["metrics"], indent=2))
        if report.get("comparison"):
            print("\nAcross releases:\n" + report["comparison"])
        return 0

    parser.print_help(sys.stderr)
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
