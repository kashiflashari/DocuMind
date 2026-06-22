"""A versioned evaluation harness.

Runs a labelled dataset through the assistant and reports:

- ``precision_at_k`` — fraction of retrieved chunks that are relevant.
- ``recall``          — fraction of expected evidence substrings retrieved.
- ``citation_validity`` — fraction of cited [n] that resolve to a real passage
  (catches hallucinated citations).
- ``groundedness``    — overlap of answer terms with the cited passages.
- ``answer_coverage`` — fraction of expected key points reflected in the answer.

Results are written to ``eval_results/<version>.json`` and compared against
previous releases so quality is tracked over time.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOP = {
    "the", "a", "an", "and", "or", "of", "to", "in", "is", "are", "for", "on",
    "with", "this", "that", "it", "as", "by", "be", "based", "here", "answer",
}


def _terms(text: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOP and len(t) > 2}


@dataclass
class EvalItem:
    question: str
    relevant_sources: list[str] = field(default_factory=list)
    relevant_substrings: list[str] = field(default_factory=list)
    expected_points: list[str] = field(default_factory=list)


def load_dataset(path: str | Path) -> list[EvalItem]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [EvalItem(**item) for item in data]


def _chunk_relevant(chunk, item: EvalItem) -> bool:
    if any(s.lower() in chunk.source.lower() for s in item.relevant_sources):
        return True
    text = chunk.text.lower()
    return any(sub.lower() in text for sub in item.relevant_substrings)


def _score_item(item: EvalItem, contexts, answer) -> dict:
    # Retrieval precision@k.
    retrieved = [rc.chunk for rc in contexts]
    relevant_flags = [_chunk_relevant(c, item) for c in retrieved]
    precision = sum(relevant_flags) / len(retrieved) if retrieved else 0.0

    # Recall over expected evidence substrings.
    joined = " ".join(c.text.lower() for c in retrieved)
    if item.relevant_substrings:
        found = sum(1 for s in item.relevant_substrings if s.lower() in joined)
        recall = found / len(item.relevant_substrings)
    else:
        recall = 1.0 if any(relevant_flags) else 0.0

    # Citation validity (no hallucinated citation numbers).
    used = [int(n) for n in re.findall(r"\[(\d+)\]", answer.text)]
    valid = [n for n in used if 1 <= n <= len(contexts)]
    citation_validity = (len(valid) / len(used)) if used else 1.0

    # Groundedness: answer terms covered by the cited passages.
    cited_text = " ".join(c.snippet for c in answer.citations) or joined
    ans_terms = _terms(answer.text)
    ground_terms = _terms(cited_text)
    groundedness = (
        len(ans_terms & ground_terms) / len(ans_terms) if ans_terms else 1.0
    )

    # Answer coverage of expected key points.
    if item.expected_points:
        covered = 0
        ans_lower = answer.text.lower()
        for point in item.expected_points:
            pt_terms = _terms(point)
            if pt_terms and len(pt_terms & _terms(ans_lower)) / len(pt_terms) >= 0.5:
                covered += 1
        answer_coverage = covered / len(item.expected_points)
    else:
        answer_coverage = float("nan")

    return {
        "question": item.question,
        "precision_at_k": round(precision, 3),
        "recall": round(recall, 3),
        "citation_validity": round(citation_validity, 3),
        "groundedness": round(groundedness, 3),
        "answer_coverage": None if answer_coverage != answer_coverage else round(answer_coverage, 3),
    }


def _mean(values: list[float]) -> float:
    vals = [v for v in values if v is not None]
    return round(sum(vals) / len(vals), 4) if vals else 0.0


def evaluate(
    assistant,
    dataset: list[EvalItem],
    version: str,
    results_dir: str | Path = "eval_results",
    top_k: int | None = None,
) -> dict:
    per_item = []
    for item in dataset:
        contexts = assistant.retriever.retrieve(item.question, top_k)
        from documind.answer import generate_answer

        answer = generate_answer(item.question, contexts, assistant.settings, model=assistant.model)
        per_item.append(_score_item(item, contexts, answer))

    metrics = {
        "precision_at_k": _mean([r["precision_at_k"] for r in per_item]),
        "recall": _mean([r["recall"] for r in per_item]),
        "citation_validity": _mean([r["citation_validity"] for r in per_item]),
        "groundedness": _mean([r["groundedness"] for r in per_item]),
        "answer_coverage": _mean([r["answer_coverage"] for r in per_item]),
    }
    report = {"version": version, "n_items": len(dataset), "metrics": metrics, "per_item": per_item}

    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{version}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["comparison"] = _comparison_table(out_dir)
    return report


def _comparison_table(results_dir: Path) -> str:
    rows = []
    for path in sorted(results_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            rows.append((data["version"], data["metrics"]))
        except Exception:
            continue
    if not rows:
        return ""
    cols = ["precision_at_k", "recall", "citation_validity", "groundedness", "answer_coverage"]
    header = "| version | " + " | ".join(cols) + " |"
    sep = "|" + "---|" * (len(cols) + 1)
    lines = [header, sep]
    for version, metrics in rows:
        cells = " | ".join(f"{metrics.get(c, 0):.3f}" for c in cols)
        lines.append(f"| {version} | {cells} |")
    return "\n".join(lines)
