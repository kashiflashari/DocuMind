from documind.evaluation import EvalItem, evaluate


def test_eval_harness_produces_versioned_results(assistant, tmp_path):
    dataset = [
        EvalItem(
            question="How does rerank improve relevance?",
            relevant_sources=["rerank.md"],
            relevant_substrings=["cross encoder"],
            expected_points=["rerank reorders candidate passages"],
        ),
        EvalItem(
            question="What is reciprocal rank fusion?",
            relevant_sources=["fusion.md"],
            relevant_substrings=["reciprocal"],
        ),
    ]
    report = evaluate(assistant, dataset, version="v0.1.0", results_dir=tmp_path)

    metrics = report["metrics"]
    for key in ("precision_at_k", "recall", "citation_validity", "groundedness"):
        assert key in metrics
        assert 0.0 <= metrics[key] <= 1.0

    # Stub answers only ever cite real passage numbers.
    assert metrics["citation_validity"] == 1.0
    # We crafted the dataset so the relevant evidence is retrievable.
    assert metrics["recall"] > 0.0

    assert (tmp_path / "v0.1.0.json").exists()
    assert "v0.1.0" in report["comparison"]


def test_eval_comparison_tracks_multiple_versions(assistant, tmp_path):
    dataset = [EvalItem(question="What is reciprocal rank fusion?", relevant_sources=["fusion.md"])]
    evaluate(assistant, dataset, version="v0.1.0", results_dir=tmp_path)
    report = evaluate(assistant, dataset, version="v0.2.0", results_dir=tmp_path)
    # Both releases appear in the comparison table.
    assert "v0.1.0" in report["comparison"]
    assert "v0.2.0" in report["comparison"]
