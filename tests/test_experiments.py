import pytest

from rag_eval_lab.experiments import compare_summaries, summarize
from rag_eval_lab.models import ScoredCase


def scored(case_id: str, answer_score: float, correctness_score: float) -> ScoredCase:
    return ScoredCase(
        id=case_id,
        question="Q",
        expected_answer="E",
        actual_answer="A",
        retrieval_context=["C"],
        metadata={"retriever_k": 2},
        scores={"answer_relevancy": answer_score, "correctness": correctness_score},
        passed={
            "answer_relevancy": answer_score >= 0.7,
            "correctness": correctness_score >= 0.7,
        },
        reasons={},
    )


def test_summary_calculates_averages_and_pass_rates() -> None:
    summary = summarize(
        "baseline",
        [scored("1", 1.0, 0.8), scored("2", 0.5, 0.6)],
        threshold=0.7,
        config={"retriever_k": 2},
    )

    assert summary.averages == {"answer_relevancy": 0.75, "correctness": 0.7}
    assert summary.pass_rates == {"answer_relevancy": 0.5, "correctness": 0.5}
    assert summary.test_count == 2


def test_comparison_reports_metric_delta_from_baseline() -> None:
    baseline = summarize("baseline", [scored("1", 0.8, 0.7)], threshold=0.7)
    candidate = summarize("candidate", [scored("1", 0.9, 0.75)], threshold=0.7)

    rows = compare_summaries([baseline, candidate], baseline="baseline")

    candidate_correctness = next(
        row for row in rows if row["experiment"] == "candidate" and row["metric"] == "correctness"
    )
    assert candidate_correctness["delta"] == pytest.approx(0.05)
