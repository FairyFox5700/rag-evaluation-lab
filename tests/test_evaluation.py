from rag_eval_lab.evaluation import MetricResult, score_cases
from rag_eval_lab.models import RagResult


class FakeMetric:
    name = "correctness"

    def measure(self, result: RagResult) -> MetricResult:
        score = 1.0 if "flippers" in result.actual_answer else 0.0
        return MetricResult(score=score, passed=score >= 0.7, reason="keyword check")


def test_score_cases_persists_scores_pass_flags_and_reasons() -> None:
    result = RagResult(
        id="q1",
        question="How do penguins swim?",
        expected_answer="With flippers",
        actual_answer="They swim with flippers.",
        retrieval_context=["Penguins use flippers."],
    )

    scored = score_cases([result], [FakeMetric()])

    assert scored[0].scores == {"correctness": 1.0}
    assert scored[0].passed == {"correctness": True}
    assert scored[0].reasons == {"correctness": "keyword check"}
