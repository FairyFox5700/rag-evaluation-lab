"""Metric adapters and scoring orchestration."""

from dataclasses import dataclass
from typing import Protocol

from .models import RagResult, ScoredCase


@dataclass(frozen=True)
class MetricResult:
    score: float
    passed: bool
    reason: str


class MetricEvaluator(Protocol):
    name: str

    def measure(self, result: RagResult) -> MetricResult: ...


def score_cases(results: list[RagResult], metrics: list[MetricEvaluator]) -> list[ScoredCase]:
    scored_cases: list[ScoredCase] = []
    for result in results:
        scores: dict[str, float] = {}
        passed: dict[str, bool] = {}
        reasons: dict[str, str] = {}
        for metric in metrics:
            outcome = metric.measure(result)
            scores[metric.name] = outcome.score
            passed[metric.name] = outcome.passed
            reasons[metric.name] = outcome.reason
        scored_cases.append(
            ScoredCase(
                **result.model_dump(),
                scores=scores,
                passed=passed,
                reasons=reasons,
            )
        )
    return scored_cases


class DeepEvalMetric:
    """Adapt one DeepEval metric to the project's stable metric contract."""

    def __init__(self, name: str, metric: object):
        self.name = name
        self.metric = metric

    def measure(self, result: RagResult) -> MetricResult:
        from deepeval.test_case import LLMTestCase

        test_case = LLMTestCase(
            input=result.question,
            actual_output=result.actual_answer,
            expected_output=result.expected_answer,
            retrieval_context=result.retrieval_context,
        )
        self.metric.measure(test_case)  # type: ignore[attr-defined]
        score = float(self.metric.score)  # type: ignore[attr-defined]
        success = bool(self.metric.is_successful())  # type: ignore[attr-defined]
        reason = str(self.metric.reason or "")  # type: ignore[attr-defined]
        return MetricResult(score=score, passed=success, reason=reason)


def build_deepeval_metrics(
    *, threshold: float = 0.7, judge_model: str = "gpt-4o-mini"
) -> list[DeepEvalMetric]:
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        ContextualRelevancyMetric,
        FaithfulnessMetric,
        GEval,
    )
    from deepeval.test_case import SingleTurnParams

    return [
        DeepEvalMetric(
            "answer_relevancy",
            AnswerRelevancyMetric(threshold=threshold, model=judge_model),
        ),
        DeepEvalMetric(
            "faithfulness",
            FaithfulnessMetric(threshold=threshold, model=judge_model),
        ),
        DeepEvalMetric(
            "contextual_relevancy",
            ContextualRelevancyMetric(threshold=threshold, model=judge_model),
        ),
        DeepEvalMetric(
            "correctness",
            GEval(
                name="Correctness",
                criteria=(
                    "Determine whether the actual output is factually correct compared with "
                    "the expected output. Penalize contradictions and unsupported additions."
                ),
                evaluation_params=[
                    SingleTurnParams.ACTUAL_OUTPUT,
                    SingleTurnParams.EXPECTED_OUTPUT,
                ],
                threshold=threshold,
                model=judge_model,
            ),
        ),
    ]
