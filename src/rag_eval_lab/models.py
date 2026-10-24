"""Validated data contracts shared by generation and evaluation."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EvalCase(StrictModel):
    id: str = Field(min_length=1)
    question: str
    expected_answer: str

    @field_validator("question", "expected_answer")
    @classmethod
    def must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value.strip()


class RagResult(EvalCase):
    actual_answer: str
    retrieval_context: list[str] = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ScoredCase(RagResult):
    scores: dict[str, float]
    passed: dict[str, bool]
    reasons: dict[str, str] = Field(default_factory=dict)

    @field_validator("scores")
    @classmethod
    def scores_must_be_unit_interval(cls, scores: dict[str, float]) -> dict[str, float]:
        invalid = {name: value for name, value in scores.items() if not 0.0 <= value <= 1.0}
        if invalid:
            raise ValueError(f"scores must be between 0 and 1: {invalid}")
        return scores


class ExperimentSummary(StrictModel):
    experiment: str = Field(min_length=1)
    test_count: int = Field(ge=1)
    threshold: float = Field(ge=0.0, le=1.0)
    config: dict[str, Any] = Field(default_factory=dict)
    averages: dict[str, float]
    pass_rates: dict[str, float]
