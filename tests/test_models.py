import pytest
from pydantic import ValidationError

from rag_eval_lab.models import EvalCase, RagResult, ScoredCase


def test_evaluation_records_validate_required_fields() -> None:
    case = EvalCase(id="habitat", question="Where?", expected_answer="South.")
    result = RagResult(
        id=case.id,
        question=case.question,
        expected_answer=case.expected_answer,
        actual_answer="In the south.",
        retrieval_context=["Penguins occur in the Southern Hemisphere."],
        metadata={"retriever_k": 2},
    )
    scored = ScoredCase(
        **result.model_dump(),
        scores={"answer_relevancy": 0.9},
        passed={"answer_relevancy": True},
        reasons={"answer_relevancy": "Addresses the question."},
    )

    assert scored.id == "habitat"
    assert scored.scores["answer_relevancy"] == pytest.approx(0.9)


def test_eval_case_rejects_blank_question() -> None:
    with pytest.raises(ValidationError):
        EvalCase(id="bad", question="   ", expected_answer="Something")


def test_scored_case_rejects_score_outside_unit_interval() -> None:
    with pytest.raises(ValidationError):
        ScoredCase(
            id="bad-score",
            question="Q",
            expected_answer="E",
            actual_answer="A",
            retrieval_context=["C"],
            scores={"correctness": 1.1},
            passed={"correctness": True},
            reasons={},
        )
