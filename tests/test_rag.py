from rag_eval_lab.models import EvalCase
from rag_eval_lab.rag import RagPipeline, run_cases


class FakeRetriever:
    def retrieve(self, question: str, k: int) -> list[str]:
        assert k == 2
        return ["Penguins swim using flippers.", "Their bodies are streamlined."]


class FakeAnswerer:
    def answer(self, question: str, context: list[str]) -> str:
        return f"Based on {len(context)} sources: penguins swim using flippers."


def test_pipeline_returns_answer_context_and_trace_metadata() -> None:
    pipeline = RagPipeline(FakeRetriever(), FakeAnswerer(), retriever_k=2)
    case = EvalCase(
        id="swimming",
        question="How do penguins swim?",
        expected_answer="They use their flippers.",
    )

    result = pipeline.run(case)

    assert result.actual_answer.startswith("Based on 2 sources")
    assert result.retrieval_context == [
        "Penguins swim using flippers.",
        "Their bodies are streamlined.",
    ]
    assert result.metadata == {"retriever_k": 2}


def test_run_cases_keeps_dataset_order() -> None:
    pipeline = RagPipeline(FakeRetriever(), FakeAnswerer(), retriever_k=2)
    cases = [
        EvalCase(id="first", question="First?", expected_answer="A"),
        EvalCase(id="second", question="Second?", expected_answer="B"),
    ]

    assert [result.id for result in run_cases(cases, pipeline)] == ["first", "second"]
