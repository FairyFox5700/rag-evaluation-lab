from rag_eval_lab.models import RagResult
from rag_eval_lab.offline import ExtractiveAnswerer, HeuristicMetric, LexicalRetriever


def test_lexical_retriever_ranks_the_most_relevant_chunk_first() -> None:
    retriever = LexicalRetriever(
        [
            "Penguins use their flippers to swim underwater.",
            "Emperor penguins incubate eggs on their feet.",
        ]
    )

    results = retriever.retrieve("How do penguins swim?", k=1)

    assert results == ["Penguins use their flippers to swim underwater."]


def test_extractive_answerer_selects_a_supported_sentence() -> None:
    answerer = ExtractiveAnswerer()

    answer = answerer.answer(
        "How do penguins swim?",
        ["Penguins cannot fly. Penguins swim using powerful flippers."],
    )

    assert answer == "Penguins swim using powerful flippers."


def test_heuristic_metrics_return_bounded_offline_scores() -> None:
    result = RagResult(
        id="q1",
        question="How do penguins swim?",
        expected_answer="Penguins swim using flippers.",
        actual_answer="Penguins swim using powerful flippers.",
        retrieval_context=[
            "Penguins swim using powerful flippers.",
            "Unrelated background about colonies, eggs, migration, weather, and ocean geography.",
        ],
    )

    outcomes = {
        name: HeuristicMetric(name=name, threshold=0.5).measure(result)
        for name in (
            "answer_relevancy",
            "faithfulness",
            "contextual_relevancy",
            "correctness",
        )
    }

    assert all(0.0 <= outcome.score <= 1.0 for outcome in outcomes.values())
    assert outcomes["faithfulness"].passed is True
    assert outcomes["contextual_relevancy"].score == 1.0
