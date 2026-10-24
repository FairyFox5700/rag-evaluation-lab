"""Deterministic local adapters used for tests and a zero-cost demo."""

import re
from collections import Counter

from .evaluation import MetricResult
from .models import RagResult

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "do",
    "does",
    "how",
    "in",
    "is",
    "of",
    "the",
    "to",
    "what",
    "where",
    "why",
}


def _tokens(text: str) -> list[str]:
    return [token for token in re.findall(r"[a-z0-9]+", text.lower()) if token not in _STOPWORDS]


def _f1(left: str, right: str) -> float:
    left_counts = Counter(_tokens(left))
    right_counts = Counter(_tokens(right))
    overlap = sum((left_counts & right_counts).values())
    if not left_counts or not right_counts or overlap == 0:
        return 0.0
    precision = overlap / sum(left_counts.values())
    recall = overlap / sum(right_counts.values())
    return 2 * precision * recall / (precision + recall)


def _query_coverage(query: str, candidate: str) -> float:
    query_tokens = _tokens(query)
    if not query_tokens:
        return 0.0
    candidate_tokens = set(_tokens(candidate))
    return sum(token in candidate_tokens for token in query_tokens) / len(query_tokens)


class LexicalRetriever:
    def __init__(self, chunks: list[str]):
        if not chunks:
            raise ValueError("At least one chunk is required")
        self.chunks = chunks

    def retrieve(self, question: str, k: int) -> list[str]:
        question_tokens = set(_tokens(question))
        ranked = sorted(
            enumerate(self.chunks),
            key=lambda item: (
                -len(question_tokens & set(_tokens(item[1]))),
                item[0],
            ),
        )
        return [chunk for _, chunk in ranked[:k]]


class ExtractiveAnswerer:
    def answer(self, question: str, context: list[str]) -> str:
        sentences = [
            sentence.strip()
            for chunk in context
            for sentence in re.split(r"(?<=[.!?])\s+", chunk)
            if sentence.strip()
        ]
        if not sentences:
            return "The retrieved context does not contain an answer."
        return max(sentences, key=lambda sentence: _f1(question, sentence))


class HeuristicMetric:
    """Transparent lexical smoke metric; not a substitute for DeepEval."""

    def __init__(self, *, name: str, threshold: float = 0.5):
        self.name = name
        self.threshold = threshold

    def measure(self, result: RagResult) -> MetricResult:
        context = " ".join(result.retrieval_context)
        if self.name == "answer_relevancy":
            score = _query_coverage(result.question, result.actual_answer)
        elif self.name == "faithfulness":
            answer_tokens = _tokens(result.actual_answer)
            context_tokens = set(_tokens(context))
            score = (
                sum(token in context_tokens for token in answer_tokens) / len(answer_tokens)
                if answer_tokens
                else 0.0
            )
        elif self.name == "contextual_relevancy":
            score = _query_coverage(result.question, context)
        elif self.name == "correctness":
            score = _f1(result.actual_answer, result.expected_answer)
        else:
            raise ValueError(f"Unsupported heuristic metric: {self.name}")
        bounded = round(max(0.0, min(1.0, score)), 6)
        return MetricResult(
            score=bounded,
            passed=bounded >= self.threshold,
            reason="Deterministic lexical smoke score (not an LLM judge).",
        )
