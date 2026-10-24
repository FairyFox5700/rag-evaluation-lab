"""Composable RAG pipeline with injectable retrieval and generation boundaries."""

from collections.abc import Iterable
from typing import Protocol

from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from .models import EvalCase, RagResult


class Retriever(Protocol):
    def retrieve(self, question: str, k: int) -> list[str]: ...


class Answerer(Protocol):
    def answer(self, question: str, context: list[str]) -> str: ...


class RagPipeline:
    def __init__(self, retriever: Retriever, answerer: Answerer, *, retriever_k: int = 3):
        if retriever_k < 1:
            raise ValueError("retriever_k must be at least 1")
        self.retriever = retriever
        self.answerer = answerer
        self.retriever_k = retriever_k

    def run(self, case: EvalCase) -> RagResult:
        context = self.retriever.retrieve(case.question, self.retriever_k)
        if not context:
            raise ValueError(f"Retriever returned no context for case {case.id}")
        answer = self.answerer.answer(case.question, context)
        return RagResult(
            **case.model_dump(),
            actual_answer=answer,
            retrieval_context=context,
            metadata={"retriever_k": self.retriever_k},
        )


class LangChainRetriever:
    def __init__(self, chunks: list[str], *, model: str = "text-embedding-3-small"):
        if not chunks:
            raise ValueError("At least one chunk is required")
        embeddings = OpenAIEmbeddings(model=model)
        self.vector_store = InMemoryVectorStore.from_texts(chunks, embedding=embeddings)

    def retrieve(self, question: str, k: int) -> list[str]:
        documents = self.vector_store.similarity_search(question, k=k)
        return [document.page_content for document in documents]


class OpenAIAnswerer:
    def __init__(self, *, model: str = "gpt-4o-mini"):
        self.model = ChatOpenAI(model=model, temperature=0)

    def answer(self, question: str, context: list[str]) -> str:
        numbered_context = "\n\n".join(
            f"[{index}] {chunk}" for index, chunk in enumerate(context, start=1)
        )
        response = self.model.invoke(
            [
                (
                    "system",
                    "Answer only from the supplied context. Be concise and factual. "
                    "If the context does not contain the answer, say so.",
                ),
                ("human", f"Context:\n{numbered_context}\n\nQuestion: {question}"),
            ]
        )
        return str(response.content)


def run_cases(cases: Iterable[EvalCase], pipeline: RagPipeline) -> list[RagResult]:
    return [pipeline.run(case) for case in cases]
