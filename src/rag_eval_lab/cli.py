"""Command-line workflows for generation, evaluation, and comparison."""

import argparse
import json
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .documents import chunk_paragraphs, extract_paragraphs
from .evaluation import build_deepeval_metrics, score_cases
from .experiments import compare_summaries, summarize
from .io import read_jsonl, write_jsonl
from .models import EvalCase, ExperimentSummary, RagResult, ScoredCase
from .offline import ExtractiveAnswerer, HeuristicMetric, LexicalRetriever
from .rag import LangChainRetriever, OpenAIAnswerer, RagPipeline, run_cases

_METRIC_NAMES = (
    "answer_relevancy",
    "faithfulness",
    "contextual_relevancy",
    "correctness",
)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(value, "model_dump"):
        value = value.model_dump()
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _load_chunks(pdf: Path, chunk_size: int, chunk_overlap: int) -> list[str]:
    return chunk_paragraphs(
        extract_paragraphs(pdf),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def _run_demo(args: argparse.Namespace) -> int:
    threshold = args.threshold
    chunks = _load_chunks(args.pdf, args.chunk_size, args.chunk_overlap)
    cases = read_jsonl(args.dataset, EvalCase)
    pipeline = RagPipeline(LexicalRetriever(chunks), ExtractiveAnswerer(), retriever_k=args.k)
    results = run_cases(cases, pipeline)
    metrics = [HeuristicMetric(name=name, threshold=threshold) for name in _METRIC_NAMES]
    scored = score_cases(results, metrics)
    summary = summarize(
        args.experiment,
        scored,
        threshold=threshold,
        config={
            "mode": "offline-smoke",
            "retriever": "lexical",
            "retriever_k": args.k,
            "chunk_size": args.chunk_size,
            "chunk_overlap": args.chunk_overlap,
        },
    )
    write_jsonl(args.output_dir / "results.jsonl", results)
    write_jsonl(args.output_dir / "scored.jsonl", scored)
    _write_json(args.output_dir / "summary.json", summary)
    print(f"Offline demo: {len(scored)} cases -> {args.output_dir}")
    return 0


def _run_live(args: argparse.Namespace) -> int:
    load_dotenv()
    chunks = _load_chunks(args.pdf, args.chunk_size, args.chunk_overlap)
    cases = read_jsonl(args.dataset, EvalCase)
    pipeline = RagPipeline(
        LangChainRetriever(chunks, model=args.embedding_model),
        OpenAIAnswerer(model=args.chat_model),
        retriever_k=args.k,
    )
    results = run_cases(cases, pipeline)
    write_jsonl(args.output, results)
    print(f"Generated {len(results)} answers -> {args.output}")
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    load_dotenv()
    results = read_jsonl(args.input, RagResult)
    metrics = build_deepeval_metrics(
        threshold=args.threshold,
        judge_model=args.judge_model,
    )
    scored = score_cases(results, metrics)
    write_jsonl(args.output, scored)
    print(f"Evaluated {len(scored)} cases -> {args.output}")
    return 0


def _summarize(args: argparse.Namespace) -> int:
    cases = read_jsonl(args.input, ScoredCase)
    summary = summarize(
        args.experiment,
        cases,
        threshold=args.threshold,
        config={"retriever_k": args.k} if args.k else {},
    )
    _write_json(args.output, summary)
    print(f"Summary -> {args.output}")
    return 0


def _compare(args: argparse.Namespace) -> int:
    summaries = [
        ExperimentSummary.model_validate_json(path.read_text(encoding="utf-8"))
        for path in args.summaries
    ]
    rows = compare_summaries(summaries, baseline=args.baseline)
    if args.output:
        _write_json(args.output, rows)
    print("experiment\tmetric\taverage\tpass_rate\tdelta")
    for row in rows:
        print(
            f"{row['experiment']}\t{row['metric']}\t{row['average']:.3f}\t"
            f"{row['pass_rate']:.1%}\t{row['delta']:+.3f}"
        )
    return 0


def _add_source_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--chunk-size", type=int, default=1_000)
    parser.add_argument("--chunk-overlap", type=int, default=150)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rag-eval", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("demo", help="Run a deterministic zero-cost smoke experiment")
    _add_source_arguments(demo)
    demo.add_argument("--output-dir", type=Path, required=True)
    demo.add_argument("--experiment", default="offline-demo")
    demo.add_argument("--threshold", type=float, default=0.5)
    demo.set_defaults(handler=_run_demo)

    run = subparsers.add_parser("run", help="Generate answers with OpenAI RAG")
    _add_source_arguments(run)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--embedding-model", default="text-embedding-3-small")
    run.add_argument("--chat-model", default="gpt-4o-mini")
    run.set_defaults(handler=_run_live)

    evaluate = subparsers.add_parser("evaluate", help="Score generated results with DeepEval")
    evaluate.add_argument("--input", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--threshold", type=float, default=0.7)
    evaluate.add_argument("--judge-model", default="gpt-4o-mini")
    evaluate.set_defaults(handler=_evaluate)

    summary = subparsers.add_parser("summarize", help="Aggregate scores and pass rates")
    summary.add_argument("--input", type=Path, required=True)
    summary.add_argument("--output", type=Path, required=True)
    summary.add_argument("--experiment", required=True)
    summary.add_argument("--threshold", type=float, default=0.7)
    summary.add_argument("--k", type=int)
    summary.set_defaults(handler=_summarize)

    compare = subparsers.add_parser("compare", help="Compare summaries against a baseline")
    compare.add_argument("--summaries", type=Path, nargs="+", required=True)
    compare.add_argument("--baseline", required=True)
    compare.add_argument("--output", type=Path)
    compare.set_defaults(handler=_compare)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
