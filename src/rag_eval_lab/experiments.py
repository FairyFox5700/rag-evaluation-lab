"""Experiment aggregation and comparison."""

from typing import Any

from .models import ExperimentSummary, ScoredCase


def summarize(
    experiment: str,
    cases: list[ScoredCase],
    *,
    threshold: float,
    config: dict[str, Any] | None = None,
) -> ExperimentSummary:
    if not cases:
        raise ValueError("Cannot summarize an empty experiment")
    metric_names = sorted(cases[0].scores)
    if any(sorted(case.scores) != metric_names for case in cases):
        raise ValueError("Every scored case must contain the same metrics")

    count = len(cases)
    averages = {
        metric: round(sum(case.scores[metric] for case in cases) / count, 6)
        for metric in metric_names
    }
    pass_rates = {
        metric: round(sum(case.passed[metric] for case in cases) / count, 6)
        for metric in metric_names
    }
    return ExperimentSummary(
        experiment=experiment,
        test_count=count,
        threshold=threshold,
        config=config or {},
        averages=averages,
        pass_rates=pass_rates,
    )


def compare_summaries(
    summaries: list[ExperimentSummary], *, baseline: str
) -> list[dict[str, str | float]]:
    by_name = {summary.experiment: summary for summary in summaries}
    if baseline not in by_name:
        raise ValueError(f"Baseline experiment not found: {baseline}")
    baseline_summary = by_name[baseline]
    rows: list[dict[str, str | float]] = []
    for summary in summaries:
        for metric, score in sorted(summary.averages.items()):
            if metric not in baseline_summary.averages:
                raise ValueError(f"Metric {metric} is missing from baseline {baseline}")
            rows.append(
                {
                    "experiment": summary.experiment,
                    "metric": metric,
                    "average": score,
                    "pass_rate": summary.pass_rates[metric],
                    "delta": round(score - baseline_summary.averages[metric], 6),
                }
            )
    return rows
