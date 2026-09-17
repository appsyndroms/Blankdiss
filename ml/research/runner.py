from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ml.config import TARGETS, WALK_FORWARD_WINDOWS
from ml.dataset import build_target, load_features

from .experiments import EXPERIMENTS
from .signals import build_signal


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "data" / "processed" / "research"

RANDOM_STATE = 42
BOOTSTRAP_ITERATIONS = 2_000


@dataclass
class ExperimentResult:
    experiment_id: str
    experiment_name: str
    signal: str
    target: str
    window: str

    n: int
    selected_n: int
    non_selected_n: int

    baseline_rate: float | None
    selected_rate: float | None
    lift: float | None

    selected_mean_return: float | None
    non_selected_mean_return: float | None
    return_difference: float | None

    selected_median_return: float | None
    non_selected_median_return: float | None

    return_ci_low: float | None
    return_ci_high: float | None

    probability_positive_difference: float | None

    status: str


def utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def safe_float(value: Any) -> float | None:
    if value is None:
        return None

    try:
        value = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(value):
        return None

    return value


def percentile_mask(
    series: pd.Series,
    fraction: float,
) -> pd.Series:
    """
    Select the upper percentile tail of a signal.

    Example:
        fraction=0.01 -> top 1%
        fraction=0.05 -> top 5%
    """
    if series.empty:
        return pd.Series(False, index=series.index)

    threshold = series.quantile(1.0 - fraction)

    return series >= threshold


def bootstrap_difference(
    selected_returns: np.ndarray,
    non_selected_returns: np.ndarray,
    *,
    iterations: int = BOOTSTRAP_ITERATIONS,
    random_state: int = RANDOM_STATE,
) -> tuple[float | None, float | None, float | None]:
    """
    Bootstrap the difference:

        mean(selected) - mean(non_selected)

    Returns:
        lower CI,
        upper CI,
        P(difference > 0)
    """
    if len(selected_returns) == 0 or len(non_selected_returns) == 0:
        return None, None, None

    rng = np.random.default_rng(random_state)

    selected_returns = np.asarray(selected_returns, dtype=float)
    non_selected_returns = np.asarray(non_selected_returns, dtype=float)

    selected_returns = selected_returns[np.isfinite(selected_returns)]
    non_selected_returns = non_selected_returns[
        np.isfinite(non_selected_returns)
    ]

    if len(selected_returns) == 0 or len(non_selected_returns) == 0:
        return None, None, None

    selected_indices = rng.integers(
        0,
        len(selected_returns),
        size=(iterations, len(selected_returns)),
    )

    non_selected_indices = rng.integers(
        0,
        len(non_selected_returns),
        size=(iterations, len(non_selected_returns)),
    )

    selected_means = selected_returns[selected_indices].mean(axis=1)
    non_selected_means = non_selected_returns[
        non_selected_indices
    ].mean(axis=1)

    differences = selected_means - non_selected_means

    low, high = np.percentile(differences, [2.5, 97.5])
    probability_positive = float(np.mean(differences > 0))

    return (
        float(low),
        float(high),
        probability_positive,
    )


def classify_status(
    *,
    selected_n: int,
    difference: float | None,
    ci_low: float | None,
    ci_high: float | None,
    probability_positive: float | None,
) -> str:
    """
    Descriptive research status.

    These labels are deliberately not investment recommendations.
    """
    if selected_n < 20:
        return "INSUFFICIENT DATA"

    if difference is None:
        return "NO SIGNAL"

    if (
        ci_low is not None
        and ci_high is not None
        and ci_low > 0
        and probability_positive is not None
        and probability_positive >= 0.95
    ):
        return "STRONG RESEARCH CANDIDATE"

    if (
        probability_positive is not None
        and probability_positive >= 0.80
        and difference > 0
    ):
        return "INTERESTING"

    if (
        ci_low is not None
        and ci_high is not None
        and ci_low <= 0 <= ci_high
    ):
        return "UNSTABLE"

    return "NO SIGNAL"


def evaluate_experiment(
    df: pd.DataFrame,
    experiment: dict[str, Any],
    target_definition: dict[str, Any],
    window: dict[str, Any],
) -> ExperimentResult:
    experiment_id = experiment["id"]
    experiment_name = experiment["name"]
    signal_name = experiment["signal"]
    target_name = experiment["target"]

    window_name = (
        f"{window['train_end']}_"
        f"{window['validation_end']}_"
        f"{window['test_end']}"
    )

    signal = build_signal(
        df,
        signal_name,
        experiment.get("signal_params", {}),
    )

    if signal is None:
        return ExperimentResult(
            experiment_id=experiment_id,
            experiment_name=experiment_name,
            signal=signal_name,
            target=target_name,
            window=window_name,
            n=0,
            selected_n=0,
            non_selected_n=0,
            baseline_rate=None,
            selected_rate=None,
            lift=None,
            selected_mean_return=None,
            non_selected_mean_return=None,
            return_difference=None,
            selected_median_return=None,
            non_selected_median_return=None,
            return_ci_low=None,
            return_ci_high=None,
            probability_positive_difference=None,
            status="SIGNAL NOT AVAILABLE",
        )

    work = df.copy()
    work["_research_signal"] = signal

    target_name_column = f"_target_{target_name}"

    target_values = build_target(
        work,
        target_definition,
    )

    work[target_name_column] = target_values

    start_date = pd.Timestamp(window["train_end"]) + pd.Timedelta(days=1)
    end_date = pd.Timestamp(window["test_end"])

    work = work[
        (work["snapshot_date"] >= start_date)
        & (work["snapshot_date"] <= end_date)
    ].copy()

    work = work[
        work["_research_signal"].notna()
        & work[target_name_column].notna()
    ].copy()

    return_column = target_definition.get("return_column")

    if return_column and return_column in work.columns:
        work["_research_return"] = pd.to_numeric(
            work[return_column],
            errors="coerce",
        )
    else:
        work["_research_return"] = np.nan

    if work.empty:
        return ExperimentResult(
            experiment_id=experiment_id,
            experiment_name=experiment_name,
            signal=signal_name,
            target=target_name,
            window=window_name,
            n=0,
            selected_n=0,
            non_selected_n=0,
            baseline_rate=None,
            selected_rate=None,
            lift=None,
            selected_mean_return=None,
            non_selected_mean_return=None,
            return_difference=None,
            selected_median_return=None,
            non_selected_median_return=None,
            return_ci_low=None,
            return_ci_high=None,
            probability_positive_difference=None,
            status="NO DATA",
        )

    fraction = experiment.get("fraction", 0.05)

    selected = percentile_mask(
        work["_research_signal"],
        fraction,
    )

    selected_rows = work[selected]
    non_selected_rows = work[~selected]

    selected_n = len(selected_rows)
    non_selected_n = len(non_selected_rows)

    target_mean = safe_float(
        pd.to_numeric(
            work[target_name_column],
            errors="coerce",
        ).mean()
    )

    selected_rate = safe_float(
        pd.to_numeric(
            selected_rows[target_name_column],
            errors="coerce",
        ).mean()
    )

    baseline_rate = target_mean

    lift = None

    if (
        baseline_rate is not None
        and baseline_rate != 0
        and selected_rate is not None
    ):
        lift = selected_rate / baseline_rate

    selected_returns = selected_rows["_research_return"].dropna().to_numpy()
    non_selected_returns = (
        non_selected_rows["_research_return"].dropna().to_numpy()
    )

    selected_mean_return = safe_float(
        selected_returns.mean()
        if len(selected_returns)
        else None
    )

    non_selected_mean_return = safe_float(
        non_selected_returns.mean()
        if len(non_selected_returns)
        else None
    )

    return_difference = None

    if (
        selected_mean_return is not None
        and non_selected_mean_return is not None
    ):
        return_difference = (
            selected_mean_return
            - non_selected_mean_return
        )

    selected_median_return = safe_float(
        np.median(selected_returns)
        if len(selected_returns)
        else None
    )

    non_selected_median_return = safe_float(
        np.median(non_selected_returns)
        if len(non_selected_returns)
        else None
    )

    (
        return_ci_low,
        return_ci_high,
        probability_positive_difference,
    ) = bootstrap_difference(
        selected_returns,
        non_selected_returns,
    )

    status = classify_status(
        selected_n=selected_n,
        difference=return_difference,
        ci_low=return_ci_low,
        ci_high=return_ci_high,
        probability_positive=probability_positive_difference,
    )

    return ExperimentResult(
        experiment_id=experiment_id,
        experiment_name=experiment_name,
        signal=signal_name,
        target=target_name,
        window=window_name,
        n=len(work),
        selected_n=selected_n,
        non_selected_n=non_selected_n,
        baseline_rate=baseline_rate,
        selected_rate=selected_rate,
        lift=lift,
        selected_mean_return=selected_mean_return,
        non_selected_mean_return=non_selected_mean_return,
        return_difference=return_difference,
        selected_median_return=selected_median_return,
        non_selected_median_return=non_selected_median_return,
        return_ci_low=return_ci_low,
        return_ci_high=return_ci_high,
        probability_positive_difference=probability_positive_difference,
        status=status,
    )


def target_definition(
    target_name: str,
) -> dict[str, Any]:
    target = TARGETS[target_name]

    if hasattr(target, "model_dump"):
        return target.model_dump()

    if hasattr(target, "__dict__"):
        return dict(target.__dict__)

    if isinstance(target, dict):
        return target

    return asdict(target)


def run_research() -> tuple[list[ExperimentResult], str]:
    run_id = utc_run_id()

    print("=" * 80)
    print("Blankdiss Research Matrix")
    print("=" * 80)
    print(f"Run ID: {run_id}")
    print(f"Experiments: {len(EXPERIMENTS)}")

    print("\nLoading feature data...")
    df = load_features()

    print(f"Rows loaded: {len(df):,}")

    if "snapshot_date" in df.columns:
        print(
            "Date range: "
            f"{df['snapshot_date'].min()} → "
            f"{df['snapshot_date'].max()}"
        )

    results: list[ExperimentResult] = []

    for experiment_index, experiment in enumerate(
        EXPERIMENTS,
        start=1,
    ):
        experiment_id = experiment["id"]
        experiment_name = experiment["name"]
        target_name = experiment["target"]

        print()
        print("-" * 80)
        print(
            f"[{experiment_index}/{len(EXPERIMENTS)}] "
            f"{experiment_id}: {experiment_name}"
        )
        print(f"Signal: {experiment['signal']}")
        print(f"Target: {target_name}")

        target_def = target_definition(target_name)

        for window_index, window in enumerate(
            WALK_FORWARD_WINDOWS,
            start=1,
        ):
            print(
                f"  Window {window_index}: "
                f"train≤{window['train_end']} "
                f"validation≤{window['validation_end']} "
                f"test≤{window['test_end']}"
            )

            result = evaluate_experiment(
                df=df,
                experiment=experiment,
                target_definition=target_def,
                window=window,
            )

            results.append(result)

            print(
                f"    n={result.n:,} "
                f"selected={result.selected_n:,} "
                f"diff={result.return_difference} "
                f"p={result.probability_positive_difference} "
                f"status={result.status}"
            )

    return results, run_id


def write_results(
    results: list[ExperimentResult],
    run_id: str,
) -> Path:
    run_dir = OUTPUT_DIR / "runs" / run_id
    latest_dir = OUTPUT_DIR / "latest"

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    latest_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_dicts = [
        asdict(result)
        for result in results
    ]

    run_jsonl = run_dir / "experiment_results.jsonl"
    latest_jsonl = latest_dir / "experiment_results.jsonl"

    for path in (run_jsonl, latest_jsonl):
        with path.open(
            "w",
            encoding="utf-8",
        ) as handle:
            for result in result_dicts:
                handle.write(
                    json.dumps(
                        result,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    summary = build_summary(
        results,
        run_id,
    )

    run_summary = run_dir / "research_summary.json"
    latest_summary = latest_dir / "research_summary.json"

    for path in (run_summary, latest_summary):
        path.write_text(
            json.dumps(
                summary,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    report = build_markdown_report(
        results,
        run_id,
    )

    run_report = run_dir / "research_report.md"
    latest_report = latest_dir / "research_report.md"

    for path in (run_report, latest_report):
        path.write_text(
            report,
            encoding="utf-8",
        )

    return run_dir


def build_summary(
    results: list[ExperimentResult],
    run_id: str,
) -> dict[str, Any]:
    status_counts: dict[str, int] = {}

    for result in results:
        status_counts[result.status] = (
            status_counts.get(result.status, 0) + 1
        )

    return {
        "run_id": run_id,
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "experiment_count": len(
            {
                result.experiment_id
                for result in results
            }
        ),
        "result_count": len(results),
        "status_counts": status_counts,
        "results": [
            asdict(result)
            for result in results
        ],
    }


def build_markdown_report(
    results: list[ExperimentResult],
    run_id: str,
) -> str:
    lines: list[str] = []

    lines.append("# Blankdiss Research Matrix")
    lines.append("")
    lines.append(f"Run: `{run_id}`")
    lines.append("")
    lines.append(
        "Deterministic exploratory signal research. "
        "Results are descriptive and are not investment recommendations."
    )
    lines.append("")

    lines.append("## Overview")
    lines.append("")
    lines.append(
        f"- Results: {len(results)}"
    )

    status_counts: dict[str, int] = {}

    for result in results:
        status_counts[result.status] = (
            status_counts.get(result.status, 0) + 1
        )

    for status, count in sorted(
        status_counts.items()
    ):
        lines.append(
            f"- {status}: {count}"
        )

    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append(
        "| Experiment | Window | n | Selected | "
        "Rate | Lift | Mean diff | CI | P(diff > 0) | Status |"
    )
    lines.append(
        "|---|---|---:|---:|---:|---:|---:|---|---:|---|"
    )

    for result in results:
        rate = (
            f"{result.selected_rate:.4f}"
            if result.selected_rate is not None
            else ""
        )

        lift = (
            f"{result.lift:.3f}"
            if result.lift is not None
            else ""
        )

        difference = (
            f"{result.return_difference:.4f}"
            if result.return_difference is not None
            else ""
        )

        if (
            result.return_ci_low is not None
            and result.return_ci_high is not None
        ):
            ci = (
                f"[{result.return_ci_low:.4f}, "
                f"{result.return_ci_high:.4f}]"
            )
        else:
            ci = ""

        probability = (
            f"{result.probability_positive_difference:.3f}"
            if result.probability_positive_difference is not None
            else ""
        )

        lines.append(
            f"| {result.experiment_id} "
            f"{result.experiment_name} "
            f"| {result.window} "
            f"| {result.n:,} "
            f"| {result.selected_n:,} "
            f"| {rate} "
            f"| {lift} "
            f"| {difference} "
            f"| {ci} "
            f"| {probability} "
            f"| {result.status} |"
        )

    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "The matrix is intended to identify reproducible research "
        "signals across walk-forward periods. A result is not considered "
        "established merely because one experiment or tail produces a "
        "positive difference."
    )
    lines.append("")
    lines.append(
        "Particular attention should be paid to:"
    )
    lines.append("")
    lines.append(
        "- consistency across walk-forward windows"
    )
    lines.append(
        "- sample size in the selected tail"
    )
    lines.append(
        "- bootstrap confidence intervals"
    )
    lines.append(
        "- probability that the return difference is positive"
    )
    lines.append(
        "- multiple-testing effects across the experiment matrix"
    )
    lines.append(
        "- placebo and negative-control experiments when available"
    )

    return "\n".join(lines)


def main() -> None:
    results, run_id = run_research()

    output_dir = write_results(
        results,
        run_id,
    )

    print()
    print("=" * 80)
    print("Research completed")
    print("=" * 80)
    print(f"Run: {run_id}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
