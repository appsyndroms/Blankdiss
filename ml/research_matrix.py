"""
Blankdiss automated research matrix.

Runs a large number of deterministic research hypotheses against the
existing feature dataset.

Important:
- This module is exploratory research, not a trading decision engine.
- Signals are calculated from information available at snapshot_date.
- Forward-return columns are targets only and are never used as features.
- Data is loaded once and reused across the entire matrix.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ml.config import (
    RANDOM_STATE,
    TARGETS,
    WALK_FORWARD_WINDOWS,
)
from ml.dataset import load_features


ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "experiments"
)

LATEST_DIR = OUTPUT_DIR / "latest"
RUNS_DIR = OUTPUT_DIR / "runs"


MIN_SIGNAL_ROWS = 50
BOOTSTRAP_ITERATIONS = 1000
RANDOM_SEED = RANDOM_STATE


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SignalDefinition:
    name: str
    description: str
    direction: str
    percentile: float | None = None
    column: str | None = None


@dataclass(frozen=True)
class ExperimentDefinition:
    experiment_id: str
    family: str
    signal: str
    target: str
    description: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
    )


def clean_name(value: str) -> str:
    return (
        value.lower()
        .replace("%", "pct")
        .replace(".", "_")
        .replace("-", "_")
        .replace(" ", "_")
        .replace("/", "_")
    )


def stable_hash(value: str) -> str:
    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()[:12]


def quantile(
    series: pd.Series,
    q: float,
) -> float | None:
    values = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if values.empty:
        return None

    return float(
        values.quantile(q)
    )


def safe_mean(
    series: pd.Series,
) -> float | None:
    values = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if values.empty:
        return None

    return float(values.mean())


def safe_median(
    series: pd.Series,
) -> float | None:
    values = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if values.empty:
        return None

    return float(values.median())


# ---------------------------------------------------------------------------
# Feature discovery
# ---------------------------------------------------------------------------


def find_column(
    frame: pd.DataFrame,
    candidates: list[str],
) -> str | None:
    """
    Return the first available column from candidates.

    This deliberately allows the research engine to work even when
    optional feature columns have not yet been added to the pipeline.
    """
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate

    return None


def numeric_columns(
    frame: pd.DataFrame,
) -> list[str]:
    result = []

    for column in frame.columns:
        if pd.api.types.is_numeric_dtype(
            frame[column]
        ):
            result.append(column)

    return result


# ---------------------------------------------------------------------------
# Signal construction
# ---------------------------------------------------------------------------


def build_signals(
    frame: pd.DataFrame,
) -> dict[str, pd.Series]:
    """
    Build reusable research signals.

    Missing optional features simply disable the corresponding hypothesis
    family rather than causing the whole research run to fail.
    """

    signals: dict[str, pd.Series] = {}

    # ---------------------------------------------------------
    # Short interest
    # ---------------------------------------------------------

    si_column = find_column(
        frame,
        [
            "short_interest_pct",
            "short_interest",
            "short_position_pct",
            "aggregate_short_pct",
            "fi_short_pct",
        ],
    )

    si_change_column = find_column(
        frame,
        [
            "short_interest_change",
            "short_interest_pct_change",
            "short_position_change",
            "aggregate_short_change",
            "fi_short_change",
        ],
    )

    if si_column:
        si = pd.to_numeric(
            frame[si_column],
            errors="coerce",
        )

        signals["si_level"] = si
        signals["si_high"] = (
            si >= si.groupby(
                frame["snapshot_date"]
            ).transform(
                lambda x: x.quantile(0.80)
            )
        )
        signals["si_extreme"] = (
            si >= si.groupby(
                frame["snapshot_date"]
            ).transform(
                lambda x: x.quantile(0.95)
            )
        )

    if si_change_column:
        si_change = pd.to_numeric(
            frame[si_change_column],
            errors="coerce",
        )

        signals["si_change"] = si_change
        signals["si_increase"] = (
            si_change > 0
        )
        signals["si_strong_increase"] = (
            si_change
            >= si_change.groupby(
                frame["snapshot_date"]
            ).transform(
                lambda x: x.quantile(0.80)
            )
        )
        signals["si_strong_decrease"] = (
            si_change
            <= si_change.groupby(
                frame["snapshot_date"]
            ).transform(
                lambda x: x.quantile(0.20)
            )
        )

    # ---------------------------------------------------------
    # Price / volatility
    # ---------------------------------------------------------

    volatility = find_column(
        frame,
        [
            "price_volatility_20d",
            "volatility_20d",
        ],
    )

    momentum_5 = find_column(
        frame,
        [
            "price_return_5d",
            "return_5d",
        ],
    )

    momentum_20 = find_column(
        frame,
        [
            "price_return_20d",
            "return_20d",
        ],
    )

    momentum_60 = find_column(
        frame,
        [
            "price_return_60d",
            "return_60d",
        ],
    )

    distance_20 = find_column(
        frame,
        [
            "price_distance_from_20d_high",
        ],
    )

    distance_60 = find_column(
        frame,
        [
            "price_distance_from_60d_high",
        ],
    )

    if volatility:
        vol = pd.to_numeric(
            frame[volatility],
            errors="coerce",
        )

        signals["volatility"] = vol

        signals["high_volatility"] = (
            vol
            >= vol.groupby(
                frame["snapshot_date"]
            ).transform(
                lambda x: x.quantile(0.80)
            )
        )

        signals["low_volatility"] = (
            vol
            <= vol.groupby(
                frame["snapshot_date"]
            ).transform(
                lambda x: x.quantile(0.20)
            )
        )

    if momentum_5:
        momentum = pd.to_numeric(
            frame[momentum_5],
            errors="coerce",
        )

        signals["momentum_5d"] = momentum
        signals["strong_momentum_5d"] = (
            momentum
            >= momentum.groupby(
                frame["snapshot_date"]
            ).transform(
                lambda x: x.quantile(0.80)
            )
        )
        signals["weak_momentum_5d"] = (
            momentum
            <= momentum.groupby(
                frame["snapshot_date"]
            ).transform(
                lambda x: x.quantile(0.20)
            )
        )

    if momentum_20:
        momentum = pd.to_numeric(
            frame[momentum_20],
            errors="coerce",
        )

        signals["momentum_20d"] = momentum
        signals["positive_momentum_20d"] = (
            momentum > 0
        )
        signals["negative_momentum_20d"] = (
            momentum < 0
        )

    if momentum_60:
        momentum = pd.to_numeric(
            frame[momentum_60],
            errors="coerce",
        )

        signals["momentum_60d"] = momentum

    if distance_20:
        distance = pd.to_numeric(
            frame[distance_20],
            errors="coerce",
        )

        signals["near_20d_high"] = (
            distance >= -0.05
        )

    if distance_60:
        distance = pd.to_numeric(
            frame[distance_60],
            errors="coerce",
        )

        signals["near_60d_high"] = (
            distance >= -0.05
        )

    # ---------------------------------------------------------
    # Simple interaction signals
    # ---------------------------------------------------------

    if (
        "si_strong_increase" in signals
        and "high_volatility" in signals
    ):
        signals["si_increase_x_high_volatility"] = (
            signals["si_strong_increase"]
            & signals["high_volatility"]
        )

    if (
        "si_strong_increase" in signals
        and "strong_momentum_5d" in signals
    ):
        signals["si_increase_x_strong_momentum"] = (
            signals["si_strong_increase"]
            & signals["strong_momentum_5d"]
        )

    if (
        "si_strong_increase" in signals
        and "weak_momentum_5d" in signals
    ):
        signals["si_increase_x_weak_momentum"] = (
            signals["si_strong_increase"]
            & signals["weak_momentum_5d"]
        )

    return signals


# ---------------------------------------------------------------------------
# Target helpers
# ---------------------------------------------------------------------------


def target_series(
    frame: pd.DataFrame,
    target_name: str,
) -> tuple[pd.Series, str]:
    target = next(
        target
        for target in TARGETS
        if target.name == target_name
    )

    if target.task == "regression":
        column = target.target_column

        if not column:
            raise ValueError(
                f"Regression target lacks target_column: "
                f"{target.name}"
            )

        return (
            pd.to_numeric(
                frame[column],
                errors="coerce",
            ),
            "regression",
        )

    values = pd.to_numeric(
        frame[target.return_column],
        errors="coerce",
    )

    if target.direction == "above":
        result = (
            values > target.threshold
        ).astype(float)

    else:
        result = (
            values <= target.threshold
        ).astype(float)

    result.loc[
        values.isna()
    ] = np.nan

    return result, "classification"


def target_return_column(
    target_name: str,
) -> str:
    target = next(
        target
        for target in TARGETS
        if target.name == target_name
    )

    return target.return_column


# ---------------------------------------------------------------------------
# OOS splitting
# ---------------------------------------------------------------------------


def get_test_mask(
    frame: pd.DataFrame,
    window,
) -> pd.Series:
    dates = frame["snapshot_date"]

    return (
        (dates > pd.Timestamp(window.validation_end))
        & (
            dates
            <= pd.Timestamp(window.test_end)
        )
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def bootstrap_rate_difference(
    signal: pd.Series,
    target: pd.Series,
    iterations: int = BOOTSTRAP_ITERATIONS,
) -> dict:
    values = pd.DataFrame(
        {
            "signal": signal,
            "target": target,
        }
    ).dropna()

    if values.empty:
        return {
            "difference": None,
            "ci_low": None,
            "ci_high": None,
            "p_positive": None,
        }

    signal_mask = values["signal"].astype(bool)

    if signal_mask.sum() < MIN_SIGNAL_ROWS:
        return {
            "difference": None,
            "ci_low": None,
            "ci_high": None,
            "p_positive": None,
        }

    signal_rate = values.loc[
        signal_mask,
        "target",
    ].mean()

    baseline_rate = values[
        "target"
    ].mean()

    difference = (
        signal_rate
        - baseline_rate
    )

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    bootstrap = np.empty(
        iterations,
        dtype=float,
    )

    target_values = values[
        "target"
    ].to_numpy()

    signal_values = values[
        "signal"
    ].astype(bool).to_numpy()

    n = len(values)

    for index in range(iterations):
        sample = rng.integers(
            0,
            n,
            size=n,
        )

        sampled_target = (
            target_values[sample]
        )

        sampled_signal = (
            signal_values[sample]
        )

        if sampled_signal.sum() == 0:
            bootstrap[index] = np.nan
            continue

        bootstrap[index] = (
            sampled_target[
                sampled_signal
            ].mean()
            - sampled_target.mean()
        )

    bootstrap = bootstrap[
        np.isfinite(bootstrap)
    ]

    if len(bootstrap) < 100:
        return {
            "difference": float(difference),
            "ci_low": None,
            "ci_high": None,
            "p_positive": None,
        }

    return {
        "difference": float(difference),
        "ci_low": float(
            np.quantile(
                bootstrap,
                0.025,
            )
        ),
        "ci_high": float(
            np.quantile(
                bootstrap,
                0.975,
            )
        ),
        "p_positive": float(
            np.mean(
                bootstrap > 0
            )
        ),
    }


def classification_metrics(
    signal: pd.Series,
    target: pd.Series,
    forward_return: pd.Series,
) -> dict:
    values = pd.DataFrame(
        {
            "signal": signal,
            "target": target,
            "return": forward_return,
        }
    ).dropna()

    if values.empty:
        return {
            "n": 0,
        }

    signal_mask = values[
        "signal"
    ].astype(bool)

    n = len(values)
    signal_n = int(
        signal_mask.sum()
    )

    if signal_n < MIN_SIGNAL_ROWS:
        return {
            "n": int(n),
            "signal_n": signal_n,
        }

    baseline = float(
        values["target"].mean()
    )

    signal_rate = float(
        values.loc[
            signal_mask,
            "target",
        ].mean()
    )

    lift = (
        signal_rate / baseline
        if baseline > 0
        else None
    )

    auc = None

    if (
        values["signal"].nunique()
        > 1
        and values["target"].nunique()
        > 1
    ):
        try:
            auc = float(
                roc_auc_score(
                    values["target"],
                    values["signal"],
                )
            )
        except ValueError:
            auc = None

    bootstrap = (
        bootstrap_rate_difference(
            values["signal"],
            values["target"],
        )
    )

    signal_returns = values.loc[
        signal_mask,
        "return",
    ]

    return {
        "n": int(n),
        "signal_n": signal_n,
        "baseline_rate": baseline,
        "signal_rate": signal_rate,
        "lift": lift,
        "auc": auc,
        "mean_return": safe_mean(
            signal_returns
        ),
        "median_return": safe_median(
            signal_returns
        ),
        "bootstrap_difference": bootstrap[
            "difference"
        ],
        "bootstrap_ci_low": bootstrap[
            "ci_low"
        ],
        "bootstrap_ci_high": bootstrap[
            "ci_high"
        ],
        "bootstrap_p_positive": bootstrap[
            "p_positive"
        ],
    }


# ---------------------------------------------------------------------------
# Research classification
# ---------------------------------------------------------------------------


def classify_result(
    metrics: dict,
) -> str:
    """
    Descriptive research status.

    This is deliberately not an investment ranking.
    """

    n = metrics.get(
        "signal_n",
        0,
    )

    lift = metrics.get(
        "lift"
    )

    ci_low = metrics.get(
        "bootstrap_ci_low"
    )

    ci_high = metrics.get(
        "bootstrap_ci_high"
    )

    if n is None or n < MIN_SIGNAL_ROWS:
        return "INSUFFICIENT_DATA"

    if lift is None:
        return "NO_SIGNAL"

    if (
        ci_low is not None
        and ci_high is not None
        and ci_low > 0
        and lift >= 1.5
    ):
        return "STRONG_RESEARCH_CANDIDATE"

    if lift >= 1.25:
        return "INTERESTING"

    if (
        ci_low is not None
        and ci_high is not None
        and ci_low < 0 < ci_high
    ):
        return "UNSTABLE"

    return "NO_SIGNAL"


# ---------------------------------------------------------------------------
# Experiment generation
# ---------------------------------------------------------------------------


def relevant_targets(
    signal_name: str,
) -> list[str]:
    """
    Keep the matrix large but sensible.

    Some signals are directional by nature and are most useful against
    downside/upside event targets rather than every target.
    """

    all_targets = [
        target.name
        for target in TARGETS
        if target.task == "classification"
    ]

    if signal_name.startswith(
        "si_"
    ):
        return [
            target
            for target in all_targets
            if (
                "down_" in target
                or "up_" in target
            )
        ]

    return all_targets


def build_experiment_matrix(
    signals: dict[str, pd.Series],
) -> list[ExperimentDefinition]:
    experiments = []

    for signal_name in signals:
        for target_name in relevant_targets(
            signal_name
        ):
            experiment_id = (
                f"{clean_name(signal_name)}"
                f"__{clean_name(target_name)}"
            )

            experiments.append(
                ExperimentDefinition(
                    experiment_id=experiment_id,
                    family=signal_name.split(
                        "_"
                    )[0],
                    signal=signal_name,
                    target=target_name,
                    description=(
                        f"{signal_name} "
                        f"against {target_name}"
                    ),
                )
            )

    return experiments


# ---------------------------------------------------------------------------
# Experiment execution
# ---------------------------------------------------------------------------


def run_experiment(
    frame: pd.DataFrame,
    experiment: ExperimentDefinition,
    signals: dict[str, pd.Series],
) -> dict:
    signal = signals[
        experiment.signal
    ]

    target, task = target_series(
        frame,
        experiment.target,
    )

    return_column = target_return_column(
        experiment.target
    )

    forward_return = pd.to_numeric(
        frame[return_column],
        errors="coerce",
    )

    window_results = []

    for window_index, window in enumerate(
        WALK_FORWARD_WINDOWS,
        start=1,
    ):
        mask = get_test_mask(
            frame,
            window,
        )

        metrics = classification_metrics(
            signal.loc[mask],
            target.loc[mask],
            forward_return.loc[mask],
        )

        metrics.update(
            {
                "window": window_index,
                "test_start": (
                    pd.Timestamp(
                        window.validation_end
                    )
                    .strftime(
                        "%Y-%m-%d"
                    )
                ),
                "test_end": (
                    pd.Timestamp(
                        window.test_end
                    )
                    .strftime(
                        "%Y-%m-%d"
                    )
                ),
            }
        )

        window_results.append(
            metrics
        )

    pooled = classification_metrics(
        signal,
        target,
        forward_return,
    )

    valid_windows = [
        result
        for result in window_results
        if result.get("signal_n", 0)
        >= MIN_SIGNAL_ROWS
    ]

    if valid_windows:
        stable_count = sum(
            result.get("lift", 0) > 1
            for result in valid_windows
        )

        consistency = (
            stable_count
            / len(valid_windows)
        )
    else:
        consistency = None

    pooled["window_consistency"] = (
        consistency
    )

    pooled["status"] = classify_result(
        pooled
    )

    return {
        "experiment_id": experiment.experiment_id,
        "family": experiment.family,
        "signal": experiment.signal,
        "target": experiment.target,
        "description": experiment.description,
        "task": task,
        "pooled": pooled,
        "windows": window_results,
    }


# ---------------------------------------------------------------------------
# Placebo tests
# ---------------------------------------------------------------------------


def run_placebo(
    frame: pd.DataFrame,
    experiment: ExperimentDefinition,
    signals: dict[str, pd.Series],
) -> dict:
    signal = signals[
        experiment.signal
    ]

    target, _ = target_series(
        frame,
        experiment.target,
    )

    return_column = target_return_column(
        experiment.target
    )

    forward_return = pd.to_numeric(
        frame[return_column],
        errors="coerce",
    )

    valid = (
        signal.notna()
        & target.notna()
    )

    rng = np.random.default_rng(
        RANDOM_SEED
        + int(
            stable_hash(
                experiment.experiment_id
            ),
            16,
        ) % 100000
    )

    shuffled_signal = (
        signal.loc[valid]
        .reset_index(drop=True)
    )

    shuffled_signal = pd.Series(
        rng.permutation(
            shuffled_signal.to_numpy()
        )
    )

    shuffled_target = (
        target.loc[valid]
        .reset_index(drop=True)
    )

    shuffled_return = (
        forward_return.loc[valid]
        .reset_index(drop=True)
    )

    metrics = classification_metrics(
        shuffled_signal,
        shuffled_target,
        shuffled_return,
    )

    return {
        "n": metrics.get("n", 0),
        "signal_n": metrics.get(
            "signal_n",
            0,
        ),
        "lift": metrics.get(
            "lift"
        ),
        "signal_rate": metrics.get(
            "signal_rate"
        ),
    }


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


def write_jsonl(
    path: Path,
    rows: list[dict],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )


def write_summary(
    path: Path,
    results: list[dict],
) -> None:
    status_counts: dict[str, int] = {}

    for result in results:
        status = result[
            "pooled"
        ].get(
            "status",
            "UNKNOWN",
        )

        status_counts[status] = (
            status_counts.get(
                status,
                0,
            )
            + 1
        )

    interesting = [
        result
        for result in results
        if result[
            "pooled"
        ].get("status")
        in {
            "STRONG_RESEARCH_CANDIDATE",
            "INTERESTING",
        }
    ]

    interesting.sort(
        key=lambda result: (
            result[
                "pooled"
            ].get("lift")
            or 0
        ),
        reverse=True,
    )

    summary = {
        "created_at": utc_now(),
        "experiment_count": len(
            results
        ),
        "status_counts": status_counts,
        "interesting_count": len(
            interesting
        ),
        "interesting_experiments": [
            {
                "experiment_id": result[
                    "experiment_id"
                ],
                "signal": result[
                    "signal"
                ],
                "target": result[
                    "target"
                ],
                "status": result[
                    "pooled"
                ].get("status"),
                "lift": result[
                    "pooled"
                ].get("lift"),
                "signal_rate": result[
                    "pooled"
                ].get("signal_rate"),
                "baseline_rate": result[
                    "pooled"
                ].get("baseline_rate"),
                "mean_return": result[
                    "pooled"
                ].get("mean_return"),
                "median_return": result[
                    "pooled"
                ].get("median_return"),
                "window_consistency": result[
                    "pooled"
                ].get(
                    "window_consistency"
                ),
            }
            for result in interesting
        ],
    }

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def write_markdown_report(
    path: Path,
    results: list[dict],
) -> None:
    interesting = [
        result
        for result in results
        if result[
            "pooled"
        ].get("status")
        in {
            "STRONG_RESEARCH_CANDIDATE",
            "INTERESTING",
        }
    ]

    interesting.sort(
        key=lambda result: (
            result[
                "pooled"
            ].get("lift")
            or 0
        ),
        reverse=True,
    )

    lines = [
        "# Blankdiss Research Matrix",
        "",
        f"Generated: {utc_now()}",
        "",
        f"Experiments: {len(results)}",
        "",
        "## Research candidates",
        "",
        "| Experiment | Target | Lift | Signal | Baseline | Mean return | Consistency | Status |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]

    for result in interesting:
        pooled = result[
            "pooled"
        ]

        lines.append(
            "| "
            f"{result['signal']} | "
            f"{result['target']} | "
            f"{pooled.get('lift', 0):.2f} | "
            f"{pooled.get('signal_rate', 0):.3f} | "
            f"{pooled.get('baseline_rate', 0):.3f} | "
            f"{pooled.get('mean_return', 0):.3f} | "
            f"{pooled.get('window_consistency', 0):.2f} | "
            f"{pooled.get('status')} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "These are exploratory research results.",
            "They are not trading rankings or investment recommendations.",
            "",
            "Results that appear interesting must be validated on a",
            "predefined holdout period before being treated as robust.",
            "",
        ]
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    run_id = datetime.now(
        timezone.utc
    ).strftime(
        "%Y%m%dT%H%M%SZ"
    )

    run_dir = RUNS_DIR / run_id

    print(
        "Loading feature dataset...",
        flush=True,
    )

    frame = load_features()

    print(
        f"Rows: {len(frame):,}",
        flush=True,
    )

    print(
        "Building reusable research signals...",
        flush=True,
    )

    signals = build_signals(
        frame
    )

    print(
        f"Signals available: {len(signals)}",
        flush=True,
    )

    experiments = build_experiment_matrix(
        signals
    )

    print(
        f"Experiments generated: "
        f"{len(experiments)}",
        flush=True,
    )

    results = []

    for index, experiment in enumerate(
        experiments,
        start=1,
    ):
        if index % 25 == 0:
            print(
                f"Running experiment "
                f"{index}/{len(experiments)}...",
                flush=True,
            )

        result = run_experiment(
            frame,
            experiment,
            signals,
        )

        result["placebo"] = run_placebo(
            frame,
            experiment,
            signals,
        )

        results.append(result)

    write_jsonl(
        run_dir
        / "experiment_results.jsonl",
        results,
    )

    write_summary(
        run_dir
        / "research_summary.json",
        results,
    )

    write_markdown_report(
        run_dir
        / "research_report.md",
        results,
    )

    latest_results = (
        LATEST_DIR
        / "experiment_results.jsonl"
    )

    latest_summary = (
        LATEST_DIR
        / "research_summary.json"
    )

    latest_report = (
        LATEST_DIR
        / "research_report.md"
    )

    write_jsonl(
        latest_results,
        results,
    )

    write_summary(
        latest_summary,
        results,
    )

    write_markdown_report(
        latest_report,
        results,
    )

    status_counts = {}

    for result in results:
        status = result[
            "pooled"
        ].get(
            "status",
            "UNKNOWN",
        )

        status_counts[status] = (
            status_counts.get(
                status,
                0,
            )
            + 1
        )

    print()
    print(
        "========================================"
    )
    print(
        "BLANKDISS RESEARCH MATRIX COMPLETE"
    )
    print(
        "========================================"
    )
    print(
        f"Experiments: {len(results)}"
    )

    for status, count in sorted(
        status_counts.items()
    ):
        print(
            f"{status}: {count}"
        )

    print()
    print(
        f"Results: {run_dir}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
