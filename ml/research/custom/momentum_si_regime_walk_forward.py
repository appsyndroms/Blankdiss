# ml/research/custom/momentum_si_regime_walk_forward.py

"""
Controlled regime test for the hypothesis:

    Does short-interest change add downside-risk information
    specifically inside negative 5-day momentum regimes?

Design
------
For each walk-forward window, split and target:

    - validation
    - test

Signals:
    - price_return_5d       -> negative momentum regime
    - short_interest_delta_pp -> high SI-change regime

Momentum regimes:
    - bottom 20%
    - bottom 10%
    - bottom 5%
    - bottom 2.5%

SI regimes:
    - top 20%
    - top 10%
    - top 5%
    - top 2.5%

Targets:
    - down_5pct_5d
    - down_7pct_5d
    - down_10pct_5d

Primary comparison:
    event rate in momentum tail
        vs.
    event rate in momentum tail + high-SI-change tail

This is intentionally a descriptive regime test rather than
a tuned predictive model. No thresholds are selected from the
validation results.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ml.config import TARGET_CONFIGS, WALK_FORWARD_WINDOWS


OUTPUT_DIR = Path("data/processed/ml/research/momentum_si_regime_walk_forward")

MOMENTUM_SIGNAL = "price_return_5d"
SI_SIGNAL = "short_interest_delta_pp"

MOMENTUM_TAILS = [0.20, 0.10, 0.05, 0.025]
SI_TAILS = [0.20, 0.10, 0.05, 0.025]

TARGETS = [
    "down_5pct_5d",
    "down_7pct_5d",
    "down_10pct_5d",
]


def wilson_interval(
    events: int,
    n: int,
    z: float = 1.959963984540054,
) -> tuple[float | None, float | None]:
    """Wilson confidence interval for a binomial proportion."""
    if n == 0:
        return None, None

    p = events / n
    denominator = 1.0 + z**2 / n
    centre = (p + z**2 / (2.0 * n)) / denominator
    margin = (
        z
        * math.sqrt(
            (p * (1.0 - p) / n) + (z**2 / (4.0 * n**2))
        )
        / denominator
    )

    return max(0.0, centre - margin), min(1.0, centre + margin)


def safe_rate(events: int, n: int) -> float | None:
    if n == 0:
        return None
    return events / n


def safe_lift(
    combined_rate: float | None,
    baseline_rate: float | None,
) -> float | None:
    if combined_rate is None or baseline_rate in (None, 0):
        return None
    return combined_rate / baseline_rate


def quantile_tail(
    series: pd.Series,
    fraction: float,
    lower: bool,
) -> pd.Series:
    """
    Cross-sectional tail mask.

    The threshold is calculated independently for each snapshot_date.
    """
    if lower:
        return series <= series.quantile(fraction)
    return series >= series.quantile(1.0 - fraction)


def add_cross_sectional_masks(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add fixed, descriptive regime masks.

    Momentum = lower tail.
    SI change = upper tail.
    """
    result = df.copy()

    if "snapshot_date" not in result.columns:
        raise ValueError("Expected column: snapshot_date")

    if MOMENTUM_SIGNAL not in result.columns:
        raise ValueError(
            f"Expected signal column: {MOMENTUM_SIGNAL}"
        )

    if SI_SIGNAL not in result.columns:
        raise ValueError(
            f"Expected signal column: {SI_SIGNAL}"
        )

    grouped = result.groupby("snapshot_date", group_keys=False)

    for fraction in MOMENTUM_TAILS:
        suffix = str(fraction).replace(".", "_")

        result[f"momentum_lower_{suffix}"] = grouped[
            MOMENTUM_SIGNAL
        ].transform(
            lambda s, f=fraction: quantile_tail(
                s,
                f,
                lower=True,
            )
        )

    for fraction in SI_TAILS:
        suffix = str(fraction).replace(".", "_")

        result[f"si_upper_{suffix}"] = grouped[
            SI_SIGNAL
        ].transform(
            lambda s, f=fraction: quantile_tail(
                s,
                f,
                lower=False,
            )
        )

    return result


def evaluate_regime(
    df: pd.DataFrame,
    target: str,
    momentum_fraction: float,
    si_fraction: float,
) -> dict[str, Any]:
    """
    Compare:

        M = negative momentum tail

    against:

        M + SI = negative momentum tail AND high SI-change tail
    """
    momentum_suffix = str(momentum_fraction).replace(".", "_")
    si_suffix = str(si_fraction).replace(".", "_")

    momentum_col = f"momentum_lower_{momentum_suffix}"
    si_col = f"si_upper_{si_suffix}"

    required = [target, momentum_col, si_col]

    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns for {target}: {missing}"
        )

    usable = df[required].dropna()

    momentum = usable[usable[momentum_col].astype(bool)]
    combined = momentum[
        momentum[si_col].astype(bool)
    ]

    momentum_events = int(momentum[target].sum())
    momentum_n = int(len(momentum))

    combined_events = int(combined[target].sum())
    combined_n = int(len(combined))

    momentum_rate = safe_rate(
        momentum_events,
        momentum_n,
    )

    combined_rate = safe_rate(
        combined_events,
        combined_n,
    )

    momentum_ci_low, momentum_ci_high = wilson_interval(
        momentum_events,
        momentum_n,
    )

    combined_ci_low, combined_ci_high = wilson_interval(
        combined_events,
        combined_n,
    )

    absolute_difference = None

    if momentum_rate is not None and combined_rate is not None:
        absolute_difference = combined_rate - momentum_rate

    return {
        "target": target,
        "momentum_tail": momentum_fraction,
        "si_tail": si_fraction,

        "momentum_n": momentum_n,
        "momentum_events": momentum_events,
        "momentum_event_rate": momentum_rate,
        "momentum_ci_low": momentum_ci_low,
        "momentum_ci_high": momentum_ci_high,

        "combined_n": combined_n,
        "combined_events": combined_events,
        "combined_event_rate": combined_rate,
        "combined_ci_low": combined_ci_low,
        "combined_ci_high": combined_ci_high,

        "absolute_event_rate_difference": absolute_difference,

        "lift": safe_lift(
            combined_rate,
            momentum_rate,
        ),
    }


def load_research_data() -> pd.DataFrame:
    """
    Load the research dataset.

    The existing research runner uses the processed research
    dataset. We deliberately keep loading isolated here so the
    experiment remains independent from momentum_incremental_si.
    """
    candidates = [
        Path("data/processed/ml/research/dataset.parquet"),
        Path("data/processed/ml/research/features.parquet"),
        Path("data/processed/ml/research/research.parquet"),
    ]

    for path in candidates:
        if path.exists():
            return pd.read_parquet(path)

    raise FileNotFoundError(
        "Could not find research parquet dataset. "
        f"Tried: {[str(p) for p in candidates]}"
    )


def resolve_window_dates(
    window_name: str,
) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    """Return train, validation and test boundaries."""
    if window_name not in WALK_FORWARD_WINDOWS:
        raise KeyError(
            f"Unknown walk-forward window: {window_name}"
        )

    config = WALK_FORWARD_WINDOWS[window_name]

    return (
        pd.Timestamp(config["train_end"]),
        pd.Timestamp(config["validation_end"]),
        pd.Timestamp(config["test_end"]),
    )


def split_window(
    df: pd.DataFrame,
    window_name: str,
) -> dict[str, pd.DataFrame]:
    """
    Split according to the existing walk-forward definitions.

    The experiment only evaluates validation and test.
    """
    if "snapshot_date" not in df.columns:
        raise ValueError("Expected column: snapshot_date")

    train_end, validation_end, test_end = resolve_window_dates(
        window_name
    )

    dates = pd.to_datetime(df["snapshot_date"])

    validation = df[
        (dates > train_end)
        & (dates <= validation_end)
    ].copy()

    test = df[
        (dates > validation_end)
        & (dates <= test_end)
    ].copy()

    return {
        "validation": validation,
        "test": test,
    }


def target_column(
    df: pd.DataFrame,
    target_name: str,
) -> str:
    """
    Resolve the target column.

    Normally the target name is already present. If TARGET_CONFIGS
    provides a configured column name, use that mapping.
    """
    if target_name in df.columns:
        return target_name

    config = TARGET_CONFIGS.get(target_name)

    if isinstance(config, dict):
        for key in (
            "column",
            "name",
            "target_column",
        ):
            value = config.get(key)

            if value and value in df.columns:
                return value

    raise ValueError(
        f"Could not resolve target column for {target_name}"
    )


def evaluate_window_split(
    df: pd.DataFrame,
    window_name: str,
    split_name: str,
) -> list[dict[str, Any]]:
    """Evaluate all fixed target/tail combinations."""
    rows: list[dict[str, Any]] = []

    for target_name in TARGETS:
        column = target_column(df, target_name)

        working = df.copy()

        if column != target_name:
            working[target_name] = working[column]

        for momentum_fraction in MOMENTUM_TAILS:
            for si_fraction in SI_TAILS:
                result = evaluate_regime(
                    working,
                    target_name,
                    momentum_fraction,
                    si_fraction,
                )

                result["window"] = window_name
                result["split"] = split_name

                rows.append(result)

    return rows


def make_report(
    results: pd.DataFrame,
) -> str:
    lines = [
        "# Momentum + SI regime walk-forward test",
        "",
        "## Hypothesis",
        "",
        (
            "Förändring i short interest förväntas inte nödvändigtvis "
            "förbättra en global modell, men kan förstärka downside-risk "
            "inom negativa momentumregimer."
        ),
        "",
        "## Design",
        "",
        "- Momentum: nedre tvärsnittstail av 5-dagars prisavkastning.",
        "- SI: övre tvärsnittstail av förändring i short interest.",
        "- Momentum tails: 20%, 10%, 5%, 2,5%.",
        "- SI tails: 20%, 10%, 5%, 2,5%.",
        "- Targets: −5%, −7%, −10% inom 5 dagar.",
        "- Validation och test utvärderas separat.",
        "- Inga trösklar väljs efter resultaten.",
        "",
        "## Tolkning",
        "",
        (
            "Primär jämförelse är event rate i momentumtalet mot "
            "event rate när både momentum- och SI-villkoret är uppfyllda."
        ),
        "",
        "## Viktig begränsning",
        "",
        (
            "2025 förekommer som testperiod i window_1 och som "
            "validation-period i window_2. Dessa observationer är "
            "därför inte oberoende mellan de två fönstren."
        ),
        "",
        "## Resultat",
        "",
    ]

    if results.empty:
        lines.append("Inga resultat.")
        return "\n".join(lines)

    display_columns = [
        "window",
        "split",
        "target",
        "momentum_tail",
        "si_tail",
        "momentum_n",
        "momentum_events",
        "momentum_event_rate",
        "combined_n",
        "combined_events",
        "combined_event_rate",
        "absolute_event_rate_difference",
        "lift",
    ]

    available = [
        column
        for column in display_columns
        if column in results.columns
    ]

    lines.append(
        results[available].to_string(
            index=False,
        )
    )

    return "\n".join(lines)


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = load_research_data()

    df["snapshot_date"] = pd.to_datetime(
        df["snapshot_date"]
    )

    df = add_cross_sectional_masks(df)

    all_results: list[dict[str, Any]] = []

    for window_name in WALK_FORWARD_WINDOWS:
        splits = split_window(
            df,
            window_name,
        )

        for split_name, split_df in splits.items():
            results = evaluate_window_split(
                split_df,
                window_name,
                split_name,
            )

            all_results.extend(results)

    results_df = pd.DataFrame(all_results)

    results_df.to_csv(
        OUTPUT_DIR / "regime_comparison.csv",
        index=False,
    )

    with open(
        OUTPUT_DIR / "results.json",
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            all_results,
            handle,
            indent=2,
            ensure_ascii=False,
            default=str,
        )

    report = make_report(results_df)

    (OUTPUT_DIR / "report.md").write_text(
        report,
        encoding="utf-8",
    )

    print(report)

    print()
    print(
        f"Results written to: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
