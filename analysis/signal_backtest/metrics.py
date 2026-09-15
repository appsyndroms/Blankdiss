
"""Mätetal för Blankdiss signal-backtest."""
from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
def percentile_value(
    values: pd.Series,
    percentile: float,
) -> float | None:
    """Beräkna en percentile på observerade värden."""
    clean = pd.to_numeric(
        values,
        errors="coerce",
    ).dropna()
    if clean.empty:
        return None
    return float(
        np.percentile(
            clean.to_numpy(),
            percentile,
        )
    )
def summarize_selection(
    selected: pd.DataFrame,
) -> dict[str, Any]:
    """Sammanfatta faktisk avkastning för ett urval."""
    returns = pd.to_numeric(
        selected["target_return"],
        errors="coerce",
    ).dropna()
    actual = pd.to_numeric(
        selected["actual"],
        errors="coerce",
    )
    if returns.empty:
        return {
            "rows": 0,
            "event_rate": None,
            "mean_return": None,
            "median_return": None,
            "p10_return": None,
            "p25_return": None,
            "p75_return": None,
            "p90_return": None,
            "min_return": None,
            "max_return": None,
        }
    return {
        "rows": int(len(returns)),
        "event_rate": float(actual.mean()),
        "mean_return": float(returns.mean()),
        "median_return": float(returns.median()),
        "p10_return": percentile_value(
            returns,
            10,
        ),
        "p25_return": percentile_value(
            returns,
            25,
        ),
        "p75_return": percentile_value(
            returns,
            75,
        ),
        "p90_return": percentile_value(
            returns,
            90,
        ),
        "min_return": float(returns.min()),
        "max_return": float(returns.max()),
    }
def build_probability_buckets(
    predictions: pd.DataFrame,
    top_fractions: tuple[float, ...],
) -> dict[str, Any]:
    """Beräkna signalprofil för olika top-N-urval."""
    predictions = predictions.sort_values(
        "probability",
        ascending=False,
        kind="mergesort",
    ).reset_index(drop=True)
    total_rows = len(predictions)
    if total_rows == 0:
        raise ValueError(
            "Kan inte bygga signalprofil från tom prediction-data."
        )
    baseline_event_rate = float(
        predictions["actual"].mean()
    )
    baseline_return = pd.to_numeric(
        predictions["target_return"],
        errors="coerce",
    ).mean()
    buckets = []
    for fraction in top_fractions:
        count = max(
            1,
            int(
                np.ceil(
                    total_rows * fraction
                )
            ),
        )
        selected = predictions.iloc[:count].copy()
        summary = summarize_selection(selected)
        event_rate = summary["event_rate"]
        if (
            event_rate is not None
            and baseline_event_rate > 0
        ):
            lift_ratio = (
                event_rate / baseline_event_rate
            )
        else:
            lift_ratio = None
        buckets.append(
            {
                "fraction": float(fraction),
                "percentage": float(fraction * 100),
                "rows": summary["rows"],
                "event_rate": event_rate,
                "lift_ratio": (
                    float(lift_ratio)
                    if lift_ratio is not None
                    else None
                ),
                "mean_return": summary["mean_return"],
                "median_return": summary["median_return"],
                "p10_return": summary["p10_return"],
                "p25_return": summary["p25_return"],
                "p75_return": summary["p75_return"],
                "p90_return": summary["p90_return"],
                "min_return": summary["min_return"],
                "max_return": summary["max_return"],
            }
        )
    return {
        "total_rows": int(total_rows),
        "baseline_event_rate": baseline_event_rate,
        "baseline_mean_return": (
            float(baseline_return)
            if pd.notna(baseline_return)
            else None
        ),
        "buckets": buckets,
    }
def summarize_year(
    predictions: pd.DataFrame,
    year: int,
    top_fractions: tuple[float, ...],
) -> dict[str, Any]:
    """Beräkna signalprofil för ett kalenderår."""
    yearly = predictions.loc[
        predictions["snapshot_date"].dt.year == year
    ].copy()
    if yearly.empty:
        return {
            "year": int(year),
            "rows": 0,
            "buckets": [],
        }
    summary = build_probability_buckets(
        yearly,
        top_fractions,
    )
    return {
        "year": int(year),
        "rows": int(len(yearly)),
        "models": sorted(
            yearly["model"]
            .dropna()
            .unique()
            .tolist()
        ),
        "test_auc_mean": float(
            yearly["test_auc"].mean()
        ),
        "validation_auc_mean": float(
            yearly["validation_auc"].mean()
        ),
        **summary,
    }
