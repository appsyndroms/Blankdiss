"""Diagnostik för Blankdiss signal-backtest."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def calibration_summary(
    predictions: pd.DataFrame,
    bins: int = 10,
) -> dict[str, Any]:
    """Mät hur väl predikterad sannolikhet motsvarar faktisk event-rate."""
    data = predictions[
        ["probability", "actual"]
    ].copy()

    data["probability"] = pd.to_numeric(
        data["probability"],
        errors="coerce",
    )
    data["actual"] = pd.to_numeric(
        data["actual"],
        errors="coerce",
    )

    data = data.dropna()

    if data.empty:
        return {
            "rows": 0,
            "brier_score": None,
            "mean_probability": None,
            "mean_actual": None,
            "bins": [],
        }

    probabilities = data[
        "probability"
    ].to_numpy(dtype=float)

    actual = data[
        "actual"
    ].to_numpy(dtype=float)

    brier = float(
        np.mean(
            (probabilities - actual) ** 2
        )
    )

    ranked = data.sort_values(
        "probability",
        ascending=True,
        kind="mergesort",
    ).reset_index(drop=True)

    ranked["bin"] = pd.qcut(
        ranked.index,
        q=min(bins, len(ranked)),
        labels=False,
        duplicates="drop",
    )

    calibration_bins = []

    for bin_id, group in ranked.groupby(
        "bin",
        sort=True,
    ):
        calibration_bins.append(
            {
                "bin": int(bin_id) + 1,
                "rows": int(len(group)),
                "mean_probability": float(
                    group["probability"].mean()
                ),
                "event_rate": float(
                    group["actual"].mean()
                ),
            }
        )

    return {
        "rows": int(len(data)),
        "brier_score": brier,
        "mean_probability": float(
            data["probability"].mean()
        ),
        "mean_actual": float(
            data["actual"].mean()
        ),
        "bins": calibration_bins,
    }


def monthly_summary(
    predictions: pd.DataFrame,
    top_fraction: float = 0.01,
) -> list[dict[str, Any]]:
    """Mät signalens stabilitet månad för månad."""
    data = predictions.copy()

    data["snapshot_date"] = pd.to_datetime(
        data["snapshot_date"]
    )

    rows = []

    for period, month in data.groupby(
        data["snapshot_date"].dt.to_period("M"),
        sort=True,
    ):
        month = month.sort_values(
            "probability",
            ascending=False,
            kind="mergesort",
        )

        count = max(
            1,
            int(
                np.ceil(
                    len(month)
                    * top_fraction
                )
            ),
        )

        top = month.iloc[:count]

        baseline = float(
            month["actual"].mean()
        )

        event_rate = float(
            top["actual"].mean()
        )

        rows.append(
            {
                "month": str(period),
                "rows": int(len(month)),
                "baseline_event_rate": baseline,
                "top_fraction": float(
                    top_fraction
                ),
                "top_rows": int(len(top)),
                "top_events": int(
                    top["actual"].sum()
                ),
                "top_event_rate": event_rate,
                "top_lift": (
                    float(
                        event_rate / baseline
                    )
                    if baseline > 0
                    else None
                ),
                "top_mean_return": float(
                    top["target_return"].mean()
                ),
            }
        )

    return rows


def security_summary(
    predictions: pd.DataFrame,
    top_fraction: float = 0.01,
    top_n: int = 20,
) -> dict[str, Any]:
    """Kontrollera om signalen drivs av få värdepapper."""
    data = predictions.sort_values(
        "probability",
        ascending=False,
        kind="mergesort",
    ).copy()

    count = max(
        1,
        int(
            np.ceil(
                len(data)
                * top_fraction
            )
        ),
    )

    top = data.iloc[:count]

    grouped = (
        top.groupby(
            "security_key",
            dropna=False,
        )
        .agg(
            rows=("actual", "size"),
            event_rate=("actual", "mean"),
            mean_probability=(
                "probability",
                "mean",
            ),
            mean_return=(
                "target_return",
                "mean",
            ),
        )
        .reset_index()
        .sort_values(
            [
                "rows",
                "event_rate",
                "security_key",
            ],
            ascending=[
                False,
                False,
                True,
            ],
            kind="mergesort",
        )
        .head(top_n)
    )

    security_counts = top[
        "security_key"
    ].value_counts(
        dropna=False
    )

    concentration = (
        security_counts / len(top)
    )

    return {
        "top_fraction": float(
            top_fraction
        ),
        "top_rows": int(
            len(top)
        ),
        "unique_securities": int(
            top["security_key"].nunique(
                dropna=False
            )
        ),
        "largest_security_share": (
            float(
                concentration.iloc[0]
            )
            if not concentration.empty
            else None
        ),
        "top_5_security_share": (
            float(
                concentration.head(5).sum()
            )
            if not concentration.empty
            else None
        ),
        "top_10_security_share": (
            float(
                concentration.head(10).sum()
            )
            if not concentration.empty
            else None
        ),
        "top_securities": [
            {
                "security_key": str(
                    row["security_key"]
                ),
                "rows": int(
                    row["rows"]
                ),
                "event_rate": float(
                    row["event_rate"]
                ),
                "mean_probability": float(
                    row["mean_probability"]
                ),
                "mean_return": float(
                    row["mean_return"]
                ),
            }
            for _, row in grouped.iterrows()
        ],
    }


def build_diagnostics(
    predictions: pd.DataFrame,
) -> dict[str, Any]:
    """Bygg samlad diagnostik för OOS-prediktionerna."""
    return {
        "calibration": calibration_summary(
            predictions
        ),
        "monthly_top_1pct": monthly_summary(
            predictions,
            top_fraction=0.01,
        ),
        "security_top_1pct": security_summary(
            predictions,
            top_fraction=0.01,
        ),
    }
