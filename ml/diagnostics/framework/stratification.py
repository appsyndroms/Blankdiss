from __future__ import annotations

from typing import Iterable

import pandas as pd

from .metrics import event_summary, return_summary


def apply_threshold(
    df: pd.DataFrame,
    column: str,
    threshold: float,
    direction: str = "ge",
) -> pd.Series:
    values = df[column]

    if direction == "ge":
        return values >= threshold

    if direction == "gt":
        return values > threshold

    if direction == "le":
        return values <= threshold

    if direction == "lt":
        return values < threshold

    raise ValueError(
        f"Okänd threshold-riktning: {direction}"
    )


def quantile_buckets(
    df: pd.DataFrame,
    column: str,
    quantiles: Iterable[float],
    thresholds: dict[float, float],
) -> pd.Series:
    """
    Applicerar thresholds som beräknats på pre-test-data.

    Thresholds får alltså inte beräknas på df här.
    """

    result = pd.Series(
        "unknown",
        index=df.index,
        dtype="object",
    )

    ordered = sorted(thresholds.items())

    for q, threshold in ordered:
        result.loc[df[column] >= threshold] = (
            f"q>={q:.3f}"
        )

    return result


def two_dimensional_stratification(
    df: pd.DataFrame,
    x_column: str,
    x_thresholds: dict[str, float],
    y_column: str,
    y_thresholds: dict[str, float],
    event_columns: Iterable[str],
    return_column: str | None = None,
) -> pd.DataFrame:
    """
    Skapar en 2D-stratifiering.

    x_thresholds/y_thresholds ska vara beräknade före testperioden.
    """

    work = df.copy()

    def classify(
        values: pd.Series,
        thresholds: dict[str, float],
    ) -> pd.Series:
        result = pd.Series(
            "below",
            index=values.index,
            dtype="object",
        )

        for label, threshold in sorted(
            thresholds.items(),
            key=lambda item: item[1],
        ):
            result.loc[values >= threshold] = label

        return result

    work["_x_bucket"] = classify(
        work[x_column],
        x_thresholds,
    )

    work["_y_bucket"] = classify(
        work[y_column],
        y_thresholds,
    )

    rows = []

    for (x_bucket, y_bucket), group in work.groupby(
        ["_x_bucket", "_y_bucket"],
        dropna=False,
    ):
        row = {
            "x_bucket": x_bucket,
            "y_bucket": y_bucket,
            "n": len(group),
        }

        for event_column in event_columns:
            summary = event_summary(
                group,
                event_column,
            )

            row[f"{event_column}_rate"] = summary[
                "event_rate"
            ]

        if return_column is not None:
            summary = return_summary(
                group[return_column]
            )

            for key, value in summary.items():
                row[f"return_{key}"] = value

        rows.append(row)

    return pd.DataFrame(rows)
