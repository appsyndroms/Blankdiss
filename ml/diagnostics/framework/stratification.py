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
    values = pd.to_numeric(
        df[column],
        errors="coerce",
    )

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
    Applicerar thresholds som beräknats före testperioden.

    Varje observation placeras i det intervall där den hör hemma.
    """

    values = pd.to_numeric(
        df[column],
        errors="coerce",
    )

    ordered = sorted(
        thresholds.items()
    )

    labels = [
        f"q{int(q * 100):02d}"
        for q, _ in ordered
    ]

    result = pd.Series(
        pd.NA,
        index=df.index,
        dtype="object",
    )

    if not ordered:
        return result

    for index, (_, threshold) in enumerate(
        ordered
    ):
        if index == 0:
            mask = values < threshold

            result.loc[mask] = (
                f"<q{int(ordered[index][0] * 100):02d}"
            )

        next_threshold = (
            ordered[index + 1][1]
            if index + 1 < len(ordered)
            else None
        )

        if next_threshold is not None:
            mask = (
                (values >= threshold)
                & (values < next_threshold)
            )

            result.loc[mask] = (
                f"{labels[index]}-"
                f"{labels[index + 1]}"
            )
        else:
            result.loc[
                values >= threshold
            ] = f">={labels[index]}"

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

    Thresholds måste vara beräknade före testperioden.
    """

    work = df.copy()

    def classify(
        values: pd.Series,
        thresholds: dict[str, float],
    ) -> pd.Series:
        numeric = pd.to_numeric(
            values,
            errors="coerce",
        )

        ordered = sorted(
            thresholds.items(),
            key=lambda item: item[1],
        )

        result = pd.Series(
            pd.NA,
            index=values.index,
            dtype="object",
        )

        if not ordered:
            return result

        for index, (
            label,
            threshold,
        ) in enumerate(ordered):
            if index == 0:
                result.loc[
                    numeric < threshold
                ] = "below"

            result.loc[
                numeric >= threshold
            ] = label

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

    for (
        x_bucket,
        y_bucket,
    ), group in work.groupby(
        [
            "_x_bucket",
            "_y_bucket",
        ],
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

            row[
                f"{event_column}_rate"
            ] = summary["event_rate"]

        if return_column is not None:
            summary = return_summary(
                group[return_column]
            )

            for key, value in summary.items():
                row[
                    f"return_{key}"
                ] = value

        rows.append(row)

    return pd.DataFrame(rows)
