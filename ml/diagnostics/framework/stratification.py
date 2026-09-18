from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from .metrics import event_summary, return_summary


@dataclass(frozen=True)
class PretestBins:
    """
    Thresholds beräknade före testperioden.

    column:
        Kolumnen som stratifieras.

    thresholds:
        Namngivna thresholds.

    Exempel:

        PretestBins(
            column="price_volatility_20d",
            thresholds={"HIGH": 0.08},
        )
    """

    column: str
    thresholds: dict[str, float]


def make_pretest_bins(
    pretest: pd.DataFrame,
    column: str,
    quantiles: Iterable[float] = (0.80,),
    labels: Iterable[str] | None = None,
) -> PretestBins:
    """
    Skapa bins från PRE-TEST-data.

    Standard:

        >= 80:e percentilen -> HIGH
        < 80:e percentilen  -> LOW
    """

    if column not in pretest.columns:
        raise KeyError(
            f"Saknar kolumn: {column}"
        )

    values = pd.to_numeric(
        pretest[column],
        errors="coerce",
    ).dropna()

    if values.empty:
        raise ValueError(
            f"Kan inte skapa bins för '{column}'."
        )

    quantiles = tuple(quantiles)

    if labels is None:
        if len(quantiles) == 1:
            labels = ("HIGH",)
        else:
            labels = tuple(
                f"Q{i + 1}"
                for i in range(len(quantiles))
            )

    labels = tuple(labels)

    if len(labels) != len(quantiles):
        raise ValueError(
            "Antalet labels måste motsvara "
            "antalet quantiles."
        )

    thresholds = {}

    for q, label in zip(
        quantiles,
        labels,
    ):
        if not 0 < q < 1:
            raise ValueError(
                f"Ogiltig quantile: {q}"
            )

        thresholds[label] = float(
            values.quantile(q)
        )

    return PretestBins(
        column=column,
        thresholds=thresholds,
    )


def classify_bins(
    df: pd.DataFrame,
    bins: PretestBins,
) -> pd.Series:
    """
    Applicerar pre-test-thresholds på data.
    """

    if bins.column not in df.columns:
        raise KeyError(
            f"Saknar kolumn: {bins.column}"
        )

    values = pd.to_numeric(
        df[bins.column],
        errors="coerce",
    )

    result = pd.Series(
        "LOW",
        index=df.index,
        dtype="object",
    )

    for label, threshold in sorted(
        bins.thresholds.items(),
        key=lambda item: item[1],
    ):
        result.loc[
            values >= threshold
        ] = label

    result.loc[
        values.isna()
    ] = "UNKNOWN"

    return result


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
    """

    result = pd.Series(
        "unknown",
        index=df.index,
        dtype="object",
    )

    ordered = sorted(
        thresholds.items()
    )

    for q, threshold in ordered:
        result.loc[
            df[column] >= threshold
        ] = f"q>={q:.3f}"

    return result


def two_dimensional_stratification(
    df: pd.DataFrame,
    x_bins: PretestBins,
    y_bins: PretestBins,
    event_columns: Iterable[str],
    return_column: str | None = None,
) -> pd.DataFrame:
    """
    Skapar en 2D-stratifiering med thresholds som
    beräknats före testperioden.
    """

    work = df.copy()

    work["_x_bucket"] = classify_bins(
        work,
        x_bins,
    )

    work["_y_bucket"] = classify_bins(
        work,
        y_bins,
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
            "x_column": x_bins.column,
            "y_column": y_bins.column,
            "x_bucket": x_bucket,
            "y_bucket": y_bucket,
            "n": len(group),
        }

        for event_column in event_columns:
            if event_column not in group.columns:
                continue

            summary = event_summary(
                group,
                event_column,
            )

            row[
                f"{event_column}_rate"
            ] = summary["event_rate"]

        if return_column is not None:
            if return_column in group.columns:
                summary = return_summary(
                    group[return_column]
                )

                for key, value in summary.items():
                    row[
                        f"return_{key}"
                    ] = value

        rows.append(row)

    return pd.DataFrame(rows)
