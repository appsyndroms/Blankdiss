"""Beräkning av framtida prisavkastning."""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.feature_config import (
    RETURN_HORIZONS,
)


def add_forward_returns(
    frame: pd.DataFrame,
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """
    Beräknar forward returns från den matchade
    signal-prisdagen.

    forward_return_Nd =
        pris vid N handelsobservationer framåt
        / pris på signal-dagen - 1

    Ingen framtida information används som feature.
    Forward returns är endast targets.
    """

    if frame.empty:
        return frame

    lookup = {
        key: group.sort_values(
            "date",
            kind="mergesort",
        ).reset_index(
            drop=True
        )
        for key, group in prices.groupby(
            "security_key",
            sort=False,
        )
    }

    frame = frame.copy()

    for horizon in RETURN_HORIZONS:
        frame[
            f"forward_return_{horizon}d"
        ] = np.nan

    for index, row in frame.iterrows():
        series = lookup.get(
            row["security_key"]
        )

        if series is None or series.empty:
            continue

        price_date = pd.to_datetime(
            row["price_date"],
            errors="coerce",
        )

        if pd.isna(price_date):
            continue

        dates = series[
            "date"
        ].to_numpy(
            dtype="datetime64[ns]"
        )

        price_date_np = np.datetime64(
            price_date.to_datetime64(),
            "ns",
        )

        entry_idx = int(
            np.searchsorted(
                dates,
                price_date_np,
                side="left",
            )
        )

        if entry_idx >= len(series):
            continue

        entry_price = float(
            series.iloc[
                entry_idx
            ]["close"]
        )

        if (
            not np.isfinite(entry_price)
            or entry_price <= 0
        ):
            continue

        for horizon in RETURN_HORIZONS:
            target_idx = (
                entry_idx + horizon
            )

            if target_idx >= len(series):
                continue

            target_price = float(
                series.iloc[
                    target_idx
                ]["close"]
            )

            if (
                not np.isfinite(target_price)
                or target_price <= 0
            ):
                continue

            frame.at[
                index,
                f"forward_return_{horizon}d",
            ] = (
                target_price
                / entry_price
                - 1.0
            )

    return frame


def validate_forward_returns(
    frame: pd.DataFrame,
) -> dict[str, dict[str, float | int]]:
    """
    Enkel sanity-check av target-fördelning.

    Funktionen används av QC senare och ändrar inte data.
    """

    result: dict[
        str,
        dict[str, float | int],
    ] = {}

    for horizon in RETURN_HORIZONS:
        column = (
            f"forward_return_{horizon}d"
        )

        if column not in frame.columns:
            continue

        values = pd.to_numeric(
            frame[column],
            errors="coerce",
        ).dropna()

        if values.empty:
            result[column] = {
                "rows": 0,
                "positive": 0,
                "negative_or_zero": 0,
                "positive_rate": float("nan"),
                "min": float("nan"),
                "median": float("nan"),
                "max": float("nan"),
            }
            continue

        result[column] = {
            "rows": int(len(values)),
            "positive": int(
                (values > 0).sum()
            ),
            "negative_or_zero": int(
                (values <= 0).sum()
            ),
            "positive_rate": float(
                (values > 0).mean()
            ),
            "min": float(values.min()),
            "median": float(values.median()),
            "max": float(values.max()),
        }

    return result
