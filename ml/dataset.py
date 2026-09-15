"""Läser och förbereder Blankdiss-data för ML."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ml.config import (
    FEATURES_PATH,
    FEATURE_EXCLUDE_COLUMNS,
    TargetConfig,
)


def load_features() -> pd.DataFrame:
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Saknar feature-data: {FEATURES_PATH}"
        )

    frame = pd.read_json(
        FEATURES_PATH,
        lines=True,
    )

    if frame.empty:
        raise ValueError(
            "Feature-dataset är tomt."
        )

    frame["snapshot_date"] = pd.to_datetime(
        frame["snapshot_date"],
        errors="coerce",
    )

    frame = frame.loc[
        frame["snapshot_date"].notna()
    ].copy()

    frame = frame.sort_values(
        [
            "snapshot_date",
            "security_key",
        ],
        kind="mergesort",
    ).reset_index(
        drop=True
    )

    return frame


def build_target(
    frame: pd.DataFrame,
    target: TargetConfig,
) -> pd.Series:
    if target.return_column not in frame.columns:
        raise ValueError(
            "Saknar target-kolumn: "
            f"{target.return_column}"
        )

    returns = pd.to_numeric(
        frame[target.return_column],
        errors="coerce",
    )

    result = pd.Series(
        np.nan,
        index=frame.index,
        dtype=float,
    )

    valid = returns.notna()

    result.loc[valid] = (
        returns.loc[valid]
        > target.threshold
    ).astype(float)

    return result


def get_feature_columns(
    frame: pd.DataFrame,
) -> list[str]:
    columns: list[str] = []

    for column in frame.columns:
        if column in FEATURE_EXCLUDE_COLUMNS:
            continue

        if column.startswith(
            "forward_return_"
        ):
            continue

        if pd.api.types.is_bool_dtype(
            frame[column]
        ):
            columns.append(column)
            continue

        if pd.api.types.is_numeric_dtype(
            frame[column]
        ):
            columns.append(column)

    if not columns:
        raise ValueError(
            "Hittade inga numeriska "
            "ML-features."
        )

    return columns


def prepare_ml_data(
    frame: pd.DataFrame,
    target: TargetConfig,
) -> tuple[
    pd.DataFrame,
    pd.Series,
    list[str],
]:
    target_values = build_target(
        frame,
        target,
    )

    feature_columns = get_feature_columns(
        frame
    )

    data = frame[
        [
            "snapshot_date",
            "security_key",
            target.return_column,
        ]
        + feature_columns
    ].copy()

    valid_target = target_values.notna()

    data = data.loc[
        valid_target
    ].copy()

    target_values = target_values.loc[
        valid_target
    ].copy()

    X = data[
        feature_columns
    ].copy()

    y = target_values.astype(
        int
    )

    X = X.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    data.loc[
        :,
        feature_columns,
    ] = X

    data["target_return"] = pd.to_numeric(
        data[target.return_column],
        errors="coerce",
    )

    data = data.drop(
        columns=[target.return_column]
    )

    return (
        data,
        y,
        feature_columns,
    )


def dataset_summary(
    frame: pd.DataFrame,
    y: pd.Series,
    feature_columns: list[str],
) -> dict[str, Any]:
    return {
        "rows": int(
            len(frame)
        ),
        "features": int(
            len(feature_columns)
        ),
        "positive": int(
            y.sum()
        ),
        "negative": int(
            (y == 0).sum()
        ),
        "positive_rate": float(
            y.mean()
        ),
        "date_start": (
            frame["snapshot_date"]
            .min()
            .strftime(
                "%Y-%m-%d"
            )
        ),
        "date_end": (
            frame["snapshot_date"]
            .max()
            .strftime(
                "%Y-%m-%d"
            )
        ),
    }
