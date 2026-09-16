"""Läser och förbereder Blankdiss-data för ML."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ml.config import (
    FEATURES_DIR,
    FEATURES_GLOB,
    FEATURE_EXCLUDE_COLUMNS,
    FI_ONLY_EXCLUDE_COLUMNS,
    PRICE_FEATURE_COLUMNS,
    TargetConfig,
)


def load_features() -> pd.DataFrame:
    """Läser alla feature-chunks som ett dataset."""
    paths = sorted(
        FEATURES_DIR.glob(FEATURES_GLOB)
    )

    if not paths:
        raise FileNotFoundError(
            "Saknar feature-data: "
            f"{FEATURES_DIR / FEATURES_GLOB}"
        )

    frames: list[pd.DataFrame] = []

    for path in paths:
        frame = pd.read_json(
            path,
            lines=True,
        )

        if not frame.empty:
            frames.append(frame)

    if not frames:
        raise ValueError(
            "Feature-dataset är tomt."
        )

    frame = pd.concat(
        frames,
        ignore_index=True,
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

    print(
        f"Läste {len(paths)} feature-chunks."
    )
    print(
        f"Feature-rader: {len(frame):,}"
    )

    return frame


def build_target(
    frame: pd.DataFrame,
    target: TargetConfig,
) -> pd.Series:
    """Bygger klassificerings- eller regressionstarget."""
    if target.task == "regression":
        if not target.target_column:
            raise ValueError(
                f"Regression-target saknar target_column: "
                f"{target.name}"
            )

        if target.target_column not in frame.columns:
            raise ValueError(
                "Saknar target-kolumn: "
                f"{target.target_column}"
            )

        return pd.to_numeric(
            frame[target.target_column],
            errors="coerce",
        ).astype(float)

    values = pd.to_numeric(
        frame[target.return_column],
        errors="coerce",
    )

    result = pd.Series(
        np.nan,
        index=frame.index,
        dtype=float,
    )

    valid = values.notna()

    if target.direction == "above":
        result.loc[valid] = (
            values.loc[valid]
            > target.threshold
        ).astype(float)

    elif target.direction == "below":
        result.loc[valid] = (
            values.loc[valid]
            <= target.threshold
        ).astype(float)

    else:
        raise ValueError(
            "Okänd target-riktning: "
            f"{target.direction}"
        )

    return result


def get_feature_columns(
    frame: pd.DataFrame,
    include_price_features: bool,
) -> list[str]:
    if include_price_features:
        excluded = FEATURE_EXCLUDE_COLUMNS
    else:
        excluded = FI_ONLY_EXCLUDE_COLUMNS

    columns: list[str] = []

    for column in frame.columns:
        if column in excluded:
            continue

        if column.startswith(
            "forward_return_"
        ):
            continue

        if column.startswith(
            "min_return_"
        ):
            continue

        if column.startswith(
            "max_return_"
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
            "Hittade inga numeriska ML-features."
        )

    return columns


def prepare_feature_set(
    frame: pd.DataFrame,
    include_price_features: bool,
) -> tuple[
    pd.DataFrame,
    list[str],
]:
    """
    Förbered en feature-set en gång.

    Resultatet kan återanvändas av flera targets. Target-specifika
    rader filtreras först när prepare_ml_data_from_feature_set()
    anropas.
    """
    feature_columns = get_feature_columns(
        frame,
        include_price_features,
    )

    required_columns = [
        "snapshot_date",
        "security_key",
        "forward_return_1d",
        "forward_return_5d",
        "forward_return_20d",
        "forward_return_60d",
        "min_return_5d",
        "max_return_5d",
    ]

    available_required_columns = [
        column
        for column in required_columns
        if column in frame.columns
    ]

    data = frame[
        available_required_columns
        + feature_columns
    ].copy()

    data = data.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    all_missing = [
        column
        for column in feature_columns
        if data[column].notna().sum() == 0
    ]

    if all_missing:
        print(
            "Tar bort helt tomma ML-features:"
        )

        for column in all_missing:
            print(
                f"  {column}"
            )

        feature_columns = [
            column
            for column in feature_columns
            if column not in all_missing
        ]

        data = data[
            available_required_columns
            + feature_columns
        ].copy()

    if not feature_columns:
        raise ValueError(
            "Alla ML-features saknar "
            "observerade värden."
        )

    return (
        data,
        feature_columns,
    )


def prepare_ml_data_from_feature_set(
    feature_set_data: pd.DataFrame,
    feature_columns: list[str],
    target: TargetConfig,
) -> tuple[
    pd.DataFrame,
    pd.Series,
    list[str],
]:
    """
    Bygg target-specifikt ML-dataset från en redan preparerad
    feature-set.

    Feature-kolumnerna och grundläggande numerisk QC återanvänds
    från feature-set-cachen.
    """
    target_values = build_target(
        feature_set_data,
        target,
    )

    if target.return_column not in feature_set_data.columns:
        raise ValueError(
            "Saknar return-kolumn: "
            f"{target.return_column}"
        )

    valid_target = target_values.notna()

    data = feature_set_data.loc[
        valid_target,
        [
            "snapshot_date",
            "security_key",
        ]
        + feature_columns
        + [
            target.return_column,
        ],
    ].copy()

    target_values = target_values.loc[
        valid_target
    ].copy()

    data["target_return"] = pd.to_numeric(
        data[target.return_column],
        errors="coerce",
    )

    data = data.drop(
        columns=[
            target.return_column
        ]
    )

    if target.task == "regression":
        y = target_values.astype(float)
    else:
        y = target_values.astype(int)

    return (
        data,
        y,
        feature_columns,
    )


def prepare_ml_data(
    frame: pd.DataFrame,
    target: TargetConfig,
    include_price_features: bool,
) -> tuple[
    pd.DataFrame,
    pd.Series,
    list[str],
]:
    """
    Bakåtkompatibel wrapper.

    Nya anrop bör använda prepare_feature_set() och
    prepare_ml_data_from_feature_set() när flera targets delar
    samma feature-set.
    """
    (
        feature_set_data,
        feature_columns,
    ) = prepare_feature_set(
        frame,
        include_price_features,
    )

    return prepare_ml_data_from_feature_set(
        feature_set_data,
        feature_columns,
        target,
    )


def dataset_summary(
    frame: pd.DataFrame,
    y: pd.Series,
    feature_columns: list[str],
    task: str = "classification",
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "rows": int(len(frame)),
        "features": int(len(feature_columns)),
        "date_start": (
            frame["snapshot_date"]
            .min()
            .strftime("%Y-%m-%d")
        ),
        "date_end": (
            frame["snapshot_date"]
            .max()
            .strftime("%Y-%m-%d")
        ),
        "task": task,
    }

    if task == "classification":
        result.update(
            {
                "positive": int(y.sum()),
                "negative": int(
                    (y == 0).sum()
                ),
                "positive_rate": float(
                    y.mean()
                ),
            }
        )
    else:
        result.update(
            {
                "target_mean": float(
                    y.mean()
                ),
                "target_median": float(
                    y.median()
                ),
                "target_min": float(
                    y.min()
                ),
                "target_max": float(
                    y.max()
                ),
            }
        )

    return result
