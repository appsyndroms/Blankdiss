"""Inläsning och konstruktion av FI-features."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from analysis.feature_config import (
    FI_REQUIRED_COLUMNS,
    THRESHOLDS,
)
from analysis.feature_utils import (
    security_key,
)


def load_fi(
    path: Path,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Saknar FI-data: {path}"
        )

    frame = pd.read_json(
        path,
        lines=True,
    )

    missing = FI_REQUIRED_COLUMNS.difference(
        frame.columns
    )

    if missing:
        raise ValueError(
            "FI-data saknar kolumner: "
            + ", ".join(
                sorted(missing)
            )
        )

    frame["snapshot_date"] = pd.to_datetime(
        frame["snapshot_date"],
        errors="coerce",
    )

    frame = frame.loc[
        frame["snapshot_date"].notna()
    ].copy()

    frame["issuer"] = (
        frame["issuer"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    frame["isin"] = frame["isin"].where(
        frame["isin"].notna(),
        None,
    )

    frame["security_key"] = [
        security_key(
            isin,
            issuer,
        )
        for isin, issuer in zip(
            frame["isin"],
            frame["issuer"],
        )
    ]

    for column in (
        "short_interest_pct",
        "active_holders",
        "max_individual_position_pct",
        "max_position_share_pct",
    ):
        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

    frame = frame.sort_values(
        [
            "security_key",
            "snapshot_date",
        ],
        kind="mergesort",
    )

    duplicates = frame.duplicated(
        [
            "security_key",
            "snapshot_date",
        ],
        keep="last",
    )

    return frame.loc[
        ~duplicates
    ].copy()


def add_fi_features(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    frame = frame.sort_values(
        [
            "security_key",
            "snapshot_date",
        ],
        kind="mergesort",
    ).copy()

    grouped = frame.groupby(
        "security_key",
        sort=False,
    )

    frame["previous_snapshot_date"] = (
        grouped["snapshot_date"].shift(1)
    )

    frame["previous_short_interest_pct"] = (
        grouped["short_interest_pct"].shift(1)
    )

    frame["previous_active_holders"] = (
        grouped["active_holders"].shift(1)
    )

    frame[
        "previous_max_individual_position_pct"
    ] = grouped[
        "max_individual_position_pct"
    ].shift(1)

    frame[
        "previous_max_position_share_pct"
    ] = grouped[
        "max_position_share_pct"
    ].shift(1)

    frame["fi_observation_gap_days"] = (
        frame["snapshot_date"]
        - frame["previous_snapshot_date"]
    ).dt.days

    frame["short_interest_delta_pp"] = (
        frame["short_interest_pct"]
        - frame["previous_short_interest_pct"]
    )

    frame["holder_delta"] = (
        frame["active_holders"]
        - frame["previous_active_holders"]
    )

    frame["max_position_delta_pp"] = (
        frame["max_individual_position_pct"]
        - frame[
            "previous_max_individual_position_pct"
        ]
    )

    frame["concentration_delta_pp"] = (
        frame["max_position_share_pct"]
        - frame[
            "previous_max_position_share_pct"
        ]
    )

    previous = frame[
        "previous_short_interest_pct"
    ]

    frame["short_interest_relative_change"] = (
        np.where(
            previous >= 0.5,
            frame["short_interest_delta_pp"]
            / previous,
            np.nan,
        )
    )

    frame["short_interest_acceleration_pp"] = (
        grouped[
            "short_interest_delta_pp"
        ].diff()
    )

    for threshold in THRESHOLDS:
        label = (
            f"{threshold:.1f}".replace(
                ".",
                "_",
            )
        )

        current = (
            frame["short_interest_pct"]
            >= threshold
        )

        frame[
            f"above_{label}pct"
        ] = current

        frame[
            f"entered_above_{label}pct"
        ] = (
            previous.notna()
            & (previous < threshold)
            & current
        )

        frame[
            f"exited_below_{label}pct"
        ] = (
            previous.notna()
            & (previous >= threshold)
            & (~current)
        )

    frame["new_visible_observation"] = (
        frame["previous_snapshot_date"].isna()
    )

    return frame
