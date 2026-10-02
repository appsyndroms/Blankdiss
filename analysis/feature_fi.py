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
    normalize_text,
    security_key,
)


def _aggregate_snapshot_files(
    snapshot_dir: Path,
) -> list[Path]:
    if not snapshot_dir.exists():
        return []

    files = sorted(
        snapshot_dir.glob("*.jsonl")
    )

    if not files:
        return []

    latest_by_date: dict[str, Path] = {}

    for path in files:
        try:
            sample = pd.read_json(
                path,
                lines=True,
            )
        except (
            OSError,
            ValueError,
            TypeError,
        ):
            continue

        if "snapshot_date" not in sample.columns:
            continue

        dates = pd.to_datetime(
            sample["snapshot_date"],
            errors="coerce",
        ).dropna()

        if dates.empty:
            continue

        snapshot_date = (
            dates.max()
            .strftime("%Y-%m-%d")
        )

        current = latest_by_date.get(
            snapshot_date
        )

        if current is None:
            latest_by_date[
                snapshot_date
            ] = path
            continue

        if path.stat().st_mtime_ns > current.stat().st_mtime_ns:
            latest_by_date[
                snapshot_date
            ] = path

    return [
        latest_by_date[key]
        for key in sorted(latest_by_date)
    ]


def _load_aggregate_snapshots(
    historical: pd.DataFrame,
) -> pd.DataFrame:
    snapshot_dir = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "raw"
        / "fi"
        / "aggregate"
        / "snapshots"
    )

    files = _aggregate_snapshot_files(
        snapshot_dir
    )

    if not files:
        return pd.DataFrame(
            columns=historical.columns
        )

    frames: list[pd.DataFrame] = []

    for path in files:
        frame = pd.read_json(
            path,
            lines=True,
        )

        required = {
            "snapshot_date",
            "position_date",
            "issuer",
            "short_interest_pct",
        }

        missing = required.difference(
            frame.columns
        )

        if missing:
            raise ValueError(
                "FI aggregate-data saknar "
                "kolumner: "
                + ", ".join(
                    sorted(missing)
                )
                + f" ({path})"
            )

        # snapshot_date beskriver när FI-snapshoten
        # hämtades/publicerades.
        #
        # position_date är däremot den faktiska
        # observationsdagen för blankningen och ska
        # användas som feature-datum.
        frame["snapshot_date"] = (
            pd.to_datetime(
                frame["position_date"],
                errors="coerce",
            )
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

        frame["short_interest_pct"] = (
            pd.to_numeric(
                frame["short_interest_pct"],
                errors="coerce",
            )
        )

        frame = frame.loc[
            frame["short_interest_pct"].notna()
        ].copy()

        if frame.empty:
            continue

        frames.append(
            frame[
                [
                    "snapshot_date",
                    "issuer",
                    "short_interest_pct",
                ]
            ]
        )

    if not frames:
        return pd.DataFrame(
            columns=historical.columns
        )

    aggregate = pd.concat(
        frames,
        ignore_index=True,
    )

    historical_issuers: dict[
        str,
        set[str],
    ] = {}

    for issuer, security in zip(
        historical["issuer"],
        historical["security_key"],
    ):
        key = normalize_text(
            issuer
        )

        if not key:
            continue

        historical_issuers.setdefault(
            key,
            set(),
        ).add(security)

    def resolve_security(
        issuer: str,
    ) -> str:
        key = normalize_text(
            issuer
        )

        matches = historical_issuers.get(
            key,
            set(),
        )

        if len(matches) == 1:
            return next(iter(matches))

        return security_key(
            None,
            issuer,
        )

    aggregate["security_key"] = [
        resolve_security(issuer)
        for issuer in aggregate["issuer"]
    ]

    aggregate["isin"] = None

    aggregate[
        "active_holders"
    ] = np.nan

    aggregate[
        "max_individual_position_pct"
    ] = np.nan

    aggregate[
        "max_position_share_pct"
    ] = np.nan

    aggregate["fi_source"] = "aggregate"

    columns = [
        "snapshot_date",
        "issuer",
        "isin",
        "security_key",
        "short_interest_pct",
        "active_holders",
        "max_individual_position_pct",
        "max_position_share_pct",
        "fi_source",
    ]

    return aggregate[columns]


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

    frame["fi_source"] = "reconstructed"

    aggregate = _load_aggregate_snapshots(
        frame
    )

    if not aggregate.empty:
        frame = pd.concat(
            [
                frame,
                aggregate,
            ],
            ignore_index=True,
        )

    frame = frame.sort_values(
        [
            "security_key",
            "snapshot_date",
            "fi_source",
        ],
        kind="mergesort",
    )

    source_priority = frame[
        "fi_source"
    ].map(
        {
            "aggregate": 0,
            "reconstructed": 1,
        }
    ).fillna(0)

    frame["_fi_source_priority"] = (
        source_priority
    )

    frame = frame.sort_values(
        [
            "security_key",
            "snapshot_date",
            "_fi_source_priority",
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

    frame = frame.loc[
        ~duplicates
    ].copy()

    return frame.drop(
        columns="_fi_source_priority"
    )


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
