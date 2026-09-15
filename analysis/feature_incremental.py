"""Inkrementell hantering av feature-data."""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from analysis.feature_config import (
    FEATURE_GLOB,
    OUTPUT_DIR,
    RETURN_HORIZONS,
)
from analysis.feature_prices import (
    attach_prices,
)
from analysis.feature_returns import (
    add_forward_returns,
)
from analysis.feature_utils import (
    feature_key,
)
def load_existing(
    path: Path | None = None,
) -> pd.DataFrame:
    """
    Läser hela det befintliga feature-datasetet.
    Datasetet består av features_*.jsonl.
    Alla chunks läses och sätts ihop till ett DataFrame.
    """
    if path is not None:
        # Bakåtkompatibilitet om någon gammal caller
        # fortfarande skickar in en sökväg.
        directory = path.parent
    else:
        directory = OUTPUT_DIR
    paths = sorted(
        directory.glob(
            FEATURE_GLOB
        )
    )
    if not paths:
        return pd.DataFrame()
    frames: list[pd.DataFrame] = []
    for feature_path in paths:
        frame = pd.read_json(
            feature_path,
            lines=True,
        )
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    frame = pd.concat(
        frames,
        ignore_index=True,
    )
    for column in (
        "snapshot_date",
        "previous_snapshot_date",
        "price_date",
    ):
        if column in frame.columns:
            frame[column] = pd.to_datetime(
                frame[column],
                errors="coerce",
            )
    return frame
def find_new_fi_rows(
    fi: pd.DataFrame,
    existing: pd.DataFrame,
) -> pd.DataFrame:
    if existing.empty:
        return fi.copy()
    existing_keys = set(
        feature_key(existing)
    )
    fi_keys = feature_key(
        fi
    )
    new_mask = ~fi_keys.isin(
        existing_keys
    )
    return fi.loc[
        new_mask
    ].copy()
def build_context_rows(
    fi: pd.DataFrame,
    new_fi: pd.DataFrame,
) -> pd.DataFrame:
    """
    Hämtar senaste befintliga FI-observation före
    den första nya observationen per bolag.
    Det är viktigt att context aldrig innehåller
    den nya observationen själv.
    """
    if new_fi.empty:
        return pd.DataFrame(
            columns=fi.columns
        )
    first_new_dates = (
        new_fi.groupby(
            "security_key"
        )["snapshot_date"]
        .min()
    )
    candidate_keys = set(
        first_new_dates.index
    )
    context_candidates = fi.loc[
        fi["security_key"].isin(
            candidate_keys
        )
    ].copy()
    context_candidates = (
        context_candidates.loc[
            context_candidates.apply(
                lambda row: (
                    row["snapshot_date"]
                    < first_new_dates.get(
                        row["security_key"],
                        pd.Timestamp.max,
                    )
                ),
                axis=1,
            )
        ]
    )
    if context_candidates.empty:
        return context_candidates
    return (
        context_candidates
        .sort_values(
            [
                "security_key",
                "snapshot_date",
            ],
            kind="mergesort",
        )
        .groupby(
            "security_key",
            sort=False,
        )
        .tail(1)
    )
def refresh_incomplete_returns(
    existing: pd.DataFrame,
    prices: pd.DataFrame,
) -> pd.DataFrame:
    if existing.empty:
        return existing
    return_columns = [
        f"forward_return_{h}d"
        for h in RETURN_HORIZONS
    ]
    missing_columns = [
        column
        for column in return_columns
        if column not in existing.columns
    ]
    if missing_columns:
        for column in missing_columns:
            existing[column] = pd.NA
    incomplete = existing[
        return_columns
    ].isna().any(axis=1)
    if not incomplete.any():
        return existing
    subset = existing.loc[
        incomplete
    ].copy()
    refreshed, _ = attach_prices(
        subset,
        prices,
    )
    if refreshed.empty:
        return existing
    refreshed = add_forward_returns(
        refreshed,
        prices,
    )
    refreshed = refreshed.set_index(
        [
            "security_key",
            "snapshot_date",
        ]
    )
    current = existing.set_index(
        [
            "security_key",
            "snapshot_date",
        ]
    )
    update_columns = [
        "price_date",
        "close",
        "close_on_signal_date",
        "days_from_fi_to_price",
        "price_match_available",
        "yahoo_symbol",
        "price_mapping_source",
        "price_return_5d",
        "price_return_20d",
        "price_return_60d",
        "price_volatility_20d",
        "price_distance_from_20d_high",
        "price_distance_from_60d_high",
        "forward_return_1d",
        "forward_return_5d",
        "forward_return_20d",
        "forward_return_60d",
    ]
    common_index = current.index.intersection(
        refreshed.index
    )
    for column in update_columns:
        if column not in refreshed.columns:
            continue
        current.loc[
            common_index,
            column,
        ] = refreshed.loc[
            common_index,
            column,
        ]
    return current.reset_index()
def merge_features(
    existing: pd.DataFrame,
    new_features: pd.DataFrame,
) -> pd.DataFrame:
    if existing.empty:
        return new_features.copy()
    if new_features.empty:
        return existing.copy()
    return pd.concat(
        [
            existing,
            new_features,
        ],
        ignore_index=True,
    )
