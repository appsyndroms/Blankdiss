from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from ml.research.discovery.engine import (
    Candidate,
    DiscoveryData,
    evaluate_candidate_on_mask,
)
RESULT_COLUMNS = [
    "group",
    "value",
    "rows_selected",
    "events",
    "event_rate",
    "baseline_event_rate",
    "lift",
    "mean_return",
    "rest_mean_return",
    "return_difference",
    "bootstrap_ci_low",
    "bootstrap_ci_high",
    "selected_fraction",
]
def _date_mask(
    dates: pd.Series,
    start: str,
    end: str,
) -> np.ndarray:
    parsed = pd.to_datetime(
        dates,
        errors="coerce",
    )
    return (
        (parsed >= pd.Timestamp(start))
        & (parsed <= pd.Timestamp(end))
    ).to_numpy()
def _evaluation_mask(
    data: DiscoveryData,
    evaluation_start: str,
    evaluation_end: str,
) -> np.ndarray:
    frame = data.frame
    if "snapshot_date" not in frame.columns:
        raise KeyError(
            "Feature dataset saknar 'snapshot_date'."
        )
    return _date_mask(
        frame["snapshot_date"],
        evaluation_start,
        evaluation_end,
    )
def _evaluate_group(
    data: DiscoveryData,
    candidate: Candidate,
    mask: np.ndarray,
    group_name: str,
    group_value: str,
) -> dict[str, Any]:
    result = evaluate_candidate_on_mask(
        data=data,
        candidate=candidate,
        base_mask=mask,
        split="subgroup",
    )
    return {
        "group": group_name,
        "value": group_value,
        "rows_selected": result["n"],
        "events": result["events"],
        "event_rate": result["event_rate"],
        "baseline_event_rate": result[
            "baseline_event_rate"
        ],
        "lift": result["lift"],
        "mean_return": result["mean_return"],
        "rest_mean_return": result[
            "rest_mean_return"
        ],
        "return_difference": result[
            "return_difference"
        ],
        "bootstrap_ci_low": result[
            "bootstrap_ci_low"
        ],
        "bootstrap_ci_high": result[
            "bootstrap_ci_high"
        ],
        "selected_fraction": result[
            "selected_fraction"
        ],
    }
def _find_column(
    frame: pd.DataFrame,
    candidates: list[str],
) -> str | None:
    for name in candidates:
        if name in frame.columns:
            return name
    return None
def analyze_periods(
    data: DiscoveryData,
    candidate: Candidate,
    evaluation_start: str,
    evaluation_end: str,
) -> pd.DataFrame:
    periods = [
        (
            "2025-H2",
            "2025-12-20",
            "2025-12-31",
        ),
        (
            "2026-H1",
            "2026-01-01",
            "2026-06-30",
        ),
        (
            "2026-H2",
            "2026-07-01",
            evaluation_end,
        ),
    ]
    rows: list[dict[str, Any]] = []
    frame = data.frame
    if "snapshot_date" not in frame.columns:
        raise KeyError(
            "Feature dataset saknar 'snapshot_date'."
        )
    dates = pd.to_datetime(
        frame["snapshot_date"],
        errors="coerce",
    )
    for value, period_start, period_end in periods:
        start = max(
            period_start,
            evaluation_start,
        )
        end = min(
            period_end,
            evaluation_end,
        )
        if start > end:
            continue
        mask = (
            (dates >= pd.Timestamp(start))
            & (dates <= pd.Timestamp(end))
        ).to_numpy()
        if not mask.any():
            continue
        rows.append(
            _evaluate_group(
                data=data,
                candidate=candidate,
                mask=mask,
                group_name="period",
                group_value=value,
            )
        )
    return pd.DataFrame(
        rows,
        columns=RESULT_COLUMNS,
    )
def analyze_sectors(
    data: DiscoveryData,
    candidate: Candidate,
    evaluation_start: str,
    evaluation_end: str,
) -> pd.DataFrame:
    frame = data.frame
    sector_column = _find_column(
        frame,
        [
            "sector",
            "sector_name",
            "gics_sector",
            "sectorName",
        ],
    )
    if sector_column is None:
        return pd.DataFrame(
            columns=RESULT_COLUMNS
        )
    oos_mask = _evaluation_mask(
        data=data,
        evaluation_start=evaluation_start,
        evaluation_end=evaluation_end,
    )
    sector_values = (
        frame[sector_column]
        .dropna()
        .astype(str)
        .unique()
    )
    rows: list[dict[str, Any]] = []
    for sector in sorted(sector_values):
        sector_mask = (
            frame[sector_column]
            .astype(str)
            .eq(sector)
            .to_numpy()
        )
        mask = oos_mask & sector_mask
        if not mask.any():
            continue
        rows.append(
            _evaluate_group(
                data=data,
                candidate=candidate,
                mask=mask,
                group_name="sector",
                group_value=sector,
            )
        )
    return pd.DataFrame(
        rows,
        columns=RESULT_COLUMNS,
    )
def analyze_market_regime(
    data: DiscoveryData,
    candidate: Candidate,
    evaluation_start: str,
    evaluation_end: str,
) -> pd.DataFrame:
    frame = data.frame
    # Only use explicitly backward-looking market
    # return features. Do not use names containing
    # "forward", because that could introduce lookahead.
    market_column = _find_column(
        frame,
        [
            "market_return_20d",
            "index_return_20d",
            "omxs30_return_20d",
        ],
    )
    if market_column is None:
        return pd.DataFrame(
            columns=RESULT_COLUMNS
        )
    oos_mask = _evaluation_mask(
        data=data,
        evaluation_start=evaluation_start,
        evaluation_end=evaluation_end,
    )
    values = pd.to_numeric(
        frame[market_column],
        errors="coerce",
    )
    regime_masks = {
        "negative": (
            values < 0
        ).to_numpy(),
        "positive": (
            values >= 0
        ).to_numpy(),
    }
    rows: list[dict[str, Any]] = []
    for regime, regime_mask in regime_masks.items():
        mask = oos_mask & regime_mask
        if not mask.any():
            continue
        rows.append(
            _evaluate_group(
                data=data,
                candidate=candidate,
                mask=mask,
                group_name="market_regime",
                group_value=regime,
            )
        )
    return pd.DataFrame(
        rows,
        columns=RESULT_COLUMNS,
    )
def run_subgroup_analysis(
    data: DiscoveryData,
    candidate: Candidate,
    evaluation_start: str,
    evaluation_end: str,
    output_dir: Path,
) -> dict[str, pd.DataFrame]:
    results = {
        "period": analyze_periods(
            data=data,
            candidate=candidate,
            evaluation_start=evaluation_start,
            evaluation_end=evaluation_end,
        ),
        "sector": analyze_sectors(
            data=data,
            candidate=candidate,
            evaluation_start=evaluation_start,
            evaluation_end=evaluation_end,
        ),
        "market_regime": analyze_market_regime(
            data=data,
            candidate=candidate,
            evaluation_start=evaluation_start,
            evaluation_end=evaluation_end,
        ),
    }
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    for name, frame in results.items():
        frame.to_csv(
            output_dir / f"{name}.csv",
            index=False,
        )
    return results
