from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from ml.config import TARGETS
from ml.research.discovery.engine import (
    Candidate,
    DiscoveryData,
    evaluate_candidate_on_mask,
)
from ml.research.discovery.config import DiscoveryConfig
def _target_return_column(target_name: str) -> str:
    for target in TARGETS:
        if target.name == target_name:
            return target.return_column
    raise KeyError(
        f"Could not resolve return column for target '{target_name}'."
    )
def _date_mask(
    dates: pd.Series,
    start: str,
    end: str,
) -> np.ndarray:
    parsed = pd.to_datetime(dates)
    return (
        (parsed >= pd.Timestamp(start))
        & (parsed <= pd.Timestamp(end))
    ).to_numpy()
def _safe_mean(values: pd.Series | np.ndarray) -> float | None:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return None
    return float(values.mean())
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
        "baseline_event_rate": result["baseline_event_rate"],
        "lift": result["lift"],
        "mean_return": result["mean_return"],
        "rest_mean_return": result["rest_mean_return"],
        "return_difference": result["return_difference"],
        "bootstrap_ci_low": result["bootstrap_ci_low"],
        "bootstrap_ci_high": result["bootstrap_ci_high"],
        "selected_fraction": result["selected_fraction"],
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
        ("2025-H2", "2025-12-20", "2025-12-31"),
        ("2026-H1", "2026-01-01", "2026-06-30"),
        ("2026-H2", "2026-07-01", evaluation_end),
    ]
    rows: list[dict[str, Any]] = []
    dates = pd.to_datetime(data.features["snapshot_date"])
    for value, start, end in periods:
        start = max(start, evaluation_start)
        end = min(end, evaluation_end)
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
                data,
                candidate,
                mask,
                "period",
                value,
            )
        )
    return pd.DataFrame(rows)
def analyze_sectors(
    data: DiscoveryData,
    candidate: Candidate,
) -> pd.DataFrame:
    features = data.features
    sector_column = _find_column(
        features,
        [
            "sector",
            "sector_name",
            "gics_sector",
            "sectorName",
        ],
    )
    if sector_column is None:
        return pd.DataFrame(
            columns=[
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
        )
    rows: list[dict[str, Any]] = []
    for sector in sorted(
        features[sector_column].dropna().astype(str).unique()
    ):
        mask = (
            features[sector_column]
            .astype(str)
            .eq(sector)
            .to_numpy()
        )
        if not mask.any():
            continue
        rows.append(
            _evaluate_group(
                data,
                candidate,
                mask,
                "sector",
                sector,
            )
        )
    return pd.DataFrame(rows)
def analyze_market_regime(
    data: DiscoveryData,
    candidate: Candidate,
) -> pd.DataFrame:
    """
    Uses the market return feature if available.
    Expected interpretation:
      positive -> positive market regime
      negative -> negative market regime
    The exact threshold is deliberately zero; no optimization is performed.
    """
    features = data.features
    market_column = _find_column(
        features,
        [
            "market_return_20d",
            "index_return_20d",
            "omxs30_return_20d",
            "market_forward_return_20d",
        ],
    )
    if market_column is None:
        return pd.DataFrame(
            columns=[
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
        )
    values = pd.to_numeric(
        features[market_column],
        errors="coerce",
    )
    masks = {
        "negative": values < 0,
        "positive": values >= 0,
    }
    rows: list[dict[str, Any]] = []
    for regime, mask_series in masks.items():
        mask = mask_series.fillna(False).to_numpy()
        if not mask.any():
            continue
        rows.append(
            _evaluate_group(
                data,
                candidate,
                mask,
                "market_regime",
                regime,
            )
        )
    return pd.DataFrame(rows)
def run_subgroup_analysis(
    data: DiscoveryData,
    candidate: Candidate,
    evaluation_start: str,
    evaluation_end: str,
    output_dir: Path,
) -> dict[str, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "period": analyze_periods(
            data,
            candidate,
            evaluation_start,
            evaluation_end,
        ),
        "sector": analyze_sectors(
            data,
            candidate,
        ),
        "market_regime": analyze_market_regime(
            data,
            candidate,
        ),
    }
    for name, frame in results.items():
        frame.to_csv(
            output_dir / f"{name}.csv",
            index=False,
        )
    return results
