from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .base import ExperimentResult
from ml.research.signals import build_signal


# ============================================================================
# MOMENTUM × SI CELL CONTEXT
# ============================================================================

# These are the previously identified cells.
# They remain locked for this follow-up analysis.
FOCUS_CELLS = (
    (1, 3),
    (10, 6),
    (6, 9),
    (9, 10),
    (7, 10),
)

# Event definition used throughout this follow-up.
EVENT_THRESHOLD = -0.05

# Event-risk bands are deliberately broad and fixed.
EVENT_RISK_BANDS = (
    ("top_5pct", 0.00, 0.05),
    ("5_20pct", 0.05, 0.20),
    ("20_50pct", 0.20, 0.50),
    ("bottom_50pct", 0.50, 1.00),
)

HORIZONS = (1, 3, 5, 10, 20)

# The 9x10 follow-up focuses on the actual SI change rather than
# treating SI decile as sufficient.
SI_CHANGE_BUCKETS = (
    "negative",
    "zero",
    "positive_normal",
    "positive_extreme",
)


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(index=frame.index, dtype=float)

    return pd.to_numeric(
        frame[column],
        errors="coerce",
    )


def _numeric_mean(frame: pd.DataFrame, column: str) -> float:
    values = _numeric(frame, column).dropna()

    return float(values.mean()) if not values.empty else float("nan")


def _numeric_median(frame: pd.DataFrame, column: str) -> float:
    values = _numeric(frame, column).dropna()

    return float(values.median()) if not values.empty else float("nan")


def _event_rate(frame: pd.DataFrame) -> float:
    values = _numeric(frame, "forward_return_5d").dropna()

    if values.empty:
        return float("nan")

    return float(
        (values <= EVENT_THRESHOLD).mean()
    )


def _cross_sectional_deciles(
    frame: pd.DataFrame,
    column: str,
) -> pd.Series:
    """Assign 0-9 deciles independently for each snapshot date."""

    values = _numeric(frame, column)

    ranks = values.groupby(
        frame["snapshot_date"]
    ).rank(
        method="first",
        pct=True,
    )

    deciles = np.ceil(
        ranks * 10
    ).astype("Int64") - 1

    deciles = deciles.clip(
        lower=0,
        upper=9,
    )

    return deciles.fillna(-1).astype(int)


def _prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()

    if "snapshot_date" not in result.columns:
        raise ValueError(
            "Momentum/SI analysis requires snapshot_date"
        )

    result["snapshot_date"] = pd.to_datetime(
        result["snapshot_date"],
        errors="coerce",
    )

    result = result.loc[
        result["snapshot_date"].notna()
    ].copy()

    result["price_momentum_5d"] = build_signal(
        result,
        "price_momentum_5d",
    )

    result["short_interest_change"] = build_signal(
        result,
        "short_interest_change",
    )

    result["momentum_decile"] = _cross_sectional_deciles(
        result,
        "price_momentum_5d",
    )

    result["si_decile"] = _cross_sectional_deciles(
        result,
        "short_interest_change",
    )

    # Prefer the existing event-risk score if the surrounding research
    # context already provides it.
    if "event_score" in result.columns:
        result["event_score"] = _numeric(
            result,
            "event_score",
        )

    return result


def _cell_mask(
    frame: pd.DataFrame,
    momentum_decile: int,
    si_decile: int,
) -> pd.Series:
    return (
        (frame["momentum_decile"] == momentum_decile - 1)
        & (frame["si_decile"] == si_decile - 1)
    )


def _momentum_control_mask(
    frame: pd.DataFrame,
    momentum_decile: int,
    exclude_si_decile: int | None = None,
) -> pd.Series:
    mask = frame["momentum_decile"] == momentum_decile - 1

    if exclude_si_decile is not None:
        mask &= frame["si_decile"] != exclude_si_decile - 1

    return mask


def _event_rate_row(
    *,
    label: str,
    frame: pd.DataFrame,
) -> dict:
    return {
        "group": label,
        "n": int(len(frame)),
        "event_rate": _event_rate(frame),
        "momentum_mean": _numeric_mean(
            frame,
            "price_momentum_5d",
        ),
        "momentum_median": _numeric_median(
            frame,
            "price_momentum_5d",
        ),
        "si_change_mean": _numeric_mean(
            frame,
            "short_interest_change",
        ),
        "si_change_median": _numeric_median(
            frame,
            "short_interest_change",
        ),
        "si_level_mean": _numeric_mean(
            frame,
            "short_interest_pct",
        ),
        "si_level_median": _numeric_median(
            frame,
            "short_interest_pct",
        ),
        "volatility_mean": _numeric_mean(
            frame,
            "volatility_20d",
        ),
        "volatility_median": _numeric_median(
            frame,
            "volatility_20d",
        ),
    }


def _delta_pp(
    focal: pd.DataFrame,
    control: pd.DataFrame,
) -> float:
    focal_rate = _event_rate(focal)
    control_rate = _event_rate(control)

    if not np.isfinite(focal_rate) or not np.isfinite(control_rate):
        return float("nan")

    return float(
        focal_rate - control_rate
    )


def _event_risk_cutoffs(
    frame: pd.DataFrame,
) -> dict[str, float]:
    """
    Calculate event-risk cutoffs cross-sectionally from the available
    frame.

    The cutoffs are descriptive here; this experiment does not fit
    another model. If event_score is absent, no bands are generated.
    """
    if "event_score" not in frame.columns:
        return {}

    values = _numeric(
        frame,
        "event_score",
    ).dropna()

    if values.empty:
        return {}

    return {
        name: float(
            values.quantile(
                1.0 - upper
            )
        )
        for name, _, upper in EVENT_RISK_BANDS
    }


def _event_risk_band(
    frame: pd.DataFrame,
    band_name: str,
    lower: float,
    upper: float,
) -> pd.Series:
    if "event_score" not in frame.columns:
        return pd.Series(
            False,
            index=frame.index,
        )

    values = _numeric(
        frame,
        "event_score",
    )

    if lower == 0.0:
        cutoff = float(
            values.quantile(
                1.0 - upper
            )
        )

        return values >= cutoff

    if upper == 1.0:
        cutoff = float(
            values.quantile(
                1.0 - lower
            )
        )

        return values < cutoff

    upper_cut = float(
        values.quantile(
            1.0 - upper
        )
    )

    lower_cut = float(
        values.quantile(
            1.0 - lower
        )
    )

    return (
        (values >= upper_cut)
        & (values < lower_cut)
    )


def _add_si_change_buckets(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    result = frame.copy()

    change = _numeric(
        result,
        "short_interest_change",
    )

    positive = change[
        change > 0
    ].dropna()

    if positive.empty:
        extreme_cutoff = float("inf")
    else:
        # Extreme = top 20% of positive SI changes.
        extreme_cutoff = float(
            positive.quantile(0.80)
        )

    result["si_change_bucket"] = np.select(
        [
            (change < 0).to_numpy(dtype=bool),
            (change == 0).to_numpy(dtype=bool),
            (change > 0).to_numpy(dtype=bool),
        ],
        [
            "negative",
            "zero",
            "positive_normal",
        ],
        default="missing",
    )

    extreme_mask = (
        change >= extreme_cutoff
    ) & change.notna()

    result.loc[
        extreme_mask,
        "si_change_bucket",
    ] = "positive_extreme"

    result["si_change_extreme_cutoff"] = extreme_cutoff

    return result


def _cell_context_rows(
    frame: pd.DataFrame,
    *,
    focus_cells: tuple[tuple[int, int], ...],
) -> list[dict]:
    rows: list[dict] = []

    for momentum_decile, si_decile in focus_cells:
        focal_mask = _cell_mask(
            frame,
            momentum_decile,
            si_decile,
        )

        focal = frame.loc[
            focal_mask
        ].copy()

        momentum_control_mask = _momentum_control_mask(
            frame,
            momentum_decile,
            exclude_si_decile=si_decile,
        )

        momentum_control = frame.loc[
            momentum_control_mask
        ].copy()

        row = _event_rate_row(
            label=f"{momentum_decile}x{si_decile}",
            frame=focal,
        )

        row.update(
            {
                "momentum_decile": momentum_decile,
                "si_decile": si_decile,
                "control_n": int(
                    len(momentum_control)
                ),
                "control_event_rate": _event_rate(
                    momentum_control
                ),
                "delta_vs_same_momentum_pp": _delta_pp(
                    focal,
                    momentum_control,
                ),
            }
        )

        rows.append(row)

    return rows


def _conditional_rows(
    frame: pd.DataFrame,
    *,
    focus_cells: tuple[tuple[int, int], ...],
) -> list[dict]:
    """
    Compare each locked cell against the same momentum decile while
    controlling for event-risk band.

    This is deliberately descriptive. It answers:
        Does SI still matter inside the same momentum + risk regime?
    """
    if "event_score" not in frame.columns:
        return []

    rows: list[dict] = []

    for momentum_decile, si_decile in focus_cells:
        focal_mask = _cell_mask(
            frame,
            momentum_decile,
            si_decile,
        )

        for band_name, lower, upper in EVENT_RISK_BANDS:
            band_mask = _event_risk_band(
                frame,
                band_name,
                lower,
                upper,
            )

            focal = frame.loc[
                focal_mask & band_mask
            ].copy()

            control_mask = (
                _momentum_control_mask(
                    frame,
                    momentum_decile,
                    exclude_si_decile=si_decile,
                )
                & band_mask
            )

            control = frame.loc[
                control_mask
            ].copy()

            rows.append(
                {
                    "momentum_decile": momentum_decile,
                    "si_decile": si_decile,
                    "cell": (
                        f"{momentum_decile}x"
                        f"{si_decile}"
                    ),
                    "event_risk_band": band_name,
                    "n": int(len(focal)),
                    "event_rate": _event_rate(
                        focal
                    ),
                    "control_n": int(
                        len(control)
                    ),
                    "control_event_rate": _event_rate(
                        control
                    ),
                    "delta_vs_same_momentum_and_risk_pp": (
                        _delta_pp(
                            focal,
                            control,
                        )
                    ),
                    "event_score_mean": _numeric_mean(
                        focal,
                        "event_score",
                    ),
                    "event_score_median": _numeric_median(
                        focal,
                        "event_score",
                    ),
                }
            )

    return rows


def _nine_by_ten_si_change_rows(
    frame: pd.DataFrame,
) -> list[dict]:
    """
    Mechanism follow-up for the 9x10 cell.

    The purpose is to determine whether 9x10 is driven by the
    SI-decile label itself or by unusually large actual SI changes.
    """
    focal = frame.loc[
        _cell_mask(
            frame,
            9,
            10,
        )
    ].copy()

    if focal.empty:
        return []

    focal = _add_si_change_buckets(
        focal
    )

    rows: list[dict] = []

    for bucket in SI_CHANGE_BUCKETS:
        local = focal.loc[
            focal["si_change_bucket"] == bucket
        ].copy()

        if local.empty:
            continue

        row = _event_rate_row(
            label=bucket,
            frame=local,
        )

        row.update(
            {
                "cell": "9x10",
                "si_change_bucket": bucket,
                "n": int(len(local)),
                "event_rate": _event_rate(
                    local
                ),
                "forward_return_5d_mean": _numeric_mean(
                    local,
                    "forward_return_5d",
                ),
                "forward_return_5d_median": _numeric_median(
                    local,
                    "forward_return_5d",
                ),
            }
        )

        rows.append(row)

    return rows


def _nine_by_ten_vs_si_level_rows(
    frame: pd.DataFrame,
) -> list[dict]:
    """
    Compare the 9x10 cell against 9x1..9 by actual SI change and
    SI level. This helps identify whether the decile is merely acting
    as a proxy for a few extreme observations.
    """
    focal = frame.loc[
        frame["momentum_decile"] == 8
    ].copy()

    if focal.empty:
        return []

    focal = _add_si_change_buckets(
        focal
    )

    rows: list[dict] = []

    for si_decile in range(1, 11):
        local = focal.loc[
            focal["si_decile"] == si_decile - 1
        ].copy()

        if local.empty:
            continue

        row = _event_rate_row(
            label=f"9x{si_decile}",
            frame=local,
        )

        row.update(
            {
                "momentum_decile": 9,
                "si_decile": si_decile,
                "cell": f"9x{si_decile}",
                "n": int(len(local)),
                "event_rate": _event_rate(
                    local
                ),
                "positive_si_change_rate": (
                    float(
                        (
                            _numeric(
                                local,
                                "short_interest_change",
                            )
                            > 0
                        ).mean()
                    )
                ),
                "extreme_si_change_rate": (
                    float(
                        (
                            local["si_change_bucket"]
                            == "positive_extreme"
                        ).mean()
                    )
                ),
            }
        )

        rows.append(row)

    return rows


def run_momentum_si_cell_context(
    context,
    *,
    focus_cells: tuple[tuple[int, int], ...] = FOCUS_CELLS,
    context_columns: tuple[str, ...] = (),
    horizons: tuple[int, ...] = HORIZONS,
) -> ExperimentResult:
    """
    Locked Momentum × SI follow-up.

    Main questions:
    1. Does each locked cell have elevated event risk?
    2. Does the elevation remain against the same momentum decile?
    3. Does it remain within the same event-risk regime?
    4. For 9x10, is the effect actually driven by extreme SI changes?
    """
    frame = _prepare_frame(
        context.test
    )

    result = ExperimentResult(
        name="momentum_si_cell_context",
        description=(
            "Fördjupad kontroll av fem låsta momentum × SI-celler. "
            "Testar först kontroll mot samma momentumdecil och därefter "
            "samma momentumdecil + event-riskregim. För 9x10 analyseras "
            "dessutom faktisk SI-förändring för att skilja decileffekt "
            "från extrema SI-hopp."
        ),
    )

    # ------------------------------------------------------------------
    # 1. Main locked-cell table
    # ------------------------------------------------------------------
    result.add_table(
        "cell_context",
        pd.DataFrame(
            _cell_context_rows(
                frame,
                focus_cells=focus_cells,
            )
        ),
    )

    # ------------------------------------------------------------------
    # 2. Conditional event-risk control
    # ------------------------------------------------------------------
    conditional = _conditional_rows(
        frame,
        focus_cells=focus_cells,
    )

    result.add_table(
        "momentum_event_risk_control",
        pd.DataFrame(
            conditional
        ),
    )

    # ------------------------------------------------------------------
    # 3. 9x10 mechanism: actual SI change
    # ------------------------------------------------------------------
    result.add_table(
        "nine_by_ten_si_change",
        pd.DataFrame(
            _nine_by_ten_si_change_rows(
                frame
            )
        ),
    )

    # ------------------------------------------------------------------
    # 4. Full momentum-9 SI surface
    # ------------------------------------------------------------------
    result.add_table(
        "momentum_9_si_surface",
        pd.DataFrame(
            _nine_by_ten_vs_si_level_rows(
                frame
            )
        ),
    )

    # ------------------------------------------------------------------
    # 5. Optional context columns
    # ------------------------------------------------------------------
    if context_columns:
        context_rows: list[dict] = []

        for momentum_decile, si_decile in focus_cells:
            local = frame.loc[
                _cell_mask(
                    frame,
                    momentum_decile,
                    si_decile,
                )
            ].copy()

            row = {
                "momentum_decile": momentum_decile,
                "si_decile": si_decile,
                "cell": (
                    f"{momentum_decile}x"
                    f"{si_decile}"
                ),
                "n": int(len(local)),
            }

            for column in context_columns:
                row[f"{column}_mean"] = _numeric_mean(
                    local,
                    column,
                )

                row[f"{column}_median"] = _numeric_median(
                    local,
                    column,
                )

            for horizon in horizons:
                column = (
                    f"forward_return_{horizon}d"
                )

                if column in local.columns:
                    row[f"{column}_mean"] = _numeric_mean(
                        local,
                        column,
                    )

                    row[f"{column}_median"] = _numeric_median(
                        local,
                        column,
                    )

            context_rows.append(row)

        result.add_table(
            "cell_price_context",
            pd.DataFrame(
                context_rows
            ),
        )

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------
    result.add_metric(
        "test_rows",
        int(len(frame)),
    )

    result.add_metric(
        "focus_cell_count",
        int(len(focus_cells)),
    )

    result.add_metric(
        "focus_cells",
        [
            f"{momentum}x{si}"
            for momentum, si in focus_cells
        ],
    )

    result.add_metric(
        "event_threshold",
        EVENT_THRESHOLD,
    )

    result.add_metric(
        "event_risk_bands",
        [
            name
            for name, _, _ in EVENT_RISK_BANDS
        ],
    )

    result.add_metric(
        "mechanism_focus",
        "9x10",
    )

    result.add_metric(
        "si_change_extreme_definition",
        "top 20% of positive SI changes within the analysed frame",
    )

    return result


# ============================================================================
# MOMENTUM × SI INCREMENTAL LOCKED OOS
# ============================================================================

# ---------------------------------------------------------------------------
# LOCKED HYPOTHESIS
# ---------------------------------------------------------------------------
# This analysis deliberately contains no data-driven selection logic.
#
# The regime and SI-change definition are fixed before the 2026 test:
#
#   momentum decile 9
#   short-interest level decile 10
#   high SI change = top 20% of positive SI changes
#
# The four groups are:
#
#   A = outside 9x10 regime + low SI change
#   B = outside 9x10 regime + high SI change
#   C = 9x10 regime + low SI change
#   D = 9x10 regime + high SI change
#
# Primary effect:
#
#   D - C
#
# The experiment is deliberately descriptive. It does not fit a model,
# search thresholds, select cells, or optimize against the 2026 data.

MOMENTUM_DECILE = 9
SI_LEVEL_DECILE = 10
SI_CHANGE_TOP_FRACTION = 0.20

DISCOVERY_END = pd.Timestamp(
    "2024-12-31"
)

LOCK_START = pd.Timestamp(
    "2025-01-01"
)

LOCK_END = pd.Timestamp(
    "2025-12-31"
)

OOS_START = pd.Timestamp(
    "2026-01-01"
)

MIN_GROUP_N = 20

TARGETS = (
    (
        "down_5pct_5d",
        "forward_return_5d",
        "below",
        -0.05,
    ),
    (
        "down_7pct_5d",
        "forward_return_5d",
        "below",
        -0.07,
    ),
    (
        "down_10pct_10d",
        "forward_return_10d",
        "below",
        -0.10,
    ),
)

RETURN_HORIZONS = (
    5,
    10,
)

SECTOR_MAP_PATH = Path(
    "data/analysis/sector_map.json"
)


def _incremental_numeric(
    frame: pd.DataFrame,
    column: str,
) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(
            index=frame.index,
            dtype=float,
        )

    return pd.to_numeric(
        frame[column],
        errors="coerce",
    )


def _incremental_cross_sectional_deciles(
    frame: pd.DataFrame,
    values: pd.Series,
) -> pd.Series:
    """
    Assign 1..10 deciles independently for every snapshot date.

    Decile 1 = lowest.
    Decile 10 = highest.
    """
    numeric = pd.to_numeric(
        values,
        errors="coerce",
    )

    ranks = numeric.groupby(
        frame["snapshot_date"]
    ).rank(
        method="first",
        pct=True,
    )

    deciles = (
        np.ceil(
            ranks * 10
        )
        .astype("Int64")
    )

    return deciles.clip(
        lower=1,
        upper=10,
    )


def _high_si_change_mask(
    frame: pd.DataFrame,
) -> pd.Series:
    """
    Locked SI-change definition.

    High SI change means the top 20% of positive SI changes
    within each snapshot date.

    This is a cross-sectional rule and therefore does not
    estimate a threshold from 2025 or 2026.
    """
    # IMPORTANT:
    # Use the already normalized signal created by
    # _incremental_prepare(). The raw feature may use another
    # underlying column name and is resolved by build_signal().
    change = _incremental_numeric(
        frame,
        "si_change",
    )

    positive = change.where(
        change > 0
    )

    rank = positive.groupby(
        frame["snapshot_date"]
    ).rank(
        method="first",
        pct=True,
    )

    return (
        positive.notna()
        & (
            rank
            > (
                1.0
                - SI_CHANGE_TOP_FRACTION
            )
        )
    )


def _incremental_prepare(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    result = frame.copy()

    result["snapshot_date"] = pd.to_datetime(
        result["snapshot_date"],
        errors="coerce",
    )

    result = result.loc[
        result["snapshot_date"].notna()
    ].copy()

    result["momentum"] = build_signal(
        result,
        "price_momentum_5d",
    )

    result["si_level"] = build_signal(
        result,
        "short_interest_level",
    )

    result["si_change"] = build_signal(
        result,
        "short_interest_change",
    )

    result["momentum_decile"] = (
        _incremental_cross_sectional_deciles(
            result,
            result["momentum"],
        )
    )

    result["si_level_decile"] = (
        _incremental_cross_sectional_deciles(
            result,
            result["si_level"],
        )
    )

    result["high_si_change"] = (
        _high_si_change_mask(
            result
        )
    )

    result["high_regime"] = (
        (
            result["momentum_decile"]
            == MOMENTUM_DECILE
        )
        & (
            result["si_level_decile"]
            == SI_LEVEL_DECILE
        )
    )

    # Four locked groups:
    #
    # A = outside high regime + low SI change
    # B = outside high regime + high SI change
    # C = high regime + low SI change
    # D = high regime + high SI change
    #
    # np.select requires boolean ndarrays. Explicit conversion here
    # also handles pandas nullable boolean values safely.
    result["group"] = np.select(
        [
            (
                result["high_regime"]
                & result["high_si_change"]
            ).fillna(False).to_numpy(dtype=bool),
            (
                result["high_regime"]
                & ~result["high_si_change"]
            ).fillna(False).to_numpy(dtype=bool),
            (
                ~result["high_regime"]
                & result["high_si_change"]
            ).fillna(False).to_numpy(dtype=bool),
        ],
        [
            "D",
            "C",
            "B",
        ],
        default="A",
    )

    return result


def _event_series(
    frame: pd.DataFrame,
    return_column: str,
    direction: str,
    threshold: float,
) -> pd.Series:
    values = _incremental_numeric(
        frame,
        return_column,
    )

    if direction == "below":
        return values <= threshold

    return values >= threshold


def _proportion_ci(
    n: int,
    events: int,
    z: float = 1.96,
) -> tuple[float, float]:
    if n <= 0:
        return (
            float("nan"),
            float("nan"),
        )

    p = events / n

    se = np.sqrt(
        max(
            p * (1.0 - p),
            0.0,
        )
        / n
    )

    return (
        p - z * se,
        p + z * se,
    )


def _difference_ci(
    n_a: int,
    p_a: float,
    n_b: int,
    p_b: float,
    z: float = 1.96,
) -> tuple[float, float]:
    if (
        n_a <= 0
        or n_b <= 0
        or not np.isfinite(p_a)
        or not np.isfinite(p_b)
    ):
        return (
            float("nan"),
            float("nan"),
        )

    se = np.sqrt(
        max(
            p_a * (1.0 - p_a),
            0.0,
        )
        / n_a
        +
        max(
            p_b * (1.0 - p_b),
            0.0,
        )
        / n_b
    )

    delta = p_a - p_b

    return (
        delta - z * se,
        delta + z * se,
    )


def _group_rows(
    frame: pd.DataFrame,
    return_column: str,
    direction: str,
    threshold: float,
) -> list[dict[str, Any]]:
    event = _event_series(
        frame,
        return_column,
        direction,
        threshold,
    )

    rows: list[dict[str, Any]] = []

    for group in (
        "A",
        "B",
        "C",
        "D",
    ):
        mask = (
            frame["group"]
            == group
        )

        valid = (
            mask
            & event.notna()
        )

        values = _incremental_numeric(
            frame.loc[valid],
            return_column,
        )

        events = int(
            event.loc[valid].sum()
        )

        n = int(
            valid.sum()
        )

        rate = (
            events / n
            if n
            else float("nan")
        )

        ci_low, ci_high = (
            _proportion_ci(
                n,
                events,
            )
        )

        rows.append(
            {
                "group": group,
                "n": n,
                "events": events,
                "event_rate": rate,
                "event_rate_pct": (
                    rate * 100
                    if np.isfinite(rate)
                    else np.nan
                ),
                "event_rate_ci95_low": ci_low,
                "event_rate_ci95_high": ci_high,
                "mean_return": (
                    float(values.mean())
                    if not values.empty
                    else np.nan
                ),
                "median_return": (
                    float(values.median())
                    if not values.empty
                    else np.nan
                ),
            }
        )

    return rows


def _dc_effect(
    rows: list[dict[str, Any]],
    target_name: str,
) -> dict[str, Any]:
    by_group = {
        row["group"]: row
        for row in rows
    }

    c = by_group.get(
        "C",
        {},
    )

    d = by_group.get(
        "D",
        {},
    )

    n_c = int(
        c.get(
            "n",
            0,
        )
    )

    n_d = int(
        d.get(
            "n",
            0,
        )
    )

    p_c = float(
        c.get(
            "event_rate",
            np.nan,
        )
    )

    p_d = float(
        d.get(
            "event_rate",
            np.nan,
        )
    )

    delta = (
        p_d - p_c
        if n_c and n_d
        else np.nan
    )

    ci_low, ci_high = (
        _difference_ci(
            n_d,
            p_d,
            n_c,
            p_c,
        )
    )

    return {
        "target": target_name,
        "c_n": n_c,
        "d_n": n_d,
        "c_event_rate": p_c,
        "d_event_rate": p_d,
        "d_minus_c": delta,
        "d_minus_c_pp": (
            delta * 100
            if np.isfinite(delta)
            else np.nan
        ),
        "d_minus_c_ci95_low": ci_low,
        "d_minus_c_ci95_high": ci_high,
        "d_minus_c_ci95_low_pp": (
            ci_low * 100
            if np.isfinite(ci_low)
            else np.nan
        ),
        "d_minus_c_ci95_high_pp": (
            ci_high * 100
            if np.isfinite(ci_high)
            else np.nan
        ),
        "meets_min_group_n": (
            n_c >= MIN_GROUP_N
            and n_d >= MIN_GROUP_N
        ),
    }


def _target_analysis(
    frame: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    group_rows: list[
        dict[str, Any]
    ] = []

    effect_rows: list[
        dict[str, Any]
    ] = []

    for (
        target_name,
        return_column,
        direction,
        threshold,
    ) in TARGETS:
        if (
            return_column
            not in frame.columns
        ):
            effect_rows.append(
                {
                    "target": target_name,
                    "status": "unavailable",
                    "return_column": return_column,
                    "reason": (
                        "Feature data saknar "
                        f"{return_column}"
                    ),
                }
            )

            continue

        rows = _group_rows(
            frame,
            return_column,
            direction,
            threshold,
        )

        for row in rows:
            row.update(
                {
                    "target": target_name,
                    "return_column": return_column,
                    "threshold": threshold,
                }
            )

        group_rows.extend(
            rows
        )

        effect = _dc_effect(
            rows,
            target_name,
        )

        effect["status"] = (
            "available"
        )

        effect["return_column"] = (
            return_column
        )

        effect["threshold"] = (
            threshold
        )

        effect_rows.append(
            effect
        )

    return (
        pd.DataFrame(
            group_rows
        ),
        pd.DataFrame(
            effect_rows
        ),
    )


def _return_comparison(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[
        dict[str, Any]
    ] = []

    for horizon in RETURN_HORIZONS:
        column = (
            f"forward_return_{horizon}d"
        )

        if column not in frame.columns:
            rows.append(
                {
                    "horizon_days": horizon,
                    "status": "unavailable",
                    "return_column": column,
                }
            )

            continue

        for group in (
            "A",
            "B",
            "C",
            "D",
        ):
            values = _incremental_numeric(
                frame.loc[
                    frame["group"]
                    == group
                ],
                column,
            ).dropna()

            rows.append(
                {
                    "horizon_days": horizon,
                    "group": group,
                    "status": "available",
                    "n": int(
                        len(values)
                    ),
                    "mean_return": (
                        float(values.mean())
                        if not values.empty
                        else np.nan
                    ),
                    "median_return": (
                        float(values.median())
                        if not values.empty
                        else np.nan
                    ),
                }
            )

        d = _incremental_numeric(
            frame.loc[
                frame["group"]
                == "D"
            ],
            column,
        ).dropna()

        c = _incremental_numeric(
            frame.loc[
                frame["group"]
                == "C"
            ],
            column,
        ).dropna()

        rows.append(
            {
                "horizon_days": horizon,
                "group": "D_minus_C",
                "status": "available",
                "n": min(
                    len(d),
                    len(c),
                ),
                "mean_return": (
                    float(
                        d.mean()
                        - c.mean()
                    )
                    if (
                        not d.empty
                        and not c.empty
                    )
                    else np.nan
                ),
                "median_return": (
                    float(
                        d.median()
                        - c.median()
                    )
                    if (
                        not d.empty
                        and not c.empty
                    )
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def _load_sector_map() -> dict[str, str]:
    if not SECTOR_MAP_PATH.exists():
        return {}

    payload = json.loads(
        SECTOR_MAP_PATH.read_text(
            encoding="utf-8"
        )
    )

    instruments = (
        payload.get(
            "instruments",
            {},
        )
        if isinstance(
            payload,
            dict,
        )
        else {}
    )

    mapping: dict[
        str,
        str,
    ] = {}

    for symbol, item in instruments.items():
        if (
            isinstance(
                item,
                dict,
            )
            and item.get("sector")
        ):
            mapping[
                str(symbol)
            ] = str(
                item["sector"]
            ).strip()

    return mapping


def _sector_relative(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """
    Sector-relative return for the same four groups.

    Benchmark:
        equal-weight sector mean excluding the stock itself,
        separately for every snapshot date.

    If the frozen sector map is unavailable, report that explicitly.
    """
    sector_map = _load_sector_map()

    symbol_column = (
        "yahoo_symbol"
        if "yahoo_symbol"
        in frame.columns
        else "security_key"
    )

    if not sector_map:
        return pd.DataFrame(
            [
                {
                    "status": "unavailable",
                    "reason": (
                        "sector_map.json saknas "
                        "eller är tom"
                    ),
                }
            ]
        )

    work = frame.copy()

    work["sector"] = (
        work[
            symbol_column
        ]
        .astype(str)
        .map(sector_map)
    )

    rows: list[
        dict[str, Any]
    ] = []

    for horizon in RETURN_HORIZONS:
        column = (
            f"forward_return_{horizon}d"
        )

        if column not in work.columns:
            rows.append(
                {
                    "horizon_days": horizon,
                    "status": "unavailable",
                    "return_column": column,
                }
            )

            continue

        work["_ret"] = _incremental_numeric(
            work,
            column,
        )

        stock = work.dropna(
            subset=[
                "snapshot_date",
                "sector",
                "_ret",
            ]
        ).copy()

        if stock.empty:
            continue

        stock_sum = (
            stock.groupby(
                [
                    "snapshot_date",
                    "sector",
                ]
            )["_ret"]
            .transform("sum")
        )

        stock_count = (
            stock.groupby(
                [
                    "snapshot_date",
                    "sector",
                ]
            )["_ret"]
            .transform("count")
        )

        stock[
            "sector_mean_ex_self"
        ] = np.where(
            stock_count > 1,
            (
                stock_sum
                - stock["_ret"]
            )
            / (
                stock_count - 1
            ),
            np.nan,
        )

        stock[
            "sector_relative"
        ] = (
            stock["_ret"]
            - stock[
                "sector_mean_ex_self"
            ]
        )

        for group in (
            "A",
            "B",
            "C",
            "D",
        ):
            values = stock.loc[
                stock["group"]
                == group,
                "sector_relative",
            ].dropna()

            rows.append(
                {
                    "horizon_days": horizon,
                    "group": group,
                    "status": "available",
                    "n": int(
                        len(values)
                    ),
                    "mean_sector_relative": (
                        float(
                            values.mean()
                        )
                        if not values.empty
                        else np.nan
                    ),
                    "median_sector_relative": (
                        float(
                            values.median()
                        )
                        if not values.empty
                        else np.nan
                    ),
                }
            )

        d = stock.loc[
            stock["group"]
            == "D",
            "sector_relative",
        ].dropna()

        c = stock.loc[
            stock["group"]
            == "C",
            "sector_relative",
        ].dropna()

        rows.append(
            {
                "horizon_days": horizon,
                "group": "D_minus_C",
                "status": "available",
                "n": min(
                    len(d),
                    len(c),
                ),
                "mean_sector_relative": (
                    float(
                        d.mean()
                        - c.mean()
                    )
                    if (
                        not d.empty
                        and not c.empty
                    )
                    else np.nan
                ),
                "median_sector_relative": (
                    float(
                        d.median()
                        - c.median()
                    )
                    if (
                        not d.empty
                        and not c.empty
                    )
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def _period_frame(
    context,
    start: pd.Timestamp | None,
    end: pd.Timestamp | None,
) -> pd.DataFrame:
    frame = context.data.copy()

    frame["snapshot_date"] = pd.to_datetime(
        frame["snapshot_date"],
        errors="coerce",
    )

    frame = frame.loc[
        frame["snapshot_date"].notna()
    ].copy()

    if start is not None:
        frame = frame.loc[
            frame["snapshot_date"]
            >= start
        ]

    if end is not None:
        frame = frame.loc[
            frame["snapshot_date"]
            <= end
        ]

    return _incremental_prepare(
        frame
    )


def _analyse_period(
    frame: pd.DataFrame,
    phase: str,
) -> dict[
    str,
    pd.DataFrame,
]:
    (
        group_table,
        effect_table,
    ) = _target_analysis(
        frame
    )

    returns = _return_comparison(
        frame
    )

    sector = _sector_relative(
        frame
    )

    for table in (
        group_table,
        effect_table,
        returns,
        sector,
    ):
        if not table.empty:
            table.insert(
                0,
                "phase",
                phase,
            )

    return {
        "groups": group_table,
        "effects": effect_table,
        "returns": returns,
        "sector_relative": sector,
    }


def run_momentum_si_incremental_locked_oos(
    context,
) -> ExperimentResult:
    """
    Locked Momentum × SI incremental OOS analysis.

    Hypothesis:
        När en aktie har hög momentum + mycket hög short interest,
        är en ytterligare ökning av short interest associerad med
        högre risk för större nedgång?

    The hypothesis, regime and SI-change definition are fixed.
    No parameter selection or optimization is performed against
    the final 2026 OOS period.
    """

    # Discovery is explicitly limited to 2022-2024.
    discovery = _period_frame(
        context,
        pd.Timestamp(
            "2022-01-01"
        ),
        DISCOVERY_END,
    )

    # 2025 is the lock/replication period.
    lock = _period_frame(
        context,
        LOCK_START,
        LOCK_END,
    )

    # 2026 is the final untouched OOS period.
    oos = _period_frame(
        context,
        OOS_START,
        (
            pd.Timestamp(
                context.test_end
            )
            if context.test_end is not None
            else None
        ),
    )

    discovery_tables = _analyse_period(
        discovery,
        "discovery_2022_2024",
    )

    lock_tables = _analyse_period(
        lock,
        "lock_2025",
    )

    oos_tables = _analyse_period(
        oos,
        "final_oos_2026",
    )

    result = ExperimentResult(
        name="momentum_si_incremental_locked_oos",
        description=(
            "Låser hypotesen hög momentum + mycket hög SI "
            "+ ytterligare SI-ökning och testar den med "
            "fyra grupper i 2025 lock-period och helt "
            "orörd 2026 OOS."
        ),
    )

    tables = (
        (
            "discovery_groups",
            discovery_tables[
                "groups"
            ],
        ),
        (
            "lock_2025_groups",
            lock_tables[
                "groups"
            ],
        ),
        (
            "oos_2026_groups",
            oos_tables[
                "groups"
            ],
        ),
        (
            "discovery_effects",
            discovery_tables[
                "effects"
            ],
        ),
        (
            "lock_2025_effects",
            lock_tables[
                "effects"
            ],
        ),
        (
            "oos_2026_effects",
            oos_tables[
                "effects"
            ],
        ),
        (
            "discovery_returns",
            discovery_tables[
                "returns"
            ],
        ),
        (
            "lock_2025_returns",
            lock_tables[
                "returns"
            ],
        ),
        (
            "oos_2026_returns",
            oos_tables[
                "returns"
            ],
        ),
        (
            "discovery_sector_relative",
            discovery_tables[
                "sector_relative"
            ],
        ),
        (
            "lock_2025_sector_relative",
            lock_tables[
                "sector_relative"
            ],
        ),
        (
            "oos_2026_sector_relative",
            oos_tables[
                "sector_relative"
            ],
        ),
    )

    for (
        table_name,
        table,
    ) in tables:
        result.add_table(
            table_name,
            table,
        )

    result.add_metric(
        "hypothesis",
        (
            "När en aktie har hög momentum + mycket "
            "hög short interest, är en ytterligare "
            "ökning av short interest associerad med "
            "högre risk för större nedgång?"
        ),
    )

    result.add_metric(
        "locked_momentum_decile",
        MOMENTUM_DECILE,
    )

    result.add_metric(
        "locked_si_level_decile",
        SI_LEVEL_DECILE,
    )

    result.add_metric(
        "locked_high_si_change_definition",
        (
            "top 20% av positiva "
            "short-interest-förändringar "
            "per snapshot_date"
        ),
    )

    result.add_metric(
        "group_definition",
        {
            "A": (
                "outside 9x10 regime "
                "+ low SI change"
            ),
            "B": (
                "outside 9x10 regime "
                "+ high SI change"
            ),
            "C": (
                "9x10 regime "
                "+ low SI change"
            ),
            "D": (
                "9x10 regime "
                "+ high SI change"
            ),
        },
    )

    result.add_metric(
        "primary_effect",
        "D_minus_C",
    )

    result.add_metric(
        "discovery_period",
        "2022-01-01..2024-12-31",
    )

    result.add_metric(
        "lock_period",
        "2025-01-01..2025-12-31",
    )

    result.add_metric(
        "final_oos_period",
        "2026-01-01..test_end",
    )

    result.add_metric(
        "test_end",
        str(
            context.test_end
        ),
    )

    result.add_metric(
        "minimum_group_n",
        MIN_GROUP_N,
    )

    result.add_metric(
        "no_parameter_selection_in_test",
        True,
    )

    return result
