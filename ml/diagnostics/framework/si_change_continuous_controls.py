from __future__ import annotations

import numpy as np
import pandas as pd

from .base import ExperimentResult
from ml.research.signals import build_signal


# ============================================================================
# SI CHANGE — CONTINUOUS + CONTROLS
# ============================================================================
#
# Research question:
#
#   Har förändringen i short interest egen information om framtida
#   downside-risk efter kontroll för tidigare momentum och SI-nivå?
#
# This experiment is deliberately separate from:
#
#   - si_change_mechanism
#   - momentum_si_incremental_locked_oos
#
# The purpose is to move from a descriptive "top 20%" analysis to a
# continuous SI-change analysis with explicit controls.
#
# Important methodological rules:
#
# 1. SI-change cutoffs are learned only from context.pretest.
# 2. The resulting cutoffs are then frozen for context.test.
# 3. Momentum and SI-level deciles are calculated cross-sectionally
#    within each snapshot date. They do not use future information.
# 4. The joint momentum × SI-level control uses the same cells for the
#    high- and low-SI-change groups.
# 5. No test-period parameter is selected after looking at test results.
#
# The diagnostic runner already provides separate walk-forward windows.
# Therefore:
#
#   window 1 -> test is 2025
#   window 2 -> test is 2026
#
# The experiment itself does not manually choose between those periods.
# ============================================================================


MOMENTUM_DECILES = tuple(range(1, 11))
SI_LEVEL_DECILES = tuple(range(1, 11))

# Fixed pretest quantiles used to create continuous SI-change buckets.
#
# The 95-100 bucket is deliberately separated because the earlier
# mechanism analysis suggested that the extreme positive tail may behave
# differently from the rest of the positive changes.
SI_CHANGE_QUANTILES = (
    0.20,
    0.40,
    0.60,
    0.80,
    0.95,
)

SI_CHANGE_BUCKETS = (
    "q00_20",
    "q20_40",
    "q40_60",
    "q60_80",
    "q80_95",
    "q95_100",
)

TARGETS = (
    (
        "down_5pct_5d",
        "forward_return_5d",
        -0.05,
    ),
    (
        "down_7pct_5d",
        "forward_return_5d",
        -0.07,
    ),
    (
        "down_10pct_10d",
        "forward_return_10d",
        -0.10,
    ),
)

# Minimum observations required on each side of a control cell before
# the cell is used in the standardized comparison.
MIN_CELL_GROUP_N = 10


# ============================================================================
# GENERIC HELPERS
# ============================================================================


def _numeric(
    frame: pd.DataFrame,
    column: str,
) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(
            np.nan,
            index=frame.index,
            dtype=float,
        )

    return pd.to_numeric(
        frame[column],
        errors="coerce",
    ).replace(
        [np.inf, -np.inf],
        np.nan,
    )


def _event_series(
    frame: pd.DataFrame,
    return_column: str,
    threshold: float,
) -> pd.Series:
    values = _numeric(
        frame,
        return_column,
    )

    return (
        values <= threshold
    )


def _event_rate(
    frame: pd.DataFrame,
    return_column: str,
    threshold: float,
) -> float:
    values = _numeric(
        frame,
        return_column,
    )

    valid = values.dropna()

    if valid.empty:
        return float("nan")

    return float(
        (valid <= threshold).mean()
    )


def _mean(
    frame: pd.DataFrame,
    column: str,
) -> float:
    values = _numeric(
        frame,
        column,
    ).dropna()

    if values.empty:
        return float("nan")

    return float(
        values.mean()
    )


def _median(
    frame: pd.DataFrame,
    column: str,
) -> float:
    values = _numeric(
        frame,
        column,
    ).dropna()

    if values.empty:
        return float("nan")

    return float(
        values.median()
    )


def _safe_delta_pp(
    high_rate: float,
    low_rate: float,
) -> float:
    if (
        pd.isna(high_rate)
        or pd.isna(low_rate)
    ):
        return float("nan")

    return float(
        (high_rate - low_rate) * 100.0
    )


def _safe_fraction(
    numerator: int,
    denominator: int,
) -> float:
    if denominator <= 0:
        return float("nan")

    return float(
        numerator / denominator
    )


# ============================================================================
# PRETEST THRESHOLDS
# ============================================================================


def _pretest_si_change_cutoffs(
    context,
) -> dict[float, float]:
    """
    Learn SI-change cutoffs from PRETEST only.

    These cutoffs are frozen and subsequently applied to TEST.

    This is important because using test quantiles would make the
    definition adapt to the very period we are trying to evaluate.
    """

    pretest = context.pretest

    values = _numeric(
        pretest,
        "short_interest_delta_pp",
    ).dropna()

    if values.empty:
        raise ValueError(
            "Kan inte beräkna SI-change-cutoffs: "
            "short_interest_delta_pp saknas eller är tom."
        )

    cutoffs: dict[float, float] = {}

    for q in SI_CHANGE_QUANTILES:
        value = values.quantile(q)

        if pd.isna(value):
            raise ValueError(
                f"SI-change cutoff blev NaN för q={q}."
            )

        cutoffs[q] = float(value)

    return cutoffs


# ============================================================================
# CROSS-SECTIONAL CONTROLS
# ============================================================================


def _cross_sectional_deciles(
    values: pd.Series,
    dates: pd.Series,
) -> pd.Series:
    """
    Assign deciles independently for every snapshot date.

    Missing values remain missing.
    """

    numeric = pd.to_numeric(
        values,
        errors="coerce",
    )

    valid = numeric.notna()

    ranks = pd.Series(
        np.nan,
        index=values.index,
        dtype=float,
    )

    if valid.any():
        ranks.loc[valid] = (
            numeric.loc[valid]
            .groupby(
                dates.loc[valid]
            )
            .rank(
                method="first",
                pct=True,
            )
        )

    deciles = np.ceil(
        ranks * 10.0
    )

    deciles = deciles.clip(
        lower=1,
        upper=10,
    )

    return deciles.astype(
        "Int64"
    )


def _apply_si_change_buckets(
    frame: pd.DataFrame,
    cutoffs: dict[float, float],
) -> pd.DataFrame:
    """
    Apply frozen pretest cutoffs to the supplied frame.

    The bucket definitions therefore remain identical between
    the different walk-forward test periods.
    """

    result = frame.copy()

    change = _numeric(
        result,
        "si_change",
    )

    q20 = cutoffs[0.20]
    q40 = cutoffs[0.40]
    q60 = cutoffs[0.60]
    q80 = cutoffs[0.80]
    q95 = cutoffs[0.95]

    result["si_change_bucket"] = pd.Series(
        pd.NA,
        index=result.index,
        dtype="string",
    )

    result.loc[
        change <= q20,
        "si_change_bucket",
    ] = "q00_20"

    result.loc[
        (change > q20)
        & (change <= q40),
        "si_change_bucket",
    ] = "q20_40"

    result.loc[
        (change > q40)
        & (change <= q60),
        "si_change_bucket",
    ] = "q40_60"

    result.loc[
        (change > q60)
        & (change <= q80),
        "si_change_bucket",
    ] = "q60_80"

    result.loc[
        (change > q80)
        & (change <= q95),
        "si_change_bucket",
    ] = "q80_95"

    result.loc[
        change > q95,
        "si_change_bucket",
    ] = "q95_100"

    return result


def _prepare_test_frame(
    context,
    cutoffs: dict[float, float],
) -> pd.DataFrame:
    """
    Build all test-period variables.

    Signals are always reconstructed from the canonical signal definitions.
    """

    result = context.test.copy()

    if result.empty:
        raise ValueError(
            "SI-change continuous controls fick "
            "ett tomt testdataset."
        )

    if "snapshot_date" not in result.columns:
        raise ValueError(
            "SI-change continuous controls kräver "
            "'snapshot_date'."
        )

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
        _cross_sectional_deciles(
            result["momentum"],
            result["snapshot_date"],
        )
    )

    result["si_level_decile"] = (
        _cross_sectional_deciles(
            result["si_level"],
            result["snapshot_date"],
        )
    )

    # Cross-sectional percentile is useful for interpreting the continuous
    # SI-change signal even when the absolute distribution changes over time.
    result["si_change_percentile"] = (
        _cross_sectional_deciles(
            result["si_change"],
            result["snapshot_date"],
        )
        .astype("Float64")
        / 10.0
    )

    result = _apply_si_change_buckets(
        result,
        cutoffs,
    )

    return result


# ============================================================================
# TABLE 1
# CONTINUOUS SI-CHANGE BUCKETS
# ============================================================================


def _continuous_si_change_table(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict] = []

    for bucket in SI_CHANGE_BUCKETS:
        subset = frame.loc[
            frame["si_change_bucket"]
            == bucket
        ]

        for (
            target_name,
            return_column,
            threshold,
        ) in TARGETS:
            valid = _numeric(
                subset,
                return_column,
            ).notna()

            target_subset = subset.loc[
                valid
            ]

            rows.append(
                {
                    "si_change_bucket": bucket,
                    "target": target_name,
                    "n": int(
                        len(target_subset)
                    ),
                    "event_rate": _event_rate(
                        target_subset,
                        return_column,
                        threshold,
                    ),
                    "mean_forward_return": _mean(
                        target_subset,
                        return_column,
                    ),
                    "median_forward_return": _median(
                        target_subset,
                        return_column,
                    ),
                    "mean_si_change_pp": _mean(
                        target_subset,
                        "si_change",
                    ),
                    "median_si_change_pp": _median(
                        target_subset,
                        "si_change",
                    ),
                    "mean_momentum": _mean(
                        target_subset,
                        "momentum",
                    ),
                    "mean_si_level": _mean(
                        target_subset,
                        "si_level",
                    ),
                }
            )

    return pd.DataFrame(
        rows
    )


# ============================================================================
# TABLE 2
# HIGH VS LOW — RAW
# ============================================================================


def _raw_high_low_table(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """
    Raw comparison:

        high = q80-q100
        low  = q00-q80

    The q80 cutoff was learned exclusively from pretest.
    """

    high = frame.loc[
        frame["si_change_bucket"].isin(
            {
                "q80_95",
                "q95_100",
            }
        )
    ]

    low = frame.loc[
        frame["si_change_bucket"].isin(
            {
                "q00_20",
                "q20_40",
                "q40_60",
                "q60_80",
            }
        )
    ]

    rows: list[dict] = []

    for (
        target_name,
        return_column,
        threshold,
    ) in TARGETS:
        high_rate = _event_rate(
            high,
            return_column,
            threshold,
        )

        low_rate = _event_rate(
            low,
            return_column,
            threshold,
        )

        rows.append(
            {
                "target": target_name,
                "high_n": int(len(high)),
                "low_n": int(len(low)),
                "high_event_rate": high_rate,
                "low_event_rate": low_rate,
                "high_minus_low_pp": _safe_delta_pp(
                    high_rate,
                    low_rate,
                ),
                "high_mean_si_change_pp": _mean(
                    high,
                    "si_change",
                ),
                "low_mean_si_change_pp": _mean(
                    low,
                    "si_change",
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================================
# TABLE 3
# STRATIFIED HIGH VS LOW
# ============================================================================


def _stratified_high_low(
    frame: pd.DataFrame,
    *,
    control: str,
) -> pd.DataFrame:
    """
    Compare high vs low SI-change after stratifying by a control.

    control:
        "momentum"
        "si_level"
        "joint"

    For the joint control, observations are grouped into:

        momentum_decile × si_level_decile

    Only cells with sufficient observations in both high and low groups
    are included.

    The final adjusted rates use identical cell weights for high and low.
    This is a standardization approach rather than a fitted regression.
    """

    if control == "momentum":
        control_columns = [
            "momentum_decile",
        ]
    elif control == "si_level":
        control_columns = [
            "si_level_decile",
        ]
    elif control == "joint":
        control_columns = [
            "momentum_decile",
            "si_level_decile",
        ]
    else:
        raise ValueError(
            f"Okänd control: {control}"
        )

    high = frame.loc[
        frame["si_change_bucket"].isin(
            {
                "q80_95",
                "q95_100",
            }
        )
    ].copy()

    low = frame.loc[
        frame["si_change_bucket"].isin(
            {
                "q00_20",
                "q20_40",
                "q40_60",
                "q60_80",
            }
        )
    ].copy()

    rows: list[dict] = []

    for (
        target_name,
        return_column,
        threshold,
    ) in TARGETS:
        high_valid = high.loc[
            _numeric(
                high,
                return_column,
            ).notna()
        ].copy()

        low_valid = low.loc[
            _numeric(
                low,
                return_column,
            ).notna()
        ].copy()

        high_cell = (
            high_valid
            .groupby(control_columns)
            .agg(
                high_n=(
                    return_column,
                    "count",
                ),
            )
        )

        low_cell = (
            low_valid
            .groupby(control_columns)
            .agg(
                low_n=(
                    return_column,
                    "count",
                ),
            )
        )

        high_rates = (
            high_valid
            .assign(
                _event=_event_series(
                    high_valid,
                    return_column,
                    threshold,
                ).astype(float)
            )
            .groupby(control_columns)["_event"]
            .mean()
            .rename(
                "high_event_rate"
            )
        )

        low_rates = (
            low_valid
            .assign(
                _event=_event_series(
                    low_valid,
                    return_column,
                    threshold,
                ).astype(float)
            )
            .groupby(control_columns)["_event"]
            .mean()
            .rename(
                "low_event_rate"
            )
        )

        cells = (
            high_cell
            .join(
                low_cell,
                how="inner",
            )
            .join(
                high_rates,
                how="inner",
            )
            .join(
                low_rates,
                how="inner",
            )
            .reset_index()
        )

        cells = cells.loc[
            (cells["high_n"] >= MIN_CELL_GROUP_N)
            & (cells["low_n"] >= MIN_CELL_GROUP_N)
        ].copy()

        if cells.empty:
            rows.append(
                {
                    "control": control,
                    "target": target_name,
                    "cells_used": 0,
                    "high_n": 0,
                    "low_n": 0,
                    "adjusted_high_event_rate": float(
                        "nan"
                    ),
                    "adjusted_low_event_rate": float(
                        "nan"
                    ),
                    "adjusted_high_minus_low_pp": float(
                        "nan"
                    ),
                }
            )

            continue

        cells["cell_n"] = (
            cells["high_n"]
            + cells["low_n"]
        )

        total_n = float(
            cells["cell_n"].sum()
        )

        cells["weight"] = (
            cells["cell_n"]
            / total_n
        )

        adjusted_high = float(
            (
                cells["weight"]
                * cells["high_event_rate"]
            ).sum()
        )

        adjusted_low = float(
            (
                cells["weight"]
                * cells["low_event_rate"]
            ).sum()
        )

        rows.append(
            {
                "control": control,
                "target": target_name,
                "cells_used": int(
                    len(cells)
                ),
                "high_n": int(
                    high_valid.shape[0]
                ),
                "low_n": int(
                    low_valid.shape[0]
                ),
                "adjusted_high_event_rate": adjusted_high,
                "adjusted_low_event_rate": adjusted_low,
                "adjusted_high_minus_low_pp": (
                    (adjusted_high - adjusted_low)
                    * 100.0
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================================
# TABLE 4
# JOINT CELL DETAIL
# ============================================================================


def _joint_cell_detail(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """
    Detailed momentum × SI-level control table.

    This is deliberately retained as a diagnostic table, not as a
    parameter-selection mechanism.
    """

    high = frame.loc[
        frame["si_change_bucket"].isin(
            {
                "q80_95",
                "q95_100",
            }
        )
    ].copy()

    low = frame.loc[
        frame["si_change_bucket"].isin(
            {
                "q00_20",
                "q20_40",
                "q40_60",
                "q60_80",
            }
        )
    ].copy()

    rows: list[dict] = []

    for momentum_decile in MOMENTUM_DECILES:
        for si_level_decile in SI_LEVEL_DECILES:
            high_cell = high.loc[
                (
                    high["momentum_decile"]
                    == momentum_decile
                )
                & (
                    high["si_level_decile"]
                    == si_level_decile
                )
            ]

            low_cell = low.loc[
                (
                    low["momentum_decile"]
                    == momentum_decile
                )
                & (
                    low["si_level_decile"]
                    == si_level_decile
                )
            ]

            for (
                target_name,
                return_column,
                threshold,
            ) in TARGETS:
                high_rate = _event_rate(
                    high_cell,
                    return_column,
                    threshold,
                )

                low_rate = _event_rate(
                    low_cell,
                    return_column,
                    threshold,
                )

                rows.append(
                    {
                        "momentum_decile": momentum_decile,
                        "si_level_decile": si_level_decile,
                        "target": target_name,
                        "high_n": int(
                            _numeric(
                                high_cell,
                                return_column,
                            ).notna().sum()
                        ),
                        "low_n": int(
                            _numeric(
                                low_cell,
                                return_column,
                            ).notna().sum()
                        ),
                        "high_event_rate": high_rate,
                        "low_event_rate": low_rate,
                        "high_minus_low_pp": _safe_delta_pp(
                            high_rate,
                            low_rate,
                        ),
                    }
                )

    return pd.DataFrame(
        rows
    )


# ============================================================================
# TABLE 5
# CONTINUOUS SI-CHANGE WITH MOMENTUM CONTROL
# ============================================================================


def _bucket_by_momentum(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """
    Shows the SI-change bucket effect separately inside each momentum
    decile.

    This is a diagnostic table rather than a parameter search.
    """

    rows: list[dict] = []

    for momentum_decile in MOMENTUM_DECILES:
        subset = frame.loc[
            frame["momentum_decile"]
            == momentum_decile
        ]

        for bucket in SI_CHANGE_BUCKETS:
            bucket_frame = subset.loc[
                subset["si_change_bucket"]
                == bucket
            ]

            for (
                target_name,
                return_column,
                threshold,
            ) in TARGETS:
                rows.append(
                    {
                        "momentum_decile": momentum_decile,
                        "si_change_bucket": bucket,
                        "target": target_name,
                        "n": int(
                            _numeric(
                                bucket_frame,
                                return_column,
                            ).notna().sum()
                        ),
                        "event_rate": _event_rate(
                            bucket_frame,
                            return_column,
                            threshold,
                        ),
                        "mean_si_change_pp": _mean(
                            bucket_frame,
                            "si_change",
                        ),
                        "mean_si_level": _mean(
                            bucket_frame,
                            "si_level",
                        ),
                    }
                )

    return pd.DataFrame(
        rows
    )


# ============================================================================
# SUMMARY
# ============================================================================


def _summary_metrics(
    context,
    frame: pd.DataFrame,
    cutoffs: dict[float, float],
) -> dict:
    return {
        "test_rows": int(
            len(frame)
        ),
        "test_start": str(
            frame["snapshot_date"].min().date()
        ),
        "test_end": str(
            frame["snapshot_date"].max().date()
        ),
        "pretest_start": str(
            context.pretest["snapshot_date"].min().date()
        ),
        "pretest_end": str(
            context.pretest["snapshot_date"].max().date()
        ),
        "si_change_cutoffs_learned_from": "pretest_only",
        "si_change_cutoffs_pp": {
            str(q): value
            for q, value in cutoffs.items()
        },
        "high_si_change_definition": (
            "SI-change above the frozen pretest "
            "80th percentile."
        ),
        "low_si_change_definition": (
            "SI-change at or below the frozen pretest "
            "80th percentile."
        ),
        "control_variables": [
            "price_momentum_5d",
            "short_interest_level",
        ],
        "control_method": (
            "cross-sectional deciles by snapshot date "
            "with stratified standardization"
        ),
        "minimum_cell_group_n": MIN_CELL_GROUP_N,
        "parameter_selection": "none_in_test",
    }


# ============================================================================
# PUBLIC ENTRY POINT
# ============================================================================


def run_si_change_continuous_controls(
    context,
) -> ExperimentResult:
    """
    Test whether SI-change contains incremental information after
    accounting for momentum and SI level.

    All SI-change bucket thresholds are learned exclusively from
    context.pretest and frozen before context.test is evaluated.
    """

    cutoffs = _pretest_si_change_cutoffs(
        context
    )

    test = _prepare_test_frame(
        context,
        cutoffs,
    )

    if test.empty:
        raise ValueError(
            "SI-change continuous controls fick "
            "ett tomt testdataset efter preparation."
        )

    continuous = _continuous_si_change_table(
        test
    )

    raw = _raw_high_low_table(
        test
    )

    momentum_control = _stratified_high_low(
        test,
        control="momentum",
    )

    si_level_control = _stratified_high_low(
        test,
        control="si_level",
    )

    joint_control = _stratified_high_low(
        test,
        control="joint",
    )

    joint_detail = _joint_cell_detail(
        test
    )

    momentum_buckets = _bucket_by_momentum(
        test
    )

    result = ExperimentResult(
        name="si_change_continuous_controls",
        description=(
            "Testar om short-interest-förändringen har "
            "inkrementell information om downside-risk "
            "efter kontroll för tidigare momentum och "
            "SI-nivå. SI-change-gränser lärs endast på "
            "pretest och fryses före testperioden."
        ),
    )

    result.add_table(
        "si_change_continuous",
        continuous,
    )

    result.add_table(
        "raw_high_vs_low",
        raw,
    )

    result.add_table(
        "momentum_control",
        momentum_control,
    )

    result.add_table(
        "si_level_control",
        si_level_control,
    )

    result.add_table(
        "joint_momentum_si_control",
        joint_control,
    )

    result.add_table(
        "joint_momentum_si_cells",
        joint_detail,
    )

    result.add_table(
        "si_change_by_momentum_decile",
        momentum_buckets,
    )

    for name, value in _summary_metrics(
        context,
        test,
        cutoffs,
    ).items():
        result.add_metric(
            name,
            value,
        )

    result.add_metric(
        "targets",
        [
            target_name
            for (
                target_name,
                _,
                _,
            ) in TARGETS
        ],
    )

    result.add_metric(
        "analysis_type",
        "continuous_incremental_control",
    )

    result.add_metric(
        "uses_test_for_parameter_selection",
        False,
    )

    result.add_metric(
        "uses_future_information_in_features",
        False,
    )

    result.add_metric(
        "research_question",
        (
            "Har SI-förändringen egen information om "
            "framtida downside-risk efter kontroll för "
            "momentum och SI-nivå?"
        ),
    )

    return result
