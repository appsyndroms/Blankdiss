from __future__ import annotations

import numpy as np
import pandas as pd

from .base import ExperimentResult
from ml.research.signals import build_signal


# ============================================================================
# SI CHANGE MECHANISM
# ============================================================================
#
# Descriptive mechanism analysis.
#
# The experiment asks:
#
#   What happens after an unusually large positive change in short interest,
#   and how does the relationship vary with prior momentum and SI level?
#
# IMPORTANT:
# - This is a separate experiment from momentum_si_incremental_locked_oos.
# - No parameters are selected against a future OOS period.
# - SI-change buckets are calculated cross-sectionally by snapshot date.
# - The high-SI-change definition is descriptive and uses the top 20% of
#   positive SI changes on each snapshot date.
#
# The experiment deliberately does NOT alter or reuse the locked hypothesis
# from momentum_si_incremental_locked_oos.
# ============================================================================


MOMENTUM_DECILES = tuple(range(1, 11))
SI_LEVEL_DECILES = tuple(range(1, 11))

HIGH_SI_CHANGE_FRACTION = 0.20

MIN_GROUP_N = 20

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

SI_CHANGE_BUCKETS = (
    "negative",
    "zero",
    "positive_0_50",
    "positive_50_80",
    "positive_80_95",
    "positive_95_100",
)


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
    events = _event_series(
        frame,
        return_column,
        threshold,
    )

    valid = events.dropna()

    if valid.empty:
        return float("nan")

    return float(
        valid.mean()
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


def _delta(
    high: float,
    low: float,
) -> float:
    if (
        pd.isna(high)
        or pd.isna(low)
    ):
        return float("nan")

    return float(
        high - low
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
# PREPARATION
# ============================================================================


def _cross_sectional_deciles(
    values: pd.Series,
    dates: pd.Series,
) -> pd.Series:
    numeric = pd.to_numeric(
        values,
        errors="coerce",
    )

    rank = numeric.groupby(
        dates
    ).rank(
        method="first",
        pct=True,
    )

    deciles = np.ceil(
        rank * 10.0
    )

    deciles = deciles.clip(
        lower=1,
        upper=10,
    )

    return deciles.astype(
        "Int64"
    )


def _prepare_frame(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    result = frame.copy()

    if "snapshot_date" not in result.columns:
        raise KeyError(
            "SI-change mechanism saknar "
            "'snapshot_date'."
        )

    result["snapshot_date"] = pd.to_datetime(
        result["snapshot_date"],
        errors="coerce",
    )

    result = result.loc[
        result["snapshot_date"].notna()
    ].copy()

    if result.empty:
        raise ValueError(
            "SI-change mechanism saknar "
            "giltiga snapshot-datum."
        )

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

    result["high_si_change"] = (
        _high_si_change_mask(
            result
        )
    )

    result["si_change_bucket"] = (
        _si_change_bucket(
            result
        )
    )

    return result


def _high_si_change_mask(
    frame: pd.DataFrame,
) -> pd.Series:
    """
    High SI change = top 20% of positive SI changes
    within each snapshot date.

    Negative and zero changes are never classified as
    high_si_change.
    """

    change = _numeric(
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
                - HIGH_SI_CHANGE_FRACTION
            )
        )
    )


def _si_change_bucket(
    frame: pd.DataFrame,
) -> pd.Series:
    """
    Creates descriptive SI-change buckets.

    Negative and zero changes are kept separately.

    Positive changes are ranked cross-sectionally by date:
      0-50%
      50-80%
      80-95%
      95-100%
    """

    change = _numeric(
        frame,
        "si_change",
    )

    result = pd.Series(
        pd.NA,
        index=frame.index,
        dtype="string",
    )

    result.loc[
        change < 0
    ] = "negative"

    result.loc[
        change == 0
    ] = "zero"

    positive = change.where(
        change > 0
    )

    positive_rank = positive.groupby(
        frame["snapshot_date"]
    ).rank(
        method="first",
        pct=True,
    )

    result.loc[
        positive.notna()
        & (positive_rank <= 0.50)
    ] = "positive_0_50"

    result.loc[
        positive.notna()
        & (positive_rank > 0.50)
        & (positive_rank <= 0.80)
    ] = "positive_50_80"

    result.loc[
        positive.notna()
        & (positive_rank > 0.80)
        & (positive_rank <= 0.95)
    ] = "positive_80_95"

    result.loc[
        positive.notna()
        & (positive_rank > 0.95)
    ] = "positive_95_100"

    return result


# ============================================================================
# TABLE 1
# SI CHANGE × MOMENTUM DECILE
# ============================================================================


def _si_change_by_momentum_decile(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict] = []

    for momentum_decile in MOMENTUM_DECILES:
        momentum_frame = frame.loc[
            frame["momentum_decile"]
            == momentum_decile
        ]

        for si_state in (
            "low_si_change",
            "high_si_change",
        ):
            if si_state == "high_si_change":
                subset = momentum_frame.loc[
                    momentum_frame[
                        "high_si_change"
                    ].fillna(False)
                ]
            else:
                subset = momentum_frame.loc[
                    ~momentum_frame[
                        "high_si_change"
                    ].fillna(False)
                ]

            for (
                target_name,
                return_column,
                threshold,
            ) in TARGETS:
                if (
                    return_column
                    not in subset.columns
                ):
                    continue

                valid_returns = _numeric(
                    subset,
                    return_column,
                ).notna()

                target_subset = subset.loc[
                    valid_returns
                ]

                rows.append(
                    {
                        "momentum_decile":
                            momentum_decile,
                        "si_change_state":
                            si_state,
                        "target":
                            target_name,
                        "n":
                            int(len(target_subset)),
                        "event_rate":
                            _event_rate(
                                target_subset,
                                return_column,
                                threshold,
                            ),
                        "mean_forward_return":
                            _mean(
                                target_subset,
                                return_column,
                            ),
                        "median_forward_return":
                            _median(
                                target_subset,
                                return_column,
                            ),
                        "mean_si_change":
                            _mean(
                                target_subset,
                                "si_change",
                            ),
                        "median_si_change":
                            _median(
                                target_subset,
                                "si_change",
                            ),
                    }
                )

    result = pd.DataFrame(
        rows
    )

    if result.empty:
        return result

    high = result.loc[
        result["si_change_state"]
        == "high_si_change"
    ][
        [
            "momentum_decile",
            "target",
            "n",
            "event_rate",
        ]
    ].rename(
        columns={
            "n": "high_n",
            "event_rate":
                "high_event_rate",
        }
    )

    low = result.loc[
        result["si_change_state"]
        == "low_si_change"
    ][
        [
            "momentum_decile",
            "target",
            "n",
            "event_rate",
        ]
    ].rename(
        columns={
            "n": "low_n",
            "event_rate":
                "low_event_rate",
        }
    )

    delta = high.merge(
        low,
        on=[
            "momentum_decile",
            "target",
        ],
        how="outer",
    )

    delta["high_minus_low_pp"] = (
        delta["high_event_rate"]
        - delta["low_event_rate"]
    ) * 100.0

    result = result.merge(
        delta[
            [
                "momentum_decile",
                "target",
                "high_n",
                "low_n",
                "high_event_rate",
                "low_event_rate",
                "high_minus_low_pp",
            ]
        ],
        on=[
            "momentum_decile",
            "target",
        ],
        how="left",
        suffixes=(
            "",
            "_comparison",
        ),
    )

    return result


# ============================================================================
# TABLE 2
# SI CHANGE QUANTILES
# ============================================================================


def _si_change_quantiles(
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
            if (
                return_column
                not in subset.columns
            ):
                continue

            valid_returns = _numeric(
                subset,
                return_column,
            ).notna()

            target_subset = subset.loc[
                valid_returns
            ]

            rows.append(
                {
                    "si_change_bucket":
                        bucket,
                    "target":
                        target_name,
                    "n":
                        int(len(target_subset)),
                    "event_rate":
                        _event_rate(
                            target_subset,
                            return_column,
                            threshold,
                        ),
                    "mean_forward_return":
                        _mean(
                            target_subset,
                            return_column,
                        ),
                    "median_forward_return":
                        _median(
                            target_subset,
                            return_column,
                        ),
                    "mean_si_change":
                        _mean(
                            target_subset,
                            "si_change",
                        ),
                    "median_si_change":
                        _median(
                            target_subset,
                            "si_change",
                        ),
                    "mean_momentum":
                        _mean(
                            target_subset,
                            "momentum",
                        ),
                    "mean_si_level":
                        _mean(
                            target_subset,
                            "si_level",
                        ),
                }
            )

    return pd.DataFrame(
        rows
    )


# ============================================================================
# TABLE 3
# MOMENTUM × SI LEVEL × HIGH SI CHANGE
# ============================================================================


def _high_si_change_momentum_surface(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict] = []

    for momentum_decile in MOMENTUM_DECILES:
        for si_level_decile in SI_LEVEL_DECILES:
            cell = frame.loc[
                (
                    frame["momentum_decile"]
                    == momentum_decile
                )
                & (
                    frame["si_level_decile"]
                    == si_level_decile
                )
            ]

            for si_state in (
                "low_si_change",
                "high_si_change",
            ):
                if si_state == "high_si_change":
                    subset = cell.loc[
                        cell[
                            "high_si_change"
                        ].fillna(False)
                    ]
                else:
                    subset = cell.loc[
                        ~cell[
                            "high_si_change"
                        ].fillna(False)
                    ]

                for (
                    target_name,
                    return_column,
                    threshold,
                ) in TARGETS:
                    if (
                        return_column
                        not in subset.columns
                    ):
                        continue

                    valid_returns = _numeric(
                        subset,
                        return_column,
                    ).notna()

                    target_subset = subset.loc[
                        valid_returns
                    ]

                    rows.append(
                        {
                            "momentum_decile":
                                momentum_decile,
                            "si_level_decile":
                                si_level_decile,
                            "si_change_state":
                                si_state,
                            "target":
                                target_name,
                            "n":
                                int(
                                    len(
                                        target_subset
                                    )
                                ),
                            "event_rate":
                                _event_rate(
                                    target_subset,
                                    return_column,
                                    threshold,
                                ),
                            "mean_forward_return":
                                _mean(
                                    target_subset,
                                    return_column,
                                ),
                            "mean_si_change":
                                _mean(
                                    target_subset,
                                    "si_change",
                                ),
                        }
                    )

    result = pd.DataFrame(
        rows
    )

    if result.empty:
        return result

    high = result.loc[
        result["si_change_state"]
        == "high_si_change"
    ][
        [
            "momentum_decile",
            "si_level_decile",
            "target",
            "n",
            "event_rate",
        ]
    ].rename(
        columns={
            "n": "high_n",
            "event_rate":
                "high_event_rate",
        }
    )

    low = result.loc[
        result["si_change_state"]
        == "low_si_change"
    ][
        [
            "momentum_decile",
            "si_level_decile",
            "target",
            "n",
            "event_rate",
        ]
    ].rename(
        columns={
            "n": "low_n",
            "event_rate":
                "low_event_rate",
        }
    )

    comparison = high.merge(
        low,
        on=[
            "momentum_decile",
            "si_level_decile",
            "target",
        ],
        how="outer",
    )

    comparison["high_minus_low_pp"] = (
        comparison["high_event_rate"]
        - comparison["low_event_rate"]
    ) * 100.0

    result = result.merge(
        comparison[
            [
                "momentum_decile",
                "si_level_decile",
                "target",
                "high_n",
                "low_n",
                "high_event_rate",
                "low_event_rate",
                "high_minus_low_pp",
            ]
        ],
        on=[
            "momentum_decile",
            "si_level_decile",
            "target",
        ],
        how="left",
        suffixes=(
            "",
            "_comparison",
        ),
    )

    return result


# ============================================================================
# SUMMARY METRICS
# ============================================================================


def _summary_metrics(
    frame: pd.DataFrame,
) -> dict:
    positive = _numeric(
        frame,
        "si_change",
    )

    positive = positive.loc[
        positive > 0
    ].dropna()

    high = frame.loc[
        frame["high_si_change"].fillna(False)
    ]

    low = frame.loc[
        ~frame["high_si_change"].fillna(False)
    ]

    return {
        "test_rows":
            int(len(frame)),
        "positive_si_change_n":
            int(len(positive)),
        "high_si_change_n":
            int(len(high)),
        "low_si_change_n":
            int(len(low)),
        "high_si_change_fraction":
            _safe_fraction(
                len(high),
                len(frame),
            ),
        "positive_si_change_fraction":
            _safe_fraction(
                len(positive),
                len(frame),
            ),
        "high_si_change_definition":
            (
                "Top 20% of positive SI changes "
                "cross-sectionally within each "
                "snapshot date."
            ),
        "momentum_deciles":
            10,
        "si_level_deciles":
            10,
        "minimum_group_n_reference":
            MIN_GROUP_N,
    }


# ============================================================================
# PUBLIC ENTRY POINT
# ============================================================================


def run_si_change_mechanism(
    context,
) -> ExperimentResult:
    """
    Descriptive analysis of the relationship between
    short-interest changes, prior momentum and SI level.

    The analysis is performed on context.test only.

    No threshold is learned from future observations and
    no parameter is optimized against the test period.
    """

    test = _prepare_frame(
        context.test
    )

    if test.empty:
        raise ValueError(
            "SI-change mechanism fick "
            "ett tomt testdataset."
        )

    momentum_rows = (
        _si_change_by_momentum_decile(
            test
        )
    )

    quantile_rows = (
        _si_change_quantiles(
            test
        )
    )

    surface_rows = (
        _high_si_change_momentum_surface(
            test
        )
    )

    if momentum_rows.empty:
        raise ValueError(
            "SI-change mechanism producerade "
            "ingen momentumanalys."
        )

    if quantile_rows.empty:
        raise ValueError(
            "SI-change mechanism producerade "
            "ingen SI-change-quantilanalys."
        )

    if surface_rows.empty:
        raise ValueError(
            "SI-change mechanism producerade "
            "ingen momentum × SI-nivå-yta."
        )

    result = ExperimentResult(
        name="si_change_mechanism",
        description=(
            "Deskriptiv mekanismanalys av hur "
            "ovanligt stora förändringar i short "
            "interest hänger samman med efterföljande "
            "downside-risk, och hur sambandet varierar "
            "med tidigare momentum och SI-nivå."
        ),
    )

    result.add_table(
        "si_change_by_momentum_decile",
        momentum_rows,
    )

    result.add_table(
        "si_change_quantiles",
        quantile_rows,
    )

    result.add_table(
        "high_si_change_momentum_surface",
        surface_rows,
    )

    for name, value in _summary_metrics(
        test
    ).items():
        result.add_metric(
            name,
            value,
        )

    result.add_metric(
        "test_start",
        str(
            test["snapshot_date"].min().date()
        ),
    )

    result.add_metric(
        "test_end",
        str(
            test["snapshot_date"].max().date()
        ),
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
        "descriptive_mechanism",
    )

    result.add_metric(
        "parameter_selection",
        "none",
    )

    return result
