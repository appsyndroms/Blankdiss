from __future__ import annotations

import numpy as np
import pandas as pd

from .base import ExperimentResult
from .si_event_risk_signal_anatomy import (
    LOCKED_EVENT_THRESHOLD,
    LOCKED_HORIZON,
    _prepare_signal,
)
from .event_risk_grid import (
    LOCKED_DOWNSIDE_TARGET,
    LOCKED_RISK_CUTOFF,
    LOCKED_SI_CHANGE_CUTOFF,
    _numeric,
)


MOMENTUM_COLUMN = "price_return_20d"

PRIMARY_GROUPS = (
    "B_high_event_high_si_nonnegative_momentum",
    "D_high_event_low_si_nonnegative_momentum",
)

CONTEXT_COLUMNS = (
    "event_score",
    "short_interest_pct_change",
    "short_interest_pct",
    "price_return_5d",
    "price_return_20d",
    "price_return_60d",
    "price_volatility_5d",
    "price_volatility_10d",
    "price_volatility_20d",
    "price_distance_from_5d_high",
    "price_distance_from_10d_high",
    "price_distance_from_20d_high",
    "price_distance_from_60d_high",
)

OPTIONAL_MOMENTUM_COLUMNS = (
    "price_return_1d",
    "price_return_3d",
    "price_return_10d",
    "price_return_30d",
)

OPTIONAL_VOLUME_COLUMNS = (
    "volume_ratio_5d",
    "volume_ratio_10d",
    "volume_ratio_20d",
    "price_volume_ratio_20d",
)


def _safe_values(
    frame: pd.DataFrame,
    column: str,
) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(
            dtype=float,
            index=frame.index,
        )

    return _numeric(
        frame,
        column,
    )


def _safe_mean(
    frame: pd.DataFrame,
    column: str,
) -> float:
    values = _safe_values(
        frame,
        column,
    ).dropna()

    if values.empty:
        return float("nan")

    return float(values.mean())


def _safe_median(
    frame: pd.DataFrame,
    column: str,
) -> float:
    values = _safe_values(
        frame,
        column,
    ).dropna()

    if values.empty:
        return float("nan")

    return float(values.median())


def _safe_quantile(
    frame: pd.DataFrame,
    column: str,
    quantile: float,
) -> float:
    values = _safe_values(
        frame,
        column,
    ).dropna()

    if values.empty:
        return float("nan")

    return float(
        values.quantile(quantile)
    )


def _downside_rate(
    frame: pd.DataFrame,
    threshold: float,
) -> float:
    values = _safe_values(
        frame,
        "forward_return_1d",
    ).dropna()

    if values.empty:
        return float("nan")

    return float(
        (
            values <= threshold
        ).mean()
    )


def _return_difference(
    frame: pd.DataFrame,
    long_column: str,
    short_column: str,
) -> float:
    long_values = _safe_values(
        frame,
        long_column,
    )

    short_values = _safe_values(
        frame,
        short_column,
    )

    values = pd.concat(
        [
            long_values.rename("long"),
            short_values.rename("short"),
        ],
        axis=1,
    ).dropna()

    if values.empty:
        return float("nan")

    return float(
        (
            values["long"]
            - values["short"]
        ).mean()
    )


def _prepare_groups(
    test: pd.DataFrame,
) -> pd.DataFrame:
    local = test.copy()

    momentum = _safe_values(
        local,
        MOMENTUM_COLUMN,
    )

    valid_momentum = momentum.notna()

    local["momentum_nonnegative"] = (
        valid_momentum
        & (momentum >= 0)
    )

    local["momentum_negative"] = (
        valid_momentum
        & (momentum < 0)
    )

    local["context_group"] = np.select(
        [
            (
                local["high_event_risk"]
                & local["high_si_change"]
                & local["momentum_nonnegative"]
            ),
            (
                local["high_event_risk"]
                & ~local["high_si_change"]
                & local["momentum_nonnegative"]
            ),
            (
                local["high_event_risk"]
                & local["high_si_change"]
                & local["momentum_negative"]
            ),
            (
                local["high_event_risk"]
                & ~local["high_si_change"]
                & local["momentum_negative"]
            ),
        ],
        [
            PRIMARY_GROUPS[0],
            PRIMARY_GROUPS[1],
            "A_high_event_high_si_negative_momentum",
            "C_high_event_low_si_negative_momentum",
        ],
        default="outside_analysis",
    )

    return local


def _context_table(
    test: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict] = []

    groups = (
        PRIMARY_GROUPS[0],
        PRIMARY_GROUPS[1],
        "A_high_event_high_si_negative_momentum",
        "C_high_event_low_si_negative_momentum",
    )

    for group in groups:
        frame = test[
            test["context_group"]
            == group
        ].copy()

        row = {
            "group": group,
            "n": int(len(frame)),
        }

        for column in CONTEXT_COLUMNS:
            row[
                f"mean_{column}"
            ] = _safe_mean(
                frame,
                column,
            )

            row[
                f"median_{column}"
            ] = _safe_median(
                frame,
                column,
            )

        for column in OPTIONAL_MOMENTUM_COLUMNS:
            if column in frame.columns:
                row[
                    f"mean_{column}"
                ] = _safe_mean(
                    frame,
                    column,
                )

                row[
                    f"median_{column}"
                ] = _safe_median(
                    frame,
                    column,
                )

        for column in OPTIONAL_VOLUME_COLUMNS:
            if column in frame.columns:
                row[
                    f"mean_{column}"
                ] = _safe_mean(
                    frame,
                    column,
                )

                row[
                    f"median_{column}"
                ] = _safe_median(
                    frame,
                    column,
                )

        row[
            "down_3pct_1d_rate"
        ] = _downside_rate(
            frame,
            -0.03,
        )

        row[
            "down_5pct_1d_rate"
        ] = _downside_rate(
            frame,
            -0.05,
        )

        row[
            "down_7pct_1d_rate"
        ] = _downside_rate(
            frame,
            LOCKED_DOWNSIDE_TARGET,
        )

        row[
            "down_10pct_1d_rate"
        ] = _downside_rate(
            frame,
            -0.10,
        )

        row[
            "mean_forward_return_1d"
        ] = _safe_mean(
            frame,
            "forward_return_1d",
        )

        row[
            "median_forward_return_1d"
        ] = _safe_median(
            frame,
            "forward_return_1d",
        )

        row[
            "mean_min_return_5d"
        ] = _safe_mean(
            frame,
            "min_return_5d",
        )

        row[
            "median_min_return_5d"
        ] = _safe_median(
            frame,
            "min_return_5d",
        )

        row[
            "mean_max_return_5d"
        ] = _safe_mean(
            frame,
            "max_return_5d",
        )

        row[
            "median_max_return_5d"
        ] = _safe_median(
            frame,
            "max_return_5d",
        )

        rows.append(row)

    return pd.DataFrame(rows)


def _primary_difference_table(
    test: pd.DataFrame,
) -> pd.DataFrame:
    high_si = test[
        test["context_group"]
        == PRIMARY_GROUPS[0]
    ].copy()

    low_si = test[
        test["context_group"]
        == PRIMARY_GROUPS[1]
    ].copy()

    variables = list(
        dict.fromkeys(
            CONTEXT_COLUMNS
            + OPTIONAL_MOMENTUM_COLUMNS
            + OPTIONAL_VOLUME_COLUMNS
        )
    )

    rows: list[dict] = []

    for column in variables:
        if (
            column not in high_si.columns
            and column not in low_si.columns
        ):
            continue

        high_values = _safe_values(
            high_si,
            column,
        ).dropna()

        low_values = _safe_values(
            low_si,
            column,
        ).dropna()

        if (
            high_values.empty
            and low_values.empty
        ):
            continue

        high_mean = (
            float(high_values.mean())
            if not high_values.empty
            else float("nan")
        )

        low_mean = (
            float(low_values.mean())
            if not low_values.empty
            else float("nan")
        )

        rows.append(
            {
                "variable": column,
                "high_si_n": int(
                    len(high_values)
                ),
                "low_si_n": int(
                    len(low_values)
                ),
                "high_si_mean": high_mean,
                "low_si_mean": low_mean,
                "difference_high_minus_low": (
                    high_mean
                    - low_mean
                    if np.isfinite(high_mean)
                    and np.isfinite(low_mean)
                    else float("nan")
                ),
                "high_si_median": (
                    float(high_values.median())
                    if not high_values.empty
                    else float("nan")
                ),
                "low_si_median": (
                    float(low_values.median())
                    if not low_values.empty
                    else float("nan")
                ),
            }
        )

    return pd.DataFrame(rows)


def _momentum_structure_table(
    test: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict] = []

    groups = (
        PRIMARY_GROUPS[0],
        PRIMARY_GROUPS[1],
    )

    for group in groups:
        frame = test[
            test["context_group"]
            == group
        ].copy()

        row = {
            "group": group,
            "n": int(len(frame)),
        }

        for column in (
            "price_return_5d",
            "price_return_20d",
            "price_return_60d",
        ):
            row[
                f"mean_{column}"
            ] = _safe_mean(
                frame,
                column,
            )

        if (
            "price_return_5d" in frame.columns
            and "price_return_20d" in frame.columns
        ):
            row[
                "return_acceleration_5d_vs_20d"
            ] = _return_difference(
                frame,
                "price_return_5d",
                "price_return_20d",
            )

        if (
            "price_return_20d" in frame.columns
            and "price_return_60d" in frame.columns
        ):
            row[
                "return_acceleration_20d_vs_60d"
            ] = _return_difference(
                frame,
                "price_return_20d",
                "price_return_60d",
            )

        for column in (
            "price_distance_from_5d_high",
            "price_distance_from_10d_high",
            "price_distance_from_20d_high",
            "price_distance_from_60d_high",
        ):
            if column in frame.columns:
                row[
                    f"mean_{column}"
                ] = _safe_mean(
                    frame,
                    column,
                )

        rows.append(row)

    return pd.DataFrame(rows)


def _risk_and_si_structure_table(
    test: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict] = []

    for group in PRIMARY_GROUPS:
        frame = test[
            test["context_group"]
            == group
        ].copy()

        rows.append(
            {
                "group": group,
                "n": int(len(frame)),
                "event_score_mean": _safe_mean(
                    frame,
                    "event_score",
                ),
                "event_score_median": _safe_median(
                    frame,
                    "event_score",
                ),
                "event_score_p25": _safe_quantile(
                    frame,
                    "event_score",
                    0.25,
                ),
                "event_score_p75": _safe_quantile(
                    frame,
                    "event_score",
                    0.75,
                ),
                "si_change_mean_pp": _safe_mean(
                    frame,
                    "short_interest_pct_change",
                ),
                "si_change_median_pp": _safe_median(
                    frame,
                    "short_interest_pct_change",
                ),
                "si_level_mean": _safe_mean(
                    frame,
                    "short_interest_pct",
                ),
                "si_level_median": _safe_median(
                    frame,
                    "short_interest_pct",
                ),
            }
        )

    return pd.DataFrame(rows)


def _outcome_table(
    test: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict] = []

    for group in PRIMARY_GROUPS:
        frame = test[
            test["context_group"]
            == group
        ].copy()

        forward = _safe_values(
            frame,
            "forward_return_1d",
        ).dropna()

        row = {
            "group": group,
            "n": int(len(frame)),
            "outcome_n": int(
                len(forward)
            ),
            "mean_forward_return_1d": (
                float(forward.mean())
                if not forward.empty
                else float("nan")
            ),
            "median_forward_return_1d": (
                float(forward.median())
                if not forward.empty
                else float("nan")
            ),
            "down_3pct_1d_rate": _downside_rate(
                frame,
                -0.03,
            ),
            "down_5pct_1d_rate": _downside_rate(
                frame,
                -0.05,
            ),
            "down_7pct_1d_rate": _downside_rate(
                frame,
                LOCKED_DOWNSIDE_TARGET,
            ),
            "down_10pct_1d_rate": _downside_rate(
                frame,
                -0.10,
            ),
            "mean_min_return_5d": _safe_mean(
                frame,
                "min_return_5d",
            ),
            "median_min_return_5d": _safe_median(
                frame,
                "min_return_5d",
            ),
            "mean_max_return_5d": _safe_mean(
                frame,
                "max_return_5d",
            ),
            "median_max_return_5d": _safe_median(
                frame,
                "max_return_5d",
            ),
        }

        rows.append(row)

    if len(rows) == 2:
        high = rows[0]
        low = rows[1]

        rows.append(
            {
                "group": "B_minus_D",
                "n": int(
                    high["n"]
                ),
                "outcome_n": int(
                    min(
                        high["outcome_n"],
                        low["outcome_n"],
                    )
                ),
                "mean_forward_return_1d": (
                    high[
                        "mean_forward_return_1d"
                    ]
                    - low[
                        "mean_forward_return_1d"
                    ]
                ),
                "median_forward_return_1d": (
                    high[
                        "median_forward_return_1d"
                    ]
                    - low[
                        "median_forward_return_1d"
                    ]
                ),
                "down_3pct_1d_rate": (
                    high[
                        "down_3pct_1d_rate"
                    ]
                    - low[
                        "down_3pct_1d_rate"
                    ]
                ),
                "down_5pct_1d_rate": (
                    high[
                        "down_5pct_1d_rate"
                    ]
                    - low[
                        "down_5pct_1d_rate"
                    ]
                ),
                "down_7pct_1d_rate": (
                    high[
                        "down_7pct_1d_rate"
                    ]
                    - low[
                        "down_7pct_1d_rate"
                    ]
                ),
                "down_10pct_1d_rate": (
                    high[
                        "down_10pct_1d_rate"
                    ]
                    - low[
                        "down_10pct_1d_rate"
                    ]
                ),
                "mean_min_return_5d": (
                    high[
                        "mean_min_return_5d"
                    ]
                    - low[
                        "mean_min_return_5d"
                    ]
                ),
                "median_min_return_5d": (
                    high[
                        "median_min_return_5d"
                    ]
                    - low[
                        "median_min_return_5d"
                    ]
                ),
                "mean_max_return_5d": (
                    high[
                        "mean_max_return_5d"
                    ]
                    - low[
                        "mean_max_return_5d"
                    ]
                ),
                "median_max_return_5d": (
                    high[
                        "median_max_return_5d"
                    ]
                    - low[
                        "median_max_return_5d"
                    ]
                ),
            }
        )

    return pd.DataFrame(rows)


def run_positive_momentum_context(
    context,
) -> ExperimentResult:
    (
        test,
        pretest,
        info,
    ) = _prepare_signal(
        context
    )

    test = _prepare_groups(
        test
    )

    primary = test[
        test["context_group"].isin(
            PRIMARY_GROUPS
        )
    ].copy()

    if primary.empty:
        raise ValueError(
            "No positive/neutral momentum "
            "signal context observations "
            "were available."
        )

    result = ExperimentResult(
        name=(
            "si_event_risk_positive_momentum_context"
        ),
        description=(
            "Mekanismanalys av B- och D-cellerna "
            "från den låsta SI × event-risk-signalen. "
            "Fokus ligger på vad som skiljer hög "
            "SI-förändring från låg SI-förändring "
            "innan signalen, när tidigare 20-dagars "
            "momentum inte är negativt."
        ),
    )

    result.add_metric(
        "analysis_type",
        "locked_positive_momentum_context",
    )

    result.add_metric(
        "parameter_selection",
        "inherited_locked_configuration",
    )

    result.add_metric(
        "is_discovery_grid",
        False,
    )

    result.add_metric(
        "uses_test_for_parameter_selection",
        False,
    )

    result.add_metric(
        "horizon_days",
        LOCKED_HORIZON,
    )

    result.add_metric(
        "event_threshold",
        LOCKED_EVENT_THRESHOLD,
    )

    result.add_metric(
        "downside_target",
        LOCKED_DOWNSIDE_TARGET,
    )

    result.add_metric(
        "risk_cutoff",
        LOCKED_RISK_CUTOFF,
    )

    result.add_metric(
        "si_change_cutoff",
        LOCKED_SI_CHANGE_CUTOFF,
    )

    result.add_metric(
        "event_model",
        info["event_model"],
    )

    result.add_metric(
        "event_features",
        info["event_features"],
    )

    result.add_metric(
        "event_validation_auc",
        info["event_validation_auc"],
    )

    result.add_metric(
        "risk_threshold",
        info["risk_threshold"],
    )

    result.add_metric(
        "si_change_threshold",
        info["si_change_threshold"],
    )

    result.add_metric(
        "momentum_column",
        MOMENTUM_COLUMN,
    )

    result.add_metric(
        "momentum_definition",
        (
            "nonnegative if "
            "price_return_20d >= 0"
        ),
    )

    result.add_metric(
        "test_rows",
        int(len(test)),
    )

    result.add_metric(
        "primary_rows",
        int(len(primary)),
    )

    result.add_metric(
        "B_rows",
        int(
            (
                test["context_group"]
                == PRIMARY_GROUPS[0]
            ).sum()
        ),
    )

    result.add_metric(
        "D_rows",
        int(
            (
                test["context_group"]
                == PRIMARY_GROUPS[1]
            ).sum()
        ),
    )

    result.add_metric(
        "research_question",
        (
            "Vad skiljer B-cellen: hög "
            "event-risk + hög SI-förändring + "
            "icke-negativt 20d momentum, från "
            "D-cellen: samma event-risk och "
            "momentum men låg SI-förändring?"
        ),
    )

    result.add_table(
        "context_by_group",
        _context_table(
            test
        ),
    )

    result.add_table(
        "B_minus_D_context_difference",
        _primary_difference_table(
            test
        ),
    )

    result.add_table(
        "momentum_structure",
        _momentum_structure_table(
            test
        ),
    )

    result.add_table(
        "risk_and_si_structure",
        _risk_and_si_structure_table(
            test
        ),
    )

    result.add_table(
        "outcomes",
        _outcome_table(
            test
        ),
    )

    result.add_metric(
        "optional_columns_checked",
        list(
            OPTIONAL_MOMENTUM_COLUMNS
            + OPTIONAL_VOLUME_COLUMNS
        ),
    )

    result.add_metric(
        "pretest_rows_used_for_thresholds",
        int(len(pretest)),
    )

    return result
