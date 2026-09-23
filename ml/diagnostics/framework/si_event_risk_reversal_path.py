from __future__ import annotations

import numpy as np
import pandas as pd

from .base import ExperimentResult
from .event_risk_grid import (
    LOCKED_DOWNSIDE_TARGET,
    LOCKED_RISK_CUTOFF,
    LOCKED_SI_CHANGE_CUTOFF,
    _numeric,
)
from .si_event_risk_positive_momentum_context import (
    PRIMARY_GROUPS,
    _prepare_groups,
)
from .si_event_risk_signal_anatomy import (
    LOCKED_EVENT_THRESHOLD,
    LOCKED_HORIZON,
    _prepare_signal,
)


TRAJECTORY_GROUPS = (
    "A_high_event_high_si_negative_momentum",
    "B_high_event_high_si_nonnegative_momentum",
    "C_high_event_low_si_negative_momentum",
    "D_high_event_low_si_nonnegative_momentum",
)

PRICE_RETURN_COLUMNS = (
    "price_return_1d",
    "price_return_3d",
    "price_return_5d",
    "price_return_10d",
    "price_return_20d",
    "price_return_30d",
    "price_return_60d",
)

DISTANCE_COLUMNS = (
    "price_distance_from_5d_high",
    "price_distance_from_10d_high",
    "price_distance_from_20d_high",
    "price_distance_from_60d_high",
)

SI_COLUMNS = (
    "short_interest_pct",
    "short_interest_pct_change",
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


def _return_difference(
    frame: pd.DataFrame,
    first_column: str,
    second_column: str,
) -> float:
    first = _safe_values(
        frame,
        first_column,
    )

    second = _safe_values(
        frame,
        second_column,
    )

    values = pd.concat(
        [
            first.rename("first"),
            second.rename("second"),
        ],
        axis=1,
    ).dropna()

    if values.empty:
        return float("nan")

    return float(
        (
            values["first"]
            - values["second"]
        ).mean()
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


def _trajectory_type(
    frame: pd.DataFrame,
) -> pd.Series:
    return_20d = _safe_values(
        frame,
        "price_return_20d",
    )

    return_60d = _safe_values(
        frame,
        "price_return_60d",
    )

    conditions = [
        (
            return_60d < 0
        )
        & (
            return_20d >= 0
        ),
        (
            return_60d >= 0
        )
        & (
            return_20d >= 0
        ),
        (
            return_60d < 0
        )
        & (
            return_20d < 0
        ),
    ]

    choices = [
        "recovery",
        "persistent_uptrend",
        "persistent_downtrend",
    ]

    result = np.select(
        conditions,
        choices,
        default="other",
    )

    result = pd.Series(
        result,
        index=frame.index,
        dtype="object",
    )

    missing = (
        return_20d.isna()
        | return_60d.isna()
    )

    result.loc[missing] = (
        "insufficient_history"
    )

    return result


def _prepare_trajectory(
    test: pd.DataFrame,
) -> pd.DataFrame:
    local = test.copy()

    local["trajectory_type"] = (
        _trajectory_type(
            local
        )
    )

    if (
        "price_return_5d" in local.columns
        and "price_return_20d" in local.columns
    ):
        local[
            "return_acceleration_5d_vs_20d"
        ] = (
            _safe_values(
                local,
                "price_return_5d",
            )
            - _safe_values(
                local,
                "price_return_20d",
            )
        )

    if (
        "price_return_20d" in local.columns
        and "price_return_60d" in local.columns
    ):
        local[
            "return_acceleration_20d_vs_60d"
        ] = (
            _safe_values(
                local,
                "price_return_20d",
            )
            - _safe_values(
                local,
                "price_return_60d",
            )
        )

    return local


def _trajectory_by_group(
    test: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict] = []

    for group in TRAJECTORY_GROUPS:
        frame = test[
            test["context_group"]
            == group
        ].copy()

        row = {
            "group": group,
            "n": int(len(frame)),
        }

        for column in PRICE_RETURN_COLUMNS:
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

        if (
            "return_acceleration_5d_vs_20d"
            in frame.columns
        ):
            row[
                "mean_return_acceleration_5d_vs_20d"
            ] = _safe_mean(
                frame,
                "return_acceleration_5d_vs_20d",
            )

            row[
                "median_return_acceleration_5d_vs_20d"
            ] = _safe_median(
                frame,
                "return_acceleration_5d_vs_20d",
            )

        if (
            "return_acceleration_20d_vs_60d"
            in frame.columns
        ):
            row[
                "mean_return_acceleration_20d_vs_60d"
            ] = _safe_mean(
                frame,
                "return_acceleration_20d_vs_60d",
            )

            row[
                "median_return_acceleration_20d_vs_60d"
            ] = _safe_median(
                frame,
                "return_acceleration_20d_vs_60d",
            )

        for column in DISTANCE_COLUMNS:
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

        for column in SI_COLUMNS:
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

        rows.append(row)

    return pd.DataFrame(rows)


def _trajectory_type_distribution(
    test: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict] = []

    for group in TRAJECTORY_GROUPS:
        frame = test[
            test["context_group"]
            == group
        ].copy()

        total = len(frame)

        counts = (
            frame["trajectory_type"]
            .value_counts(
                dropna=False
            )
        )

        for trajectory_type in (
            "recovery",
            "persistent_uptrend",
            "persistent_downtrend",
            "other",
            "insufficient_history",
        ):
            count = int(
                counts.get(
                    trajectory_type,
                    0,
                )
            )

            rows.append(
                {
                    "group": group,
                    "trajectory_type": (
                        trajectory_type
                    ),
                    "n": count,
                    "share": (
                        float(count / total)
                        if total > 0
                        else float("nan")
                    ),
                }
            )

    return pd.DataFrame(rows)


def _B_minus_D_trajectory_difference(
    test: pd.DataFrame,
) -> pd.DataFrame:
    b = test[
        test["context_group"]
        == PRIMARY_GROUPS[0]
    ].copy()

    d = test[
        test["context_group"]
        == PRIMARY_GROUPS[1]
    ].copy()

    variables = list(
        dict.fromkeys(
            PRICE_RETURN_COLUMNS
            + DISTANCE_COLUMNS
            + SI_COLUMNS
            + (
                "return_acceleration_5d_vs_20d",
                "return_acceleration_20d_vs_60d",
            )
        )
    )

    rows: list[dict] = []

    for column in variables:
        if (
            column not in b.columns
            and column not in d.columns
        ):
            continue

        b_values = _safe_values(
            b,
            column,
        ).dropna()

        d_values = _safe_values(
            d,
            column,
        ).dropna()

        if (
            b_values.empty
            and d_values.empty
        ):
            continue

        b_mean = (
            float(b_values.mean())
            if not b_values.empty
            else float("nan")
        )

        d_mean = (
            float(d_values.mean())
            if not d_values.empty
            else float("nan")
        )

        rows.append(
            {
                "variable": column,
                "B_n": int(
                    len(b_values)
                ),
                "D_n": int(
                    len(d_values)
                ),
                "B_mean": b_mean,
                "D_mean": d_mean,
                "difference_B_minus_D": (
                    b_mean - d_mean
                    if np.isfinite(b_mean)
                    and np.isfinite(d_mean)
                    else float("nan")
                ),
                "B_median": (
                    float(
                        b_values.median()
                    )
                    if not b_values.empty
                    else float("nan")
                ),
                "D_median": (
                    float(
                        d_values.median()
                    )
                    if not d_values.empty
                    else float("nan")
                ),
            }
        )

    return pd.DataFrame(rows)


def _risk_and_si_relationship(
    test: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict] = []

    for group in PRIMARY_GROUPS:
        frame = test[
            test["context_group"]
            == group
        ].copy()

        event_score = _safe_values(
            frame,
            "event_score",
        ).dropna()

        si_change = _safe_values(
            frame,
            "short_interest_pct_change",
        ).dropna()

        si_level = _safe_values(
            frame,
            "short_interest_pct",
        ).dropna()

        relationship = (
            float(
                frame[
                    [
                        "event_score",
                        "short_interest_pct_change",
                    ]
                ]
                .apply(pd.to_numeric, errors="coerce")
                .corr()
                .iloc[0, 1]
            )
            if (
                "event_score" in frame.columns
                and "short_interest_pct_change"
                in frame.columns
            )
            else float("nan")
        )

        rows.append(
            {
                "group": group,
                "n": int(len(frame)),
                "event_score_mean": (
                    float(event_score.mean())
                    if not event_score.empty
                    else float("nan")
                ),
                "event_score_median": (
                    float(event_score.median())
                    if not event_score.empty
                    else float("nan")
                ),
                "event_score_p25": (
                    float(
                        event_score.quantile(
                            0.25
                        )
                    )
                    if not event_score.empty
                    else float("nan")
                ),
                "event_score_p75": (
                    float(
                        event_score.quantile(
                            0.75
                        )
                    )
                    if not event_score.empty
                    else float("nan")
                ),
                "si_change_mean_pp": (
                    float(si_change.mean())
                    if not si_change.empty
                    else float("nan")
                ),
                "si_change_median_pp": (
                    float(si_change.median())
                    if not si_change.empty
                    else float("nan")
                ),
                "si_level_mean": (
                    float(si_level.mean())
                    if not si_level.empty
                    else float("nan")
                ),
                "si_level_median": (
                    float(si_level.median())
                    if not si_level.empty
                    else float("nan")
                ),
                "event_score_si_change_correlation": (
                    relationship
                ),
            }
        )

    return pd.DataFrame(rows)


def _outcomes(
    test: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict] = []

    for group in TRAJECTORY_GROUPS:
        frame = test[
            test["context_group"]
            == group
        ].copy()

        rows.append(
            {
                "group": group,
                "n": int(len(frame)),
                "mean_forward_return_1d": (
                    _safe_mean(
                        frame,
                        "forward_return_1d",
                    )
                ),
                "median_forward_return_1d": (
                    _safe_median(
                        frame,
                        "forward_return_1d",
                    )
                ),
                "down_3pct_1d_rate": (
                    _downside_rate(
                        frame,
                        -0.03,
                    )
                ),
                "down_5pct_1d_rate": (
                    _downside_rate(
                        frame,
                        -0.05,
                    )
                ),
                "down_7pct_1d_rate": (
                    _downside_rate(
                        frame,
                        LOCKED_DOWNSIDE_TARGET,
                    )
                ),
                "down_10pct_1d_rate": (
                    _downside_rate(
                        frame,
                        -0.10,
                    )
                ),
                "mean_min_return_5d": (
                    _safe_mean(
                        frame,
                        "min_return_5d",
                    )
                ),
                "median_min_return_5d": (
                    _safe_median(
                        frame,
                        "min_return_5d",
                    )
                ),
                "mean_max_return_5d": (
                    _safe_mean(
                        frame,
                        "max_return_5d",
                    )
                ),
                "median_max_return_5d": (
                    _safe_median(
                        frame,
                        "max_return_5d",
                    )
                ),
            }
        )

    if len(rows) == len(TRAJECTORY_GROUPS):
        b = next(
            row
            for row in rows
            if row["group"]
            == PRIMARY_GROUPS[0]
        )

        d = next(
            row
            for row in rows
            if row["group"]
            == PRIMARY_GROUPS[1]
        )

        rows.append(
            {
                "group": "B_minus_D",
                "n": int(b["n"]),
                "mean_forward_return_1d": (
                    b[
                        "mean_forward_return_1d"
                    ]
                    - d[
                        "mean_forward_return_1d"
                    ]
                ),
                "median_forward_return_1d": (
                    b[
                        "median_forward_return_1d"
                    ]
                    - d[
                        "median_forward_return_1d"
                    ]
                ),
                "down_3pct_1d_rate": (
                    b["down_3pct_1d_rate"]
                    - d["down_3pct_1d_rate"]
                ),
                "down_5pct_1d_rate": (
                    b["down_5pct_1d_rate"]
                    - d["down_5pct_1d_rate"]
                ),
                "down_7pct_1d_rate": (
                    b["down_7pct_1d_rate"]
                    - d["down_7pct_1d_rate"]
                ),
                "down_10pct_1d_rate": (
                    b["down_10pct_1d_rate"]
                    - d["down_10pct_1d_rate"]
                ),
                "mean_min_return_5d": (
                    b["mean_min_return_5d"]
                    - d["mean_min_return_5d"]
                ),
                "median_min_return_5d": (
                    b["median_min_return_5d"]
                    - d["median_min_return_5d"]
                ),
                "mean_max_return_5d": (
                    b["mean_max_return_5d"]
                    - d["mean_max_return_5d"]
                ),
                "median_max_return_5d": (
                    b["median_max_return_5d"]
                    - d["median_max_return_5d"]
                ),
            }
        )

    return pd.DataFrame(rows)


def run_reversal_path(
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

    test = _prepare_trajectory(
        test
    )

    primary = test[
        test["context_group"].isin(
            (
                PRIMARY_GROUPS[0],
                PRIMARY_GROUPS[1],
            )
        )
    ].copy()

    if primary.empty:
        raise ValueError(
            "No B/D observations were "
            "available for reversal-path analysis."
        )

    result = ExperimentResult(
        name="si_event_risk_reversal_path",
        description=(
            "Prisbananalys av den låsta "
            "SI × event-risk-signalen. "
            "Fokus ligger på om B-cellen "
            "präglas av återhämtning efter "
            "längre svaghet, jämfört med "
            "D-cellen."
        ),
    )

    result.add_metric(
        "analysis_type",
        "locked_reversal_path",
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
        "price_return_20d",
    )

    result.add_metric(
        "trajectory_definitions",
        {
            "recovery": (
                "price_return_60d < 0 and "
                "price_return_20d >= 0"
            ),
            "persistent_uptrend": (
                "price_return_60d >= 0 and "
                "price_return_20d >= 0"
            ),
            "persistent_downtrend": (
                "price_return_60d < 0 and "
                "price_return_20d < 0"
            ),
            "other": (
                "price_return_60d >= 0 and "
                "price_return_20d < 0"
            ),
        },
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
            "Har B-signalen en karakteristisk "
            "prisbanan där positivt kortsiktigt "
            "momentum uppstår efter längre "
            "svaghet, och skiljer detta B från D?"
        ),
    )

    result.add_table(
        "trajectory_by_group",
        _trajectory_by_group(
            test
        ),
    )

    result.add_table(
        "trajectory_type_distribution",
        _trajectory_type_distribution(
            test
        ),
    )

    result.add_table(
        "B_minus_D_trajectory_difference",
        _B_minus_D_trajectory_difference(
            test
        ),
    )

    result.add_table(
        "risk_and_si_relationship",
        _risk_and_si_relationship(
            test
        ),
    )

    result.add_table(
        "outcomes",
        _outcomes(
            test
        ),
    )

    result.add_metric(
        "price_return_columns_checked",
        list(
            PRICE_RETURN_COLUMNS
        ),
    )

    result.add_metric(
        "distance_columns_checked",
        list(
            DISTANCE_COLUMNS
        ),
    )

    result.add_metric(
        "pretest_rows_used_for_thresholds",
        int(len(pretest)),
    )

    return result
