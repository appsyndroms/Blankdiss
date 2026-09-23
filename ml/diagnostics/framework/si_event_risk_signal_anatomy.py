from __future__ import annotations

import numpy as np
import pandas as pd

from .base import ExperimentResult
from .event_risk_grid import (
    LOCKED_DOWNSIDE_TARGET,
    LOCKED_RISK_CUTOFF,
    LOCKED_SI_CHANGE_CUTOFF,
    _fit_event_model,
    _numeric,
    _predict_event_score,
    _prepare_frame,
    _select_event_model,
    _tail_threshold,
)


LOCKED_HORIZON = 1
LOCKED_EVENT_THRESHOLD = 0.07

SI_LAGS = (0, 1, 2)

PRE_EVENT_COLUMNS = (
    "price_return_5d",
    "price_return_20d",
    "price_return_60d",
    "short_interest_pct",
    "price_volatility_20d",
    "price_distance_from_20d_high",
    "price_distance_from_60d_high",
)


def _entity_column(frame: pd.DataFrame) -> str:
    if "yahoo_symbol" in frame.columns:
        return "yahoo_symbol"

    if "security_key" in frame.columns:
        return "security_key"

    raise KeyError(
        "Signal-anatomy diagnostic requires either "
        "'yahoo_symbol' or 'security_key'."
    )


def _add_si_lags(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    frame = frame.copy()

    group_column = _entity_column(frame)

    frame["snapshot_date"] = pd.to_datetime(
        frame["snapshot_date"],
        errors="coerce",
    )

    frame = frame.sort_values(
        [
            group_column,
            "snapshot_date",
        ],
        kind="mergesort",
    ).copy()

    change = _numeric(
        frame,
        "short_interest_pct_change",
    )

    for lag in SI_LAGS:
        frame[
            f"si_change_lag_{lag}"
        ] = (
            change
            if lag == 0
            else change.groupby(
                frame[group_column]
            ).shift(lag)
        )

    previous_date = (
        frame
        .groupby(
            group_column,
            sort=False,
        )["snapshot_date"]
        .shift(1)
    )

    frame["days_since_previous_snapshot"] = (
        frame["snapshot_date"]
        - previous_date
    ).dt.days

    return frame


def _prepare_signal(
    context,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict,
]:
    train = _prepare_frame(
        context.train,
        LOCKED_HORIZON,
    )

    validation = _prepare_frame(
        context.validation,
        LOCKED_HORIZON,
    )

    pretest = _prepare_frame(
        context.pretest,
        LOCKED_HORIZON,
    )

    test = _prepare_frame(
        context.test,
        LOCKED_HORIZON,
    )

    train = _add_si_lags(train)
    validation = _add_si_lags(validation)
    pretest = _add_si_lags(pretest)
    test = _add_si_lags(test)

    event_model = _select_event_model(
        train,
        validation,
        LOCKED_HORIZON,
        LOCKED_EVENT_THRESHOLD,
    )

    pretest_model = _fit_event_model(
        pretest,
        event_model.features,
        LOCKED_HORIZON,
        LOCKED_EVENT_THRESHOLD,
    )

    if pretest_model is None:
        raise ValueError(
            "Could not fit pretest event-risk model."
        )

    pretest = pretest.copy()
    test = test.copy()

    pretest["event_score"] = (
        _predict_event_score(
            pretest_model,
            pretest,
            event_model.features,
        )
    )

    test["event_score"] = (
        _predict_event_score(
            pretest_model,
            test,
            event_model.features,
        )
    )

    positive_change = _numeric(
        pretest,
        "short_interest_pct_change",
    )

    positive_change = positive_change[
        positive_change > 0
    ].dropna()

    if positive_change.empty:
        raise ValueError(
            "No positive SI changes available in pretest."
        )

    si_threshold = float(
        positive_change.quantile(
            1.0 - LOCKED_SI_CHANGE_CUTOFF
        )
    )

    risk_threshold = _tail_threshold(
        pretest["event_score"],
        LOCKED_RISK_CUTOFF,
    )

    if not np.isfinite(risk_threshold):
        raise ValueError(
            "Event-risk threshold is not finite."
        )

    test["high_event_risk"] = (
        test["event_score"]
        >= risk_threshold
    )

    test["high_si_change"] = (
        _numeric(
            test,
            "short_interest_pct_change",
        )
        >= si_threshold
    )

    test["signal"] = (
        test["high_event_risk"]
        & test["high_si_change"]
    )

    test["high_risk_low_si"] = (
        test["high_event_risk"]
        & ~test["high_si_change"]
    )

    return (
        test,
        pretest,
        {
            "event_model": event_model.name,
            "event_features": list(
                event_model.features
            ),
            "event_validation_auc": (
                event_model.validation_auc
            ),
            "risk_threshold": risk_threshold,
            "si_change_threshold": si_threshold,
        },
    )


def _safe_mean(
    frame: pd.DataFrame,
    column: str,
) -> float:
    if column not in frame.columns:
        return float("nan")

    values = _numeric(
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
    if column not in frame.columns:
        return float("nan")

    values = _numeric(
        frame,
        column,
    ).dropna()

    if values.empty:
        return float("nan")

    return float(values.median())


def _downside_rate(
    frame: pd.DataFrame,
    column: str,
    threshold: float,
) -> float:
    if column not in frame.columns:
        return float("nan")

    values = _numeric(
        frame,
        column,
    ).dropna()

    if values.empty:
        return float("nan")

    return float(
        (
            values <= threshold
        ).mean()
    )


def _signal_context_table(
    test: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict] = []

    groups = (
        (
            "signal",
            test[
                test["signal"]
            ].copy(),
        ),
        (
            "high_risk_low_si",
            test[
                test["high_risk_low_si"]
            ].copy(),
        ),
    )

    for label, frame in groups:
        row = {
            "group": label,
            "n": int(len(frame)),
            "mean_event_score": _safe_mean(
                frame,
                "event_score",
            ),
            "median_event_score": _safe_median(
                frame,
                "event_score",
            ),
            "mean_si_change_pp": _safe_mean(
                frame,
                "short_interest_pct_change",
            ),
            "median_si_change_pp": _safe_median(
                frame,
                "short_interest_pct_change",
            ),
            "mean_si_level": _safe_mean(
                frame,
                "short_interest_pct",
            ),
            "median_si_level": _safe_median(
                frame,
                "short_interest_pct",
            ),
        }

        for column in PRE_EVENT_COLUMNS:
            if column in {
                "short_interest_pct",
            }:
                continue

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
            "forward_return_1d",
            -0.03,
        )

        row[
            "down_5pct_1d_rate"
        ] = _downside_rate(
            frame,
            "forward_return_1d",
            -0.05,
        )

        row[
            "down_7pct_1d_rate"
        ] = _downside_rate(
            frame,
            "forward_return_1d",
            LOCKED_DOWNSIDE_TARGET,
        )

        row[
            "down_10pct_1d_rate"
        ] = _downside_rate(
            frame,
            "forward_return_1d",
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


def _outcome_timing_table(
    test: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict] = []

    if (
        "min_return_5d_date"
        not in test.columns
        or "price_date"
        not in test.columns
    ):
        return pd.DataFrame()

    local = test.copy()

    local["price_date"] = pd.to_datetime(
        local["price_date"],
        errors="coerce",
    )

    local["min_return_5d_date"] = pd.to_datetime(
        local["min_return_5d_date"],
        errors="coerce",
    )

    local["days_to_min_5d"] = (
        local["min_return_5d_date"]
        - local["price_date"]
    ).dt.days

    for label, mask in (
        (
            "signal",
            local["signal"],
        ),
        (
            "high_risk_low_si",
            local["high_risk_low_si"],
        ),
    ):
        frame = local.loc[
            mask
        ].copy()

        values = _numeric(
            frame,
            "days_to_min_5d",
        ).dropna()

        row = {
            "group": label,
            "n": int(len(frame)),
            "n_with_min_date": int(
                len(values)
            ),
            "mean_days_to_min_5d": (
                float(values.mean())
                if not values.empty
                else float("nan")
            ),
            "median_days_to_min_5d": (
                float(values.median())
                if not values.empty
                else float("nan")
            ),
            "min_return_5d_mean": _safe_mean(
                frame,
                "min_return_5d",
            ),
            "min_return_5d_median": _safe_median(
                frame,
                "min_return_5d",
            ),
        }

        rows.append(row)

    return pd.DataFrame(rows)


def _timing_lag_table(
    test: pd.DataFrame,
    si_threshold: float,
) -> pd.DataFrame:
    rows: list[dict] = []

    high_risk = test[
        test["high_event_risk"]
    ].copy()

    for lag in SI_LAGS:
        column = (
            f"si_change_lag_{lag}"
        )

        if column not in high_risk.columns:
            continue

        high_si = (
            _numeric(
                high_risk,
                column,
            )
            >= si_threshold
        )

        signal = high_risk.loc[
            high_si
        ].copy()

        control = high_risk.loc[
            ~high_si
        ].copy()

        signal_return = _numeric(
            signal,
            "forward_return_1d",
        ).dropna()

        control_return = _numeric(
            control,
            "forward_return_1d",
        ).dropna()

        row = {
            "lag": lag,
            "lag_description": (
                "current_si_change"
                if lag == 0
                else (
                    f"{lag}_prior_si_observation"
                )
            ),
            "high_risk_n": int(
                len(high_risk)
            ),
            "high_si_n": int(
                len(signal)
            ),
            "low_si_n": int(
                len(control)
            ),
            "high_si_rate": (
                float(signal_return.mean())
                if not signal_return.empty
                else float("nan")
            ),
            "low_si_rate": (
                float(control_return.mean())
                if not control_return.empty
                else float("nan")
            ),
            "high_si_down_5pct_rate": (
                float(
                    (
                        signal_return
                        <= -0.05
                    ).mean()
                )
                if not signal_return.empty
                else float("nan")
            ),
            "low_si_down_5pct_rate": (
                float(
                    (
                        control_return
                        <= -0.05
                    ).mean()
                )
                if not control_return.empty
                else float("nan")
            ),
            "high_si_down_7pct_rate": (
                float(
                    (
                        signal_return
                        <= LOCKED_DOWNSIDE_TARGET
                    ).mean()
                )
                if not signal_return.empty
                else float("nan")
            ),
            "low_si_down_7pct_rate": (
                float(
                    (
                        control_return
                        <= LOCKED_DOWNSIDE_TARGET
                    ).mean()
                )
                if not control_return.empty
                else float("nan")
            ),
            "mean_days_since_previous_snapshot": (
                _safe_mean(
                    signal,
                    "days_since_previous_snapshot",
                )
            ),
        }

        row[
            "down_7pct_effect_pp"
        ] = (
            (
                row["high_si_down_7pct_rate"]
                - row["low_si_down_7pct_rate"]
            )
            * 100.0
            if np.isfinite(
                row["high_si_down_7pct_rate"]
            )
            and np.isfinite(
                row["low_si_down_7pct_rate"]
            )
            else float("nan")
        )

        rows.append(row)

    return pd.DataFrame(rows)


def _signal_distribution_table(
    test: pd.DataFrame,
) -> pd.DataFrame:
    signal = test[
        test["signal"]
    ].copy()

    if signal.empty:
        return pd.DataFrame()

    rows = []

    for column in (
        "event_score",
        "short_interest_pct_change",
        "short_interest_pct",
        "price_return_5d",
        "price_return_20d",
        "price_return_60d",
        "price_volatility_20d",
    ):
        if column not in signal.columns:
            continue

        values = _numeric(
            signal,
            column,
        ).dropna()

        if values.empty:
            continue

        rows.append(
            {
                "variable": column,
                "n": int(len(values)),
                "mean": float(values.mean()),
                "median": float(values.median()),
                "p10": float(
                    values.quantile(0.10)
                ),
                "p25": float(
                    values.quantile(0.25)
                ),
                "p75": float(
                    values.quantile(0.75)
                ),
                "p90": float(
                    values.quantile(0.90)
                ),
                "min": float(values.min()),
                "max": float(values.max()),
            }
        )

    return pd.DataFrame(rows)


def run_signal_anatomy(
    context,
) -> ExperimentResult:
    (
        test,
        pretest,
        info,
    ) = _prepare_signal(
        context
    )

    signal = test[
        test["signal"]
    ].copy()

    high_risk = test[
        test["high_event_risk"]
    ].copy()

    if signal.empty:
        raise ValueError(
            "The locked SI × event-risk signal "
            "produced no test observations."
        )

    result = ExperimentResult(
        name="si_event_risk_signal_anatomy",
        description=(
            "Mekanismdiagnostik för den redan "
            "låsta SI × event-risk-signalen. "
            "Analyserar pre-signal-kontext, "
            "SI-förändringens timing och "
            "efterföljande prisförlopp utan "
            "att ändra de låsta parametrarna."
        ),
    )

    result.add_metric(
        "analysis_type",
        "locked_signal_anatomy",
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
        "test_rows",
        int(len(test)),
    )

    result.add_metric(
        "high_event_risk_rows",
        int(len(high_risk)),
    )

    result.add_metric(
        "signal_rows",
        int(len(signal)),
    )

    result.add_metric(
        "signal_share_of_high_event_risk",
        (
            float(
                len(signal)
                / len(high_risk)
            )
            if len(high_risk)
            else float("nan")
        ),
    )

    result.add_metric(
        "research_question",
        (
            "Är signalen specifik för aktuell "
            "SI-förändring, vilken pris- och "
            "SI-kontext föregår signalen, och "
            "hur snabbt kommer efterföljande "
            "downside?"
        ),
    )

    result.add_table(
        "signal_context",
        _signal_context_table(
            test
        ),
    )

    result.add_table(
        "si_timing_lags",
        _timing_lag_table(
            test,
            info["si_change_threshold"],
        ),
    )

    result.add_table(
        "outcome_timing",
        _outcome_timing_table(
            test
        ),
    )

    result.add_table(
        "signal_distribution",
        _signal_distribution_table(
            test
        ),
    )

    result.add_metric(
        "pretest_rows_used_for_thresholds",
        int(len(pretest)),
    )

    result.add_metric(
        "si_lags_tested",
        list(SI_LAGS),
    )

    result.add_metric(
        "pre_event_variables",
        list(PRE_EVENT_COLUMNS),
    )

    return result
