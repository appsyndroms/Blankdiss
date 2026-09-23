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

SI_CHANGE_BANDS = (
    ("0_to_25pct", 0.0, 0.25),
    ("25_to_50pct", 0.25, 0.50),
    ("50_to_75pct", 0.50, 0.75),
    ("75_to_100pct", 0.75, 1.00),
    ("over_100pct", 1.00, float("inf")),
)


def _test_halves(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = frame.copy()
    frame["snapshot_date"] = pd.to_datetime(
        frame["snapshot_date"],
        errors="coerce",
    )
    frame = frame.dropna(
        subset=["snapshot_date"]
    ).sort_values(
        "snapshot_date"
    )

    if len(frame) < 2:
        return (
            frame.iloc[0:0].copy(),
            frame.iloc[0:0].copy(),
        )

    midpoint = (
        frame["snapshot_date"].min()
        + (
            frame["snapshot_date"].max()
            - frame["snapshot_date"].min()
        )
        / 2
    )

    first = frame[
        frame["snapshot_date"] <= midpoint
    ].copy()

    second = frame[
        frame["snapshot_date"] > midpoint
    ].copy()

    return first, second


def _return_profile(
    frame: pd.DataFrame,
    label: str,
) -> dict:
    result = {
        "group": label,
        "n": len(frame),
    }

    for horizon in (
        1,
        5,
        20,
    ):
        column = (
            f"forward_return_{horizon}d"
        )

        if column not in frame.columns:
            continue

        values = _numeric(
            frame,
            column,
        ).dropna()

        if values.empty:
            continue

        result[
            f"mean_return_{horizon}d"
        ] = float(values.mean())

        result[
            f"median_return_{horizon}d"
        ] = float(values.median())

    if "forward_return_1d" in frame.columns:
        values = _numeric(
            frame,
            "forward_return_1d",
        ).dropna()

        if not values.empty:
            result["down_3pct_rate"] = float(
                (
                    values <= -0.03
                ).mean()
            )

            result["down_5pct_rate"] = float(
                (
                    values <= -0.05
                ).mean()
            )

            result["down_7pct_rate"] = float(
                (
                    values <= -0.07
                ).mean()
            )

            result["down_10pct_rate"] = float(
                (
                    values <= -0.10
                ).mean()
            )

    return result


def _si_gradient_rows(
    frame: pd.DataFrame,
) -> list[dict]:
    rows: list[dict] = []

    change = _numeric(
        frame,
        "short_interest_pct_change",
    )

    for (
        label,
        lower,
        upper,
    ) in SI_CHANGE_BANDS:
        mask = change >= lower

        if np.isfinite(upper):
            mask &= change < upper

        group = frame[
            mask
        ].copy()

        row = _return_profile(
            group,
            label,
        )

        row["si_change_lower"] = lower
        row["si_change_upper"] = upper

        rows.append(row)

    return rows


def _concentration(
    frame: pd.DataFrame,
) -> dict:
    if "yahoo_symbol" in frame.columns:
        column = "yahoo_symbol"
    elif "security_key" in frame.columns:
        column = "security_key"
    else:
        return {
            "entity_column": None,
            "unique_entities": None,
            "largest_entity_share_of_signal": None,
            "largest_entity_signal_events": None,
        }

    values = frame[
        column
    ].dropna()

    counts = values.value_counts()

    if counts.empty:
        return {
            "entity_column": column,
            "unique_entities": 0,
            "largest_entity_share_of_signal": None,
            "largest_entity_signal_events": 0,
        }

    return {
        "entity_column": column,
        "unique_entities": int(
            counts.size
        ),
        "largest_entity_share_of_signal": float(
            counts.iloc[0] / len(values)
        ),
        "largest_entity_signal_events": int(
            counts.iloc[0]
        ),
    }


def _prepare_signal(
    context,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict,
] | None:
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
        return None

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
        return None

    si_threshold = float(
        positive_change.quantile(
            1.0 - LOCKED_SI_CHANGE_CUTOFF
        )
    )

    risk_threshold = _tail_threshold(
        pretest["event_score"],
        LOCKED_RISK_CUTOFF,
    )

    if not np.isfinite(
        risk_threshold
    ):
        return None

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

    signal = test[
        test["high_event_risk"]
        & test["high_si_change"]
    ].copy()

    context_info = {
        "event_model": event_model.name,
        "event_features": list(
            event_model.features
        ),
        "event_validation_auc": (
            event_model.validation_auc
        ),
        "risk_threshold": risk_threshold,
        "si_change_threshold": si_threshold,
        "test_start": test[
            "snapshot_date"
        ].min(),
        "test_end": test[
            "snapshot_date"
        ].max(),
    }

    return (
        test,
        signal,
        context_info,
    )


def _temporal_rows(
    test: pd.DataFrame,
) -> list[dict]:
    first, second = _test_halves(
        test
    )

    rows: list[dict] = []

    for (
        label,
        frame,
    ) in (
        ("test_first_half", first),
        ("test_second_half", second),
        ("test_all", test),
    ):
        high_risk = frame[
            frame["high_event_risk"]
        ].copy()

        signal = high_risk[
            high_risk["high_si_change"]
        ].copy()

        control = high_risk[
            ~high_risk["high_si_change"]
        ].copy()

        signal_returns = _numeric(
            signal,
            "forward_return_1d",
        ).dropna()

        control_returns = _numeric(
            control,
            "forward_return_1d",
        ).dropna()

        signal_down_rate = (
            float(
                (
                    signal_returns
                    <= LOCKED_DOWNSIDE_TARGET
                ).mean()
            )
            if not signal_returns.empty
            else float("nan")
        )

        control_down_rate = (
            float(
                (
                    control_returns
                    <= LOCKED_DOWNSIDE_TARGET
                ).mean()
            )
            if not control_returns.empty
            else float("nan")
        )

        row = {
            "period": label,
            "test_n": len(frame),
            "high_risk_n": len(high_risk),
            "signal_n": len(signal),
            "high_risk_low_si_n": len(control),
            "signal_rate_within_high_risk": (
                float(
                    len(signal)
                    / len(high_risk)
                )
                if len(high_risk)
                else float("nan")
            ),
            "signal_down_7pct_rate": (
                signal_down_rate
            ),
            "high_risk_low_si_down_7pct_rate": (
                control_down_rate
            ),
            "risk_si_effect": (
                signal_down_rate
                - control_down_rate
            ),
        }

        profile = _return_profile(
            signal,
            "signal",
        )

        for key, value in profile.items():
            if key != "group":
                row[
                    f"signal_{key}"
                ] = value

        rows.append(row)

    return rows


def run_locked_robustness(
    context,
) -> ExperimentResult:
    prepared = _prepare_signal(
        context
    )

    if prepared is None:
        raise ValueError(
            "SI × event-risk "
            "robusthetsanalysen kunde inte "
            "beräknas för den aktuella "
            "walk-forward-perioden."
        )

    (
        test,
        signal,
        info,
    ) = prepared

    if signal.empty:
        raise ValueError(
            "Den låsta SI × event-risk-signalen "
            "gav inga testobservationer."
        )

    temporal_rows = _temporal_rows(
        test
    )

    gradient_rows = _si_gradient_rows(
        signal
    )

    concentration = _concentration(
        signal
    )

    concentration_row = {
        "group": "signal_all",
        **concentration,
    }

    result = ExperimentResult(
        name="si_event_risk_locked_robustness",
        description=(
            "Robusthetsanalys av den redan "
            "låsta SI × event-risk-signalen. "
            "Testperioden används endast för "
            "temporal split, SI-gradient, "
            "avkastningsprofil och "
            "koncentrationskontroll."
        ),
    )

    result.add_metric(
        "analysis_type",
        "locked_oos_robustness",
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
        "signal_n",
        len(signal),
    )

    result.add_metric(
        "signal_start",
        (
            signal[
                "snapshot_date"
            ].min().isoformat()
            if not signal.empty
            else None
        ),
    )

    result.add_metric(
        "signal_end",
        (
            signal[
                "snapshot_date"
            ].max().isoformat()
            if not signal.empty
            else None
        ),
    )

    result.add_metric(
        "test_midpoint",
        (
            (
                test[
                    "snapshot_date"
                ].min()
                + (
                    test[
                        "snapshot_date"
                    ].max()
                    - test[
                        "snapshot_date"
                    ].min()
                )
                / 2
            ).isoformat()
            if not test.empty
            else None
        ),
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

    result.add_table(
        "temporal_profile",
        pd.DataFrame(
            temporal_rows
        ),
    )

    result.add_table(
        "si_gradient",
        pd.DataFrame(
            gradient_rows
        ),
    )

    result.add_table(
        "concentration",
        pd.DataFrame(
            [concentration_row]
        ),
    )

    return result
