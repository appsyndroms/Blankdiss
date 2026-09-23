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
from .si_event_risk_signal_anatomy import (
    LOCKED_EVENT_THRESHOLD,
    LOCKED_HORIZON,
    _prepare_signal,
)


MOMENTUM_COLUMN = "price_return_20d"


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


def _rate(
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


def _momentum_label(
    frame: pd.DataFrame,
) -> pd.Series:
    values = _numeric(
        frame,
        MOMENTUM_COLUMN,
    )

    return pd.Series(
        np.where(
            values < 0.0,
            "negative",
            "neutral_or_positive",
        ),
        index=frame.index,
        dtype="object",
    )


def _add_momentum_regime(
    test: pd.DataFrame,
) -> pd.DataFrame:
    test = test.copy()

    test["momentum_regime"] = (
        _momentum_label(test)
    )

    test["momentum_negative"] = (
        _numeric(
            test,
            MOMENTUM_COLUMN,
        )
        < 0.0
    )

    return test


def _four_cell_table(
    test: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict] = []

    definitions = (
        (
            "A_high_event_high_si_negative_momentum",
            test["high_event_risk"]
            & test["high_si_change"]
            & test["momentum_negative"],
            "high_event_risk",
            "high_si_change",
            "negative",
        ),
        (
            "B_high_event_high_si_neutral_or_positive_momentum",
            test["high_event_risk"]
            & test["high_si_change"]
            & ~test["momentum_negative"],
            "high_event_risk",
            "high_si_change",
            "neutral_or_positive",
        ),
        (
            "C_high_event_low_si_negative_momentum",
            test["high_event_risk"]
            & ~test["high_si_change"]
            & test["momentum_negative"],
            "high_event_risk",
            "low_si_change",
            "negative",
        ),
        (
            "D_high_event_low_si_neutral_or_positive_momentum",
            test["high_event_risk"]
            & ~test["high_si_change"]
            & ~test["momentum_negative"],
            "high_event_risk",
            "low_si_change",
            "neutral_or_positive",
        ),
    )

    for (
        label,
        mask,
        _risk_group,
        si_group,
        momentum_group,
    ) in definitions:
        frame = test.loc[
            mask
        ].copy()

        row = {
            "cell": label,
            "n": int(len(frame)),
            "event_risk_group": _risk_group,
            "si_group": si_group,
            "momentum_group": momentum_group,
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
            "mean_price_return_5d": _safe_mean(
                frame,
                "price_return_5d",
            ),
            "median_price_return_5d": _safe_median(
                frame,
                "price_return_5d",
            ),
            "mean_price_return_20d": _safe_mean(
                frame,
                "price_return_20d",
            ),
            "median_price_return_20d": _safe_median(
                frame,
                "price_return_20d",
            ),
            "mean_price_return_60d": _safe_mean(
                frame,
                "price_return_60d",
            ),
            "median_price_return_60d": _safe_median(
                frame,
                "price_return_60d",
            ),
            "mean_event_score": _safe_mean(
                frame,
                "event_score",
            ),
            "mean_forward_return_1d": _safe_mean(
                frame,
                "forward_return_1d",
            ),
            "median_forward_return_1d": _safe_median(
                frame,
                "forward_return_1d",
            ),
            "down_3pct_1d_rate": _rate(
                frame,
                "forward_return_1d",
                -0.03,
            ),
            "down_5pct_1d_rate": _rate(
                frame,
                "forward_return_1d",
                -0.05,
            ),
            "down_7pct_1d_rate": _rate(
                frame,
                "forward_return_1d",
                LOCKED_DOWNSIDE_TARGET,
            ),
            "down_10pct_1d_rate": _rate(
                frame,
                "forward_return_1d",
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

    return pd.DataFrame(rows)


def _si_effect_by_momentum_table(
    test: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict] = []

    for momentum_label, momentum_mask in (
        (
            "negative",
            test["momentum_negative"],
        ),
        (
            "neutral_or_positive",
            ~test["momentum_negative"],
        ),
    ):
        high_si = test.loc[
            test["high_event_risk"]
            & momentum_mask
            & test["high_si_change"]
        ].copy()

        low_si = test.loc[
            test["high_event_risk"]
            & momentum_mask
            & ~test["high_si_change"]
        ].copy()

        high_rate = _rate(
            high_si,
            "forward_return_1d",
            LOCKED_DOWNSIDE_TARGET,
        )

        low_rate = _rate(
            low_si,
            "forward_return_1d",
            LOCKED_DOWNSIDE_TARGET,
        )

        effect = (
            (high_rate - low_rate) * 100.0
            if np.isfinite(high_rate)
            and np.isfinite(low_rate)
            else float("nan")
        )

        rows.append(
            {
                "momentum_group": momentum_label,
                "high_si_n": int(len(high_si)),
                "low_si_n": int(len(low_si)),
                "high_si_down_7pct_rate": high_rate,
                "low_si_down_7pct_rate": low_rate,
                "down_7pct_effect_pp": effect,
                "high_si_mean_forward_return_1d": (
                    _safe_mean(
                        high_si,
                        "forward_return_1d",
                    )
                ),
                "low_si_mean_forward_return_1d": (
                    _safe_mean(
                        low_si,
                        "forward_return_1d",
                    )
                ),
                "high_si_mean_min_return_5d": (
                    _safe_mean(
                        high_si,
                        "min_return_5d",
                    )
                ),
                "low_si_mean_min_return_5d": (
                    _safe_mean(
                        low_si,
                        "min_return_5d",
                    )
                ),
            }
        )

    return pd.DataFrame(rows)


def _momentum_effect_within_si_table(
    test: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict] = []

    for si_label, si_mask in (
        (
            "high_si",
            test["high_si_change"],
        ),
        (
            "low_si",
            ~test["high_si_change"],
        ),
    ):
        negative = test.loc[
            test["high_event_risk"]
            & si_mask
            & test["momentum_negative"]
        ].copy()

        neutral_or_positive = test.loc[
            test["high_event_risk"]
            & si_mask
            & ~test["momentum_negative"]
        ].copy()

        negative_rate = _rate(
            negative,
            "forward_return_1d",
            LOCKED_DOWNSIDE_TARGET,
        )

        non_negative_rate = _rate(
            neutral_or_positive,
            "forward_return_1d",
            LOCKED_DOWNSIDE_TARGET,
        )

        effect = (
            (negative_rate - non_negative_rate)
            * 100.0
            if np.isfinite(negative_rate)
            and np.isfinite(non_negative_rate)
            else float("nan")
        )

        rows.append(
            {
                "si_group": si_label,
                "negative_momentum_n": int(
                    len(negative)
                ),
                "neutral_or_positive_momentum_n": int(
                    len(neutral_or_positive)
                ),
                "negative_momentum_down_7pct_rate": (
                    negative_rate
                ),
                "neutral_or_positive_down_7pct_rate": (
                    non_negative_rate
                ),
                "negative_momentum_effect_pp": effect,
                "negative_momentum_mean_20d_return": (
                    _safe_mean(
                        negative,
                        "price_return_20d",
                    )
                ),
                "neutral_or_positive_mean_20d_return": (
                    _safe_mean(
                        neutral_or_positive,
                        "price_return_20d",
                    )
                ),
            }
        )

    return pd.DataFrame(rows)


def _cell_comparison_table(
    test: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict] = []

    cells = {
        "A": (
            test["high_event_risk"]
            & test["high_si_change"]
            & test["momentum_negative"]
        ),
        "B": (
            test["high_event_risk"]
            & test["high_si_change"]
            & ~test["momentum_negative"]
        ),
        "C": (
            test["high_event_risk"]
            & ~test["high_si_change"]
            & test["momentum_negative"]
        ),
        "D": (
            test["high_event_risk"]
            & ~test["high_si_change"]
            & ~test["momentum_negative"]
        ),
    }

    for label, mask in cells.items():
        frame = test.loc[
            mask
        ].copy()

        rows.append(
            {
                "cell": label,
                "n": int(len(frame)),
                "down_5pct_1d_rate": _rate(
                    frame,
                    "forward_return_1d",
                    -0.05,
                ),
                "down_7pct_1d_rate": _rate(
                    frame,
                    "forward_return_1d",
                    LOCKED_DOWNSIDE_TARGET,
                ),
                "down_10pct_1d_rate": _rate(
                    frame,
                    "forward_return_1d",
                    -0.10,
                ),
                "mean_forward_return_1d": _safe_mean(
                    frame,
                    "forward_return_1d",
                ),
                "mean_min_return_5d": _safe_mean(
                    frame,
                    "min_return_5d",
                ),
                "mean_price_return_20d": _safe_mean(
                    frame,
                    "price_return_20d",
                ),
                "mean_price_return_60d": _safe_mean(
                    frame,
                    "price_return_60d",
                ),
            }
        )

    return pd.DataFrame(rows)


def run_momentum_split(
    context,
) -> ExperimentResult:
    (
        test,
        pretest,
        info,
    ) = _prepare_signal(
        context
    )

    test = _add_momentum_regime(
        test
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
        name="si_event_risk_momentum_split",
        description=(
            "Låst 2×2-analys av SI × event-risk-"
            "signalen mot tidigare 20-dagars "
            "momentum. Syftet är att skilja "
            "SI-effekten från redan fallande "
            "aktier utan att optimera nya "
            "signalparametrar."
        ),
    )

    result.add_metric(
        "analysis_type",
        "locked_signal_momentum_split",
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
        "momentum_split",
        "negative_if_price_return_20d_lt_0",
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
        "signal_negative_momentum_rows",
        int(
            (
                signal["momentum_negative"]
            ).sum()
        ),
    )

    result.add_metric(
        "signal_neutral_or_positive_momentum_rows",
        int(
            (
                signal[
                    "momentum_negative"
                ] == False
            ).sum()
        ),
    )

    result.add_metric(
        "research_question",
        (
            "Är SI × event-risk-signalen "
            "fortfarande informativ inom redan "
            "fallande aktier, och finns samma "
            "SI-effekt när tidigare momentum "
            "inte är negativt?"
        ),
    )

    result.add_table(
        "four_momentum_si_cells",
        _four_cell_table(
            test
        ),
    )

    result.add_table(
        "si_effect_by_momentum",
        _si_effect_by_momentum_table(
            test
        ),
    )

    result.add_table(
        "momentum_effect_within_si",
        _momentum_effect_within_si_table(
            test
        ),
    )

    result.add_table(
        "cell_comparison",
        _cell_comparison_table(
            test
        ),
    )

    result.add_metric(
        "pretest_rows_used_for_thresholds",
        int(len(pretest)),
    )

    return result
