from __future__ import annotations

import numpy as np
import pandas as pd

from .base import ExperimentResult
from .event_risk import (
    _prepare_event_risk_data,
    _risk_label,
    _tail_threshold,
)


def _numeric(
    frame: pd.DataFrame,
    column: str,
) -> pd.Series:
    return pd.to_numeric(
        frame[column],
        errors="coerce",
    )


def _summary(
    frame: pd.DataFrame,
    return_column: str,
) -> dict[str, float | int]:
    values = _numeric(
        frame,
        return_column,
    ).dropna()

    if values.empty:
        return {
            "n": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "positive_rate": float("nan"),
        }

    return {
        "n": int(len(values)),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "positive_rate": float(
            (values > 0).mean()
        ),
    }


def _positive_change_threshold(
    pretest: pd.DataFrame,
    cutoff: float,
) -> float:
    values = _numeric(
        pretest,
        "short_interest_pct_change",
    )

    values = values[
        values > 0
    ].dropna()

    if values.empty:
        return float("nan")

    return float(
        values.quantile(
            1.0 - cutoff
        )
    )


def _prior_return_threshold(
    pretest: pd.DataFrame,
    column: str,
) -> float:
    values = _numeric(
        pretest,
        column,
    ).dropna()

    if values.empty:
        return float("nan")

    return float(
        values.quantile(0.50)
    )


def _build_change_groups(
    test: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    result = test.copy()

    change = _numeric(
        result,
        "short_interest_pct_change",
    )

    result["positive_si_change"] = (
        change > 0
    )

    result["high_si_change"] = (
        change >= threshold
    )

    return result


def _build_price_dynamics_table(
    test: pd.DataFrame,
    *,
    horizons: tuple[int, ...],
    si_change_cutoffs: tuple[float, ...],
    thresholds: dict[float, float],
) -> pd.DataFrame:
    rows: list[dict] = []

    for cutoff in si_change_cutoffs:
        threshold = thresholds.get(
            cutoff,
            float("nan"),
        )

        if not np.isfinite(threshold):
            continue

        frame = _build_change_groups(
            test,
            threshold,
        )

        frame = frame[
            frame["positive_si_change"]
        ].copy()

        high = frame[
            frame["high_si_change"]
        ]

        low = frame[
            ~frame["high_si_change"]
        ]

        for horizon in horizons:
            column = (
                f"forward_return_{horizon}d"
            )

            if column not in frame.columns:
                continue

            high_summary = _summary(
                high,
                column,
            )

            low_summary = _summary(
                low,
                column,
            )

            rows.append(
                {
                    "si_change_cutoff": cutoff,
                    "si_change_threshold": threshold,
                    "horizon_days": horizon,
                    "high_n": high_summary["n"],
                    "other_positive_n": low_summary["n"],
                    "high_mean_return": (
                        high_summary["mean"]
                    ),
                    "other_positive_mean_return": (
                        low_summary["mean"]
                    ),
                    "mean_return_delta": (
                        high_summary["mean"]
                        - low_summary["mean"]
                    ),
                    "high_median_return": (
                        high_summary["median"]
                    ),
                    "other_positive_median_return": (
                        low_summary["median"]
                    ),
                    "high_positive_rate": (
                        high_summary["positive_rate"]
                    ),
                    "other_positive_rate": (
                        low_summary["positive_rate"]
                    ),
                }
            )

    return pd.DataFrame(rows)


def _build_bottom_timing_table(
    test: pd.DataFrame,
    *,
    si_change_cutoffs: tuple[float, ...],
    thresholds: dict[float, float],
) -> pd.DataFrame:
    rows: list[dict] = []

    required = {
        "min_return_5d",
        "min_return_5d_date",
        "price_date",
    }

    missing = [
        column
        for column in required
        if column not in test.columns
    ]

    if missing:
        return pd.DataFrame()

    for cutoff in si_change_cutoffs:
        threshold = thresholds.get(
            cutoff,
            float("nan"),
        )

        if not np.isfinite(threshold):
            continue

        frame = _build_change_groups(
            test,
            threshold,
        )

        frame = frame[
            frame["positive_si_change"]
        ].copy()

        frame["min_return_5d"] = _numeric(
            frame,
            "min_return_5d",
        )

        frame["price_date"] = pd.to_datetime(
            frame["price_date"],
            errors="coerce",
        )

        frame["min_return_5d_date"] = (
            pd.to_datetime(
                frame["min_return_5d_date"],
                errors="coerce",
            )
        )

        frame["bottom_delay_calendar_days"] = (
            frame["min_return_5d_date"]
            - frame["price_date"]
        ).dt.days

        for label, group in (
            ("high_si_change", frame[
                frame["high_si_change"]
            ]),
            ("other_positive", frame[
                ~frame["high_si_change"]
            ]),
        ):
            returns = group[
                "min_return_5d"
            ].dropna()

            delays = group[
                "bottom_delay_calendar_days"
            ].dropna()

            rows.append(
                {
                    "si_change_cutoff": cutoff,
                    "si_change_threshold": threshold,
                    "group": label,
                    "n": int(len(group)),
                    "mean_min_return_5d": (
                        float(returns.mean())
                        if not returns.empty
                        else float("nan")
                    ),
                    "median_min_return_5d": (
                        float(returns.median())
                        if not returns.empty
                        else float("nan")
                    ),
                    "mean_bottom_delay_calendar_days": (
                        float(delays.mean())
                        if not delays.empty
                        else float("nan")
                    ),
                    "median_bottom_delay_calendar_days": (
                        float(delays.median())
                        if not delays.empty
                        else float("nan")
                    ),
                }
            )

    return pd.DataFrame(rows)


def _build_event_risk_table(
    pretest: pd.DataFrame,
    test: pd.DataFrame,
    *,
    horizons: tuple[int, ...],
    si_change_cutoffs: tuple[float, ...],
    event_risk_cutoffs: tuple[float, ...],
    thresholds: dict[float, float],
) -> pd.DataFrame:
    rows: list[dict] = []

    for si_cutoff in si_change_cutoffs:
        si_threshold = thresholds.get(
            si_cutoff,
            float("nan"),
        )

        if not np.isfinite(si_threshold):
            continue

        frame = _build_change_groups(
            test,
            si_threshold,
        )

        frame = frame[
            frame["positive_si_change"]
        ].copy()

        for risk_cutoff in event_risk_cutoffs:
            risk_threshold = _tail_threshold(
                pretest["event_score"],
                risk_cutoff,
            )

            if not np.isfinite(
                risk_threshold
            ):
                continue

            risk_label = _risk_label(
                risk_cutoff
            )

            frame["event_risk_group"] = np.where(
                frame["event_score"]
                >= risk_threshold,
                risk_label,
                "outside",
            )

            high_risk = frame[
                frame["event_risk_group"]
                == risk_label
            ]

            outside_risk = frame[
                frame["event_risk_group"]
                != risk_label
            ]

            high_risk_high_si = high_risk[
                high_risk["high_si_change"]
            ]

            high_risk_low_si = high_risk[
                ~high_risk["high_si_change"]
            ]

            outside_high_si = outside_risk[
                outside_risk["high_si_change"]
            ]

            outside_low_si = outside_risk[
                ~outside_risk["high_si_change"]
            ]

            for horizon in horizons:
                column = (
                    f"forward_return_{horizon}d"
                )

                if column not in frame.columns:
                    continue

                a = _summary(
                    high_risk_high_si,
                    column,
                )
                b = _summary(
                    high_risk_low_si,
                    column,
                )
                c = _summary(
                    outside_high_si,
                    column,
                )
                d = _summary(
                    outside_low_si,
                    column,
                )

                rows.append(
                    {
                        "si_change_cutoff": si_cutoff,
                        "si_change_threshold": si_threshold,
                        "event_risk_cutoff": risk_cutoff,
                        "event_risk_threshold": risk_threshold,
                        "horizon_days": horizon,
                        "risk_high_si_n": a["n"],
                        "risk_low_si_n": b["n"],
                        "outside_high_si_n": c["n"],
                        "outside_low_si_n": d["n"],
                        "risk_high_si_mean_return": a["mean"],
                        "risk_low_si_mean_return": b["mean"],
                        "outside_high_si_mean_return": c["mean"],
                        "outside_low_si_mean_return": d["mean"],
                        "risk_si_effect": (
                            a["mean"] - b["mean"]
                        ),
                        "outside_si_effect": (
                            c["mean"] - d["mean"]
                        ),
                        "interaction": (
                            (a["mean"] - b["mean"])
                            - (c["mean"] - d["mean"])
                        ),
                    }
                )

    return pd.DataFrame(rows)


def _build_prior_price_control_table(
    pretest: pd.DataFrame,
    test: pd.DataFrame,
    *,
    horizons: tuple[int, ...],
    si_change_cutoffs: tuple[float, ...],
    prior_return_columns: tuple[str, ...],
    thresholds: dict[float, float],
) -> pd.DataFrame:
    rows: list[dict] = []

    for prior_column in prior_return_columns:
        if prior_column not in test.columns:
            continue

        prior_threshold = _prior_return_threshold(
            pretest,
            prior_column,
        )

        if not np.isfinite(prior_threshold):
            continue

        for si_cutoff in si_change_cutoffs:
            si_threshold = thresholds.get(
                si_cutoff,
                float("nan"),
            )

            if not np.isfinite(si_threshold):
                continue

            frame = _build_change_groups(
                test,
                si_threshold,
            )

            frame = frame[
                frame["positive_si_change"]
            ].copy()

            prior = _numeric(
                frame,
                prior_column,
            )

            frame["prior_high"] = (
                prior >= prior_threshold
            )

            for prior_label, prior_group in (
                ("prior_high", frame[
                    frame["prior_high"]
                ]),
                ("prior_low", frame[
                    ~frame["prior_high"]
                ]),
            ):
                high_si = prior_group[
                    prior_group["high_si_change"]
                ]

                other = prior_group[
                    ~prior_group["high_si_change"]
                ]

                for horizon in horizons:
                    column = (
                        f"forward_return_{horizon}d"
                    )

                    if column not in frame.columns:
                        continue

                    high_values = _numeric(
                        high_si,
                        column,
                    ).dropna()

                    other_values = _numeric(
                        other,
                        column,
                    ).dropna()

                    rows.append(
                        {
                            "prior_return_column": prior_column,
                            "prior_return_threshold": prior_threshold,
                            "prior_group": prior_label,
                            "si_change_cutoff": si_cutoff,
                            "si_change_threshold": si_threshold,
                            "horizon_days": horizon,
                            "high_si_n": len(high_values),
                            "other_positive_n": len(other_values),
                            "high_si_mean_return": (
                                float(high_values.mean())
                                if not high_values.empty
                                else float("nan")
                            ),
                            "other_positive_mean_return": (
                                float(other_values.mean())
                                if not other_values.empty
                                else float("nan")
                            ),
                            "mean_return_delta": (
                                float(high_values.mean())
                                - float(other_values.mean())
                                if (
                                    not high_values.empty
                                    and not other_values.empty
                                )
                                else float("nan")
                            ),
                        }
                    )

    return pd.DataFrame(rows)


def run_si_price_dynamics(
    context,
    *,
    horizons: tuple[int, ...],
    si_change_cutoffs: tuple[float, ...],
    event_risk_cutoffs: tuple[float, ...],
    prior_return_columns: tuple[str, ...],
) -> ExperimentResult:
    (
        pretest,
        test,
        model_name,
        features,
        validation_auc,
    ) = _prepare_event_risk_data(
        context
    )

    required_columns = [
        "short_interest_pct_change",
        "event_score",
        "price_date",
        "min_return_5d",
        "min_return_5d_date",
    ]

    missing = [
        column
        for column in required_columns
        if column not in test.columns
    ]

    if missing:
        raise KeyError(
            "SI price dynamics saknar kolumner: "
            + ", ".join(missing)
        )

    thresholds = {
        cutoff: _positive_change_threshold(
            pretest,
            cutoff,
        )
        for cutoff in si_change_cutoffs
    }

    result = ExperimentResult(
        name="si_price_dynamics",
        description=(
            "Testar hur priset utvecklas före och "
            "efter förändringar i short interest, "
            "inklusive event-risk och kontroll "
            "för tidigare prisrörelse."
        ),
    )

    result.add_metric(
        "event_model",
        model_name,
    )

    result.add_metric(
        "event_features",
        list(features),
    )

    result.add_metric(
        "validation_auc",
        validation_auc,
    )

    result.add_metric(
        "si_change_thresholds",
        {
            str(key): value
            for key, value in thresholds.items()
        },
    )

    result.add_table(
        "si_change_price_dynamics",
        _build_price_dynamics_table(
            test,
            horizons=horizons,
            si_change_cutoffs=si_change_cutoffs,
            thresholds=thresholds,
        ),
    )

    result.add_table(
        "bottom_timing",
        _build_bottom_timing_table(
            test,
            si_change_cutoffs=si_change_cutoffs,
            thresholds=thresholds,
        ),
    )

    result.add_table(
        "event_risk_interaction",
        _build_event_risk_table(
            pretest,
            test,
            horizons=horizons,
            si_change_cutoffs=si_change_cutoffs,
            event_risk_cutoffs=event_risk_cutoffs,
            thresholds=thresholds,
        ),
    )

    result.add_table(
        "prior_price_control",
        _build_prior_price_control_table(
            pretest,
            test,
            horizons=horizons,
            si_change_cutoffs=si_change_cutoffs,
            prior_return_columns=prior_return_columns,
            thresholds=thresholds,
        ),
    )

    return result
