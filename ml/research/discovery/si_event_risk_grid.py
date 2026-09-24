from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
BOOTSTRAP_ITERATIONS = 2_000

EVENT_FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "volatility_20d": (
        "price_volatility_20d",
    ),
    "volatility_60d": (
        "volatility_60d",
    ),
    "volatility_20d_plus_60d": (
        "price_volatility_20d",
        "volatility_60d",
    ),
    "volatility_20d_plus_60d_plus_term_structure": (
        "price_volatility_20d",
        "volatility_60d",
        "volatility_term_structure",
    ),
}

EVENT_THRESHOLDS = (
    0.03,
    0.05,
    0.07,
    0.10,
)

RISK_CUTOFFS = (
    0.50,
    0.25,
    0.10,
    0.05,
    0.01,
)

SI_CHANGE_CUTOFFS = (
    0.50,
    0.25,
    0.10,
    0.05,
)

HORIZONS = (
    1,
    3,
    5,
    10,
    20,
)

DOWNSIDE_TARGETS = (
    -0.03,
    -0.05,
    -0.07,
    -0.10,
)


@dataclass(frozen=True)
class EventModel:
    name: str
    features: tuple[str, ...]
    validation_auc: float
    model: Pipeline


def _numeric(
    frame: pd.DataFrame,
    column: str,
) -> pd.Series:
    return pd.to_numeric(
        frame[column],
        errors="coerce",
    )


def _return_column(
    horizon: int,
) -> str:
    return f"forward_return_{horizon}d"


def add_short_interest_dynamics(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """
    Preserve the old diagnostic semantics.

    Short-interest change is calculated independently inside
    each supplied frame/split.
    """
    result = frame.copy()

    if "short_interest_pct" not in result.columns:
        raise KeyError(
            "Event-risk grid requires 'short_interest_pct'."
        )

    if "snapshot_date" not in result.columns:
        raise KeyError(
            "Event-risk grid requires 'snapshot_date'."
        )

    if "yahoo_symbol" in result.columns:
        group_column = "yahoo_symbol"
    elif "security_key" in result.columns:
        group_column = "security_key"
    else:
        raise KeyError(
            "Event-risk grid requires either "
            "'yahoo_symbol' or 'security_key'."
        )

    result["short_interest_pct"] = _numeric(
        result,
        "short_interest_pct",
    )

    result["snapshot_date"] = pd.to_datetime(
        result["snapshot_date"],
        errors="coerce",
    )

    result = result.sort_values(
        [
            group_column,
            "snapshot_date",
        ]
    ).copy()

    previous = (
        result
        .groupby(
            group_column,
            sort=False,
        )["short_interest_pct"]
        .shift(1)
    )

    result["short_interest_pct_change"] = (
        result["short_interest_pct"]
        - previous
    )

    return result


def _prepare_frame(
    frame: pd.DataFrame,
    horizon: int,
) -> pd.DataFrame:
    result = add_short_interest_dynamics(
        frame
    )

    return_column = _return_column(
        horizon
    )

    if return_column not in result.columns:
        raise KeyError(
            f"Missing required return column: "
            f"{return_column}"
        )

    result[return_column] = _numeric(
        result,
        return_column,
    )

    return result


def _build_model() -> Pipeline:
    return Pipeline(
        [
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=2_000,
                    C=1.0,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def _fit_event_model(
    frame: pd.DataFrame,
    features: tuple[str, ...],
    horizon: int,
    event_threshold: float,
) -> Pipeline | None:
    return_column = _return_column(
        horizon
    )

    missing = [
        column
        for column in features
        if column not in frame.columns
    ]

    if missing:
        return None

    values = frame[
        [
            *features,
            return_column,
        ]
    ].copy()

    for column in features:
        values[column] = _numeric(
            values,
            column,
        )

    values[return_column] = _numeric(
        values,
        return_column,
    )

    values["event"] = (
        values[return_column].abs()
        >= event_threshold
    ).astype(float)

    values = values.dropna()

    if len(values) < 100:
        return None

    if values["event"].nunique() < 2:
        return None

    model = _build_model()

    model.fit(
        values[list(features)],
        values["event"],
    )

    return model


def _predict_event_score(
    model: Pipeline,
    frame: pd.DataFrame,
    features: tuple[str, ...],
) -> pd.Series:
    scores = pd.Series(
        np.nan,
        index=frame.index,
        dtype=float,
    )

    missing = [
        column
        for column in features
        if column not in frame.columns
    ]

    if missing:
        return scores

    values = frame[
        list(features)
    ].copy()

    for column in features:
        values[column] = _numeric(
            values,
            column,
        )

    valid = values.notna().all(
        axis=1
    )

    if not valid.any():
        return scores

    scores.loc[valid] = (
        model.predict_proba(
            values.loc[
                valid,
                list(features),
            ]
        )[:, 1]
    )

    return scores


def _safe_auc(
    y_true: pd.Series,
    scores: pd.Series,
) -> float:
    values = pd.DataFrame(
        {
            "y": pd.to_numeric(
                y_true,
                errors="coerce",
            ),
            "score": pd.to_numeric(
                scores,
                errors="coerce",
            ),
        }
    ).dropna()

    if len(values) < 2:
        return float("nan")

    if values["y"].nunique() < 2:
        return float("nan")

    if values["score"].nunique() < 2:
        return 0.5

    return float(
        roc_auc_score(
            values["y"],
            values["score"],
        )
    )


def _select_event_model(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    horizon: int,
    event_threshold: float,
) -> EventModel:
    best: EventModel | None = None

    return_column = _return_column(
        horizon
    )

    validation_event = (
        _numeric(
            validation,
            return_column,
        ).abs()
        >= event_threshold
    ).astype(float)

    for name, features in EVENT_FEATURE_SETS.items():
        model = _fit_event_model(
            train,
            features,
            horizon,
            event_threshold,
        )

        if model is None:
            continue

        scores = _predict_event_score(
            model,
            validation,
            features,
        )

        auc = _safe_auc(
            validation_event,
            scores,
        )

        if np.isnan(auc):
            continue

        candidate = EventModel(
            name=name,
            features=features,
            validation_auc=float(auc),
            model=model,
        )

        if (
            best is None
            or candidate.validation_auc
            > best.validation_auc
        ):
            best = candidate

    if best is None:
        raise RuntimeError(
            "Could not select an event-risk model for "
            f"horizon={horizon}, "
            f"threshold={event_threshold}."
        )

    return best


def _tail_threshold(
    values: pd.Series,
    fraction: float,
) -> float:
    values = pd.to_numeric(
        values,
        errors="coerce",
    ).dropna()

    if values.empty:
        return float("nan")

    return float(
        values.quantile(
            1.0 - fraction
        )
    )


def _bootstrap_interaction(
    risk_low: pd.Series,
    risk_high: pd.Series,
    outside_low: pd.Series,
    outside_high: pd.Series,
) -> tuple[float, float, float, float]:
    groups = [
        pd.to_numeric(
            values,
            errors="coerce",
        )
        .dropna()
        .to_numpy(dtype=float)
        for values in (
            risk_low,
            risk_high,
            outside_low,
            outside_high,
        )
    ]

    if any(
        len(values) == 0
        for values in groups
    ):
        return (
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
        )

    (
        risk_low_values,
        risk_high_values,
        outside_low_values,
        outside_high_values,
    ) = groups

    observed = (
        risk_high_values.mean()
        - risk_low_values.mean()
        - outside_high_values.mean()
        + outside_low_values.mean()
    )

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    deltas = np.empty(
        BOOTSTRAP_ITERATIONS,
        dtype=float,
    )

    for index in range(
        BOOTSTRAP_ITERATIONS
    ):
        risk_low_sample = rng.choice(
            risk_low_values,
            size=len(risk_low_values),
            replace=True,
        )

        risk_high_sample = rng.choice(
            risk_high_values,
            size=len(risk_high_values),
            replace=True,
        )

        outside_low_sample = rng.choice(
            outside_low_values,
            size=len(outside_low_values),
            replace=True,
        )

        outside_high_sample = rng.choice(
            outside_high_values,
            size=len(outside_high_values),
            replace=True,
        )

        deltas[index] = (
            risk_high_sample.mean()
            - risk_low_sample.mean()
            - outside_high_sample.mean()
            + outside_low_sample.mean()
        )

    return (
        float(observed),
        float(
            np.quantile(
                deltas,
                0.025,
            )
        ),
        float(
            np.quantile(
                deltas,
                0.975,
            )
        ),
        float(
            (deltas > 0).mean()
        ),
    )


def _run_grid_for_horizon(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    pretest: pd.DataFrame,
    test: pd.DataFrame,
    horizon: int,
) -> list[dict]:
    return_column = _return_column(
        horizon
    )

    rows: list[dict] = []

    for event_threshold in EVENT_THRESHOLDS:
        event_model = _select_event_model(
            train,
            validation,
            horizon,
            event_threshold,
        )

        pretest_model = _fit_event_model(
            pretest,
            event_model.features,
            horizon,
            event_threshold,
        )

        if pretest_model is None:
            continue

        pretest_local = pretest.copy()
        test_local = test.copy()

        pretest_local["event_score"] = (
            _predict_event_score(
                pretest_model,
                pretest_local,
                event_model.features,
            )
        )

        test_local["event_score"] = (
            _predict_event_score(
                pretest_model,
                test_local,
                event_model.features,
            )
        )

        change = _numeric(
            pretest_local,
            "short_interest_pct_change",
        )

        positive_change = change[
            change > 0
        ].dropna()

        if positive_change.empty:
            continue

        for si_cutoff in SI_CHANGE_CUTOFFS:
            si_threshold = float(
                positive_change.quantile(
                    1.0 - si_cutoff
                )
            )

            test_change = _numeric(
                test_local,
                "short_interest_pct_change",
            )

            test_local["high_si_change"] = (
                test_change >= si_threshold
            )

            for risk_cutoff in RISK_CUTOFFS:
                risk_threshold = _tail_threshold(
                    pretest_local["event_score"],
                    risk_cutoff,
                )

                if not np.isfinite(
                    risk_threshold
                ):
                    continue

                test_local["high_event_risk"] = (
                    test_local["event_score"]
                    >= risk_threshold
                )

                for downside_target in DOWNSIDE_TARGETS:
                    target = (
                        _numeric(
                            test_local,
                            return_column,
                        )
                        <= downside_target
                    )

                    local = test_local[
                        target.notna()
                    ].copy()

                    local["down"] = (
                        target.loc[
                            local.index
                        ].astype(float)
                    )

                    local = local[
                        local[
                            "high_event_risk"
                        ].notna()
                    ]

                    groups = (
                        local[
                            local[
                                "high_event_risk"
                            ]
                            & local[
                                "high_si_change"
                            ]
                        ]["down"],
                        local[
                            local[
                                "high_event_risk"
                            ]
                            & ~local[
                                "high_si_change"
                            ]
                        ]["down"],
                        local[
                            ~local[
                                "high_event_risk"
                            ]
                            & local[
                                "high_si_change"
                            ]
                        ]["down"],
                        local[
                            ~local[
                                "high_event_risk"
                            ]
                            & ~local[
                                "high_si_change"
                            ]
                        ]["down"],
                    )

                    if any(
                        len(group) == 0
                        for group in groups
                    ):
                        continue

                    (
                        risk_high_si,
                        risk_low_si,
                        outside_high_si,
                        outside_low_si,
                    ) = groups

                    (
                        interaction,
                        ci_low,
                        ci_high,
                        p_positive,
                    ) = _bootstrap_interaction(
                        risk_low_si,
                        risk_high_si,
                        outside_low_si,
                        outside_high_si,
                    )

                    rows.append(
                        {
                            "selection_stage": (
                                "discovery_grid"
                            ),
                            "is_discovery_grid": True,
                            "horizon_days": horizon,
                            "event_threshold": (
                                event_threshold
                            ),
                            "downside_target": (
                                downside_target
                            ),
                            "risk_cutoff": (
                                risk_cutoff
                            ),
                            "risk_threshold": (
                                risk_threshold
                            ),
                            "si_change_cutoff": (
                                si_cutoff
                            ),
                            "si_change_threshold": (
                                si_threshold
                            ),
                            "event_model": (
                                event_model.name
                            ),
                            "event_features": (
                                list(
                                    event_model.features
                                )
                            ),
                            "event_validation_auc": (
                                event_model.validation_auc
                            ),
                            "risk_high_si_n": (
                                len(
                                    risk_high_si
                                )
                            ),
                            "risk_low_si_n": (
                                len(
                                    risk_low_si
                                )
                            ),
                            "outside_high_si_n": (
                                len(
                                    outside_high_si
                                )
                            ),
                            "outside_low_si_n": (
                                len(
                                    outside_low_si
                                )
                            ),
                            "risk_high_si_rate": (
                                float(
                                    risk_high_si.mean()
                                )
                            ),
                            "risk_low_si_rate": (
                                float(
                                    risk_low_si.mean()
                                )
                            ),
                            "outside_high_si_rate": (
                                float(
                                    outside_high_si.mean()
                                )
                            ),
                            "outside_low_si_rate": (
                                float(
                                    outside_low_si.mean()
                                )
                            ),
                            "risk_si_effect": (
                                float(
                                    risk_high_si.mean()
                                    - risk_low_si.mean()
                                )
                            ),
                            "outside_si_effect": (
                                float(
                                    outside_high_si.mean()
                                    - outside_low_si.mean()
                                )
                            ),
                            "interaction": interaction,
                            "interaction_ci_low": ci_low,
                            "interaction_ci_high": ci_high,
                            "interaction_p_positive": (
                                p_positive
                            ),
                        }
                    )

    return rows


def run(
    frame: pd.DataFrame,
    *,
    train_end: str | pd.Timestamp,
    validation_end: str | pd.Timestamp,
    test_end: str | pd.Timestamp,
) -> tuple[
    pd.DataFrame,
    dict[str, object],
]:
    data = frame.copy()

    data["snapshot_date"] = pd.to_datetime(
        data["snapshot_date"],
        errors="coerce",
    )

    train_end = pd.Timestamp(
        train_end
    )

    validation_end = pd.Timestamp(
        validation_end
    )

    test_end = pd.Timestamp(
        test_end
    )

    train = data[
        data["snapshot_date"] <= train_end
    ].copy()

    validation = data[
        (data["snapshot_date"] > train_end)
        & (data["snapshot_date"] <= validation_end)
    ].copy()

    pretest = data[
        data["snapshot_date"] <= validation_end
    ].copy()

    test = data[
        (data["snapshot_date"] > validation_end)
        & (data["snapshot_date"] <= test_end)
    ].copy()

    all_rows: list[dict] = []

    for horizon in HORIZONS:
        prepared_train = _prepare_frame(
            train,
            horizon,
        )

        prepared_validation = _prepare_frame(
            validation,
            horizon,
        )

        prepared_pretest = _prepare_frame(
            pretest,
            horizon,
        )

        prepared_test = _prepare_frame(
            test,
            horizon,
        )

        all_rows.extend(
            _run_grid_for_horizon(
                prepared_train,
                prepared_validation,
                prepared_pretest,
                prepared_test,
                horizon,
            )
        )

    grid = pd.DataFrame(
        all_rows
    )

    if not grid.empty:
        grid["selection_stage"] = (
            "discovery_grid"
        )

        grid["is_discovery_grid"] = True
    else:
        grid = pd.DataFrame(
            columns=[
                "selection_stage",
                "is_discovery_grid",
            ]
        )

    metrics: dict[str, object] = {
        "analysis_type": "discovery_grid",
        "horizons": list(HORIZONS),
        "event_thresholds": list(
            EVENT_THRESHOLDS
        ),
        "risk_cutoffs": list(
            RISK_CUTOFFS
        ),
        "si_change_cutoffs": list(
            SI_CHANGE_CUTOFFS
        ),
        "downside_targets": list(
            DOWNSIDE_TARGETS
        ),
        "result_rows": len(grid),
    }

    return grid, metrics
