from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from .base import ExperimentResult
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
def _return_column(horizon: int) -> str:
    return f"forward_return_{horizon}d"
def _add_short_interest_dynamics(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    frame = frame.copy()
    if "short_interest_pct" not in frame.columns:
        raise KeyError(
            "Event-risk grid requires 'short_interest_pct'."
        )
    if "snapshot_date" not in frame.columns:
        raise KeyError(
            "Event-risk grid requires 'snapshot_date'."
        )
    if "yahoo_symbol" in frame.columns:
        group_column = "yahoo_symbol"
    elif "security_key" in frame.columns:
        group_column = "security_key"
    else:
        raise KeyError(
            "Event-risk grid requires either "
            "'yahoo_symbol' or 'security_key'."
        )
    frame["short_interest_pct"] = _numeric(
        frame,
        "short_interest_pct",
    )
    frame["snapshot_date"] = pd.to_datetime(
        frame["snapshot_date"],
        errors="coerce",
    )
    frame = frame.sort_values(
        [
            group_column,
            "snapshot_date",
        ]
    ).copy()
    previous = (
        frame
        .groupby(
            group_column,
            sort=False,
        )["short_interest_pct"]
        .shift(1)
    )
    current = frame["short_interest_pct"]
    frame["short_interest_pct_change"] = (
        current - previous
    )
    return frame
def _prepare_frame(
    frame: pd.DataFrame,
    horizon: int,
) -> pd.DataFrame:
    frame = _add_short_interest_dynamics(
        frame
    )
    return_column = _return_column(
        horizon
    )
    if return_column not in frame.columns:
        raise KeyError(
            f"Missing required return column: "
            f"{return_column}"
        )
    frame[return_column] = _numeric(
        frame,
        return_column,
    )
    return frame
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
            "y": _numeric(
                pd.DataFrame(
                    {"x": y_true}
                ),
                "x",
            ),
            "score": _numeric(
                pd.DataFrame(
                    {"x": scores}
                ),
                "x",
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
    context,
    horizon: int,
) -> list[dict]:
    train = _prepare_frame(
        context.train,
        horizon,
    )
    validation = _prepare_frame(
        context.validation,
        horizon,
    )
    pretest = _prepare_frame(
        context.pretest,
        horizon,
    )
    test = _prepare_frame(
        context.test,
        horizon,
    )
    return_column = _return_column(
        horizon
    )
    rows: list[dict] = []
    for event_threshold in (
        0.03,
        0.05,
        0.07,
        0.10,
    ):
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
        change = _numeric(
            pretest,
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
                test,
                "short_interest_pct_change",
            )
            test["high_si_change"] = (
                test_change >= si_threshold
            )
            for risk_cutoff in RISK_CUTOFFS:
                risk_threshold = (
                    _tail_threshold(
                        pretest[
                            "event_score"
                        ],
                        risk_cutoff,
                    )
                )
                if not np.isfinite(
                    risk_threshold
                ):
                    continue
                test["high_event_risk"] = (
                    test["event_score"]
                    >= risk_threshold
                )
                for downside_target in (
                    DOWNSIDE_TARGETS
                ):
                    target = (
                        _numeric(
                            test,
                            return_column,
                        )
                        <= downside_target
                    )
                    local = test[
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
def run_event_risk_grid(
    context,
) -> ExperimentResult:
    all_rows: list[dict] = []
    for horizon in HORIZONS:
        rows = _run_grid_for_horizon(
            context,
            horizon,
        )
        all_rows.extend(rows)
    result = ExperimentResult(
        name="si_event_risk_grid",
        description=(
            "Systematiskt walk-forward-test av "
            "interaktionen mellan short-interest-"
            "förändring och extrem event-risk över "
            "flera horisonter, risknivåer och "
            "nedgångsmål."
        ),
    )
    result.add_metric(
        "horizons",
        list(HORIZONS),
    )
    result.add_metric(
        "event_thresholds",
        [
            0.03,
            0.05,
            0.07,
            0.10,
        ],
    )
    result.add_metric(
        "risk_cutoffs",
        list(RISK_CUTOFFS),
    )
    result.add_metric(
        "si_change_cutoffs",
        list(SI_CHANGE_CUTOFFS),
    )
    result.add_metric(
        "downside_targets",
        list(DOWNSIDE_TARGETS),
    )
    result.add_metric(
        "result_rows",
        len(all_rows),
    )
    result.add_table(
        "interaction_grid",
        pd.DataFrame(all_rows),
    )
    return result
