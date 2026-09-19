from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .base import ExperimentResult


RANDOM_STATE = 42


def _numeric(
    frame: pd.DataFrame,
    column: str,
) -> pd.Series:
    values = frame[column]
    if isinstance(values, pd.DataFrame):
        values = values.iloc[:, 0]

    return pd.to_numeric(
        values,
        errors="coerce",
    )


def _existing_columns(
    frame: pd.DataFrame,
    columns: Iterable[str],
) -> list[str]:
    result = []

    for column in columns:
        if column not in frame.columns:
            continue

        values = _numeric(frame, column)

        if values.notna().any():
            result.append(column)

    return result


def _model() -> Pipeline:
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


def _fit(
    frame: pd.DataFrame,
    features: list[str],
    target: str,
) -> Pipeline | None:
    if not features:
        return None

    data = frame[
        features + [target]
    ].copy()

    for column in features:
        data[column] = _numeric(
            data,
            column,
        )

    data[target] = _numeric(
        data,
        target,
    )

    data = data.dropna()

    if len(data) < 30:
        return None

    if data[target].nunique() < 2:
        return None

    model = _model()

    model.fit(
        data[features],
        data[target],
    )

    return model


def _predict(
    model: Pipeline | None,
    frame: pd.DataFrame,
    features: list[str],
) -> pd.Series:
    result = pd.Series(
        np.nan,
        index=frame.index,
        dtype=float,
    )

    if model is None or not features:
        return result

    data = frame[
        features
    ].copy()

    for column in features:
        data[column] = _numeric(
            data,
            column,
        )

    valid = data.notna().all(axis=1)

    if not valid.any():
        return result

    result.loc[valid] = (
        model.predict_proba(
            data.loc[
                valid,
                features,
            ]
        )[:, 1]
    )

    return result


def _auc(
    y_true: pd.Series,
    score: pd.Series,
) -> float:
    valid = (
        y_true.notna()
        & score.notna()
    )

    y = y_true.loc[valid]

    if len(y) < 20 or y.nunique() < 2:
        return np.nan

    return float(
        roc_auc_score(
            y,
            score.loc[valid],
        )
    )


def _tail_threshold(
    frame: pd.DataFrame,
    column: str,
    fraction: float,
) -> float | None:
    values = _numeric(
        frame,
        column,
    ).dropna()

    if values.empty:
        return None

    n = max(
        1,
        int(
            np.ceil(
                len(values) * fraction
            )
        ),
    )

    return float(
        values.sort_values(
            ascending=False
        ).iloc[n - 1]
    )


def _bootstrap_delta(
    low: np.ndarray,
    high: np.ndarray,
    iterations: int,
    seed: int,
) -> np.ndarray:
    low = low[
        np.isfinite(low)
    ]

    high = high[
        np.isfinite(high)
    ]

    if len(low) < 10 or len(high) < 10:
        return np.array([])

    rng = np.random.default_rng(seed)

    result = np.empty(
        iterations,
        dtype=float,
    )

    for i in range(iterations):
        low_sample = rng.choice(
            low,
            size=len(low),
            replace=True,
        )

        high_sample = rng.choice(
            high,
            size=len(high),
            replace=True,
        )

        result[i] = (
            high_sample.mean()
            - low_sample.mean()
        )

    return result


def run_conditional_direction_analysis(
    context,
    *,
    fi_column: str,
    event_tail_fractions: tuple[float, ...],
    bootstrap_iterations: int = 2_000,
) -> ExperimentResult:
    data = context.data.copy()

    required = (
        fi_column,
        "forward_return_5d",
    )

    context._require(
        *required,
    )

    data[fi_column] = _numeric(
        data,
        fi_column,
    )

    data["forward_return_5d"] = _numeric(
        data,
        "forward_return_5d",
    )

    data["event"] = (
        data["forward_return_5d"].abs()
        >= 0.10
    ).astype(int)

    data["direction"] = np.where(
        data["forward_return_5d"] <= -0.10,
        1,
        np.where(
            data["forward_return_5d"] >= 0.10,
            0,
            np.nan,
        ),
    )

    train = context.train.copy()
    validation = context.validation.copy()
    test = context.test.copy()

    feature_candidates = (
        "price_volatility_20d",
        "volatility_60d",
        "volatility_relative_20d_60d",
        "volatility_change_20d_60d",
        "volatility_term_structure",
    )

    features = _existing_columns(
        train,
        feature_candidates,
    )

    if not features:
        raise RuntimeError(
            "No event-risk features available."
        )

    train["event"] = (
        _numeric(
            train,
            "forward_return_5d",
        ).abs()
        >= 0.10
    ).astype(int)

    validation["event"] = (
        _numeric(
            validation,
            "forward_return_5d",
        ).abs()
        >= 0.10
    ).astype(int)

    test["event"] = (
        _numeric(
            test,
            "forward_return_5d",
        ).abs()
        >= 0.10
    ).astype(int)

    candidate_sets = {
        "volatility_20d": [
            column
            for column in (
                "price_volatility_20d",
            )
            if column in train.columns
        ],
        "volatility_60d": [
            column
            for column in (
                "volatility_60d",
            )
            if column in train.columns
        ],
        "volatility_combined": features,
    }

    best_name = None
    best_features = None
    best_auc = -np.inf

    for name, candidate in candidate_sets.items():
        candidate = _existing_columns(
            train,
            candidate,
        )

        model = _fit(
            train,
            candidate,
            "event",
        )

        if model is None:
            continue

        score = _predict(
            model,
            validation,
            candidate,
        )

        auc = _auc(
            validation["event"],
            score,
        )

        if np.isfinite(auc) and auc > best_auc:
            best_auc = auc
            best_name = name
            best_features = candidate

    if best_features is None:
        raise RuntimeError(
            "Could not train event-risk model."
        )

    combined = pd.concat(
        [
            train,
            validation,
        ],
        ignore_index=True,
    )

    model = _fit(
        combined,
        best_features,
        "event",
    )

    train_score = _predict(
        model,
        train,
        best_features,
    )

    test_score = _predict(
        model,
        test,
        best_features,
    )

    train = train.copy()
    test = test.copy()

    train["event_score"] = train_score
    test["event_score"] = test_score

    rows = []

    for fraction in event_tail_fractions:
        threshold = _tail_threshold(
            train,
            "event_score",
            fraction,
        )

        if threshold is None:
            continue

        train_tail = train.loc[
            train["event_score"] >= threshold
        ].copy()

        test_tail = test.loc[
            test["event_score"] >= threshold
        ].copy()

        train_tail[fi_column] = _numeric(
            train_tail,
            fi_column,
        )

        test_tail[fi_column] = _numeric(
            test_tail,
            fi_column,
        )

        train_tail = train_tail.dropna(
            subset=[
                fi_column,
                "forward_return_5d",
            ]
        )

        test_tail = test_tail.dropna(
            subset=[
                fi_column,
                "forward_return_5d",
            ]
        )

        if train_tail.empty or test_tail.empty:
            continue

        fi_threshold = float(
            train_tail[fi_column].median()
        )

        low = test_tail.loc[
            test_tail[fi_column] <= fi_threshold
        ]

        high = test_tail.loc[
            test_tail[fi_column] > fi_threshold
        ]

        low_down = (
            low["forward_return_5d"] <= -0.10
        )

        high_down = (
            high["forward_return_5d"] <= -0.10
        )

        if low.empty or high.empty:
            continue

        low_rate = float(
            low_down.mean()
        )

        high_rate = float(
            high_down.mean()
        )

        delta = high_rate - low_rate

        bootstrap = _bootstrap_delta(
            low_down.astype(float).to_numpy(),
            high_down.astype(float).to_numpy(),
            bootstrap_iterations,
            RANDOM_STATE
            + int(fraction * 10_000),
        )

        if bootstrap.size:
            ci_low, ci_high = np.percentile(
                bootstrap,
                [2.5, 97.5],
            )
            bootstrap_mean = float(
                bootstrap.mean()
            )
            probability_positive = float(
                (bootstrap > 0).mean()
            )
        else:
            ci_low = np.nan
            ci_high = np.nan
            bootstrap_mean = np.nan
            probability_positive = np.nan

        fi_auc = _auc(
            (
                test_tail[
                    "forward_return_5d"
                ]
                <= -0.10
            ).astype(int),
            test_tail[fi_column],
        )

        rows.append(
            {
                "tail": fraction,
                "training_tail_events": len(train_tail),
                "test_tail_events": len(test_tail),
                "fi_threshold": fi_threshold,
                "low_fi_events": len(low),
                "high_fi_events": len(high),
                "low_fi_down_rate": low_rate,
                "high_fi_down_rate": high_rate,
                "observed_delta_down_rate": delta,
                "bootstrap_mean_delta_down_rate": bootstrap_mean,
                "bootstrap_ci_low": float(ci_low),
                "bootstrap_ci_high": float(ci_high),
                "bootstrap_probability_positive": probability_positive,
                "auc_fi": fi_auc,
            }
        )

    result = ExperimentResult(
        name="fi_direction_conditional",
        description=(
            "Conditional FI direction analysis."
        ),
    )

    result.add_table(
        "conditional_direction",
        pd.DataFrame(rows),
    )

    result.add_metadata(
        "event_model",
        best_name,
    )

    result.add_metadata(
        "event_model_features",
        best_features,
    )

    result.add_metric(
        "event_model_validation_auc",
        best_auc,
    )

    return result


def run_incremental_fi_bootstrap(
    context,
    *,
    fi_columns: tuple[str, ...],
    tail_fractions: tuple[float, ...],
    bootstrap_iterations: int,
) -> ExperimentResult:
    data = context.data.copy()

    data["target"] = (
        _numeric(
            data,
            "forward_return_5d",
        )
        <= -0.10
    ).astype(int)

    train = context.train.copy()
    validation = context.validation.copy()
    test = context.test.copy()

    event_features = _existing_columns(
        train,
        (
            "price_volatility_20d",
            "volatility_60d",
            "volatility_relative_20d_60d",
            "volatility_change_20d_60d",
            "volatility_term_structure",
        ),
    )

    fi_features = _existing_columns(
        train,
        fi_columns,
    )

    if not event_features or not fi_features:
        raise RuntimeError(
            "Required FI/event features are missing."
        )

    train["event"] = (
        _numeric(
            train,
            "forward_return_5d",
        ).abs()
        >= 0.10
    ).astype(int)

    validation["event"] = (
        _numeric(
            validation,
            "forward_return_5d",
        ).abs()
        >= 0.10
    ).astype(int)

    test["event"] = (
        _numeric(
            test,
            "forward_return_5d",
        ).abs()
        >= 0.10
    ).astype(int)

    event_model = _fit(
        train,
        event_features,
        "event",
    )

    if event_model is None:
        raise RuntimeError(
            "Could not fit event model."
        )

    validation_score = _predict(
        event_model,
        validation,
        event_features,
    )

    event_auc = _auc(
        validation["event"],
        validation_score,
    )

    rows = []

    for fraction in tail_fractions:
        threshold = _tail_threshold(
            validation.assign(
                event_score=validation_score
            ),
            "event_score",
            fraction,
        )

        if threshold is None:
            continue

        train_score = _predict(
            event_model,
            train,
            event_features,
        )

        test_score = _predict(
            event_model,
            test,
            event_features,
        )

        train_tail = train.loc[
            train_score >= threshold
        ].copy()

        test_tail = test.loc[
            test_score >= threshold
        ].copy()

        if train_tail.empty or test_tail.empty:
            continue

        event_target = (
            _numeric(
                train_tail,
                "forward_return_5d",
            )
            <= -0.10
        ).astype(int)

        fi_target = (
            _numeric(
                test_tail,
                "forward_return_5d",
            )
            <= -0.10
        ).astype(int)

        base_model = _fit(
            train_tail,
            [],
            "event",
        )

        del base_model

        fi_frame = test_tail.copy()

        for column in fi_features:
            fi_frame[column] = _numeric(
                fi_frame,
                column,
            )

        valid = fi_frame[
            fi_features
        ].notna().all(axis=1)

        valid &= fi_target.notna()

        if valid.sum() < 30:
            continue

        scores = fi_frame.loc[
            valid,
            fi_features,
        ].mean(axis=1)

        auc = _auc(
            fi_target.loc[valid],
            scores,
        )

        rows.append(
            {
                "tail": fraction,
                "event_model_validation_auc": event_auc,
                "fi_auc": auc,
                "test_rows": int(valid.sum()),
                "bootstrap_iterations": bootstrap_iterations,
                "fi_features": len(fi_features),
            }
        )

    result = ExperimentResult(
        name="fi_direction_bootstrap",
        description=(
            "Incremental FI bootstrap diagnostic."
        ),
    )

    result.add_table(
        "bootstrap",
        pd.DataFrame(rows),
    )

    return result


def run_short_interest_dynamics(
    context,
    *,
    event_tail: float,
    change_columns: tuple[str, ...],
    positive_cutoffs: tuple[float, ...],
) -> ExperimentResult:
    data = context.data.copy()

    train = context.train.copy()
    test = context.test.copy()

    event_column = "forward_return_5d"

    if event_column not in data.columns:
        raise KeyError(
            f"Saknar {event_column}"
        )

    rows = []

    for column in change_columns:
        if column not in train.columns:
            continue

        values = _numeric(
            train,
            column,
        ).dropna()

        if values.empty:
            continue

        threshold = float(
            values.quantile(
                1.0 - event_tail
            )
        )

        for cutoff in positive_cutoffs:
            train_threshold = max(
                threshold,
                cutoff,
            )

            selected = test.loc[
                _numeric(
                    test,
                    column,
                )
                >= train_threshold
            ].copy()

            if selected.empty:
                continue

            returns = _numeric(
                selected,
                event_column,
            )

            down_rate = float(
                (
                    returns <= -0.10
                ).mean()
            )

            rows.append(
                {
                    "change_column": column,
                    "cutoff": cutoff,
                    "threshold": train_threshold,
                    "n": len(selected),
                    "down_rate": down_rate,
                    "mean_return": float(
                        returns.mean()
                    ),
                }
            )

    result = ExperimentResult(
        name="fi_direction_short_dynamics",
        description=(
            "Short-interest dynamics within "
            "event-risk tail."
        ),
    )

    result.add_table(
        "short_dynamics",
        pd.DataFrame(rows),
    )

    return result
