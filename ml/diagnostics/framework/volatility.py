from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .base import ExperimentResult


RANDOM_STATE = 42


def _numeric(
    frame: pd.DataFrame,
    column: str,
) -> pd.Series:
    return pd.to_numeric(
        frame[column],
        errors="coerce",
    )


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

    if valid.any():
        result.loc[valid] = (
            model.predict_proba(
                data.loc[
                    valid,
                    features,
                ]
            )[:, 1]
        )

    return result


def _target(
    frame: pd.DataFrame,
    target: str,
) -> pd.Series:
    if target in frame.columns:
        return _numeric(
            frame,
            target,
        ).astype(float)

    if target == "down_5pct_5d":
        return (
            _numeric(
                frame,
                "forward_return_5d",
            )
            <= -0.05
        ).astype(int)

    if target == "down_3pct_5d":
        return (
            _numeric(
                frame,
                "forward_return_5d",
            )
            <= -0.03
        ).astype(int)

    if target == "down_7pct_5d":
        return (
            _numeric(
                frame,
                "forward_return_5d",
            )
            <= -0.07
        ).astype(int)

    if target == "down_10pct_5d":
        return (
            _numeric(
                frame,
                "forward_return_5d",
            )
            <= -0.10
        ).astype(int)

    raise KeyError(
        f"Target saknas: {target}"
    )


def _auc(
    y: pd.Series,
    score: pd.Series,
) -> float:
    valid = (
        y.notna()
        & score.notna()
    )

    if valid.sum() < 20:
        return np.nan

    y_valid = y.loc[valid]

    if y_valid.nunique() < 2:
        return np.nan

    return float(
        roc_auc_score(
            y_valid,
            score.loc[valid],
        )
    )


def _evaluate(
    y: pd.Series,
    score: pd.Series,
) -> dict[str, Any]:
    valid = (
        y.notna()
        & score.notna()
    )

    if valid.sum() == 0:
        return {
            "n": 0,
            "auc": np.nan,
            "brier": np.nan,
            "log_loss": np.nan,
        }

    y_valid = y.loc[valid]
    score_valid = score.loc[valid]

    result = {
        "n": int(valid.sum()),
        "auc": _auc(
            y_valid,
            score_valid,
        ),
        "brier": float(
            brier_score_loss(
                y_valid,
                score_valid,
            )
        ),
    }

    if y_valid.nunique() >= 2:
        result["log_loss"] = float(
            log_loss(
                y_valid,
                score_valid,
                labels=[0, 1],
            )
        )
    else:
        result["log_loss"] = np.nan

    return result


def _feature_set_results(
    context,
    feature_sets: dict[str, tuple[str, ...]],
    target: str,
) -> ExperimentResult:
    train = context.train.copy()
    validation = context.validation.copy()
    test = context.test.copy()

    y_train = _target(
        train,
        target,
    )

    y_validation = _target(
        validation,
        target,
    )

    y_test = _target(
        test,
        target,
    )

    rows = []

    for name, requested in feature_sets.items():
        features = [
            column
            for column in requested
            if column in train.columns
        ]

        model = _fit(
            train.assign(
                _target=y_train
            ),
            features,
            "_target",
        )

        if model is None:
            continue

        validation_score = _predict(
            model,
            validation,
            features,
        )

        validation_metrics = _evaluate(
            y_validation,
            validation_score,
        )

        combined = pd.concat(
            [
                train,
                validation,
            ],
            ignore_index=True,
        )

        combined_target = pd.concat(
            [
                y_train,
                y_validation,
            ],
            ignore_index=True,
        )

        final_model = _fit(
            combined.assign(
                _target=combined_target
            ),
            features,
            "_target",
        )

        test_score = _predict(
            final_model,
            test,
            features,
        )

        test_metrics = _evaluate(
            y_test,
            test_score,
        )

        rows.append(
            {
                "feature_set": name,
                "features": ",".join(features),
                "validation_n": validation_metrics["n"],
                "validation_auc": validation_metrics["auc"],
                "validation_brier": validation_metrics["brier"],
                "validation_log_loss": validation_metrics["log_loss"],
                "test_n": test_metrics["n"],
                "test_auc": test_metrics["auc"],
                "test_brier": test_metrics["brier"],
                "test_log_loss": test_metrics["log_loss"],
            }
        )

    result = ExperimentResult(
        name="volatility_screening",
        description=(
            "Volatility logistic screening."
        ),
    )

    result.add_table(
        "models",
        pd.DataFrame(rows),
    )

    return result


def run_logistic_screen(
    context,
    *,
    feature_sets: dict[str, tuple[str, ...]],
    target: str,
) -> ExperimentResult:
    return _feature_set_results(
        context,
        feature_sets,
        target,
    )


def run_feature_set_comparison(
    context,
    *,
    feature_sets: dict[str, str],
    target: str,
) -> ExperimentResult:
    definitions = {
        "fi": (
            "short_interest",
            "short_interest_change",
            "short_interest_pct",
            "short_interest_delta",
            "short_interest_rank",
            "short_interest_zscore",
        ),
        "volatility": (
            "price_volatility_20d",
        ),
        "fi_plus_volatility": (
            "short_interest",
            "short_interest_change",
            "short_interest_pct",
            "short_interest_delta",
            "short_interest_rank",
            "short_interest_zscore",
            "price_volatility_20d",
        ),
    }

    resolved = {}

    for name, definition in feature_sets.items():
        if isinstance(
            definition,
            str,
        ):
            resolved[name] = definitions.get(
                definition,
                (),
            )
        else:
            resolved[name] = tuple(
                definition
            )

    return _feature_set_results(
        context,
        resolved,
        target,
    )


def run_interaction_screen(
    context,
    *,
    base_features: tuple[str, ...],
    interactions: dict[str, tuple[str, str]],
    target: str,
) -> ExperimentResult:
    train = context.train.copy()
    validation = context.validation.copy()
    test = context.test.copy()

    for name, (left, right) in interactions.items():
        for frame in (
            train,
            validation,
            test,
        ):
            if (
                left in frame.columns
                and right in frame.columns
            ):
                frame[name] = (
                    _numeric(frame, left)
                    * _numeric(frame, right)
                )

    feature_sets = {}

    base = tuple(
        column
        for column in base_features
        if column in train.columns
    )

    for name, interaction in interactions.items():
        features = base + (
            name,
        )

        features = tuple(
            column
            for column in features
            if column in train.columns
        )

        feature_sets[name] = features

    return _feature_set_results(
        context,
        feature_sets,
        target,
    )


def run_descriptive_volatility_analysis(
    context,
    *,
    volatility_column: str,
    targets: tuple[str, ...],
) -> ExperimentResult:
    pretest = context.pretest.copy()
    test = context.test.copy()

    if volatility_column not in pretest.columns:
        raise KeyError(
            f"Saknar {volatility_column}"
        )

    thresholds = {
        "q50": float(
            _numeric(
                pretest,
                volatility_column,
            ).quantile(0.50)
        ),
        "q80": float(
            _numeric(
                pretest,
                volatility_column,
            ).quantile(0.80)
        ),
        "q90": float(
            _numeric(
                pretest,
                volatility_column,
            ).quantile(0.90)
        ),
    }

    rows = []

    volatility = _numeric(
        test,
        volatility_column,
    )

    for regime, threshold in thresholds.items():
        if regime == "q50":
            mask = volatility <= threshold
        elif regime == "q80":
            mask = volatility > thresholds["q50"]
            mask &= volatility <= threshold
        else:
            mask = volatility > threshold

        subset = test.loc[
            mask
        ].copy()

        for target in targets:
            y = _target(
                subset,
                target,
            )

            returns = (
                _numeric(
                    subset,
                    "forward_return_5d",
                )
                if "forward_return_5d"
                in subset.columns
                else pd.Series(
                    np.nan,
                    index=subset.index,
                )
            )

            rows.append(
                {
                    "regime": regime,
                    "n": len(subset),
                    "target": target,
                    "event_rate": float(
                        y.mean()
                    )
                    if len(y)
                    else np.nan,
                    "mean_return": float(
                        returns.mean()
                    )
                    if returns.notna().any()
                    else np.nan,
                }
            )

    result = ExperimentResult(
        name="volatility_diagnostics",
        description=(
            "Descriptive volatility diagnostics."
        ),
    )

    result.add_table(
        "regimes",
        pd.DataFrame(rows),
    )

    result.add_metadata(
        "training_thresholds",
        thresholds,
    )

    return result


def run_directional_tail_analysis(
    context,
    *,
    volatility_column: str,
    event_tail_fractions: tuple[float, ...],
) -> ExperimentResult:
    train = context.train.copy()
    test = context.test.copy()

    train["event_size"] = _numeric(
        train,
        "forward_return_5d",
    ).abs()

    test["event_size"] = _numeric(
        test,
        "forward_return_5d",
    ).abs()

    rows = []

    for fraction in event_tail_fractions:
        threshold = float(
            train["event_size"].quantile(
                1.0 - fraction
            )
        )

        subset = test.loc[
            test["event_size"] >= threshold
        ].copy()

        if subset.empty:
            continue

        volatility = _numeric(
            subset,
            volatility_column,
        )

        direction = (
            _numeric(
                subset,
                "forward_return_5d",
            )
            <= 0
        ).astype(int)

        valid = (
            volatility.notna()
            & direction.notna()
        )

        auc = _auc(
            direction.loc[valid],
            volatility.loc[valid],
        )

        rows.append(
            {
                "event_tail": fraction,
                "training_threshold": threshold,
                "n": int(valid.sum()),
                "down_rate": float(
                    direction.loc[valid].mean()
                )
                if valid.any()
                else np.nan,
                "volatility_auc": auc,
                "mean_volatility": float(
                    volatility.loc[valid].mean()
                )
                if valid.any()
                else np.nan,
            }
        )

    result = ExperimentResult(
        name="volatility_directional_tail",
        description=(
            "Volatility direction within "
            "event-risk tails."
        ),
    )

    result.add_table(
        "directional_tail",
        pd.DataFrame(rows),
    )

    return result


def run_volatility_regime(
    context,
    *,
    volatility_column: str,
    strategies: tuple[str, ...],
) -> ExperimentResult:
    fi_features = tuple(
        column
        for column in (
            "short_interest",
            "short_interest_change",
            "short_interest_pct",
            "short_interest_delta",
            "short_interest_rank",
            "short_interest_zscore",
        )
        if column in context.train.columns
    )

    vol_features = fi_features + (
        volatility_column,
    )

    train = context.train.copy()
    validation = context.validation.copy()
    test = context.test.copy()

    y_train = (
        _numeric(
            train,
            "forward_return_5d",
        )
        <= -0.05
    ).astype(int)

    y_validation = (
        _numeric(
            validation,
            "forward_return_5d",
        )
        <= -0.05
    ).astype(int)

    y_test = (
        _numeric(
            test,
            "forward_return_5d",
        )
        <= -0.05
    ).astype(int)

    fi_model = _fit(
        train.assign(
            _target=y_train
        ),
        list(fi_features),
        "_target",
    )

    vol_model = _fit(
        train.assign(
            _target=y_train
        ),
        list(vol_features),
        "_target",
    )

    fi_validation = _predict(
        fi_model,
        validation,
        list(fi_features),
    )

    vol_validation = _predict(
        vol_model,
        validation,
        list(vol_features),
    )

    q1 = float(
        _numeric(
            train,
            volatility_column,
        ).quantile(
            1 / 3
        )
    )

    q2 = float(
        _numeric(
            train,
            volatility_column,
        ).quantile(
            2 / 3
        )
    )

    def regime(values):
        return pd.Series(
            np.where(
                values <= q1,
                "LOW",
                np.where(
                    values <= q2,
                    "MID",
                    "HIGH",
                ),
            ),
            index=values.index,
        )

    validation_regime = regime(
        _numeric(
            validation,
            volatility_column,
        )
    )

    gate = {}

    for name in (
        "LOW",
        "MID",
        "HIGH",
    ):
        mask = (
            validation_regime == name
        )

        fi_auc = _auc(
            y_validation.loc[mask],
            fi_validation.loc[mask],
        )

        vol_auc = _auc(
            y_validation.loc[mask],
            vol_validation.loc[mask],
        )

        gate[name] = (
            "vol"
            if np.isfinite(vol_auc)
            and (
                not np.isfinite(fi_auc)
                or vol_auc > fi_auc
            )
            else "fi"
        )

    fi_test = _predict(
        fi_model,
        test,
        list(fi_features),
    )

    vol_test = _predict(
        vol_model,
        test,
        list(vol_features),
    )

    test_regime = regime(
        _numeric(
            test,
            volatility_column,
        )
    )

    rows = []

    for strategy in strategies:
        if strategy == "fi_only":
            score = fi_test
        elif strategy == "fi_plus_volatility":
            score = vol_test
        elif strategy == "gate_validation":
            score = pd.Series(
                np.where(
                    test_regime.map(gate)
                    == "vol",
                    vol_test,
                    fi_test,
                ),
                index=test.index,
            )
        elif strategy == "gate_vol_mid_high":
            score = pd.Series(
                np.where(
                    test_regime == "LOW",
                    fi_test,
                    vol_test,
                ),
                index=test.index,
            )
        else:
            raise ValueError(
                f"Unknown strategy: {strategy}"
            )

        metrics = _evaluate(
            y_test,
            score,
        )

        rows.append(
            {
                "strategy": strategy,
                **metrics,
            }
        )

    result = ExperimentResult(
        name="volatility_regime",
        description=(
            "Volatility regime gate diagnostic."
        ),
    )

    result.add_table(
        "strategies",
        pd.DataFrame(rows),
    )

    result.add_table(
        "validation_gate",
        pd.DataFrame(
            [
                {
                    "regime": regime_name,
                    "selected": selected,
                }
                for regime_name, selected
                in gate.items()
            ]
        ),
    )

    result.add_metadata(
        "training_regime_q1",
        q1,
    )

    result.add_metadata(
        "training_regime_q2",
        q2,
    )

    return result
