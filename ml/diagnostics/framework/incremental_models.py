from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
MIN_MODEL_ROWS = 100


@dataclass(frozen=True)
class IncrementalModelSpec:
    name: str
    features: tuple[str, ...]


def _numeric(
    frame: pd.DataFrame,
    column: str,
) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(
            np.nan,
            index=frame.index,
            dtype=float,
        )

    return pd.to_numeric(
        frame[column],
        errors="coerce",
    ).replace(
        [np.inf, -np.inf],
        np.nan,
    )


def _build_model() -> Pipeline:
    return Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
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


def _prepare_features(
    frame: pd.DataFrame,
    features: tuple[str, ...],
) -> pd.DataFrame:
    missing = [
        column
        for column in features
        if column not in frame.columns
    ]

    if missing:
        raise KeyError(
            "Incremental model saknar kolumner: "
            + ", ".join(missing)
        )

    values = frame[
        list(features)
    ].copy()

    for column in features:
        values[column] = _numeric(
            values,
            column,
        )

    return values


def fit_model(
    frame: pd.DataFrame,
    features: tuple[str, ...],
    target_column: str,
    target_threshold: float,
) -> Pipeline:
    values = _prepare_features(
        frame,
        features,
    )

    target_values = _numeric(
        frame,
        target_column,
    )

    valid = (
        target_values.notna()
        & values.notna().any(axis=1)
    )

    values = values.loc[valid]

    target = (
        target_values.loc[valid]
        <= target_threshold
    ).astype(float)

    if len(values) < MIN_MODEL_ROWS:
        raise RuntimeError(
            "För få rader för incremental model: "
            f"{len(values)}"
        )

    if target.nunique() < 2:
        raise RuntimeError(
            "Incremental target innehåller "
            "inte båda klasserna."
        )

    model = _build_model()

    model.fit(
        values[list(features)],
        target,
    )

    return model


def predict(
    model: Pipeline,
    frame: pd.DataFrame,
    features: tuple[str, ...],
) -> pd.Series:
    values = _prepare_features(
        frame,
        features,
    )

    scores = pd.Series(
        np.nan,
        index=frame.index,
        dtype=float,
    )

    valid = values.notna().any(axis=1)

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


def classification_metrics(
    frame: pd.DataFrame,
    scores: pd.Series,
    target_column: str,
    target_threshold: float,
) -> dict[str, float | int]:
    target_values = _numeric(
        frame,
        target_column,
    )

    target = (
        target_values
        <= target_threshold
    ).astype(float)

    working = pd.DataFrame(
        {
            "target": target,
            "score": scores,
        }
    ).dropna()

    if len(working) < 2:
        return {
            "n": 0,
            "down_rate": float("nan"),
            "auc": float("nan"),
            "brier": float("nan"),
            "log_loss": float("nan"),
        }

    if working["target"].nunique() < 2:
        auc = float("nan")
    else:
        auc = float(
            roc_auc_score(
                working["target"],
                working["score"],
            )
        )

    clipped = working["score"].clip(
        1e-6,
        1.0 - 1e-6,
    )

    return {
        "n": int(len(working)),
        "down_rate": float(
            working["target"].mean()
        ),
        "auc": auc,
        "brier": float(
            brier_score_loss(
                working["target"],
                working["score"],
            )
        ),
        "log_loss": float(
            log_loss(
                working["target"],
                clipped,
                labels=[0.0, 1.0],
            )
        ),
    }


def _safe_delta(
    value: float,
    baseline: float,
) -> float:
    if (
        pd.isna(value)
        or pd.isna(baseline)
    ):
        return float("nan")

    return float(
        value - baseline
    )


def compare_incremental_models(
    *,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    pretest: pd.DataFrame,
    test: pd.DataFrame,
    model_specs: tuple[IncrementalModelSpec, ...],
    target_column: str,
    target_threshold: float,
) -> pd.DataFrame:
    rows: list[dict] = []

    for spec in model_specs:
        train_model = fit_model(
            train,
            spec.features,
            target_column,
            target_threshold,
        )

        validation_scores = predict(
            train_model,
            validation,
            spec.features,
        )

        validation_metrics = (
            classification_metrics(
                validation,
                validation_scores,
                target_column,
                target_threshold,
            )
        )

        pretest_model = fit_model(
            pretest,
            spec.features,
            target_column,
            target_threshold,
        )

        pretest_scores = predict(
            pretest_model,
            pretest,
            spec.features,
        )

        test_scores = predict(
            pretest_model,
            test,
            spec.features,
        )

        test_metrics = classification_metrics(
            test,
            test_scores,
            target_column,
            target_threshold,
        )

        rows.append(
            {
                "model": spec.name,
                "features": ",".join(
                    spec.features
                ),

                "validation_n": (
                    validation_metrics["n"]
                ),
                "validation_auc": (
                    validation_metrics["auc"]
                ),
                "validation_brier": (
                    validation_metrics["brier"]
                ),
                "validation_log_loss": (
                    validation_metrics["log_loss"]
                ),

                "test_n": (
                    test_metrics["n"]
                ),
                "test_down_rate": (
                    test_metrics["down_rate"]
                ),
                "test_auc": (
                    test_metrics["auc"]
                ),
                "test_brier": (
                    test_metrics["brier"]
                ),
                "test_log_loss": (
                    test_metrics["log_loss"]
                ),
            }
        )

    baseline = next(
        (
            row
            for row in rows
            if row["model"] == model_specs[0].name
        ),
        None,
    )

    if baseline is not None:
        for row in rows:
            row["delta_test_auc_vs_m0"] = (
                _safe_delta(
                    row["test_auc"],
                    baseline["test_auc"],
                )
            )

            row["delta_test_brier_vs_m0"] = (
                _safe_delta(
                    row["test_brier"],
                    baseline["test_brier"],
                )
            )

            row["delta_test_log_loss_vs_m0"] = (
                _safe_delta(
                    row["test_log_loss"],
                    baseline["test_log_loss"],
                )
            )

    return pd.DataFrame(rows)
