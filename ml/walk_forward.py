"""Walk-forward-träning för Blankdiss."""
from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from ml.config import (
    RANDOM_STATE,
    TEST_MIN_ROWS,
    VALIDATION_MIN_ROWS,
    WalkForwardWindow,
)
from ml.evaluate import (
    evaluate_predictions,
    return_by_probability_bucket,
    return_by_top_fraction,
)
from ml.models import build_models


def _split(
    data: pd.DataFrame,
    y: pd.Series,
    window: WalkForwardWindow,
):
    train_end = pd.Timestamp(window.train_end)
    validation_end = pd.Timestamp(window.validation_end)
    test_end = pd.Timestamp(window.test_end)

    train_mask = data["snapshot_date"] <= train_end

    validation_mask = (
        (data["snapshot_date"] > train_end)
        & (data["snapshot_date"] <= validation_end)
    )

    test_mask = (
        (data["snapshot_date"] > validation_end)
        & (data["snapshot_date"] <= test_end)
    )

    return (
        train_mask,
        validation_mask,
        test_mask,
    )


def _features_available_in_training(
    train: pd.DataFrame,
    feature_columns: list[str],
) -> list[str]:
    available: list[str] = []
    removed: list[str] = []

    for column in feature_columns:
        if train[column].notna().any():
            available.append(column)
        else:
            removed.append(column)

    if removed:
        print(
            "Tar bort features som saknar "
            "observerade värden i training:"
        )

        for column in removed:
            print(f"  {column}")

    return available


def roc_auc_safe(
    y_true,
    probabilities,
) -> float:
    from sklearn.metrics import roc_auc_score

    if len(np.unique(y_true)) < 2:
        return float("-inf")

    return float(
        roc_auc_score(
            y_true,
            probabilities,
        )
    )


def _validation_score(
    model,
    X_validation,
    y_validation,
    task: str,
    direction: str,
) -> tuple[float, np.ndarray]:
    if task == "classification":
        predictions = model.predict_proba(
            X_validation
        )[:, 1]

        score = roc_auc_safe(
            y_validation,
            predictions,
        )

        return score, predictions

    predictions = np.asarray(
        model.predict(X_validation),
        dtype=float,
    )

    y_values = np.asarray(
        y_validation,
        dtype=float,
    )

    valid = (
        np.isfinite(predictions)
        & np.isfinite(y_values)
    )

    if not valid.any():
        return float("-inf"), predictions

    errors = np.abs(
        predictions[valid]
        - y_values[valid]
    )

    score = -float(
        errors.mean()
    )

    return score, predictions


def _economic_score(
    predictions,
    task: str,
    direction: str,
) -> np.ndarray:
    predictions = np.asarray(
        predictions,
        dtype=float,
    )

    if task == "classification":
        return predictions

    if direction == "below":
        return -predictions

    return predictions


def _train_models(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    y_train: pd.Series,
    y_validation: pd.Series,
    task: str,
    direction: str,
):
    models = build_models(
        RANDOM_STATE,
        task=task,
    )

    trained = []

    for name, model in models.items():
        model.fit(
            train,
            y_train,
        )

        validation_score, _ = _validation_score(
            model,
            validation,
            y_validation,
            task,
            direction,
        )

        trained.append(
            (
                validation_score,
                name,
                model,
            )
        )

    trained.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    return trained


def train_window(
    data: pd.DataFrame,
    y: pd.Series,
    feature_columns: list[str],
    window: WalkForwardWindow,
    task: str = "classification",
    direction: str = "above",
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    (
        train_mask,
        validation_mask,
        test_mask,
    ) = _split(
        data,
        y,
        window,
    )

    train = data.loc[
        train_mask,
        feature_columns,
    ]

    validation = data.loc[
        validation_mask,
        feature_columns,
    ]

    test = data.loc[
        test_mask,
        feature_columns,
    ]

    y_train = y.loc[train_mask]
    y_validation = y.loc[validation_mask]
    y_test = y.loc[test_mask]

    if len(train) == 0:
        return [], []

    if len(validation) < VALIDATION_MIN_ROWS:
        return [], []

    if len(test) < TEST_MIN_ROWS:
        return [], []

    if task == "classification":
        if len(np.unique(y_train)) < 2:
            return [], []

        if len(np.unique(y_validation)) < 2:
            return [], []

        if len(np.unique(y_test)) < 2:
            return [], []

    available_features = (
        _features_available_in_training(
            train,
            feature_columns,
        )
    )

    if not available_features:
        return [], []

    train = train.loc[:, available_features]
    validation = validation.loc[:, available_features]
    test = test.loc[:, available_features]

    trained = _train_models(
        train,
        validation,
        y_train,
        y_validation,
        task,
        direction,
    )

    if not trained:
        return [], []

    (
        selected_validation_score,
        selected_name,
        selected_model,
    ) = trained[0]

    if task == "classification":
        selected_predictions = (
            selected_model.predict_proba(
                test
            )[:, 1]
        )
    else:
        selected_predictions = np.asarray(
            selected_model.predict(test),
            dtype=float,
        )

    selected_scores = _economic_score(
        selected_predictions,
        task,
        direction,
    )

    results: list[dict[str, Any]] = []

    returns = data.loc[
        test_mask,
        "target_return",
    ]

    for (
        validation_score,
        name,
        model,
    ) in trained:
        if task == "classification":
            test_predictions = (
                model.predict_proba(
                    test
                )[:, 1]
            )

            metrics = evaluate_predictions(
                y_test,
                test_predictions,
                task="classification",
            )

            bucket_results = (
                return_by_probability_bucket(
                    y_test,
                    test_predictions,
                    returns,
                )
            )

            ranking_results = (
                return_by_top_fraction(
                    y_test,
                    test_predictions,
                    returns,
                )
            )

        else:
            test_predictions = np.asarray(
                model.predict(test),
                dtype=float,
            )

            metrics = evaluate_predictions(
                y_test,
                test_predictions,
                task="regression",
            )

            bucket_results = []
            ranking_results = []

        results.append(
            {
                "model": name,
                "task": task,
                "direction": direction,
                "validation_score": validation_score,
                "validation_metric": (
                    "roc_auc"
                    if task == "classification"
                    else "negative_mae"
                ),
                "selected_for_oos": (
                    name == selected_name
                ),
                "test": metrics,
                "return_buckets": bucket_results,
                "ranking_buckets": ranking_results,
                "features": available_features,
                "feature_count": int(
                    len(available_features)
                ),
                "window": {
                    "train_end": window.train_end,
                    "validation_end": window.validation_end,
                    "test_end": window.test_end,
                },
                "train_rows": int(
                    len(train)
                ),
                "validation_rows": int(
                    len(validation)
                ),
                "test_rows": int(
                    len(test)
                ),
                "created_at": (
                    datetime.utcnow().isoformat()
                    + "Z"
                ),
            }
        )

    test_rows = data.loc[
        test_mask,
        [
            "snapshot_date",
            "security_key",
            "target_return",
        ],
    ]

    oos_predictions: list[
        dict[str, Any]
    ] = []

    for index, row in test_rows.reset_index(
        drop=True
    ).iterrows():
        oos_predictions.append(
            {
                "snapshot_date": (
                    pd.Timestamp(
                        row["snapshot_date"]
                    ).strftime(
                        "%Y-%m-%d"
                    )
                ),
                "security_key": str(
                    row["security_key"]
                ),
                "target_return": float(
                    row["target_return"]
                ),
                "prediction": float(
                    selected_predictions[index]
                ),
                "score": float(
                    selected_scores[index]
                ),
                "probability": (
                    float(
                        selected_predictions[index]
                    )
                    if task == "classification"
                    else None
                ),
                "model": selected_name,
                "task": task,
                "direction": direction,
                "validation_score": float(
                    selected_validation_score
                ),
                "window": {
                    "train_end": window.train_end,
                    "validation_end": (
                        window.validation_end
                    ),
                    "test_end": window.test_end,
                },
            }
        )

    return (
        results,
        oos_predictions,
    )
