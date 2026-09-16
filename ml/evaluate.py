"""Utvärdering av Blankdiss ML-modeller."""
from __future__ import annotations
from typing import Any
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
TOP_FRACTIONS = (
    0.01,
    0.05,
    0.10,
    0.20,
)
def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(
        values,
        kind="mergesort",
    )
    ranks = np.empty(
        len(values),
        dtype=float,
    )
    ranks[order] = np.arange(
        len(values),
        dtype=float,
    )
    return ranks
def _spearman(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> float | None:
    valid = (
        np.isfinite(actual)
        & np.isfinite(predicted)
    )
    actual = actual[valid]
    predicted = predicted[valid]
    if len(actual) < 2:
        return None
    actual_rank = _rankdata(actual)
    predicted_rank = _rankdata(predicted)
    actual_rank -= actual_rank.mean()
    predicted_rank -= predicted_rank.mean()
    denominator = (
        np.sqrt(
            np.sum(actual_rank ** 2)
        )
        * np.sqrt(
            np.sum(predicted_rank ** 2)
        )
    )
    if denominator == 0:
        return None
    return float(
        np.sum(
            actual_rank
            * predicted_rank
        )
        / denominator
    )
def evaluate_predictions(
    y_true,
    predictions,
    task: str = "classification",
) -> dict[str, Any]:
    if task == "regression":
        actual = np.asarray(
            y_true,
            dtype=float,
        )
        predicted = np.asarray(
            predictions,
            dtype=float,
        )
        valid = (
            np.isfinite(actual)
            & np.isfinite(predicted)
        )
        actual = actual[valid]
        predicted = predicted[valid]
        if len(actual) == 0:
            return {
                "rows": 0,
                "mae": None,
                "rmse": None,
                "mean_actual": None,
                "mean_prediction": None,
                "median_actual": None,
                "median_prediction": None,
                "spearman": None,
            }
        return {
            "rows": int(len(actual)),
            "mae": float(
                mean_absolute_error(
                    actual,
                    predicted,
                )
            ),
            "rmse": float(
                np.sqrt(
                    mean_squared_error(
                        actual,
                        predicted,
                    )
                )
            ),
            "mean_actual": float(
                actual.mean()
            ),
            "mean_prediction": float(
                predicted.mean()
            ),
            "median_actual": float(
                np.median(actual)
            ),
            "median_prediction": float(
                np.median(predicted)
            ),
            "spearman": _spearman(
                actual,
                predicted,
            ),
        }
    probabilities = np.asarray(
        predictions,
        dtype=float,
    )
    predictions_binary = (
        probabilities >= 0.5
    ).astype(int)
    result: dict[str, Any] = {
        "rows": int(len(y_true)),
        "accuracy": float(
            accuracy_score(
                y_true,
                predictions_binary,
            )
        ),
        "precision": float(
            precision_score(
                y_true,
                predictions_binary,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                predictions_binary,
                zero_division=0,
            )
        ),
        "brier_score": float(
            brier_score_loss(
                y_true,
                probabilities,
            )
        ),
    }
    unique_classes = np.unique(
        y_true
    )
    if len(unique_classes) == 2:
        result["roc_auc"] = float(
            roc_auc_score(
                y_true,
                probabilities,
            )
        )
    else:
        result["roc_auc"] = None
    result["predicted_positive_rate"] = float(
        predictions_binary.mean()
    )
    result["mean_probability"] = float(
        probabilities.mean()
    )
    return result
def return_by_probability_bucket(
    y_true,
    probabilities,
    returns,
) -> list[dict[str, Any]]:
    frame = np.column_stack(
        [
            np.asarray(
                probabilities,
                dtype=float,
            ),
            np.asarray(
                returns,
                dtype=float,
            ),
        ]
    )
    buckets = (
        (0.50, 0.55),
        (0.55, 0.60),
        (0.60, 0.65),
        (0.65, 0.70),
        (0.70, 1.01),
    )
    results = []
    for lower, upper in buckets:
        mask = (
            (frame[:, 0] >= lower)
            & (frame[:, 0] < upper)
            & np.isfinite(frame[:, 1])
        )
        selected = frame[
            mask,
            1,
        ]
        results.append(
            {
                "lower": lower,
                "upper": upper,
                "rows": int(
                    len(selected)
                ),
                "mean_return": (
                    float(
                        selected.mean()
                    )
                    if len(selected)
                    else None
                ),
                "median_return": (
                    float(
                        np.median(
                            selected
                        )
                    )
                    if len(selected)
                    else None
                ),
            }
        )
    return results
def return_by_top_fraction(
    y_true,
    probabilities,
    returns,
    fractions=TOP_FRACTIONS,
) -> list[dict[str, Any]]:
    y_array = np.asarray(
        y_true,
        dtype=float,
    )
    probability_array = np.asarray(
        probabilities,
        dtype=float,
    )
    return_array = np.asarray(
        returns,
        dtype=float,
    )
    valid = (
        np.isfinite(
            probability_array
        )
        & np.isfinite(
            return_array
        )
        & np.isfinite(
            y_array
        )
    )
    probability_array = (
        probability_array[valid]
    )
    return_array = (
        return_array[valid]
    )
    y_array = (
        y_array[valid]
    )
    rows = len(
        probability_array
    )
    if rows == 0:
        return [
            {
                "top_fraction": float(
                    fraction
                ),
                "rows": 0,
                "event_rate": None,
                "baseline_event_rate": None,
                "event_rate_lift_ratio": None,
                "mean_return": None,
                "median_return": None,
                "baseline_mean_return": None,
            }
            for fraction in fractions
        ]
    order = np.argsort(
        -probability_array,
        kind="mergesort",
    )
    baseline_event_rate = float(
        y_array.mean()
    )
    baseline_mean_return = float(
        return_array.mean()
    )
    results = []
    for fraction in fractions:
        count = max(
            1,
            int(
                np.ceil(
                    rows * fraction
                )
            ),
        )
        selected_indices = (
            order[:count]
        )
        selected_events = (
            y_array[
                selected_indices
            ]
        )
        selected_returns = (
            return_array[
                selected_indices
            ]
        )
        event_rate = float(
            selected_events.mean()
        )
        if baseline_event_rate > 0:
            lift_ratio = float(
                event_rate
                / baseline_event_rate
            )
        else:
            lift_ratio = None
        results.append(
            {
                "top_fraction": float(
                    fraction
                ),
                "rows": int(
                    count
                ),
                "event_rate": event_rate,
                "baseline_event_rate": (
                    baseline_event_rate
                ),
                "event_rate_lift_ratio": (
                    lift_ratio
                ),
                "mean_return": float(
                    selected_returns.mean()
                ),
                "median_return": float(
                    np.median(
                        selected_returns
                    )
                ),
                "baseline_mean_return": (
                    baseline_mean_return
                ),
            }
        )
    return results
