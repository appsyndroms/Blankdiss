"""Utvärdering av Blankdiss ML-modeller."""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_predictions(
    y_true,
    probabilities,
) -> dict[str, Any]:
    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    result: dict[str, Any] = {
        "rows": int(len(y_true)),
        "accuracy": float(
            accuracy_score(
                y_true,
                predictions,
            )
        ),
        "precision": float(
            precision_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                predictions,
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

    result[
        "predicted_positive_rate"
    ] = float(
        predictions.mean()
    )

    result[
        "mean_probability"
    ] = float(
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
