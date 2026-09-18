from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


def binary_event_rate(
    y: pd.Series | np.ndarray,
) -> float:
    values = pd.Series(y).dropna()

    if len(values) == 0:
        return float("nan")

    return float(values.mean())


def lift(
    event_rate: float,
    baseline_rate: float,
) -> float:
    if baseline_rate <= 0 or pd.isna(baseline_rate):
        return float("nan")

    return float(event_rate / baseline_rate)


def classification_metrics(
    y_true,
    y_score,
) -> dict[str, float]:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    mask = np.isfinite(y_score) & np.isfinite(y_true)

    y_true = y_true[mask]
    y_score = y_score[mask]

    if len(y_true) == 0:
        return {}

    result: dict[str, float] = {}

    if len(np.unique(y_true)) >= 2:
        result["auc"] = float(
            roc_auc_score(y_true, y_score)
        )

        result["average_precision"] = float(
            average_precision_score(y_true, y_score)
        )

        clipped = np.clip(y_score, 1e-8, 1 - 1e-8)

        result["log_loss"] = float(
            log_loss(y_true, clipped)
        )

        result["brier"] = float(
            brier_score_loss(y_true, clipped)
        )

    return result


def return_summary(
    returns: pd.Series | np.ndarray,
) -> dict[str, float]:
    values = pd.Series(returns).dropna()

    if len(values) == 0:
        return {}

    return {
        "n": int(len(values)),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "p01": float(values.quantile(0.01)),
        "p05": float(values.quantile(0.05)),
        "p10": float(values.quantile(0.10)),
        "p90": float(values.quantile(0.90)),
        "p95": float(values.quantile(0.95)),
        "p99": float(values.quantile(0.99)),
    }


def event_summary(
    df: pd.DataFrame,
    event_column: str,
    baseline_rate: float | None = None,
) -> dict[str, float]:
    if event_column not in df.columns:
        raise KeyError(
            f"Saknar event-kolumn: {event_column}"
        )

    values = df[event_column].dropna()

    if len(values) == 0:
        return {
            "n": 0,
            "event_rate": float("nan"),
            "lift": float("nan"),
        }

    rate = float(values.mean())

    result = {
        "n": int(len(values)),
        "event_rate": rate,
    }

    if baseline_rate is not None:
        result["lift"] = lift(rate, baseline_rate)

    return result
