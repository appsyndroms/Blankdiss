from __future__ import annotations
import hashlib
from typing import Any
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from ml.research.bootstrap import bootstrap_mean_ci
from ml.research.cache import ResearchCache, _tail_key
from ml.research.experiments import Experiment
from ml.research.state import _stable_seed
def _safe_auc(
    y_true: np.ndarray,
    scores: np.ndarray,
) -> float | None:
    """
    Calculate ROC AUC if both classes are present.
    """
    valid = (
        np.isfinite(y_true)
        & np.isfinite(scores)
    )
    if not np.any(valid):
        return None
    y = y_true[valid]
    s = scores[valid]
    if np.unique(y).size < 2:
        return None
    try:
        return float(
            roc_auc_score(y, s)
        )
    except ValueError:
        return None
def _binary_metrics(
    target: np.ndarray,
    selected: np.ndarray,
) -> dict[str, Any]:
    """
    Calculate event rate and lift for a selected signal tail.
    """
    valid = np.isfinite(target)
    if not np.any(valid):
        return {
            "n": 0,
            "events": 0,
            "event_rate": None,
            "baseline_event_rate": None,
            "lift": None,
        }
    y = target[valid].astype(float)
    selection = selected[valid]
    events = y > 0
    n = int(
        selection.sum()
    )
    selected_events = int(
        events[selection].sum()
    )
    baseline_rate = float(
        events.mean()
    )
    if n == 0:
        return {
            "n": 0,
            "events": 0,
            "event_rate": None,
            "baseline_event_rate": baseline_rate,
            "lift": None,
        }
    event_rate = (
        selected_events / n
    )
    lift = (
        event_rate / baseline_rate
        if baseline_rate > 0
        else None
    )
    return {
        "n": n,
        "events": selected_events,
        "event_rate": float(event_rate),
        "baseline_event_rate": baseline_rate,
        "lift": (
            float(lift)
            if lift is not None
            else None
        ),
    }
def _return_metrics(
    returns: np.ndarray | None,
    selected: np.ndarray,
    *,
    seed: int,
) -> dict[str, Any]:
    """
    Calculate return statistics for selected observations.
    """
    if returns is None:
        return {
            "return_n": 0,
            "mean_return": None,
            "median_return": None,
            "bootstrap_ci_low": None,
            "bootstrap_ci_high": None,
        }
    valid = (
        np.isfinite(returns)
        & selected
    )
    values = returns[valid]
    if values.size == 0:
        return {
            "return_n": 0,
            "mean_return": None,
            "median_return": None,
            "bootstrap_ci_low": None,
            "bootstrap_ci_high": None,
        }
    mean_return = float(
        np.mean(values)
    )
    median_return = float(
        np.median(values)
    )
    ci_low, ci_high = bootstrap_mean_ci(
        values,
        seed=seed,
    )
    return {
        "return_n": int(values.size),
        "mean_return": mean_return,
        "median_return": median_return,
        "bootstrap_ci_low": (
            float(ci_low)
            if ci_low is not None
            else None
        ),
        "bootstrap_ci_high": (
            float(ci_high)
            if ci_high is not None
            else None
        ),
    }
def _return_metrics_without_bootstrap(
    returns: np.ndarray | None,
    selected: np.ndarray,
) -> dict[str, Any]:
    """
    Calculate the old evaluator's return metrics without bootstrap.
    Used by the migration comparison because bootstrap semantics are
    intentionally not compared between the old evaluator and the new
    Research Engine.
    """
    if returns is None:
        return {
            "return_n": 0,
            "mean_return": None,
            "median_return": None,
            "bootstrap_ci_low": None,
            "bootstrap_ci_high": None,
        }
    valid = (
        np.isfinite(returns)
        & selected
    )
    values = returns[valid]
    if values.size == 0:
        return {
            "return_n": 0,
            "mean_return": None,
            "median_return": None,
            "bootstrap_ci_low": None,
            "bootstrap_ci_high": None,
        }
    return {
        "return_n": int(values.size),
        "mean_return": float(
            np.mean(values)
        ),
        "median_return": float(
            np.median(values)
        ),
        "bootstrap_ci_low": None,
        "bootstrap_ci_high": None,
    }
def evaluate_experiment(
    frame: pd.DataFrame,
    cache: ResearchCache,
    experiment: Experiment,
    window_name: str,
    split_name: str,
    *,
    bootstrap: bool = True,
) -> dict[str, Any]:
    """
    Evaluate one experiment on one walk-forward split.
    The DataFrame is kept in the signature for API compatibility and
    metadata access, but the hot path operates on cached NumPy arrays.
    bootstrap=False is used by the migration comparator because the
    old and new implementations intentionally use different bootstrap
    definitions.
    """
    signal = cache.signals[
        experiment.signal_name
    ]
    target = cache.targets[
        experiment.target_name
    ]
    window_mask = cache.window_masks[
        window_name
    ][split_name]
    tail_key = _tail_key(
        experiment.signal_name,
        experiment.tail_direction,
        experiment.tail_fraction,
    )
    selected = cache.tail_masks[
        tail_key
    ]
    mask = (
        window_mask
        & selected
        & np.isfinite(signal)
        & np.isfinite(target)
    )
    target_config = cache.target_configs[
        experiment.target_name
    ]
    return_column = getattr(
        target_config,
        "return_column",
        None,
    )
    returns = (
        cache.returns.get(return_column)
        if return_column
        else None
    )
    auc_mask = (
        window_mask
        & np.isfinite(signal)
        & np.isfinite(target)
    )
    auc = _safe_auc(
        target[auc_mask],
        signal[auc_mask],
    )
    binary = _binary_metrics(
        target[window_mask],
        selected[window_mask],
    )
    if bootstrap:
        seed = _stable_seed(
            experiment.experiment_id,
            window_name,
            split_name,
        )
        returns_metrics = _return_metrics(
            returns,
            window_mask & selected,
            seed=seed,
        )
    else:
        returns_metrics = (
            _return_metrics_without_bootstrap(
                returns,
                window_mask & selected,
            )
        )
    result: dict[str, Any] = {
        "experiment_id": (
            experiment.experiment_id
        ),
        "signal_name": (
            experiment.signal_name
        ),
        "target_name": (
            experiment.target_name
        ),
        "tail_fraction": (
            experiment.tail_fraction
        ),
        "tail_direction": (
            experiment.tail_direction
        ),
        "window": window_name,
        "split": split_name,
        "auc": auc,
        **binary,
        **returns_metrics,
    }
    result["n_valid_auc"] = int(
        auc_mask.sum()
    )
    result["n_valid_target"] = int(
        np.isfinite(
            target[window_mask]
        ).sum()
    )
    result["selected_fraction"] = (
        float(
            selected[window_mask].mean()
        )
        if window_mask.any()
        else None
    )
    result["n_valid"] = int(
        mask.sum()
    )
    return result
