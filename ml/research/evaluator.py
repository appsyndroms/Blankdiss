"""Fast NumPy-based evaluation of Blankdiss research experiments."""

from __future__ import annotations

import hashlib
from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ml.config import RANDOM_STATE, TEST_MIN_ROWS

from .bootstrap import bootstrap_mean_difference
from .cache import ResearchCache
from .experiments import Experiment


def evaluate_experiment(
    frame: pd.DataFrame,
    experiment: Experiment,
    cache: ResearchCache,
    *,
    train_end: str,
    validation_end: str,
    test_end: str,
) -> dict[str, Any]:
    """
    Evaluate one experiment on one OOS test window.

    Heavy pandas operations have already been performed by the cache.
    """

    signal = cache.get_signal(
        experiment.signal_name
    )

    target = cache.get_target(
        experiment.target_name
    )

    tail_mask = cache.get_tail_mask(
        experiment.signal_name,
        experiment.tail_direction,
        experiment.tail_fraction,
    )

    test_mask = cache.get_window_mask(
        train_end,
        validation_end,
        test_end,
    )

    # ------------------------------------------------------------
    # Valid observations
    # ------------------------------------------------------------

    valid = (
        test_mask
        & np.isfinite(signal)
        & np.isfinite(target)
    )

    n_total = int(valid.sum())

    base_result = {
        **asdict(experiment),
        "train_end": train_end,
        "validation_end": validation_end,
        "test_end": test_end,
    }

    if n_total < TEST_MIN_ROWS:
        return {
            **base_result,
            "status": "INSUFFICIENT_DATA",
            "n": n_total,
        }

    signal_values = signal[valid]
    target_values = target[valid]
    selected = tail_mask[valid]

    n_tail = int(selected.sum())

    if n_tail == 0 or n_tail == n_total:
        return {
            **base_result,
            "status": "INSUFFICIENT_TAIL",
            "n": n_total,
            "n_tail": n_tail,
        }

    # ------------------------------------------------------------
    # AUC
    #
    # For lower-tail experiments, invert the signal so that
    # increasing score always means "more of the tested tail".
    # ------------------------------------------------------------

    auc = _calculate_auc(
        signal_values,
        target_values,
        experiment.tail_direction,
    )

    baseline = float(
        np.mean(target_values)
    )

    tail_target = target_values[selected]

    hit_rate = float(
        np.mean(tail_target)
    )

    lift = (
        hit_rate / baseline
        if baseline != 0
        else None
    )

    # ------------------------------------------------------------
    # Returns
    # ------------------------------------------------------------

    target_config = cache.get_target_config(
        experiment.target_name
    )

    mean_return = None
    median_return = None
    return_difference = None
    ci_lower = None
    ci_upper = None

    return_column = target_config.return_column

    if return_column in frame.columns:
        returns = pd.to_numeric(
            frame[return_column],
            errors="coerce",
        ).to_numpy(
            dtype=np.float64,
            copy=False,
        )

        returns = returns[valid]

        return_valid = np.isfinite(
            returns
        )

        returns = returns[return_valid]
        return_selected = selected[
            return_valid
        ]

        tail_returns = returns[
            return_selected
        ]

        rest_returns = returns[
            ~return_selected
        ]

        if len(tail_returns):
            mean_return = float(
                np.mean(tail_returns)
            )

            median_return = float(
                np.median(tail_returns)
            )

        if (
            len(tail_returns)
            and len(rest_returns)
        ):
            return_difference = float(
                np.mean(tail_returns)
                - np.mean(rest_returns)
            )

            ci_lower, ci_upper = (
                bootstrap_mean_difference(
                    tail_returns,
                    rest_returns,
                    seed=_experiment_seed(
                        experiment.experiment_id,
                        train_end,
                    ),
                )
            )

    status = classify_result(
        auc=auc,
        baseline=baseline,
        hit_rate=hit_rate,
        return_difference=return_difference,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        n_tail=n_tail,
    )

    return {
        **base_result,
        "status": status,
        "n": n_total,
        "n_tail": n_tail,
        "auc": auc,
        "baseline": baseline,
        "hit_rate": hit_rate,
        "lift": lift,
        "mean_return": mean_return,
        "median_return": median_return,
        "return_difference": return_difference,
        "bootstrap_ci_lower": ci_lower,
        "bootstrap_ci_upper": ci_upper,
    }


def _calculate_auc(
    signal: np.ndarray,
    target: np.ndarray,
    direction: str,
) -> float | None:
    if np.unique(target).size < 2:
        return None

    score = (
        signal
        if direction == "upper"
        else -signal
    )

    try:
        return float(
            roc_auc_score(
                target.astype(np.int8),
                score,
            )
        )
    except ValueError:
        return None


def classify_result(
    *,
    auc: float | None,
    baseline: float,
    hit_rate: float,
    return_difference: float | None,
    ci_lower: float | None,
    ci_upper: float | None,
    n_tail: int,
) -> str:
    """
    Descriptive research status.

    These labels are not investment recommendations.
    """

    if n_tail < 20:
        return "INSUFFICIENT_DATA"

    if auc is None:
        return "NO_SIGNAL"

    meaningful_return = (
        return_difference is not None
        and ci_lower is not None
        and ci_upper is not None
        and ci_lower > 0
    )

    meaningful_hit_rate = (
        baseline > 0
        and hit_rate > baseline
    )

    if auc >= 0.65 and (
        meaningful_return
        or meaningful_hit_rate
    ):
        return "STRONG RESEARCH CANDIDATE"

    if auc >= 0.60:
        return "INTERESTING"

    return "NO_SIGNAL"


def _experiment_seed(
    experiment_id: str,
    train_end: str,
) -> int:
    """
    Stable seed across Python processes and CI runs.

    Do not use Python's built-in hash() here because hash randomization
    can produce different values between processes.
    """

    value = (
        f"{RANDOM_STATE}:"
        f"{experiment_id}:"
        f"{train_end}"
    ).encode("utf-8")

    digest = hashlib.sha256(
        value
    ).digest()

    return int.from_bytes(
        digest[:8],
        byteorder="little",
        signed=False,
    ) % (2**32)
