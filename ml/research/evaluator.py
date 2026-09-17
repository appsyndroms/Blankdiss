from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ml.config import RANDOM_STATE, TEST_MIN_ROWS

from .bootstrap import bootstrap_mean_difference
from .experiments import Experiment
from .cache import ResearchCache


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
    Evaluate one experiment on one walk-forward window.

    Only the OOS/test period is evaluated here.
    """

    target = cache.get_target(
        experiment.target_name
    )

    signal = cache.get_signal(
        experiment.signal_name
    )

    tail_mask = cache.get_tail_mask(
        experiment.signal_name,
        experiment.tail_direction,
        experiment.tail_fraction,
    )

    test_period = (
        (frame["snapshot_date"] > validation_end)
        & (frame["snapshot_date"] <= test_end)
    )

    valid = (
        test_period
        & signal.notna()
        & target.notna()
    )

    if valid.sum() < TEST_MIN_ROWS:
        return {
            **asdict(experiment),
            "train_end": train_end,
            "validation_end": validation_end,
            "test_end": test_end,
            "status": "INSUFFICIENT_DATA",
            "n": int(valid.sum()),
        }

    signal_values = signal.loc[valid]
    target_values = target.loc[valid]
    tail_values = tail_mask.loc[valid]

    # ------------------------------------------------------------
    # Classification metrics
    # ------------------------------------------------------------

    auc = None

    unique_targets = target_values.nunique()

    if unique_targets >= 2:
        try:
            auc = float(
                roc_auc_score(
                    target_values.astype(int),
                    signal_values,
                )
            )
        except ValueError:
            auc = None

    baseline = float(target_values.mean())

    selected = tail_values.astype(bool)

    n_tail = int(selected.sum())
    n_total = int(len(selected))

    if n_tail == 0 or n_tail == n_total:
        return {
            **asdict(experiment),
            "train_end": train_end,
            "validation_end": validation_end,
            "test_end": test_end,
            "status": "INSUFFICIENT_TAIL",
            "n": n_total,
            "n_tail": n_tail,
            "auc": auc,
            "baseline": baseline,
        }

    tail_target = target_values.loc[selected]
    rest_target = target_values.loc[~selected]

    hit_rate = float(tail_target.mean())

    # ------------------------------------------------------------
    # Returns
    # ------------------------------------------------------------

    return_column = _return_column_for_target(
        experiment.target_name
    )

    mean_return = None
    median_return = None
    return_difference = None
    ci_lower = None
    ci_upper = None

    if return_column in frame.columns:
        returns = frame.loc[
            valid,
            return_column,
        ].astype(float)

        finite = returns.notna()

        returns = returns.loc[finite]
        selected_returns = tail_values.loc[
            returns.index
        ].astype(bool)

        tail_returns = returns.loc[
            selected_returns
        ].to_numpy()

        rest_returns = returns.loc[
            ~selected_returns
        ].to_numpy()

        if len(tail_returns):
            mean_return = float(
                np.mean(tail_returns)
            )
            median_return = float(
                np.median(tail_returns)
            )

        if len(tail_returns) and len(rest_returns):
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

    lift = None

    if baseline != 0:
        lift = hit_rate / baseline

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
        **asdict(experiment),
        "train_end": train_end,
        "validation_end": validation_end,
        "test_end": test_end,
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
    Descriptive classification only.

    These labels describe research behavior and are not investment
    recommendations.
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


def _return_column_for_target(
    target_name: str,
) -> str:
    if target_name.endswith("_5d"):
        return "max_return_5d"

    if target_name.endswith("_20d"):
        return "max_return_20d"

    if target_name.endswith("_60d"):
        return "max_return_60d"

    return "max_return_5d"


def _experiment_seed(
    experiment_id: str,
    train_end: str,
) -> int:
    value = (
        f"{RANDOM_STATE}:"
        f"{experiment_id}:"
        f"{train_end}"
    )

    return abs(hash(value)) % (2**32)
