from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from ml.research.discovery.engine import (
    Candidate,
    DiscoveryData,
    evaluate_candidate_on_mask,
)


def _permuted_data(
    data: DiscoveryData,
    candidate: Candidate,
    oos_mask: np.ndarray,
    rng: np.random.Generator,
    metric: str,
) -> DiscoveryData:
    targets = dict(data.targets)
    returns = dict(data.returns)

    target = targets[candidate.target_name]
    return_values = returns[candidate.target_name]

    if metric == "lift":
        valid = oos_mask & np.isfinite(target)

        shuffled = target[valid].copy()
        rng.shuffle(shuffled)

        permuted_target = target.copy()
        permuted_target[valid] = shuffled

        targets[candidate.target_name] = permuted_target

    elif metric == "return_difference":
        valid = oos_mask & np.isfinite(return_values)

        shuffled = return_values[valid].copy()
        rng.shuffle(shuffled)

        permuted_returns = return_values.copy()
        permuted_returns[valid] = shuffled

        returns[candidate.target_name] = permuted_returns

    else:
        raise ValueError(
            "metric måste vara 'lift' eller 'return_difference'."
        )

    return replace(
        data,
        targets=targets,
        returns=returns,
    )


def run_frozen_null_test(
    data: DiscoveryData,
    candidate: Candidate,
    oos_mask: np.ndarray,
    observed_result: dict[str, Any],
    permutations: int,
    seed: int,
    metric: str,
) -> dict[str, Any]:
    if permutations < 1:
        raise ValueError("permutations måste vara >= 1.")

    if metric not in {"lift", "return_difference"}:
        raise ValueError(
            "metric måste vara 'lift' eller 'return_difference'."
        )

    observed_value = observed_result.get(metric)

    if observed_value is None or not np.isfinite(observed_value):
        raise ValueError(
            f"Observed metric '{metric}' är inte giltig."
        )

    rng = np.random.default_rng(seed)
    null_values: list[float] = []

    for permutation_index in range(1, permutations + 1):
        permuted_data = _permuted_data(
            data=data,
            candidate=candidate,
            oos_mask=oos_mask,
            rng=rng,
            metric=metric,
        )

        result = evaluate_candidate_on_mask(
            data=permuted_data,
            candidate=candidate,
            base_mask=oos_mask,
            split="oos_null",
        )

        value = result.get(metric)

        if value is not None and np.isfinite(value):
            null_values.append(float(value))

    if not null_values:
        raise RuntimeError(
            "Frozen null-testet producerade inga giltiga permutationer."
        )

    null_array = np.asarray(
        null_values,
        dtype=float,
    )

    exceedances = int(
        np.sum(null_array >= observed_value)
    )

    p_value = (
        exceedances + 1
    ) / (
        len(null_array) + 1
    )

    return {
        "metric": metric,
        "observed": float(observed_value),
        "permutations_requested": int(permutations),
        "permutations_valid": int(len(null_array)),
        "seed": int(seed),
        "exceedances": exceedances,
        "p_value": float(p_value),
        "null_mean": float(null_array.mean()),
        "null_std": float(null_array.std()),
        "null_min": float(null_array.min()),
        "null_max": float(null_array.max()),
        "null_percentile_95": float(
            np.percentile(null_array, 95)
        ),
        "null_percentile_99": float(
            np.percentile(null_array, 99)
        ),
    }
