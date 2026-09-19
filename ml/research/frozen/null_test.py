from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ml.research.discovery.engine import (
    Candidate,
    DiscoveryData,
    evaluate_candidate_on_mask,
)
from ml.research.discovery.targets import (
    build_target,
    _target_return_column,
)


def _permuted_data(
    data: DiscoveryData,
    target_name: str,
    mask: np.ndarray,
    rng: np.random.Generator,
) -> DiscoveryData:
    """
    Permuterar endast outcome-värden inom OOS-masken.

    Signaler och stressvariabler lämnas orörda.
    """

    target = data.targets[target_name]

    return_values = data.returns[target_name].copy()

    valid = (
        mask
        & np.isfinite(return_values)
    )

    shuffled = return_values[
        valid
    ].copy()

    rng.shuffle(shuffled)

    permuted_returns = return_values.copy()

    permuted_returns[
        valid
    ] = shuffled

    target_config = next(
        target_config
        for target_config in (
            # build_target behöver rätt target-konfiguration.
            # DiscoveryData innehåller inte target-konfigurationen,
            # så den hämtas via engine-data indirekt i runnern.
        )
    )

    raise RuntimeError(
        "Internal error: _permuted_data must be supplied "
        "with target configuration."
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
    """
    Frozen null-test för exakt en kandidat.

    Ingen candidate search sker under permutationerna.
    """

    if permutations < 1:
        raise ValueError(
            "permutations måste vara >= 1."
        )

    if metric not in {
        "lift",
        "return_difference",
    }:
        raise ValueError(
            "metric måste vara 'lift' "
            "eller 'return_difference'."
        )

    observed_value = observed_result.get(
        metric
    )

    if observed_value is None or not np.isfinite(
        observed_value
    ):
        raise ValueError(
            f"Observed metric '{metric}' är inte giltig."
        )

    rng = np.random.default_rng(
        seed
    )

    target = data.targets[
        candidate.target_name
    ]

    returns = data.returns[
        candidate.target_name
    ]

    valid = (
        oos_mask
        & np.isfinite(target)
        & np.isfinite(returns)
    )

    original_returns = returns.copy()

    null_values: list[float] = []

    for permutation_index in range(
        1,
        permutations + 1,
    ):
        permuted_returns = original_returns.copy()

        shuffled = permuted_returns[
            valid
        ].copy()

        rng.shuffle(
            shuffled
        )

        permuted_returns[
            valid
        ] = shuffled

        selected_mask = _candidate_selected_mask(
            data=data,
            candidate=candidate,
        )

        selected = (
            oos_mask
            & selected_mask
            & np.isfinite(permuted_returns)
        )

        rest = (
            oos_mask
            & ~selected_mask
            & np.isfinite(permuted_returns)
        )

        if metric == "return_difference":
            selected_values = permuted_returns[
                selected
            ]

            rest_values = permuted_returns[
                rest
            ]

            if (
                not len(selected_values)
                or not len(rest_values)
            ):
                continue

            value = float(
                selected_values.mean()
                - rest_values.mean()
            )

        else:
            selected_values = target[
                selected
            ]

            baseline_values = target[
                oos_mask
                & np.isfinite(target)
            ]

            if (
                not len(selected_values)
                or not len(baseline_values)
            ):
                continue

            event_rate = float(
                (selected_values > 0).mean()
            )

            baseline_rate = float(
                (baseline_values > 0).mean()
            )

            if baseline_rate <= 0:
                continue

            value = (
                event_rate
                - baseline_rate
            )

        null_values.append(value)

        if (
            permutation_index == 1
            or permutation_index % 100 == 0
            or permutation_index == permutations
        ):
            print(
                "Frozen null progress: "
                f"{permutation_index:,}/{permutations:,}",
                flush=True,
            )

    if not null_values:
        raise RuntimeError(
            "Frozen null-testet producerade inga giltiga permutationer."
        )

    null_array = np.asarray(
        null_values,
        dtype=float,
    )

    exceedances = int(
        np.sum(
            null_array >= observed_value
        )
    )

    p_value = (
        exceedances + 1
    ) / (
        len(null_array) + 1
    )

    return {
        "metric": metric,
        "observed": float(observed_value),
        "permutations_requested": permutations,
        "permutations_valid": len(null_values),
        "seed": seed,
        "p_value": float(p_value),
        "null_mean": float(
            null_array.mean()
        ),
        "null_std": float(
            null_array.std()
        ),
        "null_min": float(
            null_array.min()
        ),
        "null_max": float(
            null_array.max()
        ),
    }


def _candidate_selected_mask(
    data: DiscoveryData,
    candidate: Candidate,
) -> np.ndarray:
    from ml.research.discovery.validation import tail_mask

    signal = data.signals[
        candidate.signal_name
    ]

    stress = data.stress_signals[
        candidate.stress_feature
    ]

    signal_mask = tail_mask(
        signal,
        candidate.signal_tail,
        direction="upper",
    )

    stress_mask = tail_mask(
        stress,
        candidate.stress_tail,
        direction=candidate.stress_direction,
    )

    return (
        signal_mask
        & stress_mask
    )
