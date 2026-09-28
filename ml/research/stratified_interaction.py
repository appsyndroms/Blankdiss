from __future__ import annotations

from typing import Any

import numpy as np

from .analysis_utils import stable_seed
from .bootstrap import (
    bootstrap_binary_rate_difference_in_differences,
)
from .cache import ResearchCache, _tail_key
from .spec import SignalSpec


def _build_disjoint_bands(
    cache: ResearchCache,
    signal: SignalSpec,
) -> list[tuple[str, float, float, np.ndarray]]:
    fractions = tuple(sorted(signal.bins))
    bands = []
    previous = 0.0

    for fraction in fractions:
        current = cache.tail_masks[
            _tail_key(
                signal.name,
                signal.direction,
                fraction,
            )
        ]

        if previous == 0.0:
            mask = current.copy()
        else:
            previous_mask = cache.tail_masks[
                _tail_key(
                    signal.name,
                    signal.direction,
                    previous,
                )
            ]

            mask = current & ~previous_mask

        bands.append(
            (
                f"{signal.direction}_{previous:g}_{fraction:g}",
                previous,
                fraction,
                mask,
            )
        )

        previous = fraction

    return bands


def _momentum_label(signal: SignalSpec) -> str:
    """
    Return a stable human-readable label for a momentum signal.

    Examples:
        price_momentum_5d  -> 5d
        price_momentum_20d -> 20d
        price_momentum_60d -> 60d

    The analysis itself remains generic: the label is derived from
    the signal supplied by the research spec rather than assuming
    a specific momentum horizon.
    """
    prefix = "price_momentum_"

    if signal.name.startswith(prefix):
        return signal.name[len(prefix):]

    return signal.name


def _rate(
    target: np.ndarray,
    mask: np.ndarray,
) -> tuple[int, int, float | None]:
    selected = target[mask]
    n = int(selected.shape[0])

    if n == 0:
        return 0, 0, None

    events = int((selected > 0).sum())

    return n, events, events / n


def _volatility_effect(
    target: np.ndarray,
    stratum_mask: np.ndarray,
    volatility_mask: np.ndarray,
    window_mask: np.ndarray,
) -> dict[str, Any]:
    high = (
        stratum_mask
        & volatility_mask
        & window_mask
    )

    normal = (
        stratum_mask
        & ~volatility_mask
        & window_mask
    )

    high_n, high_events, high_rate = _rate(
        target,
        high,
    )

    normal_n, normal_events, normal_rate = _rate(
        target,
        normal,
    )

    effect = None

    if (
        high_rate is not None
        and normal_rate is not None
    ):
        effect = high_rate - normal_rate

    return {
        "high_n": high_n,
        "high_events": high_events,
        "high_rate": high_rate,
        "normal_n": normal_n,
        "normal_events": normal_events,
        "normal_rate": normal_rate,
        "volatility_effect": effect,
        "high_mask": high,
        "normal_mask": normal,
    }


def _contrast_row(
    target: np.ndarray,
    window_mask: np.ndarray,
    spec_id: str,
    target_name: str,
    window_name: str,
    split_name: str,
    *,
    contrast_type: str,
    reference_label: str,
    comparison_label: str,
    reference: dict[str, Any],
    comparison: dict[str, Any],
    bootstrap: bool,
    bootstrap_iterations: int,
) -> dict[str, Any]:
    reference_effect = reference[
        "volatility_effect"
    ]

    comparison_effect = comparison[
        "volatility_effect"
    ]

    difference = None

    if (
        reference_effect is not None
        and comparison_effect is not None
    ):
        difference = (
            comparison_effect
            - reference_effect
        )

    ci_low = None
    ci_high = None

    if bootstrap:
        seed = stable_seed(
            spec_id,
            target_name,
            window_name,
            split_name,
            contrast_type,
            reference_label,
            comparison_label,
        )

        ci_low, ci_high = (
            bootstrap_binary_rate_difference_in_differences(
                target[window_mask],
                reference["high_mask"][
                    window_mask
                ],
                reference["normal_mask"][
                    window_mask
                ],
                comparison["high_mask"][
                    window_mask
                ],
                comparison["normal_mask"][
                    window_mask
                ],
                iterations=bootstrap_iterations,
                seed=seed,
            )
        )

    return {
        "analysis": "stratified_interaction",
        "target": target_name,
        "window": window_name,
        "split": split_name,
        "contrast_type": contrast_type,
        "reference_stratum": reference_label,
        "comparison_stratum": comparison_label,
        "reference_volatility_effect": reference_effect,
        "comparison_volatility_effect": comparison_effect,
        "interaction_difference_in_differences": difference,
        "bootstrap_ci_low": ci_low,
        "bootstrap_ci_high": ci_high,
        "reference_high_n": reference["high_n"],
        "reference_high_events": reference[
            "high_events"
        ],
        "reference_high_rate": reference[
            "high_rate"
        ],
        "reference_normal_n": reference[
            "normal_n"
        ],
        "reference_normal_events": reference[
            "normal_events"
        ],
        "reference_normal_rate": reference[
            "normal_rate"
        ],
        "comparison_high_n": comparison[
            "high_n"
        ],
        "comparison_high_events": comparison[
            "high_events"
        ],
        "comparison_high_rate": comparison[
            "high_rate"
        ],
        "comparison_normal_n": comparison[
            "normal_n"
        ],
        "comparison_normal_events": comparison[
            "normal_events"
        ],
        "comparison_normal_rate": comparison[
            "normal_rate"
        ],
    }


def analyse_stratified_interaction(
    cache: ResearchCache,
    signals: tuple[SignalSpec, ...],
    target_name: str,
    fractions: tuple[float, ...],
    window_name: str,
    split_name: str,
    *,
    bootstrap: bool,
    bootstrap_iterations: int,
    spec_id: str,
) -> list[dict[str, Any]]:
    """
    Formal difference-in-differences test of whether the
    high-volatility effect changes with momentum strength.

    The first two supplied signals are treated as the two
    momentum dimensions. The third supplied signal is treated
    as the volatility dimension.

    We compare:

      1. First momentum signal: weak vs strong, holding each
         band of the second momentum signal fixed.
      2. Second momentum signal: weak vs strong, holding each
         band of the first momentum signal fixed.
      3. Jointly weak momentum (weak/weak) vs jointly strong
         momentum (strong/strong).

    In every contrast:

        DID = volatility_effect(comparison)
              - volatility_effect(reference)

    where:

        volatility_effect =
            P(event | high volatility)
            - P(event | normal volatility)

    The implementation is intentionally generic with respect
    to momentum horizon. For example, the same analysis can
    operate on:

        price_momentum_5d  x price_momentum_60d
        price_momentum_5d  x price_momentum_20d
        price_momentum_20d x price_momentum_60d
    """
    if len(signals) != 3 or len(fractions) != 3:
        raise ValueError(
            "stratified_interaction kräver exakt tre signaler."
        )

    if (
        len(signals[0].bins) < 3
        or len(signals[1].bins) < 3
    ):
        raise ValueError(
            "De två momentum-signalerna måste ha minst tre bins."
        )

    target = cache.targets[target_name]

    window_mask = cache.window_masks[
        window_name
    ][split_name]

    first_bands = _build_disjoint_bands(
        cache,
        signals[0],
    )

    second_bands = _build_disjoint_bands(
        cache,
        signals[1],
    )

    if (
        len(first_bands) < 3
        or len(second_bands) < 3
    ):
        raise ValueError(
            "Minst tre momentumregimer krävs."
        )

    volatility_mask = cache.tail_masks[
        _tail_key(
            signals[2].name,
            signals[2].direction,
            fractions[2],
        )
    ]

    cells: dict[
        tuple[int, int],
        dict[str, Any],
    ] = {}

    for i, (
        _,
        _,
        _,
        first_mask,
    ) in enumerate(first_bands):
        for j, (
            _,
            _,
            _,
            second_mask,
        ) in enumerate(second_bands):
            cells[(i, j)] = _volatility_effect(
                target,
                first_mask & second_mask,
                volatility_mask,
                window_mask,
            )

    results: list[dict[str, Any]] = []

    strong_i = 0
    weak_i = len(first_bands) - 1

    strong_j = 0
    weak_j = len(second_bands) - 1

    first_label = _momentum_label(signals[0])
    second_label = _momentum_label(signals[1])

    # ------------------------------------------------------------
    # First momentum dimension:
    #
    # Compare weak first momentum with strong first momentum
    # while holding the second momentum band fixed.
    # ------------------------------------------------------------
    for j, (
        second_band_label,
        _,
        _,
        _,
    ) in enumerate(second_bands):
        reference = cells[(strong_i, j)]
        comparison = cells[(weak_i, j)]

        results.append(
            _contrast_row(
                target,
                window_mask,
                spec_id,
                target_name,
                window_name,
                split_name,
                contrast_type=(
                    f"momentum_{first_label}_weak_vs_strong"
                ),
                reference_label=(
                    f"{first_bands[strong_i][0]}"
                    f"__{second_band_label}"
                ),
                comparison_label=(
                    f"{first_bands[weak_i][0]}"
                    f"__{second_band_label}"
                ),
                reference=reference,
                comparison=comparison,
                bootstrap=bootstrap,
                bootstrap_iterations=(
                    bootstrap_iterations
                ),
            )
        )

    # ------------------------------------------------------------
    # Second momentum dimension:
    #
    # Compare weak second momentum with strong second momentum
    # while holding the first momentum band fixed.
    # ------------------------------------------------------------
    for i, (
        first_band_label,
        _,
        _,
        _,
    ) in enumerate(first_bands):
        reference = cells[(i, strong_j)]
        comparison = cells[(i, weak_j)]

        results.append(
            _contrast_row(
                target,
                window_mask,
                spec_id,
                target_name,
                window_name,
                split_name,
                contrast_type=(
                    f"momentum_{second_label}_weak_vs_strong"
                ),
                reference_label=(
                    f"{first_band_label}"
                    f"__{second_bands[strong_j][0]}"
                ),
                comparison_label=(
                    f"{first_band_label}"
                    f"__{second_bands[weak_j][0]}"
                ),
                reference=reference,
                comparison=comparison,
                bootstrap=bootstrap,
                bootstrap_iterations=(
                    bootstrap_iterations
                ),
            )
        )

    # ------------------------------------------------------------
    # Joint contrast:
    #
    # Strong first + strong second versus weak first + weak
    # second.
    # ------------------------------------------------------------
    reference = cells[
        (strong_i, strong_j)
    ]

    comparison = cells[
        (weak_i, weak_j)
    ]

    results.append(
        _contrast_row(
            target,
            window_mask,
            spec_id,
            target_name,
            window_name,
            split_name,
            contrast_type="joint_weak_vs_strong",
            reference_label=(
                f"{first_bands[strong_i][0]}"
                f"__{second_bands[strong_j][0]}"
            ),
            comparison_label=(
                f"{first_bands[weak_i][0]}"
                f"__{second_bands[weak_j][0]}"
            ),
            reference=reference,
            comparison=comparison,
            bootstrap=bootstrap,
            bootstrap_iterations=(
                bootstrap_iterations
            ),
        )
    )

    return results
