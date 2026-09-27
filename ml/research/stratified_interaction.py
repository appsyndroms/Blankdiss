from __future__ import annotations
from typing import Any
import numpy as np
from .analysis_utils import stable_seed
from .bootstrap import MIN_ROWS
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
            _tail_key(signal.name, signal.direction, fraction)
        ]
        if previous == 0.0:
            mask = current.copy()
        else:
            previous_mask = cache.tail_masks[
                _tail_key(signal.name, signal.direction, previous)
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
    high = stratum_mask & volatility_mask & window_mask
    normal = stratum_mask & ~volatility_mask & window_mask
    high_n, high_events, high_rate = _rate(target, high)
    normal_n, normal_events, normal_rate = _rate(target, normal)
    effect = None
    if high_rate is not None and normal_rate is not None:
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
def _bootstrap_effect_contrast(
    target: np.ndarray,
    effect_a_high: np.ndarray,
    effect_a_normal: np.ndarray,
    effect_b_high: np.ndarray,
    effect_b_normal: np.ndarray,
    *,
    iterations: int,
    seed: int,
) -> tuple[float | None, float | None]:
    groups = [
        target[effect_a_high],
        target[effect_a_normal],
        target[effect_b_high],
        target[effect_b_normal],
    ]
    if any(len(values) < MIN_ROWS for values in groups):
        return None, None
    rng = np.random.default_rng(seed)
    values = np.empty(iterations, dtype=np.float64)
    offset = 0
    while offset < iterations:
        current = min(100, iterations - offset)
        sampled_rates = []
        for group in groups:
            indices = rng.integers(
                0,
                len(group),
                size=(current, len(group)),
            )
            sampled_rates.append(
                (group[indices] > 0).mean(axis=1)
            )
        reference_effect = (
            sampled_rates[0] - sampled_rates[1]
        )
        comparison_effect = (
            sampled_rates[2] - sampled_rates[3]
        )
        values[offset : offset + current] = (
            comparison_effect - reference_effect
        )
        offset += current
    low, high = np.quantile(
        values,
        [0.025, 0.975],
    )
    return float(low), float(high)
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
    reference_effect = reference["volatility_effect"]
    comparison_effect = comparison["volatility_effect"]
    difference = None
    if (
        reference_effect is not None
        and comparison_effect is not None
    ):
        difference = comparison_effect - reference_effect
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
        ci_low, ci_high = _bootstrap_effect_contrast(
            target[window_mask],
            reference["high_mask"][window_mask],
            reference["normal_mask"][window_mask],
            comparison["high_mask"][window_mask],
            comparison["normal_mask"][window_mask],
            iterations=bootstrap_iterations,
            seed=seed,
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
        "reference_high_events": reference["high_events"],
        "reference_high_rate": reference["high_rate"],
        "reference_normal_n": reference["normal_n"],
        "reference_normal_events": reference["normal_events"],
        "reference_normal_rate": reference["normal_rate"],
        "comparison_high_n": comparison["high_n"],
        "comparison_high_events": comparison["high_events"],
        "comparison_high_rate": comparison["high_rate"],
        "comparison_normal_n": comparison["normal_n"],
        "comparison_normal_events": comparison["normal_events"],
        "comparison_normal_rate": comparison["normal_rate"],
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
    We compare:
      1. 5d momentum: weak vs strong, holding each 60d band fixed.
      2. 60d momentum: weak vs strong, holding each 5d band fixed.
      3. Jointly weak momentum (weak/weak) vs jointly strong
         momentum (strong/strong).
    In every contrast:
        DID = volatility_effect(comparison)
              - volatility_effect(reference)
    where:
        volatility_effect =
            P(event | high volatility)
            - P(event | normal volatility)
    """
    if len(signals) != 3 or len(fractions) != 3:
        raise ValueError(
            "stratified_interaction kräver exakt tre signaler."
        )
    if len(signals[0].bins) < 3 or len(signals[1].bins) < 3:
        raise ValueError(
            "De två momentum-signalerna måste ha minst tre bins."
        )
    target = cache.targets[target_name]
    window_mask = cache.window_masks[window_name][split_name]
    first_bands = _build_disjoint_bands(
        cache,
        signals[0],
    )
    second_bands = _build_disjoint_bands(
        cache,
        signals[1],
    )
    if len(first_bands) < 3 or len(second_bands) < 3:
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
    for i, (_, _, _, first_mask) in enumerate(first_bands):
        for j, (_, _, _, second_mask) in enumerate(second_bands):
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
    # ------------------------------------------------------------
    # 5d momentum:
    #
    # Compare weak 5d momentum with strong 5d momentum while
    # holding the 60d momentum band fixed.
    # ------------------------------------------------------------
    for j, (
        second_label,
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
                contrast_type="momentum_5d_weak_vs_strong",
                reference_label=(
                    f"{first_bands[strong_i][0]}"
                    f"__{second_label}"
                ),
                comparison_label=(
                    f"{first_bands[weak_i][0]}"
                    f"__{second_label}"
                ),
                reference=reference,
                comparison=comparison,
                bootstrap=bootstrap,
                bootstrap_iterations=bootstrap_iterations,
            )
        )
    # ------------------------------------------------------------
    # 60d momentum:
    #
    # Compare weak 60d momentum with strong 60d momentum while
    # holding the 5d momentum band fixed.
    # ------------------------------------------------------------
    for i, (
        first_label,
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
                contrast_type="momentum_60d_weak_vs_strong",
                reference_label=(
                    f"{first_label}"
                    f"__{second_bands[strong_j][0]}"
                ),
                comparison_label=(
                    f"{first_label}"
                    f"__{second_bands[weak_j][0]}"
                ),
                reference=reference,
                comparison=comparison,
                bootstrap=bootstrap,
                bootstrap_iterations=bootstrap_iterations,
            )
        )
    # ------------------------------------------------------------
    # Joint contrast:
    #
    # Strong 5d + strong 60d versus weak 5d + weak 60d.
    # ------------------------------------------------------------
    reference = cells[(strong_i, strong_j)]
    comparison = cells[(weak_i, weak_j)]
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
            bootstrap_iterations=bootstrap_iterations,
        )
    )
    return results
