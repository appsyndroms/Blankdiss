from __future__ import annotations

from typing import Any

from .analysis_utils import (
    regime_rate,
    stable_seed,
)
from .bootstrap import (
    bootstrap_binary_rate_difference,
)
from .cache import (
    ResearchCache,
    _tail_key,
)
from .spec import SignalSpec


def analyse_multi_regime_comparison(
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
) -> dict[str, Any]:
    """
    Testar om en kombinerad multi-signal-regim har
    annan downside-risk än den första signalens
    baseline.

    Baseline:
        första signalens tail.

    Combined:
        AND av alla signalers tails.
    """
    if len(signals) < 3:
        raise ValueError(
            "multi_regime_comparison kräver "
            "minst tre signaler."
        )

    if len(signals) != len(fractions):
        raise ValueError(
            "Number of signals must match "
            "number of fractions."
        )

    target = cache.targets[target_name]

    window_mask = cache.window_masks[
        window_name
    ][split_name]

    masks = [
        cache.tail_masks[
            _tail_key(
                signal.name,
                signal.direction,
                fraction,
            )
        ]
        for signal, fraction in zip(
            signals,
            fractions,
        )
    ]

    baseline_mask = masks[0]

    combined_mask = masks[0].copy()

    for mask in masks[1:]:
        combined_mask &= mask

    baseline_in_window = (
        baseline_mask
        & window_mask
    )

    combined_in_window = (
        combined_mask
        & window_mask
    )

    baseline_metrics = regime_rate(
        target,
        baseline_in_window,
    )

    combined_metrics = regime_rate(
        target,
        combined_in_window,
    )

    baseline_rate = (
        baseline_metrics["event_rate"]
    )

    combined_rate = (
        combined_metrics["event_rate"]
    )

    absolute_difference = None

    if (
        baseline_rate is not None
        and combined_rate is not None
    ):
        absolute_difference = (
            combined_rate
            - baseline_rate
        )

    lift = None

    if (
        baseline_rate is not None
        and baseline_rate > 0
        and combined_rate is not None
    ):
        lift = (
            combined_rate
            / baseline_rate
        )

    seed = stable_seed(
        spec_id,
        *(
            part
            for signal, fraction in zip(
                signals,
                fractions,
            )
            for part in (
                signal.name,
                signal.direction,
                fraction,
            )
        ),
        target_name,
        window_name,
        split_name,
    )

    ci_low = None
    ci_high = None

    if bootstrap:
        (
            ci_low,
            ci_high,
        ) = bootstrap_binary_rate_difference(
            target[window_mask],
            baseline_mask[window_mask],
            combined_in_window[window_mask],
            iterations=bootstrap_iterations,
            seed=seed,
        )

    return {
        "analysis": "multi_regime_comparison",
        "baseline_signal": (
            signals[0].name
        ),
        "baseline_direction": (
            signals[0].direction
        ),
        "baseline_fraction": (
            fractions[0]
        ),
        "incremental_signals": [
            {
                "name": signal.name,
                "direction": signal.direction,
                "fraction": fraction,
            }
            for signal, fraction in zip(
                signals[1:],
                fractions[1:],
            )
        ],
        "target": target_name,
        "window": window_name,
        "split": split_name,
        "baseline_n": (
            baseline_metrics["n"]
        ),
        "baseline_events": (
            baseline_metrics["events"]
        ),
        "baseline_event_rate": (
            baseline_rate
        ),
        "combined_n": (
            combined_metrics["n"]
        ),
        "combined_events": (
            combined_metrics["events"]
        ),
        "combined_event_rate": (
            combined_rate
        ),
        "absolute_event_rate_difference": (
            absolute_difference
        ),
        "lift": lift,
        "bootstrap_ci_low": ci_low,
        "bootstrap_ci_high": ci_high,
    }
