from __future__ import annotations

from typing import Any

from .analysis_utils import (
    regime_rate,
    stable_seed,
)
from .bootstrap import (
    bootstrap_binary_rate_difference_between_groups,
)
from .cache import (
    ResearchCache,
    _tail_key,
)
from .spec import SignalSpec


def analyse_nested_regime_comparison(
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
    Testar om signal 3 tillför information efter att
    signal 1 och signal 2 redan identifierat en baseline-regim.

    Baseline:
        signal 1 AND signal 2

    Incremental group:
        signal 1 AND signal 2 AND signal 3

    Comparator group:
        signal 1 AND signal 2 AND NOT signal 3

    Skillnaden mäter event_rate(incremental)
    minus event_rate(comparator).
    """
    if len(signals) != 3:
        raise ValueError(
            "nested_regime_comparison kräver "
            "exakt tre signaler."
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

    baseline_mask = (
        masks[0]
        & masks[1]
    )

    incremental_mask = (
        baseline_mask
        & masks[2]
    )

    comparator_mask = (
        baseline_mask
        & ~masks[2]
    )

    incremental_in_window = (
        incremental_mask
        & window_mask
    )

    comparator_in_window = (
        comparator_mask
        & window_mask
    )

    baseline_in_window = (
        baseline_mask
        & window_mask
    )

    baseline_metrics = regime_rate(
        target,
        baseline_in_window,
    )

    incremental_metrics = regime_rate(
        target,
        incremental_in_window,
    )

    comparator_metrics = regime_rate(
        target,
        comparator_in_window,
    )

    incremental_rate = (
        incremental_metrics["event_rate"]
    )

    comparator_rate = (
        comparator_metrics["event_rate"]
    )

    absolute_difference = None

    if (
        incremental_rate is not None
        and comparator_rate is not None
    ):
        absolute_difference = (
            incremental_rate
            - comparator_rate
        )

    lift = None

    if (
        comparator_rate is not None
        and comparator_rate > 0
        and incremental_rate is not None
    ):
        lift = (
            incremental_rate
            / comparator_rate
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
        ) = (
            bootstrap_binary_rate_difference_between_groups(
                target[window_mask],
                comparator_mask[window_mask],
                incremental_mask[window_mask],
                iterations=bootstrap_iterations,
                seed=seed,
            )
        )

    return {
        "analysis": "nested_regime_comparison",
        "baseline_signals": [
            {
                "name": signal.name,
                "direction": signal.direction,
                "fraction": fraction,
            }
            for signal, fraction in zip(
                signals[:2],
                fractions[:2],
            )
        ],
        "incremental_signal": {
            "name": signals[2].name,
            "direction": signals[2].direction,
            "fraction": fractions[2],
        },
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
            baseline_metrics["event_rate"]
        ),
        "incremental_n": (
            incremental_metrics["n"]
        ),
        "incremental_events": (
            incremental_metrics["events"]
        ),
        "incremental_event_rate": (
            incremental_rate
        ),
        "comparator_n": (
            comparator_metrics["n"]
        ),
        "comparator_events": (
            comparator_metrics["events"]
        ),
        "comparator_event_rate": (
            comparator_rate
        ),
        "absolute_event_rate_difference": (
            absolute_difference
        ),
        "lift": lift,
        "bootstrap_ci_low": ci_low,
        "bootstrap_ci_high": ci_high,
    }
