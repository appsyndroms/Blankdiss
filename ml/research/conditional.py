from __future__ import annotations

import hashlib
from typing import Any

from .bootstrap import (
    bootstrap_binary_rate_difference_between_groups,
)
from .cache import ResearchCache, _tail_key
from .spec import SignalSpec


def _stable_seed(
    *parts: object,
) -> int:
    payload = "|".join(
        str(part)
        for part in parts
    ).encode("utf-8")

    digest = hashlib.sha256(
        payload
    ).digest()

    return int.from_bytes(
        digest[:8],
        byteorder="little",
        signed=False,
    ) % (2**32 - 1)


def _regime_rate(
    target,
    mask,
) -> dict[str, Any]:
    selected = target[mask]
    n = int(selected.shape[0])

    if n == 0:
        return {
            "n": 0,
            "events": 0,
            "event_rate": None,
        }

    events = int(selected.sum())

    return {
        "n": n,
        "events": events,
        "event_rate": events / n,
    }


def analyse_conditional_regime_comparison(
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
    Testar om den sista signalen tillför information
    efter att alla tidigare signaler redan definierat
    en baseline-regim.

    Baseline:
        signal 1 AND signal 2 AND ... AND signal N-1

    Incremental group:
        baseline AND signal N

    Comparator group:
        baseline AND NOT signal N

    Skillnaden mäter:

        event_rate(incremental)
        - event_rate(comparator)

    Detta är en generell conditional/nested-regimanalys.
    Antalet baseline-signaler är inte låst till två.
    """

    if len(signals) < 2:
        raise ValueError(
            "conditional_regime_comparison kräver "
            "minst två signaler."
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

    baseline_mask = masks[0].copy()

    for mask in masks[1:-1]:
        baseline_mask &= mask

    incremental_mask = (
        baseline_mask
        & masks[-1]
    )

    comparator_mask = (
        baseline_mask
        & ~masks[-1]
    )

    baseline_in_window = (
        baseline_mask
        & window_mask
    )

    incremental_in_window = (
        incremental_mask
        & window_mask
    )

    comparator_in_window = (
        comparator_mask
        & window_mask
    )

    baseline_metrics = _regime_rate(
        target,
        baseline_in_window,
    )

    incremental_metrics = _regime_rate(
        target,
        incremental_in_window,
    )

    comparator_metrics = _regime_rate(
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

    ci_low = None
    ci_high = None

    if bootstrap:
        seed = _stable_seed(
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
        "analysis": (
            "conditional_regime_comparison"
        ),
        "baseline_signals": [
            {
                "name": signal.name,
                "direction": signal.direction,
                "fraction": fraction,
            }
            for signal, fraction in zip(
                signals[:-1],
                fractions[:-1],
            )
        ],
        "incremental_signal": {
            "name": signals[-1].name,
            "direction": signals[-1].direction,
            "fraction": fractions[-1],
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
