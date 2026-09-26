from __future__ import annotations

import hashlib
from typing import Any

from .bootstrap import (
    bootstrap_binary_rate_difference,
    bootstrap_binary_rate_difference_between_groups,
)
from .cache import ResearchCache, _tail_key
from .spec import ResearchSpec, SignalSpec


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


def _analyse_tail(
    cache: ResearchCache,
    signal: SignalSpec,
    fraction: float,
    target_name: str,
    window_name: str,
    split_name: str,
) -> dict[str, Any]:
    target = cache.targets[target_name]

    window_mask = cache.window_masks[
        window_name
    ][split_name]

    tail_mask = cache.tail_masks[
        _tail_key(
            signal.name,
            signal.direction,
            fraction,
        )
    ]

    mask = (
        window_mask
        & tail_mask
    )

    metrics = _regime_rate(
        target,
        mask,
    )

    return {
        "analysis": "tail",
        "signal": signal.name,
        "direction": signal.direction,
        "fraction": fraction,
        "target": target_name,
        "window": window_name,
        "split": split_name,
        "n": metrics["n"],
        "events": metrics["events"],
        "event_rate": metrics["event_rate"],
    }


def _analyse_interaction(
    cache: ResearchCache,
    signals: tuple[SignalSpec, ...],
    fractions: tuple[float, ...],
    target_name: str,
    window_name: str,
    split_name: str,
) -> dict[str, Any]:
    if len(signals) != 2:
        raise ValueError(
            "interaction requires exactly two signals."
        )

    target = cache.targets[target_name]

    window_mask = cache.window_masks[
        window_name
    ][split_name]

    first_mask = cache.tail_masks[
        _tail_key(
            signals[0].name,
            signals[0].direction,
            fractions[0],
        )
    ]

    second_mask = cache.tail_masks[
        _tail_key(
            signals[1].name,
            signals[1].direction,
            fractions[1],
        )
    ]

    first = (
        window_mask
        & first_mask
    )

    second = (
        window_mask
        & second_mask
    )

    combined = first & second

    first_metrics = _regime_rate(
        target,
        first,
    )

    second_metrics = _regime_rate(
        target,
        second,
    )

    combined_metrics = _regime_rate(
        target,
        combined,
    )

    return {
        "analysis": "interaction",
        "signal_1": signals[0].name,
        "signal_1_direction": (
            signals[0].direction
        ),
        "signal_1_fraction": fractions[0],
        "signal_2": signals[1].name,
        "signal_2_direction": (
            signals[1].direction
        ),
        "signal_2_fraction": fractions[1],
        "target": target_name,
        "window": window_name,
        "split": split_name,
        "signal_1_n": first_metrics["n"],
        "signal_1_events": first_metrics["events"],
        "signal_1_event_rate": (
            first_metrics["event_rate"]
        ),
        "signal_2_n": second_metrics["n"],
        "signal_2_events": second_metrics["events"],
        "signal_2_event_rate": (
            second_metrics["event_rate"]
        ),
        "combined_n": combined_metrics["n"],
        "combined_events": (
            combined_metrics["events"]
        ),
        "combined_event_rate": (
            combined_metrics["event_rate"]
        ),
    }


def _analyse_regime_comparison(
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
    if len(signals) != 2:
        raise ValueError(
            "regime_comparison requires exactly "
            "two signals."
        )

    target = cache.targets[target_name]

    window_mask = cache.window_masks[
        window_name
    ][split_name]

    baseline_mask = cache.tail_masks[
        _tail_key(
            signals[0].name,
            signals[0].direction,
            fractions[0],
        )
    ]

    incremental_mask = cache.tail_masks[
        _tail_key(
            signals[1].name,
            signals[1].direction,
            fractions[1],
        )
    ]

    baseline_in_window = (
        baseline_mask
        & window_mask
    )

    combined_in_window = (
        baseline_mask
        & incremental_mask
        & window_mask
    )

    baseline_metrics = _regime_rate(
        target,
        baseline_in_window,
    )

    combined_metrics = _regime_rate(
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

    seed = _stable_seed(
        spec_id,
        signals[0].name,
        signals[0].direction,
        fractions[0],
        signals[1].name,
        signals[1].direction,
        fractions[1],
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
        "analysis": "regime_comparison",
        "baseline_signal": (
            signals[0].name
        ),
        "baseline_direction": (
            signals[0].direction
        ),
        "baseline_fraction": fractions[0],
        "incremental_signal": (
            signals[1].name
        ),
        "incremental_direction": (
            signals[1].direction
        ),
        "incremental_fraction": fractions[1],
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


def _analyse_multi_regime_comparison(
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

    baseline_metrics = _regime_rate(
        target,
        baseline_in_window,
    )

    combined_metrics = _regime_rate(
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


def _analyse_nested_regime_comparison(
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


def run_spec(
    cache: ResearchCache,
    spec: ResearchSpec,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []

    if spec.analysis.type == "tail":
        for signal in spec.signals:
            for fraction in signal.bins:
                for target_name in spec.targets:
                    for window_name in spec.windows:
                        for split_name in spec.splits:
                            results.append(
                                _analyse_tail(
                                    cache,
                                    signal,
                                    fraction,
                                    target_name,
                                    window_name,
                                    split_name,
                                )
                            )

    elif spec.analysis.type == "interaction":
        if len(spec.signals) != 2:
            raise ValueError(
                "interaction requires exactly "
                "two signals."
            )

        fractions = tuple(
            signal.bins[0]
            for signal in spec.signals
        )

        for target_name in spec.targets:
            for window_name in spec.windows:
                for split_name in spec.splits:
                    results.append(
                        _analyse_interaction(
                            cache,
                            spec.signals,
                            fractions,
                            target_name,
                            window_name,
                            split_name,
                        )
                    )

    elif spec.analysis.type == "regime_comparison":
        if len(spec.signals) != 2:
            raise ValueError(
                "regime_comparison requires exactly "
                "two signals."
            )

        fractions = tuple(
            signal.bins[0]
            for signal in spec.signals
        )

        bootstrap = spec.analysis.bootstrap

        for target_name in spec.targets:
            for window_name in spec.windows:
                for split_name in spec.splits:
                    results.append(
                        _analyse_regime_comparison(
                            cache,
                            spec.signals,
                            target_name,
                            fractions,
                            window_name,
                            split_name,
                            bootstrap=bootstrap,
                            bootstrap_iterations=(
                                spec.analysis
                                .bootstrap_iterations
                            ),
                            spec_id=spec.id,
                        )
                    )

    elif spec.analysis.type == "nested_regime_comparison":
        if len(spec.signals) != 3:
            raise ValueError(
                "nested_regime_comparison requires "
                "exactly three signals."
            )

        fractions = tuple(
            signal.bins[0]
            for signal in spec.signals
        )

        bootstrap = spec.analysis.bootstrap

        for target_name in spec.targets:
            for window_name in spec.windows:
                for split_name in spec.splits:
                    results.append(
                        _analyse_nested_regime_comparison(
                            cache,
                            spec.signals,
                            target_name,
                            target_name,
                            fractions,
                            window_name,
                            split_name,
                            bootstrap=bootstrap,
                            bootstrap_iterations=(
                                spec.analysis
                                .bootstrap_iterations
                            ),
                            spec_id=spec.id,
                        )
                    )

    elif spec.analysis.type == "multi_regime_comparison":
        if len(spec.signals) < 3:
            raise ValueError(
                "multi_regime_comparison requires "
                "at least three signals."
            )

        fractions = tuple(
            signal.bins[0]
            for signal in spec.signals
        )

        bootstrap = spec.analysis.bootstrap

        for target_name in spec.targets:
            for window_name in spec.windows:
                for split_name in spec.splits:
                    results.append(
                        _analyse_multi_regime_comparison(
                            cache,
                            spec.signals,
                            target_name,
                            fractions,
                            window_name,
                            split_name,
                            bootstrap=bootstrap,
                            bootstrap_iterations=(
                                spec.analysis
                                .bootstrap_iterations
                            ),
                            spec_id=spec.id,
                        )
                    )

    else:
        raise ValueError(
            f"Unsupported analysis type: "
            f"{spec.analysis.type}"
        )

    return {
        "spec_id": spec.id,
        "question": spec.question,
        "mode": spec.mode,
        "analysis": spec.analysis.type,
        "results": results,
    }
