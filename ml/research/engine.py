from __future__ import annotations

from itertools import product
from typing import Any

from .analysis_utils import regime_rate
from .cache import ResearchCache, _tail_key
from .conditional import (
    analyse_conditional_regime_comparison,
)
from .derived_metrics import apply_derived_metrics
from .interaction import analyse_interaction
from .multi_regime import analyse_multi_regime_comparison
from .nested_regime import analyse_nested_regime_comparison
from .regime import analyse_regime_comparison
from .spec import ResearchSpec, SignalSpec
from .stratified_interaction import (
    analyse_stratified_interaction,
)
from .stratified_regime import (
    analyse_stratified_regime_comparison,
)


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

    metrics = regime_rate(
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
        fractions = tuple(
            signal.bins[0]
            for signal in spec.signals
        )

        bootstrap = spec.analysis.bootstrap

        for target_name in spec.targets:
            for window_name in spec.windows:
                for split_name in spec.splits:
                    results.append(
                        analyse_interaction(
                            cache,
                            spec.signals,
                            fractions,
                            target_name,
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

    elif spec.analysis.type == "regime_comparison":
        fractions = tuple(
            signal.bins[0]
            for signal in spec.signals
        )

        bootstrap = spec.analysis.bootstrap

        for target_name in spec.targets:
            for window_name in spec.windows:
                for split_name in spec.splits:
                    results.append(
                        analyse_regime_comparison(
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
        fractions = tuple(
            signal.bins[0]
            for signal in spec.signals
        )

        bootstrap = spec.analysis.bootstrap

        for target_name in spec.targets:
            for window_name in spec.windows:
                for split_name in spec.splits:
                    results.append(
                        analyse_nested_regime_comparison(
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

    elif spec.analysis.type == "conditional_regime_comparison":
        bootstrap = spec.analysis.bootstrap

        baseline_fraction_options = [
            signal.bins
            for signal in spec.signals[:-1]
        ]

        for baseline_fractions in product(
            *baseline_fraction_options
        ):
            for incremental_fraction in (
                spec.signals[-1].bins
            ):
                fractions = (
                    *baseline_fractions,
                    incremental_fraction,
                )

                for target_name in spec.targets:
                    for window_name in spec.windows:
                        for split_name in spec.splits:
                            results.append(
                                analyse_conditional_regime_comparison(
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

    elif spec.analysis.type == "multi_regime_comparison":
        fractions = tuple(
            signal.bins[0]
            for signal in spec.signals
        )

        bootstrap = spec.analysis.bootstrap

        for target_name in spec.targets:
            for window_name in spec.windows:
                for split_name in spec.splits:
                    results.append(
                        analyse_multi_regime_comparison(
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

    elif (
        spec.analysis.type
        == "stratified_regime_comparison"
    ):
        fractions = tuple(
            signal.bins[0]
            for signal in spec.signals
        )

        bootstrap = spec.analysis.bootstrap

        for target_name in spec.targets:
            for window_name in spec.windows:
                for split_name in spec.splits:
                    results.extend(
                        analyse_stratified_regime_comparison(
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

    elif (
        spec.analysis.type
        == "stratified_interaction"
    ):
        fractions = tuple(
            signal.bins[0]
            for signal in spec.signals
        )

        bootstrap = spec.analysis.bootstrap

        for target_name in spec.targets:
            for window_name in spec.windows:
                for split_name in spec.splits:
                    results.extend(
                        analyse_stratified_interaction(
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

    results = apply_derived_metrics(
        spec,
        results,
    )

    return {
        "spec_id": spec.id,
        "question": spec.question,
        "mode": spec.mode,
        "analysis": spec.analysis.type,
        "results": results,
    }
