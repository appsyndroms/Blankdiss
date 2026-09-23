from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from ml.research.bootstrap import (
    bootstrap_mean_difference,
)
from ml.research.cache import ResearchCache
from ml.research.spec import (
    ResearchSpec,
    SignalSpec,
)


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


def _tail_key(
    signal: SignalSpec,
    fraction: float,
) -> str:
    return (
        f"{signal.name}|"
        f"{signal.direction}|"
        f"{fraction}"
    )


def _binary_metrics(
    target: np.ndarray,
    selected: np.ndarray,
) -> dict[str, Any]:
    valid = np.isfinite(target)

    if not valid.any():
        return {
            "n": 0,
            "events": 0,
            "event_rate": None,
            "baseline_event_rate": None,
            "lift": None,
        }

    y = target[valid]
    selection = selected[valid]

    events = y > 0

    baseline_rate = float(
        events.mean()
    )

    n = int(
        selection.sum()
    )

    if n == 0:
        return {
            "n": 0,
            "events": 0,
            "event_rate": None,
            "baseline_event_rate": baseline_rate,
            "lift": None,
        }

    event_count = int(
        events[selection].sum()
    )

    event_rate = (
        event_count / n
    )

    lift = (
        event_rate / baseline_rate
        if baseline_rate > 0
        else None
    )

    return {
        "n": n,
        "events": event_count,
        "event_rate": float(event_rate),
        "baseline_event_rate": baseline_rate,
        "lift": (
            float(lift)
            if lift is not None
            else None
        ),
    }


def _return_metrics(
    returns: np.ndarray | None,
    selected: np.ndarray,
    *,
    bootstrap: bool,
    bootstrap_iterations: int,
    seed: int,
) -> dict[str, Any]:
    if returns is None:
        return {
            "return_n": 0,
            "mean_return": None,
            "median_return": None,
            "return_difference": None,
            "bootstrap_ci_low": None,
            "bootstrap_ci_high": None,
        }

    selected_valid = (
        selected
        & np.isfinite(returns)
    )

    rest_valid = (
        ~selected
        & np.isfinite(returns)
    )

    selected_values = returns[
        selected_valid
    ]

    rest_values = returns[
        rest_valid
    ]

    if selected_values.size == 0:
        return {
            "return_n": 0,
            "mean_return": None,
            "median_return": None,
            "return_difference": None,
            "bootstrap_ci_low": None,
            "bootstrap_ci_high": None,
        }

    mean_return = float(
        selected_values.mean()
    )

    median_return = float(
        np.median(selected_values)
    )

    return_difference = None

    if rest_values.size:
        return_difference = float(
            mean_return
            - rest_values.mean()
        )

    ci_low = None
    ci_high = None

    if bootstrap:
        ci_low, ci_high = (
            bootstrap_mean_difference(
                selected_values,
                rest_values,
                iterations=(
                    bootstrap_iterations
                ),
                seed=seed,
            )
        )

    return {
        "return_n": int(
            selected_values.size
        ),
        "mean_return": mean_return,
        "median_return": median_return,
        "return_difference": (
            return_difference
        ),
        "bootstrap_ci_low": ci_low,
        "bootstrap_ci_high": ci_high,
    }


def _analyse_tail(
    cache: ResearchCache,
    signal: SignalSpec,
    target_name: str,
    fraction: float,
    window_name: str,
    split_name: str,
    *,
    bootstrap: bool,
    bootstrap_iterations: int,
    spec_id: str,
) -> dict[str, Any]:
    signal_values = cache.signals[
        signal.name
    ]

    target = cache.targets[
        target_name
    ]

    window_mask = cache.window_masks[
        window_name
    ][split_name]

    selected = cache.tail_masks[
        _tail_key(
            signal,
            fraction,
        )
    ]

    valid = (
        window_mask
        & selected
        & np.isfinite(signal_values)
        & np.isfinite(target)
    )

    target_config = cache.target_configs[
        target_name
    ]

    return_column = getattr(
        target_config,
        "return_column",
        None,
    )

    returns = (
        cache.returns.get(
            return_column
        )
        if return_column
        else None
    )

    metrics = _binary_metrics(
        target[window_mask],
        selected[window_mask],
    )

    seed = _stable_seed(
        spec_id,
        signal.name,
        target_name,
        fraction,
        window_name,
        split_name,
    )

    return_metrics = _return_metrics(
        returns,
        window_mask & selected,
        bootstrap=bootstrap,
        bootstrap_iterations=(
            bootstrap_iterations
        ),
        seed=seed,
    )

    return {
        "analysis": "tail",
        "signal": signal.name,
        "direction": signal.direction,
        "fraction": fraction,
        "target": target_name,
        "window": window_name,
        "split": split_name,
        "n_valid": int(
            valid.sum()
        ),
        **metrics,
        **return_metrics,
    }


def _analyse_interaction(
    cache: ResearchCache,
    x: SignalSpec,
    y: SignalSpec,
    target_name: str,
    x_fraction: float,
    y_fraction: float,
    window_name: str,
    split_name: str,
    *,
    bootstrap: bool,
    bootstrap_iterations: int,
    spec_id: str,
) -> dict[str, Any]:
    target = cache.targets[
        target_name
    ]

    window_mask = cache.window_masks[
        window_name
    ][split_name]

    x_mask = cache.tail_masks[
        _tail_key(
            x,
            x_fraction,
        )
    ]

    y_mask = cache.tail_masks[
        _tail_key(
            y,
            y_fraction,
        )
    ]

    selected = (
        x_mask
        & y_mask
    )

    valid = (
        window_mask
        & selected
        & np.isfinite(target)
    )

    target_config = cache.target_configs[
        target_name
    ]

    return_column = getattr(
        target_config,
        "return_column",
        None,
    )

    returns = (
        cache.returns.get(
            return_column
        )
        if return_column
        else None
    )

    metrics = _binary_metrics(
        target[window_mask],
        selected[window_mask],
    )

    seed = _stable_seed(
        spec_id,
        x.name,
        y.name,
        target_name,
        x_fraction,
        y_fraction,
        window_name,
        split_name,
    )

    return_metrics = _return_metrics(
        returns,
        window_mask & selected,
        bootstrap=bootstrap,
        bootstrap_iterations=(
            bootstrap_iterations
        ),
        seed=seed,
    )

    return {
        "analysis": "interaction",
        "signal_x": x.name,
        "direction_x": x.direction,
        "fraction_x": x_fraction,
        "signal_y": y.name,
        "direction_y": y.direction,
        "fraction_y": y_fraction,
        "target": target_name,
        "window": window_name,
        "split": split_name,
        "n_valid": int(
            valid.sum()
        ),
        **metrics,
        **return_metrics,
    }


def run_spec(
    cache: ResearchCache,
    spec: ResearchSpec,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []

    bootstrap = (
        spec.mode == "deep"
        and spec.analysis.bootstrap
    )

    if spec.analysis.type == "tail":
        for signal in spec.signals:
            for target_name in spec.targets:
                for fraction in signal.bins:
                    for window_name in spec.windows:
                        for split_name in spec.splits:
                            results.append(
                                _analyse_tail(
                                    cache,
                                    signal,
                                    target_name,
                                    fraction,
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

    elif spec.analysis.type == "interaction":
        if len(spec.signals) != 2:
            raise ValueError(
                "interaction kräver exakt två signaler."
            )

        x, y = spec.signals

        for target_name in spec.targets:
            for x_fraction in x.bins:
                for y_fraction in y.bins:
                    for window_name in spec.windows:
                        for split_name in spec.splits:
                            results.append(
                                _analyse_interaction(
                                    cache,
                                    x,
                                    y,
                                    target_name,
                                    x_fraction,
                                    y_fraction,
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
        "id": spec.id,
        "question": spec.question,
        "mode": spec.mode,
        "metadata": spec.metadata,
        "results": results,
    }
