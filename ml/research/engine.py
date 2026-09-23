from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd

from ml.config import TARGETS
from ml.research.bootstrap import bootstrap_mean_ci
from ml.research.cache import (
    ResearchCache,
    build_research_cache,
)
from ml.research.experiments import Experiment
from ml.research.spec import ResearchSpec
from ml.research.signals import (
    build_signal,
    tail_mask,
)


def _target_map():
    return {
        target.name: target
        for target in TARGETS
    }


def _build_cache_for_spec(
    frame: pd.DataFrame,
    spec: ResearchSpec,
) -> ResearchCache:
    """
    Bygg en minimal ResearchCache för en research-spec.

    Vi använder den befintliga cachemotorn.
    """
    experiments: list[Experiment] = []

    for signal in spec.signals:
        for target in spec.targets:
            experiments.append(
                Experiment(
                    experiment_id=(
                        f"{spec.id}__"
                        f"{signal.name}__"
                        f"{target}"
                    ),
                    signal_name=signal.name,
                    target_name=target,
                    tail_fraction=0.20,
                    tail_direction=signal.direction,
                )
            )

    return build_research_cache(
        frame,
        experiments,
    )


def _quantile_thresholds(
    values: np.ndarray,
    quantiles: tuple[float, ...],
) -> dict[float, float]:
    valid = values[
        np.isfinite(values)
    ]

    if valid.size == 0:
        return {
            q: float("nan")
            for q in quantiles
        }

    return {
        q: float(
            np.quantile(
                valid,
                q,
            )
        )
        for q in quantiles
    }


def _bucket(
    values: np.ndarray,
    thresholds: dict[float, float],
) -> np.ndarray:
    result = np.full(
        values.shape,
        "LOW",
        dtype=object,
    )

    valid = np.isfinite(values)

    ordered = sorted(
        thresholds.items(),
        key=lambda item: item[1],
    )

    for q, threshold in ordered:
        label = f"Q{int(q * 100):02d}"
        result[
            valid
            & (values >= threshold)
        ] = label

    result[~valid] = "UNKNOWN"

    return result


def _event_metrics(
    target: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    valid = (
        np.isfinite(target)
        & mask
    )

    if not np.any(valid):
        return {
            "n": 0,
            "events": 0,
            "event_rate": None,
            "baseline_rate": None,
            "lift": None,
        }

    y = target[valid]

    event = y > 0

    n = int(
        event.size
    )

    events = int(
        event.sum()
    )

    rate = (
        events / n
        if n
        else None
    )

    baseline_mask = np.isfinite(target)

    baseline = target[
        baseline_mask
    ]

    baseline_rate = (
        float(
            (baseline > 0).mean()
        )
        if baseline.size
        else None
    )

    lift = (
        rate / baseline_rate
        if (
            rate is not None
            and baseline_rate
            and baseline_rate > 0
        )
        else None
    )

    return {
        "n": n,
        "events": events,
        "event_rate": rate,
        "baseline_rate": baseline_rate,
        "lift": lift,
    }


def _return_metrics(
    returns: np.ndarray | None,
    mask: np.ndarray,
    *,
    bootstrap: bool,
    seed: int,
) -> dict[str, Any]:
    if returns is None:
        return {}

    valid = (
        np.isfinite(returns)
        & mask
    )

    values = returns[valid]

    if values.size == 0:
        return {
            "return_n": 0,
            "mean_return": None,
            "median_return": None,
        }

    result = {
        "return_n": int(
            values.size
        ),
        "mean_return": float(
            np.mean(values)
        ),
        "median_return": float(
            np.median(values)
        ),
    }

    if bootstrap:
        low, high = bootstrap_mean_ci(
            values,
            seed=seed,
        )

        result.update(
            {
                "bootstrap_ci_low": low,
                "bootstrap_ci_high": high,
            }
        )

    return result


def _analyse_interaction(
    cache: ResearchCache,
    spec: ResearchSpec,
    window_name: str,
    target_name: str,
) -> list[dict[str, Any]]:
    if len(spec.signals) != 2:
        raise ValueError(
            "interaction kräver exakt två signaler."
        )

    x_spec, y_spec = spec.signals

    x = cache.signals[x_spec.name]
    y = cache.signals[y_spec.name]

    window_mask = cache.window_masks[
        window_name
    ]["test"]

    target = cache.targets[
        target_name
    ]

    target_config = cache.target_configs[
        target_name
    ]

    returns = cache.returns.get(
        getattr(
            target_config,
            "return_column",
            None,
        )
    )

    pretest_mask = cache.window_masks[
        window_name
    ]["train"] | cache.window_masks[
        window_name
    ]["validation"]

    x_thresholds = _quantile_thresholds(
        x[pretest_mask],
        spec.analysis.bins,
    )

    y_thresholds = _quantile_thresholds(
        y[pretest_mask],
        spec.analysis.bins,
    )

    x_bucket = _bucket(
        x,
        x_thresholds,
    )

    y_bucket = _bucket(
        y,
        y_thresholds,
    )

    rows: list[dict[str, Any]] = []

    for xb in sorted(
        set(x_bucket[window_mask])
    ):
        for yb in sorted(
            set(y_bucket[window_mask])
        ):
            cell_mask = (
                window_mask
                & (x_bucket == xb)
                & (y_bucket == yb)
            )

            metrics = _event_metrics(
                target,
                cell_mask,
            )

            metrics.update(
                _return_metrics(
                    returns,
                    cell_mask,
                    bootstrap=(
                        spec.analysis.bootstrap
                    ),
                    seed=(
                        hash(
                            (
                                spec.id,
                                window_name,
                                target_name,
                                xb,
                                yb,
                            )
                        )
                        & 0xFFFFFFFF
                    ),
                )
            )

            rows.append(
                {
                    "research_id": spec.id,
                    "window": window_name,
                    "target": target_name,
                    "x": x_spec.name,
                    "y": y_spec.name,
                    "x_bucket": xb,
                    "y_bucket": yb,
                    **metrics,
                }
            )

    return rows


def run_spec(
    frame: pd.DataFrame,
    spec: ResearchSpec,
) -> dict[str, Any]:
    """
    Kör en deklarativ research specification.

    SCAN:
        snabb, utan bootstrap.

    DEEP:
        samma analys men med bootstrap om
        spec.analysis.bootstrap=True.
    """
    if spec.mode not in {
        "scan",
        "deep",
    }:
        raise ValueError(
            f"Okänt research mode: {spec.mode}"
        )

    cache = _build_cache_for_spec(
        frame,
        spec,
    )

    results: list[dict[str, Any]] = []

    for window_name in spec.windows:
        if window_name not in cache.window_masks:
            raise ValueError(
                f"Okänt walk-forward-fönster: "
                f"{window_name}"
            )

        for target_name in spec.targets:
            if spec.analysis.type == "interaction":
                results.extend(
                    _analyse_interaction(
                        cache,
                        spec,
                        window_name,
                        target_name,
                    )
                )

            else:
                raise ValueError(
                    "Okänd analysis type: "
                    f"{spec.analysis.type}"
                )

    return {
        "id": spec.id,
        "question": spec.question,
        "mode": spec.mode,
        "analysis": asdict(
            spec.analysis
        ),
        "signals": [
            asdict(signal)
            for signal in spec.signals
        ],
        "targets": list(
            spec.targets
        ),
        "windows": list(
            spec.windows
        ),
        "results": results,
        "metadata": spec.metadata,
    }
