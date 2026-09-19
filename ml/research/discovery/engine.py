from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ml.config import (
    TARGETS,
    WALK_FORWARD_WINDOWS,
)
from ml.dataset import (
    build_target,
    load_features,
)
from ml.research.bootstrap import (
    bootstrap_mean_difference,
)
from ml.research.signals import (
    build_signal,
    tail_mask,
)

from .config import DiscoveryConfig


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    target_name: str
    signal_name: str
    signal_tail: float
    stress_feature: str
    stress_tail: float
    stress_direction: str


@dataclass(frozen=True)
class DiscoveryData:
    frame: pd.DataFrame
    targets: dict[str, np.ndarray]
    signals: dict[str, np.ndarray]
    stress_signals: dict[str, np.ndarray]
    windows: dict[str, dict[str, np.ndarray]]
    returns: dict[str, np.ndarray]


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


def _target_map() -> dict[str, Any]:
    return {
        target.name: target
        for target in TARGETS
    }


def _validate_targets(
    config: DiscoveryConfig,
) -> dict[str, Any]:
    available = _target_map()

    unknown = sorted(
        set(config.targets)
        - set(available)
    )

    if unknown:
        raise ValueError(
            "Okända discovery-targets: "
            + ", ".join(unknown)
        )

    return {
        name: available[name]
        for name in config.targets
    }


def _validate_signals(
    config: DiscoveryConfig,
) -> None:
    for signal_name in config.signals:
        build_signal(
            pd.DataFrame(
                {
                    "snapshot_date": [],
                }
            ),
            signal_name,
        )


def _build_window_masks(
    frame: pd.DataFrame,
) -> dict[str, dict[str, np.ndarray]]:
    dates = pd.to_datetime(
        frame["snapshot_date"]
    )

    masks: dict[
        str,
        dict[str, np.ndarray],
    ] = {}

    for index, window in enumerate(
        WALK_FORWARD_WINDOWS,
        start=1,
    ):
        train_end = pd.Timestamp(
            window.train_end
        )
        validation_end = pd.Timestamp(
            window.validation_end
        )
        test_end = pd.Timestamp(
            window.test_end
        )

        train_mask = (
            dates <= train_end
        )

        validation_mask = (
            (dates > train_end)
            & (dates <= validation_end)
        )

        test_mask = (
            (dates > validation_end)
            & (dates <= test_end)
        )

        masks[
            f"window_{index}"
        ] = {
            "train": train_mask.to_numpy(),
            "validation": validation_mask.to_numpy(),
            "test": test_mask.to_numpy(),
        }

    return masks


def prepare_data(
    config: DiscoveryConfig,
) -> DiscoveryData:
    frame = load_features()

    targets_config = _validate_targets(
        config
    )

    signals = {
        name: build_signal(
            frame,
            name,
        ).to_numpy(dtype=float)
        for name in config.signals
    }

    stress_signals = {
        name: build_signal(
            frame,
            name,
        ).to_numpy(dtype=float)
        for name in config.stress_features
    }

    targets = {
        name: build_target(
            frame,
            target,
        ).to_numpy(dtype=float)
        for name, target
        in targets_config.items()
    }

    return_columns = {
        target.return_column
        for target in targets_config.values()
    }

    returns = {
        column: pd.to_numeric(
            frame[column],
            errors="coerce",
        ).to_numpy(dtype=float)
        for column in return_columns
        if column in frame.columns
    }

    return DiscoveryData(
        frame=frame,
        targets=targets,
        signals=signals,
        stress_signals=stress_signals,
        windows=_build_window_masks(
            frame
        ),
        returns=returns,
    )


def build_candidates(
    config: DiscoveryConfig,
) -> list[Candidate]:
    candidates: list[Candidate] = []

    for target_name in config.targets:
        for signal_name in config.signals:
            for stress_feature in config.stress_features:
                stress_direction = (
                    config.stress_directions[
                        stress_feature
                    ]
                )

                for signal_tail in config.tails:
                    for stress_tail in config.tails:
                        candidate_id = (
                            f"{signal_name}"
                            f"__{_tail_name(signal_tail)}"
                            f"__{stress_feature}"
                            f"__{stress_direction}"
                            f"__{_tail_name(stress_tail)}"
                            f"__{target_name}"
                        )

                        candidates.append(
                            Candidate(
                                candidate_id=candidate_id,
                                target_name=target_name,
                                signal_name=signal_name,
                                signal_tail=signal_tail,
                                stress_feature=stress_feature,
                                stress_tail=stress_tail,
                                stress_direction=stress_direction,
                            )
                        )

    return candidates


def _tail_name(
    fraction: float,
) -> str:
    if fraction == 0.20:
        return "20pct"

    if fraction == 0.10:
        return "10pct"

    if fraction == 0.05:
        return "5pct"

    if fraction == 0.025:
        return "2_5pct"

    if fraction == 0.01:
        return "1pct"

    return str(
        fraction
    ).replace(
        ".",
        "_",
    )


def _evaluate_candidate(
    data: DiscoveryData,
    candidate: Candidate,
    window_name: str,
) -> dict[str, Any]:
    target = data.targets[
        candidate.target_name
    ]

    signal = data.signals[
        candidate.signal_name
    ]

    stress = data.stress_signals[
        candidate.stress_feature
    ]

    window_mask = data.windows[
        window_name
    ]["test"]

    frame = data.frame

    signal_tail_mask = tail_mask(
        frame,
        pd.Series(
            signal,
            index=frame.index,
        ),
        candidate.signal_tail,
        direction="upper",
    ).to_numpy()

    stress_tail_mask = tail_mask(
        frame,
        pd.Series(
            stress,
            index=frame.index,
        ),
        candidate.stress_tail,
        direction=candidate.stress_direction,
    ).to_numpy()

    selected = (
        window_mask
        & signal_tail_mask
        & stress_tail_mask
        & np.isfinite(target)
    )

    rest = (
        window_mask
        & ~selected
        & np.isfinite(target)
    )

    selected_target = target[selected]
    rest_target = target[rest]

    n = int(
        selected.sum()
    )

    events = (
        selected_target > 0
    )

    event_count = int(
        events.sum()
    )

    event_rate = (
        float(
            events.mean()
        )
        if n
        else None
    )

    baseline_mask = (
        window_mask
        & np.isfinite(target)
    )

    baseline_target = target[
        baseline_mask
    ]

    baseline_events = (
        baseline_target > 0
    )

    baseline_rate = (
        float(
            baseline_events.mean()
        )
        if baseline_target.size
        else None
    )

    lift = (
        event_rate / baseline_rate
        if (
            event_rate is not None
            and baseline_rate
            and baseline_rate > 0
        )
        else None
    )

    return_column = _target_return_column(
        candidate.target_name
    )

    returns = data.returns.get(
        return_column
    )

    mean_return = None
    rest_mean_return = None
    return_difference = None
    ci_low = None
    ci_high = None

    if returns is not None:
        selected_returns = returns[
            selected
            & np.isfinite(returns)
        ]

        rest_returns = returns[
            rest
            & np.isfinite(returns)
        ]

        if selected_returns.size:
            mean_return = float(
                selected_returns.mean()
            )

        if rest_returns.size:
            rest_mean_return = float(
                rest_returns.mean()
            )

        if (
            mean_return is not None
            and rest_mean_return is not None
        ):
            return_difference = (
                mean_return
                - rest_mean_return
            )

            ci_low, ci_high = (
                bootstrap_mean_difference(
                    selected_returns,
                    rest_returns,
                    seed=_stable_seed(
                        candidate.candidate_id,
                        window_name,
                    ),
                )
            )

    return {
        "candidate_id": candidate.candidate_id,
        "target_name": candidate.target_name,
        "signal_name": candidate.signal_name,
        "signal_tail": candidate.signal_tail,
        "stress_feature": candidate.stress_feature,
        "stress_tail": candidate.stress_tail,
        "stress_direction": candidate.stress_direction,
        "window": window_name,
        "split": "test",
        "n": n,
        "events": event_count,
        "event_rate": event_rate,
        "baseline_event_rate": baseline_rate,
        "lift": lift,
        "mean_return": mean_return,
        "rest_mean_return": rest_mean_return,
        "return_difference": return_difference,
        "bootstrap_ci_low": ci_low,
        "bootstrap_ci_high": ci_high,
        "selected_fraction": (
            float(
                selected[window_mask].mean()
            )
            if window_mask.any()
            else None
        ),
    }


def _target_return_column(
    target_name: str,
) -> str:
    target = _target_map().get(
        target_name
    )

    if target is None:
        raise ValueError(
            f"Okänd target: {target_name}"
        )

    return target.return_column


def evaluate_candidates(
    data: DiscoveryData,
    candidates: list[Candidate],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    window_names = list(
        data.windows
    )

    total = (
        len(candidates)
        * len(window_names)
    )

    completed = 0

    for candidate in candidates:
        for window_name in window_names:
            result = _evaluate_candidate(
                data,
                candidate,
                window_name,
            )

            results.append(
                result
            )

            completed += 1

            if (
                completed == 1
                or completed % 100 == 0
                or completed == total
            ):
                print(
                    "Discovery progress: "
                    f"{completed:,}/{total:,}",
                    flush=True,
                )

    return results


def pool_results(
    results: list[dict[str, Any]],
    min_rows_per_window: int = 1,
) -> list[dict[str, Any]]:
    grouped: dict[
        str,
        list[dict[str, Any]],
    ] = {}

    for result in results:
        grouped.setdefault(
            result["candidate_id"],
            [],
        ).append(result)

    pooled: list[dict[str, Any]] = []

    for candidate_id, rows in grouped.items():
        first = rows[0]

        valid_rows = [
            row
            for row in rows
            if row["n"] >= min_rows_per_window
        ]

        lift_values = _values(
            valid_rows,
            "lift",
        )

        event_rates = _values(
            valid_rows,
            "event_rate",
        )

        return_differences = _values(
            valid_rows,
            "return_difference",
        )

        mean_returns = _values(
            valid_rows,
            "mean_return",
        )

        window_metrics = [
            {
                "window": row["window"],
                "n": row["n"],
                "events": row["events"],
                "event_rate": row["event_rate"],
                "baseline_event_rate": row[
                    "baseline_event_rate"
                ],
                "lift": row["lift"],
                "mean_return": row[
                    "mean_return"
                ],
                "rest_mean_return": row[
                    "rest_mean_return"
                ],
                "return_difference": row[
                    "return_difference"
                ],
                "bootstrap_ci_low": row[
                    "bootstrap_ci_low"
                ],
                "bootstrap_ci_high": row[
                    "bootstrap_ci_high"
                ],
            }
            for row in valid_rows
        ]

        pooled.append(
            {
                "candidate_id": candidate_id,
                "target_name": first["target_name"],
                "signal_name": first["signal_name"],
                "signal_tail": first["signal_tail"],
                "stress_feature": first[
                    "stress_feature"
                ],
                "stress_tail": first[
                    "stress_tail"
                ],
                "stress_direction": first[
                    "stress_direction"
                ],
                "windows": len(rows),
                "valid_windows": len(
                    valid_rows
                ),
                "min_n": min(
                    (
                        row["n"]
                        for row in rows
                    ),
                    default=0,
                ),
                "max_n": max(
                    (
                        row["n"]
                        for row in rows
                    ),
                    default=0,
                ),
                "lift": _mean(
                    lift_values
                ),
                "event_rate": _mean(
                    event_rates
                ),
                "mean_return": _mean(
                    mean_returns
                ),
                "return_difference": _mean(
                    return_differences
                ),
                "lift_min": min(
                    lift_values,
                    default=None,
                ),
                "lift_max": max(
                    lift_values,
                    default=None,
                ),
                "lift_spread": (
                    max(lift_values)
                    - min(lift_values)
                    if lift_values
                    else None
                ),
                "return_difference_min": min(
                    return_differences,
                    default=None,
                ),
                "return_difference_max": max(
                    return_differences,
                    default=None,
                ),
                "return_difference_spread": (
                    max(return_differences)
                    - min(return_differences)
                    if return_differences
                    else None
                ),
                "window_metrics": window_metrics,
                "stable_lift_windows": sum(
                    1
                    for row in valid_rows
                    if (
                        row.get("lift")
                        is not None
                        and row["lift"] > 1.0
                    )
                ),
                "negative_return_windows": sum(
                    1
                    for row in valid_rows
                    if (
                        row.get(
                            "return_difference"
                        )
                        is not None
                        and row[
                            "return_difference"
                        ] < 0
                    )
                ),
            }
        )

    return pooled


def find_candidates(
    pooled: list[dict[str, Any]],
    config: DiscoveryConfig,
) -> list[dict[str, Any]]:
    validation = config.validation

    findings = []

    for row in pooled:
        if (
            row["valid_windows"]
            < validation.min_positive_windows
        ):
            continue

        lift_ok = (
            row["lift"] is not None
            and row["lift"]
            >= validation.min_lift
        )

        return_ok = (
            row["return_difference"]
            is not None
            and row["return_difference"]
            <= validation.max_return_difference
        )

        if not (
            lift_ok
            or return_ok
        ):
            continue

        score = 0.0

        if row["lift"] is not None:
            score += (
                row["lift"] - 1.0
            )

        if (
            row["return_difference"]
            is not None
        ):
            score += max(
                0.0,
                -row["return_difference"],
            )

        findings.append(
            {
                **row,
                "discovery_score": score,
                "lift_signal": lift_ok,
                "return_signal": return_ok,
            }
        )

    findings.sort(
        key=lambda row: (
            row["discovery_score"],
            row.get(
                "lift",
                0.0,
            ) or 0.0,
        ),
        reverse=True,
    )

    return findings[
        : validation.max_findings
    ]


def run_discovery(
    config: DiscoveryConfig,
) -> tuple[
    DiscoveryData,
    list[Candidate],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    print(
        "Preparing discovery data...",
        flush=True,
    )

    data = prepare_data(
        config
    )

    candidates = build_candidates(
        config
    )

    print(
        f"Discovery candidates: "
        f"{len(candidates):,}",
        flush=True,
    )

    results = evaluate_candidates(
        data,
        candidates,
    )

    pooled = pool_results(
        results,
        min_rows_per_window=(
            config.validation
            .min_rows_per_window
        ),
    )

    findings = find_candidates(
        pooled,
        config,
    )

    print(
        f"Discovery findings: "
        f"{len(findings):,}",
        flush=True,
    )

    return (
        data,
        candidates,
        results,
        pooled,
        findings,
    )


def _values(
    rows: list[dict[str, Any]],
    key: str,
) -> list[float]:
    return [
        float(row[key])
        for row in rows
        if row.get(key) is not None
    ]


def _mean(
    values: list[float],
) -> float | None:
    if not values:
        return None

    return sum(values) / len(values)
