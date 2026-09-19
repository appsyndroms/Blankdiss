from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ml.research.config import TARGETS, WALK_FORWARD_WINDOWS
from ml.research.discovery.config import DiscoveryConfig
from ml.research.discovery.features import build_signal, load_features
from ml.research.discovery.metrics import bootstrap_mean_difference
from ml.research.discovery.targets import build_target, _target_return_column
from ml.research.discovery.validation import tail_mask


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


def _stable_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False)


def _target_map() -> dict[str, Any]:
    return {target.name: target for target in TARGETS}


def _validate_targets(config: DiscoveryConfig) -> None:
    available = _target_map()

    missing = [
        target_name
        for target_name in config.targets
        if target_name not in available
    ]

    if missing:
        raise ValueError(
            f"Unknown target(s): {', '.join(missing)}"
        )


def _validate_signals(config: DiscoveryConfig) -> None:
    if not config.signals:
        raise ValueError("Discovery requires at least one signal.")

    if not config.stress_features:
        raise ValueError("Discovery requires at least one stress feature.")


def _build_window_masks(
    frame: pd.DataFrame,
) -> dict[str, dict[str, np.ndarray]]:
    snapshot_dates = pd.to_datetime(
        frame["snapshot_date"],
        errors="coerce",
    )

    windows: dict[str, dict[str, np.ndarray]] = {}

    for window in WALK_FORWARD_WINDOWS:
        train_start = pd.Timestamp(window.train_start)
        train_end = pd.Timestamp(window.train_end)
        validation_start = pd.Timestamp(window.validation_start)
        validation_end = pd.Timestamp(window.validation_end)
        test_start = pd.Timestamp(window.test_start)
        test_end = pd.Timestamp(window.test_end)

        windows[window.name] = {
            "train": (
                (snapshot_dates >= train_start)
                & (snapshot_dates <= train_end)
            ).to_numpy(),
            "validation": (
                (snapshot_dates >= validation_start)
                & (snapshot_dates <= validation_end)
            ).to_numpy(),
            "test": (
                (snapshot_dates >= test_start)
                & (snapshot_dates <= test_end)
            ).to_numpy(),
        }

    return windows


def prepare_data(config: DiscoveryConfig) -> DiscoveryData:
    _validate_targets(config)
    _validate_signals(config)

    frame = load_features()

    targets: dict[str, np.ndarray] = {}
    returns: dict[str, np.ndarray] = {}

    for target_name in config.targets:
        target = _target_map()[target_name]

        target_values = build_target(frame, target)
        targets[target_name] = np.asarray(
            target_values,
            dtype=float,
        )

        return_column = _target_return_column(target)

        if return_column not in frame.columns:
            raise ValueError(
                f"Return column '{return_column}' for target "
                f"'{target_name}' is missing from feature frame."
            )

        returns[target_name] = pd.to_numeric(
            frame[return_column],
            errors="coerce",
        ).to_numpy(dtype=float)

    signals: dict[str, np.ndarray] = {}

    for signal_name in config.signals:
        signals[signal_name] = np.asarray(
            build_signal(frame, signal_name),
            dtype=float,
        )

    stress_signals: dict[str, np.ndarray] = {}

    for stress_feature in config.stress_features:
        stress_signals[stress_feature] = np.asarray(
            build_signal(frame, stress_feature),
            dtype=float,
        )

    windows = _build_window_masks(frame)

    return DiscoveryData(
        frame=frame,
        targets=targets,
        signals=signals,
        stress_signals=stress_signals,
        windows=windows,
        returns=returns,
    )


def build_candidates(
    config: DiscoveryConfig,
) -> list[Candidate]:
    candidates: list[Candidate] = []

    for target_name in config.targets:
        for signal_name in config.signals:
            for stress_feature in config.stress_features:
                stress_direction = config.stress_directions.get(
                    stress_feature,
                    "upper",
                )

                for signal_tail in config.tails:
                    for stress_tail in config.tails:
                        candidate_id = (
                            f"{target_name}__"
                            f"{signal_name}__"
                            f"{_tail_name(signal_tail)}__"
                            f"{stress_feature}__"
                            f"{_tail_name(stress_tail)}__"
                            f"{stress_direction}"
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


def _tail_name(tail: float) -> str:
    return f"{tail:.4f}".replace(".", "p")


def evaluate_candidate_on_mask(
    data: DiscoveryData,
    candidate: Candidate,
    base_mask: np.ndarray,
    split: str,
) -> dict[str, Any]:
    """
    Evaluate one exact candidate on an arbitrary observation mask.

    This is the shared metric implementation used by discovery and by
    frozen/OOS evaluation. Keeping the calculation here ensures that
    discovery and frozen validation use exactly the same definitions.
    """
    target = data.targets[candidate.target_name]

    signal = data.signals[candidate.signal_name]
    stress_signal = data.stress_signals[candidate.stress_feature]

    signal_mask = tail_mask(
        signal,
        candidate.signal_tail,
        direction="upper",
    )

    stress_mask = tail_mask(
        stress_signal,
        candidate.stress_tail,
        direction=candidate.stress_direction,
    )

    finite_target = np.isfinite(target)

    selected = (
        base_mask
        & signal_mask
        & stress_mask
        & finite_target
    )

    rest = (
        base_mask
        & ~selected
        & finite_target
    )

    n = int(selected.sum())
    rest_n = int(rest.sum())

    events = int((target[selected] > 0).sum())

    event_rate = (
        events / n
        if n > 0
        else float("nan")
    )

    baseline_values = target[
        base_mask & finite_target
    ]

    baseline_event_rate = (
        float((baseline_values > 0).mean())
        if len(baseline_values)
        else float("nan")
    )

    lift = (
        event_rate - baseline_event_rate
        if np.isfinite(event_rate)
        and np.isfinite(baseline_event_rate)
        else float("nan")
    )

    return_values = data.returns[candidate.target_name]

    selected_returns = return_values[selected]
    rest_returns = return_values[rest]

    selected_returns = selected_returns[
        np.isfinite(selected_returns)
    ]

    rest_returns = rest_returns[
        np.isfinite(rest_returns)
    ]

    mean_return = (
        float(selected_returns.mean())
        if len(selected_returns)
        else float("nan")
    )

    rest_mean_return = (
        float(rest_returns.mean())
        if len(rest_returns)
        else float("nan")
    )

    return_difference = (
        mean_return - rest_mean_return
        if np.isfinite(mean_return)
        and np.isfinite(rest_mean_return)
        else float("nan")
    )

    bootstrap_ci_low = float("nan")
    bootstrap_ci_high = float("nan")

    if len(selected_returns) and len(rest_returns):
        seed = _stable_seed(
            candidate.candidate_id,
            split,
        )

        bootstrap = bootstrap_mean_difference(
            selected_returns,
            rest_returns,
            seed=seed,
        )

        if bootstrap is not None:
            bootstrap_ci_low = float(
                bootstrap[0]
            )
            bootstrap_ci_high = float(
                bootstrap[1]
            )

    total_base_n = int(
        (base_mask & finite_target).sum()
    )

    selected_fraction = (
        n / total_base_n
        if total_base_n > 0
        else float("nan")
    )

    return {
        "candidate_id": candidate.candidate_id,
        "target_name": candidate.target_name,
        "signal_name": candidate.signal_name,
        "signal_tail": candidate.signal_tail,
        "stress_feature": candidate.stress_feature,
        "stress_tail": candidate.stress_tail,
        "stress_direction": candidate.stress_direction,
        "window": split,
        "split": split,
        "n": n,
        "rest_n": rest_n,
        "events": events,
        "event_rate": event_rate,
        "baseline_event_rate": baseline_event_rate,
        "lift": lift,
        "mean_return": mean_return,
        "rest_mean_return": rest_mean_return,
        "return_difference": return_difference,
        "bootstrap_ci_low": bootstrap_ci_low,
        "bootstrap_ci_high": bootstrap_ci_high,
        "selected_fraction": selected_fraction,
    }


def _evaluate_candidate(
    data: DiscoveryData,
    candidate: Candidate,
    window_name: str,
) -> dict[str, Any]:
    return evaluate_candidate_on_mask(
        data=data,
        candidate=candidate,
        base_mask=data.windows[window_name]["test"],
        split=window_name,
    )


def evaluate_candidates(
    data: DiscoveryData,
    candidates: list[Candidate],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    for candidate in candidates:
        for window_name in data.windows:
            result = _evaluate_candidate(
                data=data,
                candidate=candidate,
                window_name=window_name,
            )

            results.append(result)

    return results


def pool_results(
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not results:
        return []

    frame = pd.DataFrame(results)

    group_columns = [
        "candidate_id",
        "target_name",
        "signal_name",
        "signal_tail",
        "stress_feature",
        "stress_tail",
        "stress_direction",
    ]

    pooled: list[dict[str, Any]] = []

    for keys, group in frame.groupby(
        group_columns,
        dropna=False,
    ):
        (
            candidate_id,
            target_name,
            signal_name,
            signal_tail,
            stress_feature,
            stress_tail,
            stress_direction,
        ) = keys

        n = int(group["n"].sum())
        events = int(group["events"].sum())

        event_rate = (
            events / n
            if n > 0
            else float("nan")
        )

        baseline_event_rate = (
            float(
                np.average(
                    group["baseline_event_rate"],
                    weights=group["n"],
                )
            )
            if n > 0
            else float("nan")
        )

        lift = (
            event_rate - baseline_event_rate
            if np.isfinite(event_rate)
            and np.isfinite(baseline_event_rate)
            else float("nan")
        )

        selected_returns = group[
            "mean_return"
        ].to_numpy(dtype=float)

        rest_returns = group[
            "rest_mean_return"
        ].to_numpy(dtype=float)

        valid_selected = np.isfinite(selected_returns)
        valid_rest = np.isfinite(rest_returns)

        weighted_mean_return = (
            float(
                np.average(
                    selected_returns[valid_selected],
                    weights=group.loc[
                        valid_selected,
                        "n",
                    ],
                )
            )
            if valid_selected.any()
            else float("nan")
        )

        weighted_rest_mean_return = (
            float(
                np.average(
                    rest_returns[valid_rest],
                    weights=group.loc[
                        valid_rest,
                        "rest_n",
                    ],
                )
            )
            if valid_rest.any()
            else float("nan")
        )

        return_difference = (
            weighted_mean_return
            - weighted_rest_mean_return
            if np.isfinite(weighted_mean_return)
            and np.isfinite(weighted_rest_mean_return)
            else float("nan")
        )

        pooled.append(
            {
                "candidate_id": candidate_id,
                "target_name": target_name,
                "signal_name": signal_name,
                "signal_tail": signal_tail,
                "stress_feature": stress_feature,
                "stress_tail": stress_tail,
                "stress_direction": stress_direction,
                "window": "pooled",
                "split": "pooled",
                "n": n,
                "events": events,
                "event_rate": event_rate,
                "baseline_event_rate": baseline_event_rate,
                "lift": lift,
                "mean_return": weighted_mean_return,
                "rest_mean_return": weighted_rest_mean_return,
                "return_difference": return_difference,
            }
        )

    return pooled


def find_candidates(
    results: list[dict[str, Any]],
    config: DiscoveryConfig,
) -> list[dict[str, Any]]:
    if not results:
        return []

    frame = pd.DataFrame(results)

    if "lift" not in frame.columns:
        return []

    frame = frame[
        np.isfinite(
            pd.to_numeric(
                frame["lift"],
                errors="coerce",
            )
        )
    ].copy()

    if frame.empty:
        return []

    frame = frame.sort_values(
        ["lift", "n"],
        ascending=[False, False],
    )

    return frame.head(
        config.validation.top_k
    ).to_dict("records")


def run_discovery(
    config: DiscoveryConfig,
) -> dict[str, Any]:
    data = prepare_data(config)

    candidates = build_candidates(config)

    results = evaluate_candidates(
        data=data,
        candidates=candidates,
    )

    pooled = pool_results(results)

    selected = find_candidates(
        pooled,
        config,
    )

    return {
        "results": results,
        "pooled": pooled,
        "selected": selected,
        "candidate_count": len(candidates),
    }
