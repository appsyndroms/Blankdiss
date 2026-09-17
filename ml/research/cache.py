from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

from ml.config import TARGETS, WALK_FORWARD_WINDOWS
from ml.dataset import TargetConfig, build_target
from ml.research.experiments import Experiment
from ml.research.signals import build_signal, tail_mask


@dataclass
class ResearchCache:
    """
    Precomputed arrays used by the research matrix.

    The expensive work is done once here:
      - signal construction
      - target construction
      - cross-sectional ranks / tail masks
      - walk-forward masks
      - return arrays used for return statistics

    Individual experiments should therefore mostly consist of NumPy
    indexing and metric calculations.
    """

    signals: Dict[str, np.ndarray]
    targets: Dict[str, np.ndarray]
    returns: Dict[str, np.ndarray]
    tail_masks: Dict[str, np.ndarray]
    window_masks: Dict[str, Dict[str, np.ndarray]]
    target_configs: Dict[str, TargetConfig]


def _window_name(index: int) -> str:
    return f"window_{index + 1}"


def _build_window_masks(
    frame: pd.DataFrame,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Build boolean masks for each walk-forward window.

    WalkForwardWindow is a dataclass/object, so its fields must be accessed
    through attributes rather than dictionary indexing.
    """
    snapshot_dates = frame["snapshot_date"].to_numpy(dtype="datetime64[ns]")

    result: Dict[str, Dict[str, np.ndarray]] = {}

    for index, window in enumerate(WALK_FORWARD_WINDOWS):
        train_end = np.datetime64(window.train_end)
        validation_end = np.datetime64(window.validation_end)
        test_end = np.datetime64(window.test_end)

        train_mask = snapshot_dates <= train_end

        validation_mask = (
            (snapshot_dates > train_end)
            & (snapshot_dates <= validation_end)
        )

        test_mask = (
            (snapshot_dates > validation_end)
            & (snapshot_dates <= test_end)
        )

        result[_window_name(index)] = {
            "train": train_mask,
            "validation": validation_mask,
            "test": test_mask,
        }

    return result


def _build_tail_masks(
    frame: pd.DataFrame,
    experiments: list[Experiment],
) -> Dict[str, np.ndarray]:
    """
    Build every unique signal/tail combination only once.
    """
    result: Dict[str, np.ndarray] = {}
    seen: set[str] = set()

    for experiment in experiments:
        key = (
            f"{experiment.signal_name}|"
            f"{experiment.tail_direction}|"
            f"{experiment.tail_fraction}"
        )

        if key in seen:
            continue

        seen.add(key)

        signal = build_signal(frame, experiment.signal_name)

        result[key] = tail_mask(
            frame,
            signal,
            experiment.tail_fraction,
            experiment.tail_direction,
        )

    return result


def _build_signals(
    frame: pd.DataFrame,
    experiments: list[Experiment],
) -> Dict[str, np.ndarray]:
    """
    Build each unique signal exactly once.
    """
    signal_names = sorted(
        {experiment.signal_name for experiment in experiments}
    )

    return {
        signal_name: build_signal(frame, signal_name)
        for signal_name in signal_names
    }


def _build_targets(
    frame: pd.DataFrame,
    experiments: list[Experiment],
) -> tuple[Dict[str, np.ndarray], Dict[str, TargetConfig]]:
    """
    Build each unique target exactly once.
    """
    target_names = sorted(
        {experiment.target_name for experiment in experiments}
    )

    target_configs = {
        target.name: target
        for target in TARGETS
        if target.name in target_names
    }

    missing = sorted(set(target_names) - set(target_configs))

    if missing:
        raise ValueError(
            "Unknown research targets: "
            + ", ".join(missing)
        )

    targets: Dict[str, np.ndarray] = {}

    for target_name in target_names:
        target_config = target_configs[target_name]

        target = build_target(
            frame,
            target_config,
        )

        targets[target_name] = pd.to_numeric(
            target,
            errors="coerce",
        ).to_numpy(dtype=float)

    return targets, target_configs


def _build_returns(
    frame: pd.DataFrame,
    target_configs: Dict[str, TargetConfig],
) -> Dict[str, np.ndarray]:
    """
    Cache return arrays once per return column.

    Some TargetConfig entries, notably regression targets, may not have
    a return_column. Those targets simply do not receive a return array.
    """
    columns = sorted(
        {
            target.return_column
            for target in target_configs.values()
            if getattr(target, "return_column", None)
            and target.return_column in frame.columns
        }
    )

    return {
        column: pd.to_numeric(
            frame[column],
            errors="coerce",
        ).to_numpy(dtype=float)
        for column in columns
    }


def build_research_cache(
    frame: pd.DataFrame,
    experiments: list[Experiment],
) -> ResearchCache:
    """
    Build all reusable research data structures.

    This function is deliberately expensive and should only be called once
    per research run.
    """
    signals = _build_signals(
        frame,
        experiments,
    )

    targets, target_configs = _build_targets(
        frame,
        experiments,
    )

    returns = _build_returns(
        frame,
        target_configs,
    )

    tail_masks = _build_tail_masks(
        frame,
        experiments,
    )

    window_masks = _build_window_masks(
        frame,
    )

    return ResearchCache(
        signals=signals,
        targets=targets,
        returns=returns,
        tail_masks=tail_masks,
        window_masks=window_masks,
        target_configs=target_configs,
    )


def get_tail_mask_key(
    experiment: Experiment,
) -> str:
    return (
        f"{experiment.signal_name}|"
        f"{experiment.tail_direction}|"
        f"{experiment.tail_fraction}"
    )
