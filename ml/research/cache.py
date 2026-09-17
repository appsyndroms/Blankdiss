"""Precomputation and caching for Blankdiss research."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

from ml.config import TARGETS, WALK_FORWARD_WINDOWS, TargetConfig
from ml.dataset import build_target

from .experiments import TAIL_FRACTIONS
from .signals import build_signal


@dataclass(frozen=True)
class ResearchCache:
    """
    Immutable-ish cache of all expensive research preprocessing.

    Expensive work is deliberately done once:

    - signal construction
    - target construction
    - cross-sectional ranking
    - tail masks
    - walk-forward test masks

    Experiment evaluation then operates primarily on NumPy arrays.
    """

    signals: Dict[str, np.ndarray]
    targets: Dict[str, np.ndarray]
    tail_masks: Dict[tuple[str, str, float], np.ndarray]
    window_masks: Dict[str, np.ndarray]
    target_configs: Dict[str, TargetConfig]

    @classmethod
    def build(
        cls,
        frame: pd.DataFrame,
        signal_names: list[str],
        target_names: list[str],
    ) -> "ResearchCache":
        """Build all reusable research state once."""

        target_configs = {
            target.name: target
            for target in TARGETS
        }

        signals: Dict[str, np.ndarray] = {}
        targets: Dict[str, np.ndarray] = {}
        tail_masks: Dict[
            tuple[str, str, float],
            np.ndarray,
        ] = {}

        # ---------------------------------------------------------
        # Signals
        # ---------------------------------------------------------

        for signal_name in signal_names:
            signal = build_signal(
                frame,
                signal_name,
            )

            signals[signal_name] = (
                signal.to_numpy(
                    dtype=np.float64,
                    copy=False,
                )
            )

        # ---------------------------------------------------------
        # Targets
        # ---------------------------------------------------------

        for target_name in target_names:
            target = target_configs.get(
                target_name
            )

            if target is None:
                raise ValueError(
                    "Unknown target requested by "
                    f"research matrix: {target_name}"
                )

            target_series = build_target(
                frame,
                target,
            )

            targets[target_name] = (
                target_series.to_numpy(
                    dtype=np.float64,
                    copy=False,
                )
            )

        # ---------------------------------------------------------
        # Cross-sectional ranks
        #
        # One groupby/rank operation per signal.
        # Previously this could be repeated for every experiment.
        # ---------------------------------------------------------

        snapshot_dates = frame[
            "snapshot_date"
        ]

        for signal_name, signal in signals.items():
            rank = _cross_sectional_rank(
                snapshot_dates,
                signal,
            )

            directions = _directions_for_signal(
                signal_name
            )

            for direction in directions:
                for fraction in TAIL_FRACTIONS:
                    tail_masks[
                        (
                            signal_name,
                            direction,
                            fraction,
                        )
                    ] = _tail_from_rank(
                        rank,
                        fraction,
                        direction,
                    )

        # ---------------------------------------------------------
        # Walk-forward test masks
        #
        # These are independent of the experiment and therefore
        # should never be recomputed inside the experiment loop.
        # ---------------------------------------------------------

        window_masks: Dict[
            str,
            np.ndarray,
        ] = {}

        dates = snapshot_dates.to_numpy()

        for window in WALK_FORWARD_WINDOWS:
            key = _window_key(
                window["train_end"],
                window["validation_end"],
                window["test_end"],
            )

            mask = (
                (dates > np.datetime64(
                    window["validation_end"]
                ))
                & (
                    dates <= np.datetime64(
                        window["test_end"]
                    )
                )
            )

            window_masks[key] = mask

        return cls(
            signals=signals,
            targets=targets,
            tail_masks=tail_masks,
            window_masks=window_masks,
            target_configs=target_configs,
        )

    def get_signal(
        self,
        signal_name: str,
    ) -> np.ndarray:
        return self.signals[signal_name]

    def get_target(
        self,
        target_name: str,
    ) -> np.ndarray:
        return self.targets[target_name]

    def get_tail_mask(
        self,
        signal_name: str,
        direction: str,
        fraction: float,
    ) -> np.ndarray:
        return self.tail_masks[
            (
                signal_name,
                direction,
                fraction,
            )
        ]

    def get_window_mask(
        self,
        train_end: str,
        validation_end: str,
        test_end: str,
    ) -> np.ndarray:
        return self.window_masks[
            _window_key(
                train_end,
                validation_end,
                test_end,
            )
        ]

    def get_target_config(
        self,
        target_name: str,
    ) -> TargetConfig:
        return self.target_configs[target_name]


def _cross_sectional_rank(
    dates: pd.Series,
    signal: np.ndarray,
) -> np.ndarray:
    """
    Calculate percentile rank within each snapshot date.

    The operation is intentionally performed once per signal.
    """

    valid = np.isfinite(signal)

    if not valid.any():
        return np.full(
            len(signal),
            np.nan,
            dtype=np.float64,
        )

    working = pd.DataFrame(
        {
            "snapshot_date": dates.to_numpy(
                copy=False
            )[valid],
            "signal": signal[valid],
        }
    )

    ranked = (
        working
        .groupby(
            "snapshot_date",
            sort=False,
            observed=True,
        )["signal"]
        .rank(
            pct=True,
            method="average",
        )
        .to_numpy(
            dtype=np.float64,
        )
    )

    result = np.full(
        len(signal),
        np.nan,
        dtype=np.float64,
    )

    result[valid] = ranked

    return result


def _tail_from_rank(
    rank: np.ndarray,
    fraction: float,
    direction: str,
) -> np.ndarray:
    if direction == "upper":
        return np.isfinite(rank) & (
            rank >= 1.0 - fraction
        )

    if direction == "lower":
        return np.isfinite(rank) & (
            rank <= fraction
        )

    raise ValueError(
        f"Unsupported tail direction: {direction}"
    )


def _directions_for_signal(
    signal_name: str,
) -> tuple[str, ...]:
    """
    Return all directions actually defined by the research matrix.

    Kept here as a small optimization and consistency check rather
    than deriving directions independently in multiple places.
    """

    if signal_name in {
        "price_momentum_5d",
        "price_momentum_20d",
        "price_momentum_60d",
        "distance_from_20d_high",
        "distance_from_60d_high",
    }:
        return ("upper", "lower")

    return ("upper",)


def _window_key(
    train_end: str,
    validation_end: str,
    test_end: str,
) -> str:
    return (
        f"{train_end}|"
        f"{validation_end}|"
        f"{test_end}"
    )
