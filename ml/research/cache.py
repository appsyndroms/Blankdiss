from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

from ml.config import TARGETS
from ml.dataset import build_target
from .experiments import TAIL_FRACTIONS
from .signals import build_signal, signal_direction


@dataclass
class ResearchCache:
    """
    Precomputed research data.

    The purpose is to ensure that expensive operations such as:
      - signal construction
      - target construction
      - cross-sectional ranking

    are performed once rather than once per experiment.
    """

    signals: Dict[str, pd.Series]
    targets: Dict[str, pd.Series]
    tail_masks: Dict[tuple[str, str, float], pd.Series]

    @classmethod
    def build(
        cls,
        frame: pd.DataFrame,
        signal_names: list[str],
        target_names: list[str],
    ) -> "ResearchCache":
        signals = {}
        targets = {}
        tail_masks = {}

        # ------------------------------------------------------------
        # Signals
        # ------------------------------------------------------------

        for signal_name in signal_names:
            signal = build_signal(frame, signal_name)

            # Keep index aligned with the source frame.
            signals[signal_name] = signal

        # ------------------------------------------------------------
        # Targets
        # ------------------------------------------------------------

        target_by_name = {
            target.name: target
            for target in TARGETS
        }

        for target_name in target_names:
            target_config = target_by_name.get(target_name)

            if target_config is None:
                raise ValueError(
                    f"Unknown target requested by research matrix: "
                    f"{target_name}"
                )

            targets[target_name] = build_target(
                frame,
                target_config,
            )

        # ------------------------------------------------------------
        # Cross-sectional tail masks
        #
        # This is deliberately precomputed.
        #
        # 10 signals × up to 2 directions × 5 fractions is cheap
        # compared with doing the same groupby/rank operation for
        # every experiment.
        # ------------------------------------------------------------

        for signal_name, signal in signals.items():
            directions = {signal_direction(signal_name)}

            # Momentum signals can explicitly be tested in both
            # directions.
            if signal_name.startswith("price_momentum_"):
                directions.update({"upper", "lower"})

            if signal_name.startswith("distance_from_"):
                directions.update({"upper", "lower"})

            for direction in directions:
                ranks = _cross_sectional_rank(
                    frame["snapshot_date"],
                    signal,
                )

                for fraction in TAIL_FRACTIONS:
                    mask = _tail_from_rank(
                        ranks,
                        fraction,
                        direction,
                    )

                    tail_masks[
                        (signal_name, direction, fraction)
                    ] = mask

        return cls(
            signals=signals,
            targets=targets,
            tail_masks=tail_masks,
        )

    def get_signal(self, signal_name: str) -> pd.Series:
        return self.signals[signal_name]

    def get_target(self, target_name: str) -> pd.Series:
        return self.targets[target_name]

    def get_tail_mask(
        self,
        signal_name: str,
        direction: str,
        fraction: float,
    ) -> pd.Series:
        return self.tail_masks[
            (signal_name, direction, fraction)
        ]


def _cross_sectional_rank(
    dates: pd.Series,
    signal: pd.Series,
) -> pd.Series:
    """
    Percentile rank within each snapshot date.

    The expensive groupby/rank operation is performed once per
    signal instead of once per experiment.
    """

    work = pd.DataFrame(
        {
            "snapshot_date": dates,
            "signal": signal,
        },
        index=signal.index,
    )

    return work.groupby(
        "snapshot_date",
        sort=False,
        observed=True,
    )["signal"].rank(
        pct=True,
        method="average",
    )


def _tail_from_rank(
    ranks: pd.Series,
    fraction: float,
    direction: str,
) -> pd.Series:
    if direction == "upper":
        return ranks >= (1.0 - fraction)

    if direction == "lower":
        return ranks <= fraction

    raise ValueError(
        f"Unsupported tail direction: {direction}"
    )
