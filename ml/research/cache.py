from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ml.config import TARGETS, WALK_FORWARD_WINDOWS
from ml.dataset import build_target
from ml.research.signals import build_signal, tail_mask


@dataclass(frozen=True)
class ResearchRequirement:
    signal_name: str
    target_name: str
    tail_fraction: float
    tail_direction: str


@dataclass
class ResearchCache:
    signals: dict[str, np.ndarray]
    targets: dict[str, np.ndarray]
    returns: dict[str, np.ndarray]
    tail_masks: dict[str, np.ndarray]
    window_masks: dict[str, dict[str, np.ndarray]]
    target_configs: dict[str, object]


def _tail_key(
    signal_name: str,
    direction: str,
    fraction: float,
) -> str:
    return (
        f"{signal_name}|"
        f"{direction}|"
        f"{fraction}"
    )


def _build_window_masks(
    frame: pd.DataFrame,
) -> dict[str, dict[str, np.ndarray]]:
    dates = pd.to_datetime(
        frame["snapshot_date"],
        errors="coerce",
    )

    result: dict[
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

        test_mask = (
            (dates > validation_end)
            & (dates <= test_end)
        ).to_numpy()

        result[f"window_{index}"] = {
            "test": test_mask,
        }

    return result


def build_research_cache(
    frame: pd.DataFrame,
    requirements: list[ResearchRequirement],
) -> ResearchCache:
    if frame.empty:
        raise ValueError(
            "Research cache kan inte byggas från ett tomt dataset."
        )

    if "snapshot_date" not in frame.columns:
        raise ValueError(
            "Research cache kräver snapshot_date."
        )

    target_configs = {
        target.name: target
        for target in TARGETS
    }

    required_signal_names = sorted(
        {
            requirement.signal_name
            for requirement in requirements
        }
    )

    required_target_names = sorted(
        {
            requirement.target_name
            for requirement in requirements
        }
    )

    unknown_targets = (
        set(required_target_names)
        - set(target_configs)
    )

    if unknown_targets:
        raise ValueError(
            "Okända research targets: "
            + ", ".join(sorted(unknown_targets))
        )

    signals: dict[str, np.ndarray] = {}

    for signal_name in required_signal_names:
        print(
            f"  Building signal: {signal_name}",
            flush=True,
        )

        series = build_signal(
            frame,
            signal_name,
        )

        signals[signal_name] = (
            series.to_numpy(
                dtype=float,
                na_value=np.nan,
            )
        )

    targets: dict[str, np.ndarray] = {}
    returns: dict[str, np.ndarray] = {}

    for target_name in required_target_names:
        print(
            f"  Building target: {target_name}",
            flush=True,
        )

        target_config = target_configs[
            target_name
        ]

        target_series = build_target(
            frame,
            target_config,
        )

        targets[target_name] = (
            target_series.to_numpy(
                dtype=float,
                na_value=np.nan,
            )
        )

        return_column = (
            target_config.return_column
        )

        if return_column not in returns:
            returns[return_column] = (
                pd.to_numeric(
                    frame[return_column],
                    errors="coerce",
                )
                .replace(
                    [np.inf, -np.inf],
                    np.nan,
                )
                .to_numpy(
                    dtype=float,
                    na_value=np.nan,
                )
            )

    tail_masks: dict[
        str,
        np.ndarray,
    ] = {}

    unique_tail_requirements = {
        (
            requirement.signal_name,
            requirement.tail_direction,
            requirement.tail_fraction,
        )
        for requirement in requirements
    }

    for (
        signal_name,
        direction,
        fraction,
    ) in sorted(unique_tail_requirements):
        print(
            "  Building tail: "
            f"{signal_name} "
            f"{direction} "
            f"{fraction}",
            flush=True,
        )

        signal_series = pd.Series(
            signals[signal_name],
            index=frame.index,
        )

        mask = tail_mask(
            frame,
            signal_series,
            fraction,
            direction=direction,
        )

        tail_masks[
            _tail_key(
                signal_name,
                direction,
                fraction,
            )
        ] = mask.to_numpy(
            dtype=bool
        )

    window_masks = _build_window_masks(
        frame
    )

    return ResearchCache(
        signals=signals,
        targets=targets,
        returns=returns,
        tail_masks=tail_masks,
        window_masks=window_masks,
        target_configs=target_configs,
    )
