from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from ml.config import (
    TARGETS,
    TargetConfig,
    WALK_FORWARD_WINDOWS,
)
from ml.dataset import build_target
from ml.research.signals import (
    build_signal,
    tail_mask,
)
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
    target_configs: dict[str, TargetConfig]
    returns: dict[str, np.ndarray]
    tail_masks: dict[str, np.ndarray]
    window_masks: dict[
        str,
        dict[str, np.ndarray],
    ]
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
        result[f"window_{index}"] = {
            "train": (
                dates <= train_end
            ).to_numpy(),
            "validation": (
                (dates > train_end)
                & (dates <= validation_end)
            ).to_numpy(),
            "test": (
                (dates > validation_end)
                & (dates <= test_end)
            ).to_numpy(),
        }
    return result
def _target_config_map() -> dict[str, TargetConfig]:
    return {
        target.name: target
        for target in TARGETS
    }
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
def _build_tail_key(
    signal_name: str,
    direction: str,
    fraction: float,
) -> str:
    """
    Backward-compatible alias for the original helper name.
    """
    return _tail_key(
        signal_name,
        direction,
        fraction,
    )
def build_research_cache(
    frame: pd.DataFrame,
    requirements: list[ResearchRequirement],
) -> ResearchCache:
    """
    Bygger ett gemensamt cache-lager för en research-körning.
    Alla specs som körs i samma runner-process använder samma
    signaler, targets, tail-masker och walk-forward-masker.
    """
    target_configs = _target_config_map()
    signals: dict[str, np.ndarray] = {}
    targets: dict[str, np.ndarray] = {}
    returns: dict[str, np.ndarray] = {}
    tail_masks: dict[str, np.ndarray] = {}
    required_signal_names = {
        requirement.signal_name
        for requirement in requirements
    }
    required_target_names = {
        requirement.target_name
        for requirement in requirements
    }
    for signal_name in sorted(
        required_signal_names
    ):
        signal = build_signal(
            frame,
            signal_name,
        )
        signals[signal_name] = (
            signal.to_numpy(
                dtype=np.float64
            )
        )
    for target_name in sorted(
        required_target_names
    ):
        if target_name not in target_configs:
            raise ValueError(
                "Okänd research target: "
                f"{target_name}"
            )
        target_config = target_configs[
            target_name
        ]
        target = build_target(
            frame,
            target_config,
        )
        targets[target_name] = (
            target.to_numpy(
                dtype=np.float64
            )
        )
        return_column = (
            target_config.return_column
        )
        if return_column not in returns:
            if return_column not in frame.columns:
                raise ValueError(
                    "Saknar return-kolumn: "
                    f"{return_column}"
                )
            values = pd.to_numeric(
                frame[return_column],
                errors="coerce",
            )
            returns[return_column] = (
                values.to_numpy(
                    dtype=np.float64
                )
            )
    for requirement in requirements:
        signal = pd.Series(
            signals[requirement.signal_name],
            index=frame.index,
        )
        mask = tail_mask(
            frame,
            signal,
            requirement.tail_fraction,
            requirement.tail_direction,
        )
        key = _tail_key(
            requirement.signal_name,
            requirement.tail_direction,
            requirement.tail_fraction,
        )
        tail_masks[key] = (
            mask.to_numpy(
                dtype=bool
            )
        )
    window_masks = _build_window_masks(
        frame
    )
    return ResearchCache(
        signals=signals,
        targets=targets,
        target_configs=target_configs,
        returns=returns,
        tail_masks=tail_masks,
        window_masks=window_masks,
    )
