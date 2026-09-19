from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from typing import Any

import yaml


@dataclass(frozen=True)
class ValidationConfig:
    min_rows_per_window: int = 20
    min_positive_windows: int = 2
    min_lift: float = 1.10
    max_return_difference: float = 0.0
    max_findings: int = 25


@dataclass(frozen=True)
class DiscoveryConfig:
    enabled: bool
    discovery_end_date: date
    targets: tuple[str, ...]
    signals: tuple[str, ...]
    stress_features: tuple[str, ...]
    tails: tuple[float, ...]
    stress_directions: dict[str, str]
    validation: ValidationConfig


def _require_list(
    config: dict[str, Any],
    key: str,
) -> list[Any]:
    value = config.get(key)
    if not isinstance(value, list):
        raise ValueError(
            f"Discovery config '{key}' måste vara en lista."
        )
    if not value:
        raise ValueError(
            f"Discovery config '{key}' får inte vara tom."
        )
    return value


def _parse_date(
    value: Any,
    name: str,
) -> date:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"Discovery config '{name}' måste vara ett datum."
        )

    try:
        return date.fromisoformat(value.strip())
    except ValueError as exc:
        raise ValueError(
            f"Ogiltigt datum för '{name}': {value}"
        ) from exc


def _parse_validation(
    config: dict[str, Any],
) -> ValidationConfig:
    raw = config.get("validation", {})

    if not isinstance(raw, dict):
        raise ValueError(
            "Discovery config 'validation' måste vara ett objekt."
        )

    result = ValidationConfig(
        min_rows_per_window=int(
            raw.get("min_rows_per_window", 20)
        ),
        min_positive_windows=int(
            raw.get("min_positive_windows", 2)
        ),
        min_lift=float(
            raw.get("min_lift", 1.10)
        ),
        max_return_difference=float(
            raw.get("max_return_difference", 0.0)
        ),
        max_findings=int(
            raw.get("max_findings", 25)
        ),
    )

    if result.min_rows_per_window < 1:
        raise ValueError(
            "min_rows_per_window måste vara >= 1."
        )

    if result.min_positive_windows < 1:
        raise ValueError(
            "min_positive_windows måste vara >= 1."
        )

    if result.min_lift <= 0:
        raise ValueError(
            "min_lift måste vara > 0."
        )

    if result.max_findings < 1:
        raise ValueError(
            "max_findings måste vara >= 1."
        )

    return result


def load_discovery_config() -> DiscoveryConfig:
    raw_config = os.environ.get("DISCOVERY_CONFIG")

    if not raw_config:
        raise RuntimeError(
            "DISCOVERY_CONFIG saknas. "
            "Definiera discovery-konfigurationen i workflowets env."
        )

    config = yaml.safe_load(raw_config)

    if not isinstance(config, dict):
        raise ValueError(
            "DISCOVERY_CONFIG måste innehålla ett YAML-objekt."
        )

    targets = tuple(
        str(value)
        for value in _require_list(
            config,
            "targets",
        )
    )

    signals = tuple(
        str(value)
        for value in _require_list(
            config,
            "signals",
        )
    )

    stress_features = tuple(
        str(value)
        for value in _require_list(
            config,
            "stress_features",
        )
    )

    tails = tuple(
        float(value)
        for value in _require_list(
            config,
            "tails",
        )
    )

    for tail in tails:
        if not 0 < tail <= 1:
            raise ValueError(
                f"Ogiltig tail-fraktion: {tail}"
            )

    raw_directions = config.get(
        "stress_directions",
        {},
    )

    if not isinstance(raw_directions, dict):
        raise ValueError(
            "stress_directions måste vara ett objekt."
        )

    stress_directions = {
        str(key): str(value)
        for key, value in raw_directions.items()
    }

    for feature in stress_features:
        direction = stress_directions.get(feature)

        if direction not in {
            "upper",
            "lower",
        }:
            raise ValueError(
                "Saknar giltig stress-riktning för "
                f"'{feature}'."
            )

    discovery_end_date = _parse_date(
        config.get("discovery_end_date"),
        "discovery_end_date",
    )

    validation = _parse_validation(
        config
    )

    return DiscoveryConfig(
        enabled=bool(
            config.get(
                "enabled",
                True,
            )
        ),
        discovery_end_date=discovery_end_date,
        targets=targets,
        signals=signals,
        stress_features=stress_features,
        tails=tails,
        stress_directions=stress_directions,
        validation=validation,
    )
