from __future__ import annotations

from dataclasses import dataclass
from typing import Any


VALID_DIRECTIONS = {
    "upper",
    "lower",
}

VALID_ANALYSIS_TYPES = {
    "interaction",
    "tail",
    "regime_comparison",
    "multi_regime_comparison",
}


@dataclass(frozen=True)
class SignalSpec:
    name: str
    direction: str
    bins: tuple[float, ...]


@dataclass(frozen=True)
class AnalysisSpec:
    type: str
    bootstrap: bool = False
    bootstrap_iterations: int = 2000


@dataclass(frozen=True)
class MetadataSpec:
    stage: str | None
    purpose: str | None
    raw: dict[str, Any]


@dataclass(frozen=True)
class ResearchSpec:
    id: str
    question: str
    mode: str
    signals: tuple[SignalSpec, ...]
    targets: tuple[str, ...]
    analysis: AnalysisSpec
    windows: tuple[str, ...]
    splits: tuple[str, ...]
    metadata: MetadataSpec


def _as_tuple(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        return tuple(value)
    return (value,)


def parse_signal(data: dict[str, Any]) -> SignalSpec:
    name = data["name"]
    direction = data["direction"]
    bins = tuple(
        float(value)
        for value in _as_tuple(data.get("bins"))
    )

    if direction not in VALID_DIRECTIONS:
        raise ValueError(
            f"Invalid signal direction: {direction}"
        )

    if not bins:
        raise ValueError(
            f"Signal '{name}' must define at least one bin."
        )

    for fraction in bins:
        if not 0 < fraction <= 1:
            raise ValueError(
                f"Invalid bin fraction for '{name}': "
                f"{fraction}"
            )

    return SignalSpec(
        name=name,
        direction=direction,
        bins=bins,
    )


def parse_analysis(data: dict[str, Any]) -> AnalysisSpec:
    analysis_type = data["type"]

    if analysis_type not in VALID_ANALYSIS_TYPES:
        raise ValueError(
            f"Invalid analysis type: {analysis_type}"
        )

    bootstrap = bool(
        data.get("bootstrap", False)
    )

    bootstrap_iterations = int(
        data.get("bootstrap_iterations", 2000)
    )

    if bootstrap_iterations <= 0:
        raise ValueError(
            "bootstrap_iterations must be > 0."
        )

    return AnalysisSpec(
        type=analysis_type,
        bootstrap=bootstrap,
        bootstrap_iterations=bootstrap_iterations,
    )


def parse_metadata(
    data: dict[str, Any] | None,
) -> MetadataSpec:
    raw = dict(data or {})

    return MetadataSpec(
        stage=raw.get("stage"),
        purpose=raw.get("purpose"),
        raw=raw,
    )


def parse_spec(data: dict[str, Any]) -> ResearchSpec:
    signals = tuple(
        parse_signal(signal)
        for signal in data.get("signals", [])
    )

    if not signals:
        raise ValueError(
            "Research spec must define at least one signal."
        )

    targets = tuple(
        str(target)
        for target in _as_tuple(
            data.get("targets")
        )
    )

    if not targets:
        raise ValueError(
            "Research spec must define at least one target."
        )

    windows = tuple(
        str(window)
        for window in _as_tuple(
            data.get("windows")
        )
    )

    splits = tuple(
        str(split)
        for split in _as_tuple(
            data.get("splits")
        )
    )

    if not windows:
        raise ValueError(
            "Research spec must define at least one window."
        )

    if not splits:
        raise ValueError(
            "Research spec must define at least one split."
        )

    analysis = parse_analysis(
        data["analysis"]
    )

    if (
        analysis.type == "multi_regime_comparison"
        and len(signals) < 3
    ):
        raise ValueError(
            "multi_regime_comparison requires "
            "at least three signals."
        )

    return ResearchSpec(
        id=str(data["id"]),
        question=str(data["question"]),
        mode=str(data.get("mode", "deep")),
        signals=signals,
        targets=targets,
        analysis=analysis,
        windows=windows,
        splits=splits,
        metadata=parse_metadata(
            data.get("metadata")
        ),
    )
