from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SignalSpec:
    name: str
    direction: str = "upper"


@dataclass(frozen=True)
class AnalysisSpec:
    type: str = "interaction"
    bins: tuple[float, ...] = (0.80,)
    bootstrap: bool = False
    relative_to: tuple[str, ...] = ()


@dataclass(frozen=True)
class ResearchSpec:
    id: str
    question: str

    signals: tuple[SignalSpec, ...]

    targets: tuple[str, ...]

    analysis: AnalysisSpec = field(
        default_factory=AnalysisSpec
    )

    mode: str = "scan"

    windows: tuple[str, ...] = (
        "window_1",
        "window_2",
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


def load_spec(path: str) -> ResearchSpec:
    """
    Läs en YAML research specification.

    PyYAML finns redan i requirements.txt.
    """
    import yaml

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as handle:
        payload = yaml.safe_load(handle)

    if not isinstance(payload, dict):
        raise ValueError(
            "Research spec måste vara ett YAML-objekt."
        )

    signals = tuple(
        SignalSpec(
            name=item["name"],
            direction=item.get(
                "direction",
                "upper",
            ),
        )
        for item in payload.get(
            "signals",
            [],
        )
    )

    if not signals:
        raise ValueError(
            "Research spec måste innehålla minst "
            "en signal."
        )

    targets = tuple(
        payload.get(
            "targets",
            [],
        )
    )

    if not targets:
        raise ValueError(
            "Research spec måste innehålla minst "
            "ett target."
        )

    analysis_payload = payload.get(
        "analysis",
        {},
    )

    analysis = AnalysisSpec(
        type=analysis_payload.get(
            "type",
            "interaction",
        ),
        bins=tuple(
            analysis_payload.get(
                "bins",
                [0.80],
            )
        ),
        bootstrap=bool(
            analysis_payload.get(
                "bootstrap",
                False,
            )
        ),
        relative_to=tuple(
            analysis_payload.get(
                "relative_to",
                [],
            )
        ),
    )

    return ResearchSpec(
        id=payload["id"],
        question=payload.get(
            "question",
            "",
        ),
        signals=signals,
        targets=targets,
        analysis=analysis,
        mode=payload.get(
            "mode",
            "scan",
        ),
        windows=tuple(
            payload.get(
                "windows",
                [
                    "window_1",
                    "window_2",
                ],
            )
        ),
        metadata=payload.get(
            "metadata",
            {},
        ),
    )
