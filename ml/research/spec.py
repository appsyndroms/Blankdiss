from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


VALID_MODES = {"scan", "deep"}
VALID_ANALYSIS_TYPES = {
    "interaction",
    "tail",
}


@dataclass(frozen=True)
class SignalSpec:
    name: str
    direction: str = "upper"
    bins: tuple[float, ...] = (0.10,)


@dataclass(frozen=True)
class AnalysisSpec:
    type: str = "interaction"
    bootstrap: bool = False
    bootstrap_iterations: int = 2000


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

    splits: tuple[str, ...] = (
        "test",
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


def _tuple_floats(
    values: Any,
) -> tuple[float, ...]:
    if values is None:
        return ()

    return tuple(
        float(value)
        for value in values
    )


def load_spec(
    path: str | Path,
) -> ResearchSpec:
    path = Path(path)

    payload = yaml.safe_load(
        path.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(payload, dict):
        raise ValueError(
            f"Research spec måste vara ett objekt: {path}"
        )

    spec_id = payload.get("id")
    question = payload.get("question")

    if not spec_id:
        raise ValueError(
            f"Research spec saknar id: {path}"
        )

    if not question:
        raise ValueError(
            f"Research spec saknar question: {path}"
        )

    mode = str(
        payload.get(
            "mode",
            "scan",
        )
    ).lower()

    if mode not in VALID_MODES:
        raise ValueError(
            f"Ogiltigt mode '{mode}' i {path}"
        )

    raw_signals = payload.get(
        "signals",
        [],
    )

    if not raw_signals:
        raise ValueError(
            f"Research spec saknar signals: {path}"
        )

    signals = []

    for item in raw_signals:
        if not isinstance(item, dict):
            raise ValueError(
                f"Ogiltig signaldefinition i {path}"
            )

        name = item.get("name")

        if not name:
            raise ValueError(
                f"Signal saknar name i {path}"
            )

        direction = item.get(
            "direction",
            "upper",
        )

        bins = _tuple_floats(
            item.get(
                "bins",
                (0.10,),
            )
        )

        if not bins:
            raise ValueError(
                f"Signal '{name}' saknar bins."
            )

        for fraction in bins:
            if not 0 < fraction <= 1:
                raise ValueError(
                    f"Ogiltig bin {fraction} "
                    f"för signal '{name}'."
                )

        signals.append(
            SignalSpec(
                name=str(name),
                direction=str(direction),
                bins=bins,
            )
        )

    targets = tuple(
        str(target)
        for target in payload.get(
            "targets",
            [],
        )
    )

    if not targets:
        raise ValueError(
            f"Research spec saknar targets: {path}"
        )

    raw_analysis = payload.get(
        "analysis",
        {},
    )

    analysis = AnalysisSpec(
        type=str(
            raw_analysis.get(
                "type",
                "interaction",
            )
        ),
        bootstrap=bool(
            raw_analysis.get(
                "bootstrap",
                False,
            )
        ),
        bootstrap_iterations=int(
            raw_analysis.get(
                "bootstrap_iterations",
                2000,
            )
        ),
    )

    if analysis.type not in VALID_ANALYSIS_TYPES:
        raise ValueError(
            f"Okänd analysis.type "
            f"'{analysis.type}' i {path}"
        )

    windows = tuple(
        str(window)
        for window in payload.get(
            "windows",
            (
                "window_1",
                "window_2",
            ),
        )
    )

    splits = tuple(
        str(split)
        for split in payload.get(
            "splits",
            ("test",),
        )
    )

    return ResearchSpec(
        id=str(spec_id),
        question=str(question),
        signals=tuple(signals),
        targets=targets,
        analysis=analysis,
        mode=mode,
        windows=windows,
        splits=splits,
        metadata=dict(
            payload.get(
                "metadata",
                {},
            )
        ),
    )
