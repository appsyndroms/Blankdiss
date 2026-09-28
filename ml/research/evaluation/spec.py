from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class EvaluationPeriod:
    start: str
    end: str


@dataclass(frozen=True)
class WalkForwardWindow:
    name: str
    start: str
    end: str


@dataclass(frozen=True)
class WalkForwardSpec:
    enabled: bool
    windows: tuple[WalkForwardWindow, ...]


@dataclass(frozen=True)
class EvaluationAnalysis:
    type: str
    bootstrap: bool
    bootstrap_iterations: int


@dataclass(frozen=True)
class EvaluationSpec:
    schema_version: int
    id: str
    version: int
    candidate_id: str
    candidate_version: int
    candidate_fingerprint: str | None
    evaluation_period: EvaluationPeriod
    targets: tuple[str, ...]
    metrics: tuple[str, ...]
    analysis: EvaluationAnalysis
    walk_forward: WalkForwardSpec
    metadata: dict

    def __post_init__(self) -> None:
        if self.schema_version < 1:
            raise ValueError("schema_version måste vara >= 1.")

        if not self.id:
            raise ValueError("Evaluation saknar id.")

        if self.version < 1:
            raise ValueError("Evaluation version måste vara >= 1.")

        if not self.candidate_id:
            raise ValueError("Evaluation saknar candidate id.")

        if self.candidate_version < 1:
            raise ValueError(
                "candidate_version måste vara >= 1."
            )

        if not self.targets:
            raise ValueError(
                "Evaluation måste ha minst ett target."
            )

        if not self.metrics:
            raise ValueError(
                "Evaluation måste ha minst ett metric."
            )

        if self.analysis.type != "interaction":
            raise ValueError(
                "Den nya evaluation-adaptern stöder "
                "för närvarande endast analysis.type=interaction."
            )

        if self.analysis.bootstrap_iterations < 1:
            raise ValueError(
                "bootstrap_iterations måste vara > 0."
            )


def load_evaluation(
    path: str | Path,
) -> EvaluationSpec:
    path = Path(path)

    payload = yaml.safe_load(
        path.read_text(encoding="utf-8")
    )

    if not isinstance(payload, dict):
        raise ValueError(
            f"Evaluation måste vara ett objekt: {path}"
        )

    forbidden = {
        "parameters",
        "candidate_parameters",
        "search_space",
        "optimization",
    }

    present_forbidden = forbidden.intersection(
        payload
    )

    if present_forbidden:
        names = ", ".join(
            sorted(present_forbidden)
        )
        raise ValueError(
            "Evaluation får inte innehålla "
            f"candidate-optimeringsfält: {names}"
        )

    candidate = payload.get("candidate")

    if not isinstance(candidate, dict):
        raise ValueError(
            f"Evaluation saknar candidate-objekt: {path}"
        )

    candidate_id = candidate.get("id")
    candidate_version = candidate.get("version")

    if not candidate_id:
        raise ValueError(
            f"Evaluation candidate saknar id: {path}"
        )

    if candidate_version is None:
        raise ValueError(
            f"Evaluation candidate saknar version: {path}"
        )

    fingerprint = candidate.get("fingerprint")

    period = payload.get("evaluation_period")

    if not isinstance(period, dict):
        raise ValueError(
            f"Evaluation saknar evaluation_period: {path}"
        )

    start = period.get("start")
    end = period.get("end")

    if not start or not end:
        raise ValueError(
            "evaluation_period måste ha start och end."
        )

    targets = tuple(
        str(value)
        for value in payload.get("targets", [])
    )

    metrics = tuple(
        str(value)
        for value in payload.get("metrics", [])
    )

    raw_analysis = payload.get("analysis", {})

    if not isinstance(raw_analysis, dict):
        raise ValueError(
            "analysis måste vara ett objekt."
        )

    analysis = EvaluationAnalysis(
        type=str(
            raw_analysis.get(
                "type",
                "interaction",
            )
        ).lower(),
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

    raw_walk_forward = payload.get(
        "walk_forward",
        {},
    )

    if not isinstance(raw_walk_forward, dict):
        raise ValueError(
            "walk_forward måste vara ett objekt."
        )

    windows = []

    for index, raw_window in enumerate(
        raw_walk_forward.get("windows", []),
        start=1,
    ):
        if not isinstance(raw_window, dict):
            raise ValueError(
                "Varje walk-forward-window måste vara ett objekt."
            )

        windows.append(
            WalkForwardWindow(
                name=str(
                    raw_window.get(
                        "name",
                        f"window_{index}",
                    )
                ),
                start=str(raw_window["start"]),
                end=str(raw_window["end"]),
            )
        )

    walk_forward = WalkForwardSpec(
        enabled=bool(
            raw_walk_forward.get(
                "enabled",
                False,
            )
        ),
        windows=tuple(windows),
    )

    if walk_forward.enabled and not walk_forward.windows:
        raise ValueError(
            "walk_forward.enabled=true kräver windows."
        )

    metadata = payload.get("metadata", {})

    if not isinstance(metadata, dict):
        raise ValueError(
            "metadata måste vara ett objekt."
        )

    return EvaluationSpec(
        schema_version=int(
            payload.get("schema_version", 1)
        ),
        id=str(payload["id"]),
        version=int(payload["version"]),
        candidate_id=str(candidate_id),
        candidate_version=int(candidate_version),
        candidate_fingerprint=(
            str(fingerprint)
            if fingerprint
            else None
        ),
        evaluation_period=EvaluationPeriod(
            start=str(start),
            end=str(end),
        ),
        targets=targets,
        metrics=metrics,
        analysis=analysis,
        walk_forward=walk_forward,
        metadata=dict(metadata),
    )
