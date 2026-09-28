from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class EvaluationPeriod:
    start: str
    end: str


@dataclass(frozen=True)
class WalkForwardSpec:
    enabled: bool
    windows: tuple[EvaluationPeriod, ...]


@dataclass(frozen=True)
class EvaluationSpec:
    """
    Specification of how a frozen candidate is evaluated.

    This object deliberately contains no candidate parameters.
    Candidate parameters belong exclusively to CandidateSpec.
    """

    id: str
    version: int

    candidate_id: str
    candidate_version: int

    evaluation_period: EvaluationPeriod

    targets: tuple[str, ...]

    metrics: tuple[str, ...]

    walk_forward: WalkForwardSpec

    metadata: dict[str, Any]


def _require_string(
    value: Any,
    field_name: str,
) -> str:
    if value is None:
        raise ValueError(
            f"{field_name} saknas."
        )

    value = str(value)

    if not value:
        raise ValueError(
            f"{field_name} får inte vara tom."
        )

    return value


def _require_mapping(
    value: Any,
    field_name: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(
            f"{field_name} måste vara ett objekt."
        )

    return value


def _period_from_payload(
    value: Any,
    field_name: str,
) -> EvaluationPeriod:
    mapping = _require_mapping(
        value,
        field_name,
    )

    return EvaluationPeriod(
        start=_require_string(
            mapping.get("start"),
            f"{field_name}.start",
        ),
        end=_require_string(
            mapping.get("end"),
            f"{field_name}.end",
        ),
    )


def load_evaluation(
    path: str | Path,
) -> EvaluationSpec:
    path = Path(path)

    payload = yaml.safe_load(
        path.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(payload, dict):
        raise ValueError(
            f"Evaluation spec måste vara ett objekt: {path}"
        )

    evaluation_id = _require_string(
        payload.get("id"),
        "id",
    )

    version = int(
        payload.get(
            "version",
            0,
        )
    )

    if version < 1:
        raise ValueError(
            "Evaluation version måste vara >= 1."
        )

    candidate = _require_mapping(
        payload.get("candidate"),
        "candidate",
    )

    candidate_id = _require_string(
        candidate.get("id"),
        "candidate.id",
    )

    candidate_version = int(
        candidate.get(
            "version",
            0,
        )
    )

    if candidate_version < 1:
        raise ValueError(
            "candidate.version måste vara >= 1."
        )

    evaluation_period = _period_from_payload(
        payload.get("evaluation_period"),
        "evaluation_period",
    )

    targets = tuple(
        str(value)
        for value in payload.get(
            "targets",
            [],
        )
    )

    if not targets:
        raise ValueError(
            "Evaluation måste ha minst ett target."
        )

    metrics = tuple(
        str(value)
        for value in payload.get(
            "metrics",
            [],
        )
    )

    if not metrics:
        raise ValueError(
            "Evaluation måste ha minst ett metric."
        )

    raw_walk_forward = payload.get(
        "walk_forward",
        {},
    )

    walk_forward_mapping = _require_mapping(
        raw_walk_forward,
        "walk_forward",
    )

    walk_forward_enabled = bool(
        walk_forward_mapping.get(
            "enabled",
            False,
        )
    )

    raw_windows = walk_forward_mapping.get(
        "windows",
        [],
    )

    windows: list[EvaluationPeriod] = []

    if raw_windows:
        if not isinstance(
            raw_windows,
            list,
        ):
            raise ValueError(
                "walk_forward.windows måste vara en lista."
            )

        for index, raw_window in enumerate(
            raw_windows,
            start=1,
        ):
            windows.append(
                _period_from_payload(
                    raw_window,
                    f"walk_forward.windows[{index}]",
                )
            )

    if walk_forward_enabled and not windows:
        raise ValueError(
            "walk_forward.enabled=true kräver windows."
        )

    metadata = payload.get(
        "metadata",
        {},
    )

    if not isinstance(
        metadata,
        dict,
    ):
        raise ValueError(
            "metadata måste vara ett objekt."
        )

    return EvaluationSpec(
        id=evaluation_id,
        version=version,
        candidate_id=candidate_id,
        candidate_version=candidate_version,
        evaluation_period=evaluation_period,
        targets=targets,
        metrics=metrics,
        walk_forward=WalkForwardSpec(
            enabled=walk_forward_enabled,
            windows=tuple(windows),
        ),
        metadata=dict(metadata),
    )


def validate_candidate_reference(
    evaluation: EvaluationSpec,
    candidate_id: str,
    candidate_version: int,
) -> None:
    """
    Prevents evaluation from silently referring to another candidate.
    """

    if evaluation.candidate_id != candidate_id:
        raise ValueError(
            "Evaluation refererar till fel candidate_id: "
            f"{evaluation.candidate_id}; "
            f"förväntade {candidate_id}."
        )

    if evaluation.candidate_version != candidate_version:
        raise ValueError(
            "Evaluation refererar till fel candidate_version: "
            f"{evaluation.candidate_version}; "
            f"förväntade {candidate_version}."
        )
