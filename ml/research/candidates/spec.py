from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


VALID_STATUSES = {
    "draft",
    "tested",
    "frozen",
    "retired",
}


@dataclass(frozen=True)
class CandidateFeature:
    name: str


@dataclass(frozen=True)
class CandidateTarget:
    name: str


@dataclass(frozen=True)
class CandidatePeriod:
    start: str
    end: str


@dataclass(frozen=True)
class CandidateSpec:
    """
    Immutable representation of a Blankdiss research candidate.

    The object itself is immutable. More importantly, a frozen candidate
    represents a research decision that must not be changed by evaluation.
    """

    id: str
    version: int
    question: str

    created_at: str
    discovery_cutoff: str
    freeze_at: str

    status: str

    features: tuple[CandidateFeature, ...]
    parameters: dict[str, Any]

    target: CandidateTarget
    training_period: CandidatePeriod

    provenance: dict[str, Any]

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("Candidate måste ha id.")

        if self.version < 1:
            raise ValueError(
                "Candidate version måste vara >= 1."
            )

        if not self.question:
            raise ValueError(
                "Candidate måste ha question."
            )

        if self.status not in VALID_STATUSES:
            raise ValueError(
                f"Ogiltig candidate status: {self.status}"
            )

        if not self.features:
            raise ValueError(
                "Candidate måste ha minst en feature."
            )

        if not self.parameters:
            raise ValueError(
                "Candidate måste ha parameters."
            )

        _validate_datetime(
            self.created_at,
            "created_at",
        )

        _validate_datetime(
            self.discovery_cutoff,
            "discovery_cutoff",
        )

        _validate_datetime(
            self.freeze_at,
            "freeze_at",
        )

        if (
            _parse_datetime(self.freeze_at)
            < _parse_datetime(self.discovery_cutoff)
        ):
            raise ValueError(
                "freeze_at får inte ligga före discovery_cutoff."
            )

        if (
            _parse_datetime(self.discovery_cutoff)
            < _parse_datetime(
                self.created_at
            )
        ):
            raise ValueError(
                "discovery_cutoff får inte ligga före created_at."
            )


def _parse_datetime(value: str) -> datetime:
    normalized = value.replace(
        "Z",
        "+00:00",
    )

    parsed = datetime.fromisoformat(
        normalized
    )

    if parsed.tzinfo is None:
        raise ValueError(
            f"Datetime måste innehålla timezone: {value}"
        )

    return parsed


def _validate_datetime(
    value: str,
    field_name: str,
) -> None:
    try:
        _parse_datetime(value)
    except ValueError as exc:
        raise ValueError(
            f"Ogiltigt {field_name}: {value}"
        ) from exc


def _require_mapping(
    value: Any,
    field_name: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(
            f"{field_name} måste vara ett objekt."
        )

    return value


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


def load_candidate(
    path: str | Path,
) -> CandidateSpec:
    """
    Loads and validates a candidate YAML specification.

    Evaluation should consume this object rather than reconstructing
    candidate parameters from an evaluation specification.
    """

    path = Path(path)

    payload = yaml.safe_load(
        path.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(payload, dict):
        raise ValueError(
            f"Candidate spec måste vara ett objekt: {path}"
        )

    candidate_id = _require_string(
        payload.get("id"),
        "id",
    )

    version = int(
        payload.get(
            "version",
            0,
        )
    )

    question = _require_string(
        payload.get("question"),
        "question",
    )

    created_at = _require_string(
        payload.get("created_at"),
        "created_at",
    )

    discovery_cutoff = _require_string(
        payload.get("discovery_cutoff"),
        "discovery_cutoff",
    )

    freeze_at = _require_string(
        payload.get("freeze_at"),
        "freeze_at",
    )

    status = str(
        payload.get(
            "candidate_status",
            "",
        )
    ).lower()

    raw_features = payload.get(
        "features"
    )

    if not isinstance(
        raw_features,
        list,
    ) or not raw_features:
        raise ValueError(
            "Candidate måste ha minst en feature."
        )

    features: list[CandidateFeature] = []

    for item in raw_features:
        if isinstance(item, str):
            name = item
        elif isinstance(item, dict):
            name = item.get("name")
        else:
            raise ValueError(
                "Ogiltig feature-definition."
            )

        features.append(
            CandidateFeature(
                name=_require_string(
                    name,
                    "feature.name",
                )
            )
        )

    parameters = _require_mapping(
        payload.get("parameters"),
        "parameters",
    )

    raw_target = _require_mapping(
        payload.get("target"),
        "target",
    )

    target = CandidateTarget(
        name=_require_string(
            raw_target.get("name"),
            "target.name",
        )
    )

    raw_period = _require_mapping(
        payload.get("training_period"),
        "training_period",
    )

    training_period = CandidatePeriod(
        start=_require_string(
            raw_period.get("start"),
            "training_period.start",
        ),
        end=_require_string(
            raw_period.get("end"),
            "training_period.end",
        ),
    )

    provenance = payload.get(
        "provenance",
        {},
    )

    if not isinstance(
        provenance,
        dict,
    ):
        raise ValueError(
            "provenance måste vara ett objekt."
        )

    return CandidateSpec(
        id=candidate_id,
        version=version,
        question=question,
        created_at=created_at,
        discovery_cutoff=discovery_cutoff,
        freeze_at=freeze_at,
        status=status,
        features=tuple(features),
        parameters=dict(parameters),
        target=target,
        training_period=training_period,
        provenance=dict(provenance),
    )


def require_frozen(
    candidate: CandidateSpec,
) -> CandidateSpec:
    """
    Ensures that an evaluation can only consume a frozen candidate.
    """

    if candidate.status != "frozen":
        raise ValueError(
            "Endast frozen candidates får användas för prospective evaluation. "
            f"Candidate '{candidate.id}' har status '{candidate.status}'."
        )

    return candidate


def candidate_identity(
    candidate: CandidateSpec,
) -> tuple[str, int]:
    return (
        candidate.id,
        candidate.version,
    )
