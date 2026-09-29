from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import yaml


VALID_STATUSES = {
    "draft",
    "frozen",
    "retired",
}

VALID_ANALYSIS_TYPES = {
    "interaction",
    "tail",
    "regime_comparison",
    "multi_regime_comparison",
    "nested_regime_comparison",
    "conditional_regime_comparison",
    "stratified_regime_comparison",
    "stratified_interaction",
}


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType(
            {
                key: _freeze(item)
                for key, item in value.items()
            }
        )

    if isinstance(value, list):
        return tuple(
            _freeze(item)
            for item in value
        )

    if isinstance(value, tuple):
        return tuple(
            _freeze(item)
            for item in value
        )

    return value


def _parse_datetime(
    value: str,
    field_name: str,
) -> datetime:
    normalized = value.replace(
        "Z",
        "+00:00",
    )

    try:
        parsed = datetime.fromisoformat(
            normalized
        )
    except ValueError as exc:
        raise ValueError(
            f"Ogiltigt datetime-värde i "
            f"{field_name}: {value}"
        ) from exc

    if parsed.tzinfo is None:
        raise ValueError(
            f"{field_name} måste innehålla "
            f"timezone: {value}"
        )

    return parsed


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
class CandidateAnalysis:
    type: str = "tail"
    bootstrap: bool = False
    bootstrap_iterations: int = 2000

    def __post_init__(self) -> None:
        if self.type not in VALID_ANALYSIS_TYPES:
            raise ValueError(
                f"Ogiltig candidate analysis.type: "
                f"{self.type}"
            )

        if self.bootstrap_iterations < 1:
            raise ValueError(
                "bootstrap_iterations måste vara > 0."
            )


@dataclass(frozen=True)
class CandidateSpec:
    schema_version: int
    id: str
    version: int
    question: str

    created_at: str
    discovery_cutoff: str
    freeze_at: str

    status: str

    features: tuple[
        CandidateFeature,
        ...
    ]

    parameters: Mapping[str, Any]

    target: CandidateTarget

    analysis: CandidateAnalysis

    training_period: CandidatePeriod

    provenance: Mapping[str, Any]

    fingerprint_algorithm: str | None
    fingerprint_value: str | None

    def __post_init__(self) -> None:
        if self.schema_version < 1:
            raise ValueError(
                "schema_version måste vara >= 1."
            )

        if not self.id:
            raise ValueError(
                "Candidate saknar id."
            )

        if self.version < 1:
            raise ValueError(
                "Candidate version måste vara >= 1."
            )

        if not self.question:
            raise ValueError(
                "Candidate saknar question."
            )

        if self.status not in VALID_STATUSES:
            raise ValueError(
                f"Ogiltig candidate_status: "
                f"{self.status}"
            )

        if not self.features:
            raise ValueError(
                "Candidate måste ha minst "
                "en feature."
            )

        _parse_datetime(
            self.created_at,
            "created_at",
        )

        discovery_cutoff = _parse_datetime(
            self.discovery_cutoff,
            "discovery_cutoff",
        )

        freeze_at = _parse_datetime(
            self.freeze_at,
            "freeze_at",
        )

        created_at = _parse_datetime(
            self.created_at,
            "created_at",
        )

        if created_at > discovery_cutoff:
            raise ValueError(
                "created_at kan inte ligga efter "
                "discovery_cutoff."
            )

        if discovery_cutoff > freeze_at:
            raise ValueError(
                "discovery_cutoff kan inte ligga "
                "efter freeze_at."
            )

        training_start = _parse_datetime(
            self.training_period.start,
            "training_period.start",
        )

        training_end = _parse_datetime(
            self.training_period.end,
            "training_period.end",
        )

        if training_end < training_start:
            raise ValueError(
                "training_period.end kan inte "
                "ligga före training_period.start."
            )

        if training_end > discovery_cutoff:
            raise ValueError(
                "training_period.end kan inte ligga "
                "efter discovery_cutoff."
            )

        object.__setattr__(
            self,
            "parameters",
            _freeze(
                dict(self.parameters)
            ),
        )

        object.__setattr__(
            self,
            "provenance",
            _freeze(
                dict(self.provenance)
            ),
        )

        if self.status == "frozen":
            if not self.fingerprint_algorithm:
                raise ValueError(
                    "Frozen candidate måste ha "
                    "fingerprint algorithm."
                )

            if (
                self.fingerprint_algorithm.lower()
                != "sha256"
            ):
                raise ValueError(
                    "Frozen candidate måste använda "
                    "fingerprint algorithm=sha256."
                )

            if not self.fingerprint_value:
                raise ValueError(
                    "Frozen candidate måste ha "
                    "fingerprint value."
                )


def load_candidate(
    path: str | Path,
) -> CandidateSpec:
    path = Path(path)

    payload = yaml.safe_load(
        path.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(payload, dict):
        raise ValueError(
            f"Candidate måste vara ett objekt: "
            f"{path}"
        )

    raw_features = payload.get(
        "features",
        [],
    )

    if not raw_features:
        raise ValueError(
            f"Candidate saknar features: {path}"
        )

    features: list[CandidateFeature] = []

    for item in raw_features:
        if isinstance(item, str):
            name = item
        elif isinstance(item, dict):
            name = item.get("name")
        else:
            raise ValueError(
                f"Ogiltig featuredefinition "
                f"i {path}"
            )

        if not name:
            raise ValueError(
                f"Feature saknar name i {path}"
            )

        features.append(
            CandidateFeature(
                name=str(name)
            )
        )

    target_payload = payload.get(
        "target"
    )

    if not isinstance(
        target_payload,
        dict,
    ):
        raise ValueError(
            f"Candidate saknar target: {path}"
        )

    target_name = target_payload.get(
        "name"
    )

    if not target_name:
        raise ValueError(
            f"Target saknar name: {path}"
        )

    raw_analysis = payload.get(
        "analysis"
    )

    if not isinstance(
        raw_analysis,
        dict,
    ):
        raise ValueError(
            f"Candidate saknar analysis: {path}"
        )

    analysis_type = str(
        raw_analysis.get(
            "type",
            "tail",
        )
    ).lower()

    if analysis_type not in VALID_ANALYSIS_TYPES:
        raise ValueError(
            f"Okänd candidate analysis.type "
            f"'{analysis_type}' i {path}"
        )

    bootstrap = bool(
        raw_analysis.get(
            "bootstrap",
            False,
        )
    )

    bootstrap_iterations = int(
        raw_analysis.get(
            "bootstrap_iterations",
            2000,
        )
    )

    analysis = CandidateAnalysis(
        type=analysis_type,
        bootstrap=bootstrap,
        bootstrap_iterations=(
            bootstrap_iterations
        ),
    )

    training_payload = payload.get(
        "training_period"
    )

    if not isinstance(
        training_payload,
        dict,
    ):
        raise ValueError(
            f"Candidate saknar "
            f"training_period: {path}"
        )

    training_start = training_payload.get(
        "start"
    )

    training_end = training_payload.get(
        "end"
    )

    if not training_start or not training_end:
        raise ValueError(
            f"training_period måste ha "
            f"start och end: {path}"
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
            f"provenance måste vara ett "
            f"objekt: {path}"
        )

    fingerprint = payload.get(
        "fingerprint",
        {},
    )

    if fingerprint is None:
        fingerprint = {}

    if not isinstance(
        fingerprint,
        dict,
    ):
        raise ValueError(
            f"fingerprint måste vara ett "
            f"objekt: {path}"
        )

    return CandidateSpec(
        schema_version=int(
            payload.get(
                "schema_version",
                1,
            )
        ),
        id=str(
            payload["id"]
        ),
        version=int(
            payload["version"]
        ),
        question=str(
            payload["question"]
        ),
        created_at=str(
            payload["created_at"]
        ),
        discovery_cutoff=str(
            payload["discovery_cutoff"]
        ),
        freeze_at=str(
            payload["freeze_at"]
        ),
        status=str(
            payload.get(
                "candidate_status",
                "draft",
            )
        ).lower(),
        features=tuple(features),
        parameters=_freeze(
            payload.get(
                "parameters",
                {},
            )
        ),
        target=CandidateTarget(
            name=str(target_name)
        ),
        analysis=analysis,
        training_period=CandidatePeriod(
            start=str(training_start),
            end=str(training_end),
        ),
        provenance=_freeze(
            provenance
        ),
        fingerprint_algorithm=(
            str(
                fingerprint["algorithm"]
            )
            if fingerprint.get("algorithm")
            else None
        ),
        fingerprint_value=(
            str(
                fingerprint["value"]
            )
            if fingerprint.get("value")
            else None
        ),
    )


def require_frozen(
    candidate: CandidateSpec,
) -> None:
    if candidate.status != "frozen":
        raise ValueError(
            f"Candidate '{candidate.id}' "
            f"v{candidate.version} är inte frozen."
        )


def candidate_identity(
    candidate: CandidateSpec,
) -> tuple[str, int]:
    return (
        candidate.id,
        candidate.version,
    )
