from __future__ import annotations

import hashlib
import json
from typing import Any

from research.candidates.spec import (
    CandidateSpec,
)


def candidate_definition_payload(
    candidate: CandidateSpec,
) -> dict[str, Any]:
    return {
        "schema_version": (
            candidate.schema_version
        ),
        "id": candidate.id,
        "version": candidate.version,
        "question": candidate.question,
        "created_at": candidate.created_at,
        "discovery_cutoff": (
            candidate.discovery_cutoff
        ),
        "freeze_at": candidate.freeze_at,
        "features": [
            {
                "name": feature.name
            }
            for feature in candidate.features
        ],
        "parameters": dict(
            candidate.parameters
        ),
        "target": {
            "name": candidate.target.name
        },
        "training_period": {
            "start": (
                candidate.training_period.start
            ),
            "end": (
                candidate.training_period.end
            ),
        },
        "provenance": dict(
            candidate.provenance
        ),
    }


def candidate_fingerprint(
    candidate: CandidateSpec,
) -> str:
    payload = candidate_definition_payload(
        candidate
    )

    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    return hashlib.sha256(
        encoded
    ).hexdigest()


def verify_candidate_identity(
    candidate: CandidateSpec,
    expected_id: str,
    expected_version: int,
) -> None:
    if candidate.id != expected_id:
        raise ValueError(
            "Candidate identity mismatch: "
            f"expected id '{expected_id}', "
            f"got '{candidate.id}'."
        )

    if candidate.version != expected_version:
        raise ValueError(
            "Candidate version mismatch: "
            f"expected {expected_version}, "
            f"got {candidate.version}."
        )


def verify_candidate_frozen(
    candidate: CandidateSpec,
) -> None:
    if candidate.status != "frozen":
        raise ValueError(
            "Candidate måste vara frozen "
            "innan evaluation."
        )


def verify_candidate_fingerprint(
    candidate: CandidateSpec,
) -> None:
    if not candidate.fingerprint_value:
        raise ValueError(
            "Candidate saknar förväntad "
            "fingerprint."
        )

    actual = candidate_fingerprint(
        candidate
    )

    if actual != candidate.fingerprint_value:
        raise ValueError(
            "Candidate fingerprint mismatch: "
            f"expected {candidate.fingerprint_value}, "
            f"got {actual}."
        )


def verify_candidate(
    candidate: CandidateSpec,
) -> None:
    verify_candidate_frozen(
        candidate
    )

    verify_candidate_fingerprint(
        candidate
    )


def candidate_snapshot(
    candidate: CandidateSpec,
) -> dict[str, Any]:
    return {
        "candidate_id": candidate.id,
        "candidate_version": (
            candidate.version
        ),
        "candidate_fingerprint": (
            candidate_fingerprint(candidate)
        ),
        "candidate": (
            candidate_definition_payload(
                candidate
            )
        ),
    }
