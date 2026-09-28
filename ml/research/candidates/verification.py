from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from typing import Any

from research.candidates.spec import CandidateSpec


def candidate_payload(
    candidate: CandidateSpec,
) -> dict[str, Any]:
    """
    Returnerar kandidatens kanoniska innehåll.

    Runtime-/verifieringsmetadata ska inte ingå här.
    Det som hashash är själva forskningsdefinitionen.
    """

    return {
        "id": candidate.id,
        "version": candidate.version,
        "question": candidate.question,
        "created_at": candidate.created_at,
        "discovery_cutoff": candidate.discovery_cutoff,
        "freeze_at": candidate.freeze_at,
        "status": candidate.status,
        "features": [
            {
                "name": feature.name,
            }
            for feature in candidate.features
        ],
        "parameters": candidate.parameters,
        "target": {
            "name": candidate.target.name,
        },
        "training_period": {
            "start": candidate.training_period.start,
            "end": candidate.training_period.end,
        },
        "provenance": candidate.provenance,
    }


def candidate_fingerprint(
    candidate: CandidateSpec,
) -> str:
    """
    SHA-256 fingerprint för kandidatens definition.

    Samma kandidatdefinition ska alltid ge samma fingerprint.
    """

    payload = candidate_payload(candidate)

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
            f"expected version '{expected_version}', "
            f"got '{candidate.version}'."
        )


def verify_candidate_frozen(
    candidate: CandidateSpec,
) -> None:
    if candidate.status != "frozen":
        raise ValueError(
            "Prospective evaluation kräver en frozen candidate. "
            f"Candidate '{candidate.id}' har status "
            f"'{candidate.status}'."
        )


def verify_candidate_fingerprint(
    candidate: CandidateSpec,
    expected_fingerprint: str,
) -> None:
    actual = candidate_fingerprint(candidate)

    if actual != expected_fingerprint:
        raise ValueError(
            "Candidate fingerprint mismatch. "
            "Candidate-definitionen har ändrats."
        )


def candidate_snapshot(
    candidate: CandidateSpec,
) -> dict[str, Any]:
    """
    Skapar en reproducerbar snapshot av kandidatens identity
    och definition.

    Den kan sparas tillsammans med evaluation-resultatet.
    """

    return {
        "candidate_id": candidate.id,
        "candidate_version": candidate.version,
        "candidate_fingerprint": candidate_fingerprint(
            candidate
        ),
        "candidate": candidate_payload(
            candidate
        ),
    }
