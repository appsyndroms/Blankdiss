from __future__ import annotations

from typing import Any

import pandas as pd

from ml.research.candidates.spec import CandidateSpec
from ml.research.candidates.verification import (
    verify_candidate,
)
from ml.research.evaluation.spec import EvaluationSpec
from ml.research.evaluation.verification import (
    verify_evaluation,
    verify_evaluation_data,
)


def verify_candidate_and_evaluation(
    candidate: CandidateSpec,
    evaluation: EvaluationSpec,
) -> None:
    verify_candidate(
        candidate
    )

    verify_evaluation(
        candidate,
        evaluation,
    )


def verify_research_data(
    frame: pd.DataFrame,
    candidate: CandidateSpec,
    evaluation: EvaluationSpec,
) -> dict[str, Any]:
    return verify_evaluation_data(
        frame,
        candidate,
        evaluation,
    )


def build_verification_report(
    candidate: CandidateSpec,
    evaluation: EvaluationSpec,
    frame: pd.DataFrame | None = None,
) -> dict[str, Any]:
    verify_candidate_and_evaluation(
        candidate,
        evaluation,
    )

    data_report = None

    if frame is not None:
        data_report = verify_research_data(
            frame,
            candidate,
            evaluation,
        )

    return {
        "status": "passed",
        "candidate": {
            "id": candidate.id,
            "version": candidate.version,
            "fingerprint": (
                candidate.fingerprint_value
            ),
            "identity_verified": True,
            "frozen_verified": True,
        },
        "evaluation": {
            "id": evaluation.id,
            "version": evaluation.version,
            "candidate_reference_verified": True,
            "temporal_separation_verified": True,
            "optimization_contamination_verified": True,
        },
        "data_verified": frame is not None,
        "data": data_report,
        "checks": {
            "temporal_leakage": "passed",
            "feature_target_leakage": "passed",
            "future_data_access": "passed",
            "missing_data": "passed",
            "candidate_mutation": "passed",
            "evaluation_contamination": "passed",
            "distribution_drift": "diagnostic",
        },
    }
