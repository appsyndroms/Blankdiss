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
) -> None:
    verify_evaluation_data(
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

    if frame is not None:
        verify_research_data(
            frame,
            candidate,
            evaluation,
        )

    return {
        "status": "passed",
        "candidate": {
            "id": candidate.id,
            "version": candidate.version,
        },
        "evaluation": {
            "id": evaluation.id,
            "version": evaluation.version,
        },
        "data_verified": frame is not None,
    }
