from __future__ import annotations

from datetime import date, datetime, timezone

from research.candidates.spec import CandidateSpec
from research.candidates.verification import (
    candidate_fingerprint,
)
from research.evaluation.spec import (
    EvaluationSpec,
)


def _parse_boundary(
    value: str,
    field_name: str,
) -> datetime:
    if len(value) == 10:
        try:
            parsed_date = date.fromisoformat(
                value
            )
        except ValueError as exc:
            raise ValueError(
                f"Ogiltigt datum i "
                f"{field_name}: {value}"
            ) from exc

        return datetime(
            parsed_date.year,
            parsed_date.month,
            parsed_date.day,
            tzinfo=timezone.utc,
        )

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


def verify_candidate_reference(
    candidate: CandidateSpec,
    evaluation: EvaluationSpec,
) -> None:
    if candidate.id != evaluation.candidate_id:
        raise ValueError(
            "Candidate identity mismatch: "
            f"evaluation refererar "
            f"{evaluation.candidate_id}, "
            f"men laddad candidate är "
            f"{candidate.id}."
        )

    if (
        candidate.version
        != evaluation.candidate_version
    ):
        raise ValueError(
            "Candidate version mismatch: "
            f"evaluation refererar "
            f"v{evaluation.candidate_version}, "
            f"men laddad candidate är "
            f"v{candidate.version}."
        )


def verify_candidate_fingerprint_reference(
    candidate: CandidateSpec,
    evaluation: EvaluationSpec,
) -> None:
    actual = candidate_fingerprint(
        candidate
    )

    if candidate.fingerprint_value != actual:
        raise ValueError(
            "Candidate YAML har en fingerprint "
            "som inte matchar dess definition."
        )

    if (
        evaluation.candidate_fingerprint
        != actual
    ):
        raise ValueError(
            "Evaluation candidate fingerprint "
            "matchar inte den frysta kandidaten."
        )


def verify_temporal_separation(
    candidate: CandidateSpec,
    evaluation: EvaluationSpec,
) -> None:
    freeze_at = _parse_boundary(
        candidate.freeze_at,
        "candidate.freeze_at",
    )

    start = _parse_boundary(
        evaluation.evaluation_period.start,
        "evaluation_period.start",
    )

    end = _parse_boundary(
        evaluation.evaluation_period.end,
        "evaluation_period.end",
    )

    if end < start:
        raise ValueError(
            "Evaluation-periodens slut ligger "
            "före starten."
        )

    if start <= freeze_at:
        raise ValueError(
            "Temporal separation violation: "
            "evaluation måste börja efter "
            "candidate.freeze_at."
        )


def verify_training_period(
    candidate: CandidateSpec,
) -> None:
    start = _parse_boundary(
        candidate.training_period.start,
        "training_period.start",
    )

    end = _parse_boundary(
        candidate.training_period.end,
        "training_period.end",
    )

    cutoff = _parse_boundary(
        candidate.discovery_cutoff,
        "candidate.discovery_cutoff",
    )

    if end < start:
        raise ValueError(
            "training_period.end ligger före "
            "training_period.start."
        )

    if end > cutoff:
        raise ValueError(
            "training_period.end kan inte ligga "
            "efter discovery_cutoff."
        )


def verify_walk_forward_window(
    candidate: CandidateSpec,
    start: str,
    end: str,
) -> None:
    freeze_at = _parse_boundary(
        candidate.freeze_at,
        "candidate.freeze_at",
    )

    window_start = _parse_boundary(
        start,
        "walk_forward.start",
    )

    window_end = _parse_boundary(
        end,
        "walk_forward.end",
    )

    if window_end < window_start:
        raise ValueError(
            "Walk-forward-windowens slut ligger "
            "före starten."
        )

    if window_start <= freeze_at:
        raise ValueError(
            "Walk-forward-window måste börja "
            "efter candidate.freeze_at."
        )


def verify_all_walk_forward_windows(
    candidate: CandidateSpec,
    evaluation: EvaluationSpec,
) -> None:
    if not evaluation.walk_forward.enabled:
        return

    for window in evaluation.walk_forward.windows:
        verify_walk_forward_window(
            candidate,
            window.start,
            window.end,
        )


def verify_no_evaluation_optimization(
    evaluation: EvaluationSpec,
) -> None:
    if evaluation.metadata.get(
        "parameter_optimization",
        False,
    ):
        raise ValueError(
            "Evaluation får inte vara "
            "parameter-optimerande."
        )


def verify_evaluation(
    candidate: CandidateSpec,
    evaluation: EvaluationSpec,
) -> None:
    if candidate.status != "frozen":
        raise ValueError(
            "Evaluation kräver en frozen candidate."
        )

    verify_candidate_reference(
        candidate,
        evaluation,
    )

    verify_candidate_fingerprint_reference(
        candidate,
        evaluation,
    )

    verify_training_period(
        candidate,
    )

    verify_temporal_separation(
        candidate,
        evaluation,
    )

    verify_all_walk_forward_windows(
        candidate,
        evaluation,
    )

    verify_no_evaluation_optimization(
        evaluation,
    )
