from __future__ import annotations

from datetime import datetime

from research.candidates.spec import CandidateSpec
from research.evaluation.spec import EvaluationSpec


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
            f"{field_name} måste innehålla timezone: "
            f"{value}"
        )

    return parsed


def verify_temporal_separation(
    candidate: CandidateSpec,
    evaluation: EvaluationSpec,
) -> None:
    """
    Kontrollerar att evaluation börjar efter kandidatens freeze.

    Detta är den centrala spärren mot att en prospective evaluation
    råkar använda information som kandidaten inte borde ha haft
    tillgång till vid freeze.
    """

    freeze_at = _parse_datetime(
        candidate.freeze_at,
        "candidate.freeze_at",
    )

    evaluation_start = _parse_datetime(
        evaluation.evaluation_period.start,
        "evaluation_period.start",
    )

    evaluation_end = _parse_datetime(
        evaluation.evaluation_period.end,
        "evaluation_period.end",
    )

    if evaluation_end < evaluation_start:
        raise ValueError(
            "Evaluation-periodens slut ligger före starten."
        )

    if evaluation_start <= freeze_at:
        raise ValueError(
            "Temporal separation violation: "
            "evaluation måste börja efter candidate.freeze_at. "
            f"freeze_at={candidate.freeze_at}, "
            f"evaluation_start="
            f"{evaluation.evaluation_period.start}"
        )


def verify_training_period(
    candidate: CandidateSpec,
) -> None:
    """
    Kontrollerar kandidatens training-period internt.
    """

    start = _parse_datetime(
        candidate.training_period.start,
        "training_period.start",
    )

    end = _parse_datetime(
        candidate.training_period.end,
        "training_period.end",
    )

    if end < start:
        raise ValueError(
            "Candidate training_period slutar före den börjar."
        )

    discovery_cutoff = _parse_datetime(
        candidate.discovery_cutoff,
        "candidate.discovery_cutoff",
    )

    if end > discovery_cutoff:
        raise ValueError(
            "Candidate training_period får inte sträcka sig "
            "förbi discovery_cutoff."
        )


def verify_evaluation_targets(
    candidate: CandidateSpec,
    evaluation: EvaluationSpec,
) -> None:
    """
    Evaluation får inte byta kandidatens target.

    Flera targets är tillåtna i evaluation, men kandidatens
    definierade target måste finnas med.
    """

    if candidate.target.name not in evaluation.targets:
        raise ValueError(
            "Evaluation saknar kandidatens target: "
            f"{candidate.target.name}"
        )


def verify_candidate_reference(
    candidate: CandidateSpec,
    evaluation: EvaluationSpec,
) -> None:
    if evaluation.candidate_id != candidate.id:
        raise ValueError(
            "Evaluation refererar till fel candidate_id: "
            f"{evaluation.candidate_id}; "
            f"candidate={candidate.id}"
        )

    if evaluation.candidate_version != candidate.version:
        raise ValueError(
            "Evaluation refererar till fel candidate_version: "
            f"{evaluation.candidate_version}; "
            f"candidate={candidate.version}"
        )


def verify_evaluation(
    candidate: CandidateSpec,
    evaluation: EvaluationSpec,
) -> None:
    """
    Full verification före prospective evaluation.

    Ordningen är medveten:

    1. frozen
    2. identity
    3. training period
    4. temporal separation
    5. target
    """

    if candidate.status != "frozen":
        raise ValueError(
            "Evaluation kräver en frozen candidate."
        )

    verify_candidate_reference(
        candidate,
        evaluation,
    )

    verify_training_period(
        candidate
    )

    verify_temporal_separation(
        candidate,
        evaluation,
    )

    verify_evaluation_targets(
        candidate,
        evaluation,
    )


def verify_walk_forward_window(
    candidate: CandidateSpec,
    start: str,
    end: str,
) -> None:
    """
    Kontrollerar ett individuellt walk-forward-fönster.

    Varje framtida evaluation-window måste börja efter freeze.
    """

    freeze_at = _parse_datetime(
        candidate.freeze_at,
        "candidate.freeze_at",
    )

    window_start = _parse_datetime(
        start,
        "walk_forward.start",
    )

    window_end = _parse_datetime(
        end,
        "walk_forward.end",
    )

    if window_end < window_start:
        raise ValueError(
            "Walk-forward window slutar före det börjar."
        )

    if window_start <= freeze_at:
        raise ValueError(
            "Walk-forward window börjar inte efter "
            "candidate.freeze_at."
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
