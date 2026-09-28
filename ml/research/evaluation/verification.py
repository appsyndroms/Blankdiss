from __future__ import annotations

from datetime import date, datetime, timezone

import pandas as pd

from ml.config import TARGETS
from ml.research.candidates.spec import CandidateSpec
from ml.research.candidates.verification import (
    candidate_fingerprint,
)
from ml.research.evaluation.spec import (
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
    discovery_cutoff = _parse_boundary(
        candidate.discovery_cutoff,
        "candidate.discovery_cutoff",
    )

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

    if discovery_cutoff >= freeze_at:
        raise ValueError(
            "Temporal separation violation: "
            "discovery_cutoff måste ligga "
            "före freeze_at."
        )

    if freeze_at >= start:
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
    evaluation: EvaluationSpec | None = None,
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

    if evaluation is not None:
        evaluation_start = _parse_boundary(
            evaluation.evaluation_period.start,
            "evaluation_period.start",
        )

        evaluation_end = _parse_boundary(
            evaluation.evaluation_period.end,
            "evaluation_period.end",
        )

        if window_start < evaluation_start:
            raise ValueError(
                "Walk-forward-window måste ligga "
                "inom evaluation_period."
            )

        if window_end > evaluation_end:
            raise ValueError(
                "Walk-forward-window måste ligga "
                "inom evaluation_period."
            )


def verify_all_walk_forward_windows(
    candidate: CandidateSpec,
    evaluation: EvaluationSpec,
) -> None:
    if not evaluation.walk_forward.enabled:
        return

    windows = evaluation.walk_forward.windows

    if not windows:
        raise ValueError(
            "walk_forward.enabled=true "
            "kräver minst ett window."
        )

    names = [
        window.name
        for window in windows
    ]

    if len(names) != len(set(names)):
        raise ValueError(
            "Walk-forward-windows måste ha "
            "unika namn."
        )

    parsed_windows = []

    for window in windows:
        start = _parse_boundary(
            window.start,
            f"walk_forward.{window.name}.start",
        )

        end = _parse_boundary(
            window.end,
            f"walk_forward.{window.name}.end",
        )

        verify_walk_forward_window(
            candidate,
            window.start,
            window.end,
            evaluation,
        )

        parsed_windows.append(
            (
                window.name,
                start,
                end,
            )
        )

    for previous, current in zip(
        parsed_windows,
        parsed_windows[1:],
    ):
        previous_name, _, previous_end = previous
        current_name, current_start, _ = current

        if current_start <= previous_end:
            raise ValueError(
                "Walk-forward-windows får inte "
                "överlappa eller ligga i fel "
                "kronologisk ordning: "
                f"{previous_name} -> {current_name}."
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

    if evaluation.metadata.get(
        "optimize_parameters",
        False,
    ):
        raise ValueError(
            "Evaluation får inte optimera "
            "kandidatens parametrar."
        )


def verify_evaluation_period_data(
    frame: pd.DataFrame,
    evaluation: EvaluationSpec,
) -> None:
    if "snapshot_date" not in frame.columns:
        raise ValueError(
            "Feature-data saknar snapshot_date."
        )

    dates = pd.to_datetime(
        frame["snapshot_date"],
        errors="coerce",
    )

    if dates.notna().sum() == 0:
        raise ValueError(
            "Feature-data saknar giltiga "
            "snapshot_date-värden."
        )

    start = pd.Timestamp(
        evaluation.evaluation_period.start
    )
    end = pd.Timestamp(
        evaluation.evaluation_period.end
    )

    if start.tzinfo is not None:
        dates = (
            dates.dt.tz_localize(
                "UTC",
                ambiguous="NaT",
                nonexistent="NaT",
            )
            if dates.dt.tz is None
            else dates
        )

    evaluation_mask = (
        (dates >= start)
        & (dates <= end)
    )

    if int(evaluation_mask.sum()) == 0:
        raise ValueError(
            "Evaluation-perioden innehåller "
            "inga feature rows."
        )

    max_date = dates.max()

    if pd.notna(max_date) and max_date < end:
        raise ValueError(
            "Feature-data når inte evaluation-periodens "
            f"slutdatum: max={max_date}, end={end}."
        )


def verify_candidate_features(
    frame: pd.DataFrame,
    candidate: CandidateSpec,
) -> None:
    if "snapshot_date" not in frame.columns:
        raise ValueError(
            "Feature-data saknar snapshot_date."
        )

    missing = [
        feature.name
        for feature in candidate.features
        if feature.name not in frame.columns
    ]

    if missing:
        raise ValueError(
            "Candidate refererar till saknade "
            "features: "
            + ", ".join(sorted(missing))
        )

    target_names = {
        target.name
        for target in TARGETS
    }

    if candidate.target.name not in target_names:
        raise ValueError(
            "Candidate refererar till okänd "
            f"target: {candidate.target.name}"
        )


def verify_evaluation_data(
    frame: pd.DataFrame,
    candidate: CandidateSpec,
    evaluation: EvaluationSpec,
) -> None:
    verify_candidate_features(
        frame,
        candidate,
    )

    verify_evaluation_period_data(
        frame,
        evaluation,
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
