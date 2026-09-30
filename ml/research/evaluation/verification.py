from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any

import pandas as pd

from ml.config import (
    FEATURE_EXCLUDE_COLUMNS,
    FI_ONLY_EXCLUDE_COLUMNS,
    PRICE_FEATURE_COLUMNS,
    TARGETS,
)
from ml.research.candidates.spec import CandidateSpec
from ml.research.candidates.verification import (
    candidate_fingerprint,
)
from ml.research.evaluation.spec import (
    EvaluationSpec,
)
from ml.research.signals import build_signal


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


def _as_utc_timestamp(
    value: str,
) -> pd.Timestamp:
    timestamp = pd.Timestamp(
        _parse_boundary(
            value,
            "timestamp",
        )
    )

    if timestamp.tzinfo is None:
        return timestamp.tz_localize(
            "UTC"
        )

    return timestamp.tz_convert(
        "UTC"
    )


def _normalise_dates(
    values: pd.Series,
) -> pd.Series:
    parsed = pd.to_datetime(
        values,
        errors="coerce",
        utc=True,
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
        "candidate.training_period.start",
    )

    end = _parse_boundary(
        candidate.training_period.end,
        "candidate.training_period.end",
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

    if evaluation.walk_forward.mode == "rolling":
        return

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

    if evaluation.metadata.get(
        "hyperparameter_search",
        False,
    ):
        raise ValueError(
            "Evaluation får inte innehålla "
            "hyperparameter search."
        )

    if evaluation.metadata.get(
        "fit_on_evaluation",
        False,
    ):
        raise ValueError(
            "Evaluation får inte träna eller "
            "anpassa modellen på evaluation-data."
        )

    if evaluation.metadata.get(
        "use_evaluation_for_selection",
        False,
    ):
        raise ValueError(
            "Evaluation-data får inte användas "
            "för kandidatselektion."
        )


def verify_evaluation_period_data(
    frame: pd.DataFrame,
    candidate: CandidateSpec,
    evaluation: EvaluationSpec,
) -> dict[str, Any]:
    if "snapshot_date" not in frame.columns:
        raise ValueError(
            "Feature-data saknar snapshot_date."
        )

    dates = _normalise_dates(
        frame["snapshot_date"]
    )

    if dates.notna().sum() == 0:
        raise ValueError(
            "Feature-data saknar giltiga "
            "snapshot_date-värden."
        )

    start = _as_utc_timestamp(
        evaluation.evaluation_period.start
    )

    end = _as_utc_timestamp(
        evaluation.evaluation_period.end
    )

    evaluation_mask = (
        (dates >= start)
        & (dates <= end)
    )

    evaluation_rows = int(
        evaluation_mask.sum()
    )

    if evaluation_rows == 0:
        raise ValueError(
            "Evaluation-perioden innehåller "
            "inga feature rows."
        )

    evaluation_dates = dates[
        evaluation_mask
    ]

    max_date = evaluation_dates.max()
    min_date = evaluation_dates.min()

    feature_coverage: dict[str, Any] = {}

    for feature in candidate.features:
        signal = build_signal(
            frame,
            feature.name,
        )

        usable_mask = (
            evaluation_mask
            & signal.notna()
        )

        usable_dates = dates[
            usable_mask
        ]

        if usable_dates.empty:
            raise ValueError(
                "Candidate feature saknar "
                "användbara observationer i "
                "evaluation-perioden: "
                f"{feature.name}"
            )

        feature_min_date = (
            usable_dates.min()
        )
        feature_max_date = (
            usable_dates.max()
        )

        if (
            pd.isna(feature_max_date)
            or feature_max_date < end
        ):
            raise ValueError(
                "Candidate feature når inte "
                "evaluation-periodens slutdatum: "
                f"feature={feature.name}, "
                f"max={feature_max_date}, "
                f"end={end}."
            )

        feature_coverage[
            feature.name
        ] = {
            "usable_rows": int(
                usable_mask.sum()
            ),
            "min_snapshot_date": (
                feature_min_date.isoformat()
                if pd.notna(feature_min_date)
                else None
            ),
            "max_snapshot_date": (
                feature_max_date.isoformat()
                if pd.notna(feature_max_date)
                else None
            ),
        }

    return {
        "rows": evaluation_rows,
        "min_snapshot_date": (
            min_date.isoformat()
            if pd.notna(min_date)
            else None
        ),
        "max_snapshot_date": (
            max_date.isoformat()
            if pd.notna(max_date)
            else None
        ),
        "feature_coverage": feature_coverage,
    }


def _future_data_columns(
    candidate: CandidateSpec,
) -> list[str]:
    configured_forbidden = (
        set(FEATURE_EXCLUDE_COLUMNS)
        | set(FI_ONLY_EXCLUDE_COLUMNS)
    )

    target_return_columns = {
        target.return_column
        for target in TARGETS
    }

    target_columns = {
        target.target_column
        for target in TARGETS
        if target.target_column is not None
    }

    forbidden = (
        configured_forbidden
        | target_return_columns
        | target_columns
    )

    future_tokens = (
        "forward_return",
        "future_return",
        "future_",
        "_future",
        "target",
        "label",
        "outcome",
    )

    result = []

    for feature in candidate.features:
        name = feature.name

        lowered = name.lower()

        if name in forbidden:
            result.append(name)
            continue

        if any(
            token in lowered
            for token in future_tokens
        ):
            result.append(name)

    return sorted(set(result))


def verify_feature_target_leakage(
    candidate: CandidateSpec,
) -> None:
    leakage_columns = _future_data_columns(
        candidate
    )

    if leakage_columns:
        raise ValueError(
            "Candidate innehåller feature(s) "
            "som kan vara target/future-data: "
            + ", ".join(leakage_columns)
        )


def verify_future_data_access(
    frame: pd.DataFrame,
    candidate: CandidateSpec,
) -> None:
    """
    Verifierar att kandidatens signaler kan byggas.

    Kandidatens feature-namn är semantiska signalnamn, inte de
    fysiska kolumnnamnen i feature-datasetet. build_signal()
    använder samma signalregister som research-körningen och
    säkerställer därmed att verifieringen testar samma kontrakt.
    """
    for feature in candidate.features:
        build_signal(
            frame,
            feature.name,
        )


def verify_missing_data(
    frame: pd.DataFrame,
    candidate: CandidateSpec,
) -> dict[str, Any]:
    rows = len(frame)

    if rows == 0:
        raise ValueError(
            "Feature-data innehåller inga rows."
        )

    missingness: dict[str, Any] = {}
    unusable = []

    for feature in candidate.features:
        name = feature.name

        series = build_signal(
            frame,
            name,
        )
        missing_count = int(
            series.isna().sum()
        )

        missing_fraction = (
            missing_count / rows
            if rows
            else 1.0
        )

        missingness[name] = {
            "missing_rows": missing_count,
            "missing_fraction": missing_fraction,
        }

        if missing_count == rows:
            unusable.append(name)

    if unusable:
        raise ValueError(
            "Candidate features innehåller "
            "endast saknade värden: "
            + ", ".join(sorted(unusable))
        )

    return missingness


def verify_candidate_features(
    frame: pd.DataFrame,
    candidate: CandidateSpec,
) -> None:
    if "snapshot_date" not in frame.columns:
        raise ValueError(
            "Feature-data saknar snapshot_date."
        )

    for feature in candidate.features:
        try:
            build_signal(
                frame,
                feature.name,
            )
        except ValueError as exc:
            raise ValueError(
                "Candidate refererar till ogiltig "
                f"feature-signal '{feature.name}': "
                f"{exc}"
            ) from exc

    target_names = {
        target.name
        for target in TARGETS
    }

    if candidate.target.name not in target_names:
        raise ValueError(
            "Candidate refererar till okänd "
            f"target: {candidate.target.name}"
        )


def _distribution_summary(
    series: pd.Series,
) -> dict[str, Any]:
    numeric = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if numeric.empty:
        return {
            "n": 0,
            "mean": None,
            "std": None,
            "q05": None,
            "q50": None,
            "q95": None,
        }

    return {
        "n": int(len(numeric)),
        "mean": float(numeric.mean()),
        "std": (
            float(numeric.std())
            if len(numeric) > 1
            else 0.0
        ),
        "q05": float(
            numeric.quantile(0.05)
        ),
        "q50": float(
            numeric.quantile(0.50)
        ),
        "q95": float(
            numeric.quantile(0.95)
        ),
    }


def _distribution_change(
    training: dict[str, Any],
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    training_mean = training["mean"]
    evaluation_mean = evaluation["mean"]

    training_std = training["std"]

    if (
        training_mean is None
        or evaluation_mean is None
    ):
        mean_delta = None
    else:
        mean_delta = (
            evaluation_mean
            - training_mean
        )

    if (
        training_std is None
        or training_std == 0
        or mean_delta is None
    ):
        standardized_mean_delta = None
    else:
        standardized_mean_delta = (
            mean_delta
            / training_std
        )

    return {
        "mean_delta": mean_delta,
        "standardized_mean_delta": (
            standardized_mean_delta
        ),
        "median_delta": (
            None
            if (
                training["q50"] is None
                or evaluation["q50"] is None
            )
            else (
                evaluation["q50"]
                - training["q50"]
            )
        ),
    }


def distribution_diagnostics(
    frame: pd.DataFrame,
    candidate: CandidateSpec,
    evaluation: EvaluationSpec,
) -> dict[str, Any]:
    dates = _normalise_dates(
        frame["snapshot_date"]
    )

    training_start = _as_utc_timestamp(
        candidate.training_period.start
    )
    training_end = _as_utc_timestamp(
        candidate.training_period.end
    )

    evaluation_start = _as_utc_timestamp(
        evaluation.evaluation_period.start
    )
    evaluation_end = _as_utc_timestamp(
        evaluation.evaluation_period.end
    )

    training_mask = (
        (dates >= training_start)
        & (dates <= training_end)
    )

    evaluation_mask = (
        (dates >= evaluation_start)
        & (dates <= evaluation_end)
    )

    diagnostics = {}

    for feature in candidate.features:
        name = feature.name

        training_summary = (
            _distribution_summary(
                build_signal(
                    frame.loc[
                        training_mask,
                    ],
                    name,
                )
            )
        )

        evaluation_summary = (
            _distribution_summary(
                build_signal(
                    frame.loc[
                        evaluation_mask,
                    ],
                    name,
                )
            )
        )

        diagnostics[name] = {
            "training": training_summary,
            "evaluation": evaluation_summary,
            "change": _distribution_change(
                training_summary,
                evaluation_summary,
            ),
        }

    return diagnostics


def verify_evaluation_data(
    frame: pd.DataFrame,
    candidate: CandidateSpec,
    evaluation: EvaluationSpec,
) -> dict[str, Any]:
    verify_candidate_features(
        frame,
        candidate,
    )

    verify_feature_target_leakage(
        candidate
    )

    verify_future_data_access(
        frame,
        candidate,
    )

    period = verify_evaluation_period_data(
        frame,
        candidate,
        evaluation,
    )

    missing_data = verify_missing_data(
        frame,
        candidate,
    )

    distributions = distribution_diagnostics(
        frame,
        candidate,
        evaluation,
    )

    return {
        "evaluation_period": period,
        "missing_data": missing_data,
        "distribution_changes": distributions,
    }


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
