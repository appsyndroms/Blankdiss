from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Mapping

import pandas as pd

from ml.dataset import load_features
from ml.research.cache import (
    ResearchRequirement,
    build_research_cache,
)
from ml.research.candidates.spec import (
    load_candidate,
)
from ml.research.candidates.verification import (
    candidate_fingerprint,
    candidate_snapshot,
)
from ml.research.engine import run_spec
from ml.research.evaluation.spec import (
    EvaluationPeriod,
    EvaluationSpec,
    WalkForwardSpec,
    WalkForwardWindow,
    load_evaluation,
)
from ml.research.evaluation.verification import (
    verify_evaluation,
)
from ml.research.reporting import write_json
from ml.research.spec import (
    AnalysisSpec,
    ResearchSpec,
    SignalSpec,
)
from ml.research.verification import (
    build_verification_report,
)
from ml.research.walk_forward import (
    aggregate_walk_forward,
)


ROOT = Path(__file__).resolve().parents[3]

DEFAULT_OUTPUT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "ml"
    / "research"
    / "evaluation"
)


def _candidate_signal(
    candidate,
    feature_name: str,
) -> SignalSpec:
    parameters = candidate.parameters

    parameter = parameters.get(
        feature_name
    )

    if parameter is None:
        parameter = parameters

    if not isinstance(
        parameter,
        Mapping,
    ):
        raise ValueError(
            "Candidate parameter måste vara "
            "ett objekt: "
            f"{feature_name}"
        )

    direction = str(
        parameter.get(
            "direction",
            "upper",
        )
    ).lower()

    quantile = parameter.get(
        "quantile"
    )

    if quantile is None:
        raise ValueError(
            "Candidate parameter saknar "
            f"quantile: {feature_name}"
        )

    return SignalSpec(
        name=feature_name,
        direction=direction,
        bins=(float(quantile),),
    )


def _build_research_spec(
    candidate,
    evaluation: EvaluationSpec,
    *,
    window_name: str,
) -> ResearchSpec:
    signals = tuple(
        _candidate_signal(
            candidate,
            feature.name,
        )
        for feature in candidate.features
    )

    return ResearchSpec(
        id=(
            f"{evaluation.id}"
            f"__{window_name}"
        ),
        question=candidate.question,
        signals=signals,
        targets=(
            candidate.target.name,
        ),
        analysis=AnalysisSpec(
            type=candidate.analysis.type,
            bootstrap=candidate.analysis.bootstrap,
            bootstrap_iterations=(
                candidate.analysis
                .bootstrap_iterations
            ),
        ),
        mode="deep",
        windows=(window_name,),
        splits=("test",),
        metadata={
            "source": (
                "ml.research.evaluation.runner"
            ),
            "candidate_id": candidate.id,
            "candidate_version": (
                candidate.version
            ),
            "candidate_fingerprint": (
                candidate_fingerprint(candidate)
            ),
            "evaluation_id": evaluation.id,
            "evaluation_version": (
                evaluation.version
            ),
            "window_name": window_name,
        },
    )


def _build_evaluation_mask(
    frame: pd.DataFrame,
    start: str,
    end: str,
) -> object:
    dates = pd.to_datetime(
        frame["snapshot_date"],
        errors="coerce",
    )

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    return (
        (dates >= start_ts)
        & (dates <= end_ts)
    ).to_numpy()


def _requirements(
    spec: ResearchSpec,
) -> list[ResearchRequirement]:
    return [
        ResearchRequirement(
            signal_name=signal.name,
            target_name=target_name,
            tail_fraction=fraction,
            tail_direction=signal.direction,
        )
        for signal in spec.signals
        for target_name in spec.targets
        for fraction in signal.bins
    ]


def _data_fingerprint(
    frame: pd.DataFrame,
    candidate,
) -> str:
    columns = [
        "snapshot_date",
        *(
            feature.name
            for feature in candidate.features
        ),
    ]

    columns = list(
        dict.fromkeys(
            column
            for column in columns
            if column in frame.columns
        )
    )

    hashed = pd.util.hash_pandas_object(
        frame[columns],
        index=True,
    )

    digest = hashlib.sha256()
    digest.update(
        hashed.to_numpy(
            dtype="uint64"
        ).tobytes()
    )

    return digest.hexdigest()


def _run_material(
    candidate,
    evaluation: EvaluationSpec,
    data_fingerprint: str,
) -> str:
    return "|".join(
        [
            candidate_fingerprint(candidate),
            evaluation.id,
            str(evaluation.version),
            data_fingerprint,
        ]
    )


def _run_hash(
    candidate,
    evaluation: EvaluationSpec,
    data_fingerprint: str,
) -> str:
    material = _run_material(
        candidate,
        evaluation,
        data_fingerprint,
    )

    return hashlib.sha256(
        material.encode("utf-8")
    ).hexdigest()


def _find_duplicate_run(
    output_dir: Path,
    candidate,
    evaluation: EvaluationSpec,
    data_fingerprint: str,
) -> Path | None:
    expected_candidate_fingerprint = (
        candidate_fingerprint(candidate)
    )

    expected_hash = _run_hash(
        candidate,
        evaluation,
        data_fingerprint,
    )

    if not output_dir.exists():
        return None

    for manifest_path in output_dir.glob(
        "*/manifest.json"
    ):
        try:
            with manifest_path.open(
                "r",
                encoding="utf-8",
            ) as handle:
                manifest = json.load(handle)
        except (
            OSError,
            json.JSONDecodeError,
        ):
            continue

        if (
            manifest.get(
                "candidate_fingerprint"
            )
            != expected_candidate_fingerprint
        ):
            continue

        if (
            manifest.get("evaluation_id")
            != evaluation.id
        ):
            continue

        if (
            manifest.get("evaluation_version")
            != evaluation.version
        ):
            continue

        if (
            manifest.get("data_fingerprint")
            != data_fingerprint
        ):
            continue

        if manifest.get("run_hash") == expected_hash:
            return manifest_path.parent

    return None


def _run_window(
    frame: pd.DataFrame,
    cache,
    candidate,
    evaluation: EvaluationSpec,
    *,
    window_name: str,
    start: str,
    end: str,
) -> dict:
    spec = _build_research_spec(
        candidate,
        evaluation,
        window_name=window_name,
    )

    cache.window_masks[
        window_name
    ] = {
        "test": _build_evaluation_mask(
            frame,
            start,
            end,
        )
    }

    result = run_spec(
        cache,
        spec,
    )

    result["evaluation_period"] = {
        "start": start,
        "end": end,
    }

    result["candidate"] = {
        "id": candidate.id,
        "version": candidate.version,
        "fingerprint": candidate_fingerprint(
            candidate
        ),
    }

    result["evaluation"] = {
        "id": evaluation.id,
        "version": evaluation.version,
    }

    result["window"] = {
        "name": window_name,
        "start": start,
        "end": end,
    }

    return result


def _windows(
    evaluation: EvaluationSpec,
) -> tuple[WalkForwardWindow, ...]:
    if evaluation.walk_forward.enabled:
        return evaluation.walk_forward.windows

    return (
        WalkForwardWindow(
            name="evaluation",
            start=(
                evaluation
                .evaluation_period
                .start
            ),
            end=(
                evaluation
                .evaluation_period
                .end
            ),
        ),
    )


def _latest_feature_date(
    frame: pd.DataFrame,
) -> date:
    if "snapshot_date" not in frame.columns:
        raise ValueError(
            "Feature-data saknar snapshot_date."
        )

    dates = pd.to_datetime(
        frame["snapshot_date"],
        errors="coerce",
        utc=True,
    )

    if dates.notna().sum() == 0:
        raise ValueError(
            "Feature-data saknar giltiga "
            "snapshot_date-värden."
        )

    return dates.max().date()


def _effective_evaluation(
    evaluation: EvaluationSpec,
    latest_feature_date: date,
) -> EvaluationSpec:
    """
    Säkerställer att evaluation aldrig använder ett datum
    som ännu inte finns i feature-datasetet.

    Marknadsdata är den faktiska källan till evaluationens
    effektiva slutdatum. Om en konfigurerad evaluation pekar
    längre fram än senaste tillgängliga feature-datum kapas
    evaluation och eventuella walk-forward-fönster dit.
    """

    latest = latest_feature_date.isoformat()

    configured_end = pd.Timestamp(
        evaluation.evaluation_period.end
    ).date()

    if configured_end <= latest_feature_date:
        return evaluation

    evaluation_period = EvaluationPeriod(
        start=evaluation.evaluation_period.start,
        end=latest,
    )

    windows = []

    for window in evaluation.walk_forward.windows:
        window_end = pd.Timestamp(
            window.end
        ).date()

        if window_end > latest_feature_date:
            window = replace(
                window,
                end=latest,
            )

        windows.append(window)

    walk_forward = WalkForwardSpec(
        enabled=evaluation.walk_forward.enabled,
        mode=evaluation.walk_forward.mode,
        windows=tuple(windows),
    )

    resolved_id = evaluation.id

    if evaluation.walk_forward.mode == "rolling":
        suffix = f"_{latest}"

        if not resolved_id.endswith(suffix):
            resolved_id = (
                f"{resolved_id}{suffix}"
            )

    return replace(
        evaluation,
        id=resolved_id,
        evaluation_period=evaluation_period,
        walk_forward=walk_forward,
    )


def _log_evaluation_context(
    latest_feature_date: date,
    evaluation: EvaluationSpec,
) -> None:
    print(
        "==========================================",
        flush=True,
    )
    print(
        "PROSPECTIVE EVALUATION DATUM",
        flush=True,
    )
    print(
        "==========================================",
        flush=True,
    )
    print(
        "Senaste tillgängliga feature-datum: "
        f"{latest_feature_date.isoformat()}",
        flush=True,
    )
    print(
        "Effektivt evaluation-slutdatum: "
        f"{evaluation.evaluation_period.end}",
        flush=True,
    )

    if evaluation.walk_forward.enabled:
        print(
            "Walk-forward windows:",
            flush=True,
        )

        for window in evaluation.walk_forward.windows:
            print(
                f"  {window.name}: "
                f"{window.start} -> {window.end}",
                flush=True,
            )

    print(
        "==========================================",
        flush=True,
    )


def run_evaluation(
    candidate_path: str | Path,
    evaluation_path: str | Path,
    *,
    output_dir: str | Path | None = None,
) -> Path:
    candidate = load_candidate(
        candidate_path
    )

    frame = load_features()

    latest_feature_date = _latest_feature_date(
        frame
    )

    evaluation = load_evaluation(
        evaluation_path,
        as_of=latest_feature_date,
    )

    evaluation = _effective_evaluation(
        evaluation,
        latest_feature_date,
    )

    _log_evaluation_context(
        latest_feature_date,
        evaluation,
    )

    verify_evaluation(
        candidate,
        evaluation,
    )

    verification = build_verification_report(
        candidate,
        evaluation,
        frame,
    )

    windows = _windows(
        evaluation
    )

    if not windows:
        raise ValueError(
            "Evaluation saknar windows."
        )

    first_spec = _build_research_spec(
        candidate,
        evaluation,
        window_name=windows[0].name,
    )

    requirements = _requirements(
        first_spec
    )

    cache = build_research_cache(
        frame,
        requirements,
    )

    results = []

    for window in windows:
        results.append(
            _run_window(
                frame,
                cache,
                candidate,
                evaluation,
                window_name=window.name,
                start=window.start,
                end=window.end,
            )
        )

    data_fingerprint = _data_fingerprint(
        frame,
        candidate,
    )

    destination_root = Path(
        output_dir
        if output_dir is not None
        else DEFAULT_OUTPUT_DIR
    )

    duplicate = _find_duplicate_run(
        destination_root,
        candidate,
        evaluation,
        data_fingerprint,
    )

    if duplicate is not None:
        raise FileExistsError(
            "Duplicate evaluation run detected. "
            f"Existing run: {duplicate}"
        )

    run_hash = _run_hash(
        candidate,
        evaluation,
        data_fingerprint,
    )

    run_id = (
        datetime.now(
            timezone.utc
        ).strftime(
            "%Y%m%dT%H%M%SZ"
        )
        + "-"
        + run_hash[:16]
    )

    destination = (
        destination_root
        / run_id
    )

    if destination.exists():
        raise FileExistsError(
            "Duplicate evaluation run detected: "
            f"{run_id}"
        )

    destination.mkdir(
        parents=True,
        exist_ok=False,
    )

    aggregation = aggregate_walk_forward(
        results
    )

    payload = {
        "run_id": run_id,
        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "candidate": candidate_snapshot(
            candidate
        ),
        "evaluation": {
            "id": evaluation.id,
            "version": evaluation.version,
            "candidate_id": (
                evaluation.candidate_id
            ),
            "candidate_version": (
                evaluation.candidate_version
            ),
            "candidate_fingerprint": (
                evaluation.candidate_fingerprint
            ),
            "evaluation_period": {
                "start": (
                    evaluation
                    .evaluation_period
                    .start
                ),
                "end": (
                    evaluation
                    .evaluation_period
                    .end
                ),
            },
            "walk_forward": {
                "enabled": (
                    evaluation
                    .walk_forward
                    .enabled
                ),
                "windows": [
                    {
                        "name": window.name,
                        "start": window.start,
                        "end": window.end,
                    }
                    for window
                    in evaluation
                    .walk_forward
                    .windows
                ],
            },
            "metrics": list(
                evaluation.metrics
            ),
            "metadata": evaluation.metadata,
        },
        "verification": verification,
        "data": {
            "feature_rows": int(
                len(frame)
            ),
            "latest_feature_date": (
                latest_feature_date.isoformat()
            ),
            "fingerprint": data_fingerprint,
        },
        "walk_forward": aggregation,
        "results": results,
    }

    output_path = (
        destination
        / "evaluation.json"
    )

    write_json(
        output_path,
        payload,
    )

    write_json(
        destination
        / "manifest.json",
        {
            "run_id": run_id,
            "run_hash": run_hash,
            "candidate_id": candidate.id,
            "candidate_version": (
                candidate.version
            ),
            "candidate_fingerprint": (
                candidate_fingerprint(
                    candidate
                )
            ),
            "evaluation_id": evaluation.id,
            "evaluation_version": (
                evaluation.version
            ),
            "data_fingerprint": data_fingerprint,
            "result": str(
                output_path.relative_to(
                    ROOT
                )
            ),
            "verification": "passed",
        },
    )

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Blankdiss prospective "
            "candidate evaluation runner."
        )
    )

    parser.add_argument(
        "candidate",
        help="Path till frozen candidate YAML.",
    )

    parser.add_argument(
        "evaluation",
        help="Path till evaluation YAML.",
    )

    parser.add_argument(
        "--output-dir",
        help="Override evaluation output directory.",
    )

    args = parser.parse_args()

    output = run_evaluation(
        args.candidate,
        args.evaluation,
        output_dir=args.output_dir,
    )

    print(
        f"Evaluation complete: {output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
