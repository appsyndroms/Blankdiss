from __future__ import annotations

import argparse
import hashlib
from datetime import datetime, timezone
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
    EvaluationSpec,
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


def run_evaluation(
    candidate_path: str | Path,
    evaluation_path: str | Path,
    *,
    output_dir: str | Path | None = None,
) -> Path:
    candidate = load_candidate(
        candidate_path
    )

    evaluation = load_evaluation(
        evaluation_path
    )

    verify_evaluation(
        candidate,
        evaluation,
    )

    frame = load_features()

    verification = build_verification_report(
        candidate,
        evaluation,
        frame,
    )

    windows = _windows(
        evaluation
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

    run_material = "|".join(
        [
            candidate_fingerprint(candidate),
            evaluation.id,
            str(evaluation.version),
            data_fingerprint,
        ]
    )

    run_hash = hashlib.sha256(
        run_material.encode("utf-8")
    ).hexdigest()[:16]

    run_id = (
        datetime.now(
            timezone.utc
        ).strftime(
            "%Y%m%dT%H%M%SZ"
        )
        + "-"
        + run_hash
    )

    destination = (
        Path(
            output_dir
            if output_dir is not None
            else DEFAULT_OUTPUT_DIR
        )
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
