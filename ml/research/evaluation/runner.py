from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ml.dataset import load_features
from ml.research.cache import (
    ResearchRequirement,
    build_research_cache,
)
from ml.research.engine import run_spec
from ml.research.reporting import write_json
from ml.research.spec import (
    AnalysisSpec,
    ResearchSpec,
    SignalSpec,
)

from research.candidates.spec import load_candidate
from research.candidates.verification import (
    candidate_fingerprint,
    candidate_snapshot,
    verify_candidate_frozen,
)
from research.evaluation.spec import (
    EvaluationSpec,
    WalkForwardWindow,
    load_evaluation,
)
from research.evaluation.verification import (
    verify_evaluation,
)


ROOT = Path(__file__).resolve().parents[2]

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

    if feature_name not in parameters:
        raise ValueError(
            "Candidate saknar parameters för feature: "
            f"{feature_name}"
        )

    parameter = parameters[feature_name]

    if not isinstance(parameter, dict):
        raise ValueError(
            "Candidate parameter måste vara ett objekt: "
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
            "Candidate parameter saknar quantile: "
            f"{feature_name}"
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
        targets=(candidate.target.name,),
        analysis=AnalysisSpec(
            type=evaluation.analysis.type,
            bootstrap=evaluation.analysis.bootstrap,
            bootstrap_iterations=(
                evaluation.analysis.bootstrap_iterations
            ),
        ),
        mode="deep",
        windows=(window_name,),
        splits=("test",),
        metadata={
            "source": "research.evaluation.runner",
            "candidate_id": candidate.id,
            "candidate_version": candidate.version,
            "candidate_fingerprint": (
                candidate_fingerprint(candidate)
            ),
            "evaluation_id": evaluation.id,
            "evaluation_version": evaluation.version,
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
        (dates > start_ts)
        & (dates <= end_ts)
    ).to_numpy()


def _run_window(
    frame: pd.DataFrame,
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

    requirements = [
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

    cache = build_research_cache(
        frame,
        requirements,
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

    return result


def _windows(
    evaluation: EvaluationSpec,
) -> tuple[WalkForwardWindow, ...]:
    if evaluation.walk_forward.enabled:
        return evaluation.walk_forward.windows

    return (
        WalkForwardWindow(
            name="evaluation",
            start=evaluation.evaluation_period.start,
            end=evaluation.evaluation_period.end,
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

    verify_candidate_frozen(
        candidate
    )

    verify_evaluation(
        candidate,
        evaluation,
    )

    frame = load_features()

    windows = _windows(
        evaluation
    )

    results = []

    for window in windows:
        results.append(
            _run_window(
                frame,
                candidate,
                evaluation,
                window_name=window.name,
                start=window.start,
                end=window.end,
            )
        )

    run_id = (
        datetime.now(
            timezone.utc
        ).strftime(
            "%Y%m%dT%H%M%SZ"
        )
    )

    destination = Path(
        output_dir
        if output_dir is not None
        else DEFAULT_OUTPUT_DIR
    ) / run_id

    destination.mkdir(
        parents=True,
        exist_ok=True,
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
            "candidate_id": evaluation.candidate_id,
            "candidate_version": (
                evaluation.candidate_version
            ),
            "candidate_fingerprint": (
                evaluation.candidate_fingerprint
            ),
            "evaluation_period": {
                "start": (
                    evaluation.evaluation_period.start
                ),
                "end": (
                    evaluation.evaluation_period.end
                ),
            },
            "walk_forward": {
                "enabled": (
                    evaluation.walk_forward.enabled
                ),
                "windows": [
                    {
                        "name": window.name,
                        "start": window.start,
                        "end": window.end,
                    }
                    for window
                    in evaluation.walk_forward.windows
                ],
            },
            "metrics": list(
                evaluation.metrics
            ),
            "analysis": {
                "type": evaluation.analysis.type,
                "bootstrap": evaluation.analysis.bootstrap,
                "bootstrap_iterations": (
                    evaluation.analysis.bootstrap_iterations
                ),
            },
        },
        "verification": {
            "status": "passed",
        },
        "feature_rows": int(
            len(frame)
        ),
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
            "candidate_version": candidate.version,
            "candidate_fingerprint": (
                candidate_fingerprint(candidate)
            ),
            "evaluation_id": evaluation.id,
            "evaluation_version": evaluation.version,
            "result": str(
                output_path.relative_to(ROOT)
            ),
            "verification": "passed",
        },
    )

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Blankdiss prospective candidate evaluation runner."
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

    args = parser.parse_args()

    output = run_evaluation(
        args.candidate,
        args.evaluation,
    )

    print(
        f"Evaluation complete: {output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
