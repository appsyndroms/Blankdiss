from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ml.research.discovery.config import (
    DiscoveryConfig,
    ValidationConfig,
)
from ml.research.discovery.engine import (
    Candidate,
    evaluate_candidate_on_mask,
    prepare_data,
)

from .config import load_config


ROOT = Path(
    "data/processed/ml/research/frozen"
)


def _build_candidate(config) -> Candidate:
    candidate = config.candidate

    return Candidate(
        candidate_id=candidate.candidate_id,
        target_name=candidate.target_name,
        signal_name=candidate.signal_name,
        signal_tail=candidate.signal_tail,
        stress_feature=candidate.stress_feature,
        stress_tail=candidate.stress_tail,
        stress_direction=candidate.stress_direction,
    )


def _build_discovery_config(config) -> DiscoveryConfig:
    candidate = config.candidate

    return DiscoveryConfig(
        enabled=True,
        targets=(candidate.target_name,),
        signals=(candidate.signal_name,),
        stress_features=(candidate.stress_feature,),
        tails=(
            candidate.signal_tail,
            candidate.stress_tail,
        ),
        stress_directions={
            candidate.stress_feature:
                candidate.stress_direction,
        },
        validation=ValidationConfig(),
    )


def _write_json(
    path: Path,
    payload,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
            default=str,
        )


def run() -> None:
    config = load_config()

    print(
        "=== Blankdiss Frozen Hypothesis OOS ===",
        flush=True,
    )

    print(
        f"Hypothesis: {config.hypothesis_id}",
        flush=True,
    )

    print(
        f"Candidate: {config.candidate.candidate_id}",
        flush=True,
    )

    print(
        "Evaluation: "
        f"{config.evaluation.start_date} -> "
        f"{config.evaluation.end_date}",
        flush=True,
    )

    discovery_config = _build_discovery_config(
        config
    )

    data = prepare_data(
        discovery_config
    )

    dates = pd.to_datetime(
        data.frame["snapshot_date"]
    )

    evaluation_mask = (
        (dates >= pd.Timestamp(
            config.evaluation.start_date
        ))
        & (
            dates <= pd.Timestamp(
                config.evaluation.end_date
            )
        )
    ).to_numpy()

    if not evaluation_mask.any():
        raise RuntimeError(
            "OOS-perioden innehåller inga feature-rader."
        )

    candidate = _build_candidate(
        config
    )

    result = evaluate_candidate_on_mask(
        data=data,
        candidate=candidate,
        base_mask=evaluation_mask,
        split="oos",
    )

    timestamp = datetime.now(
        timezone.utc
    ).strftime(
        "%Y%m%dT%H%M%SZ"
    )

    run_dir = ROOT / timestamp
    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata = {
        "created_at_utc": timestamp,
        "hypothesis_id": config.hypothesis_id,
        "question": config.question,
        "discovery_run": config.discovery_run,
        "discovery_end_date": (
            config.discovery_end_date.isoformat()
        ),
        "evaluation_start_date": (
            config.evaluation.start_date.isoformat()
        ),
        "evaluation_end_date": (
            config.evaluation.end_date.isoformat()
        ),
        "feature_rows": len(data.frame),
        "evaluation_rows": int(
            evaluation_mask.sum()
        ),
        "candidate": {
            "candidate_id": candidate.candidate_id,
            "target_name": candidate.target_name,
            "signal_name": candidate.signal_name,
            "signal_tail": candidate.signal_tail,
            "stress_feature": candidate.stress_feature,
            "stress_tail": candidate.stress_tail,
            "stress_direction": candidate.stress_direction,
        },
    }

    _write_json(
        run_dir / "result.json",
        result,
    )

    _write_json(
        run_dir / "metadata.json",
        metadata,
    )

    report = "\n".join(
        [
            "# Blankdiss Frozen Hypothesis OOS",
            "",
            f"- Hypothesis: `{config.hypothesis_id}`",
            f"- Candidate: `{candidate.candidate_id}`",
            f"- Discovery run: `{config.discovery_run}`",
            f"- Discovery end: `{config.discovery_end_date}`",
            f"- OOS start: `{config.evaluation.start_date}`",
            f"- OOS end: `{config.evaluation.end_date}`",
            f"- OOS rows: {result['n']}",
            f"- Events: {result['events']}",
            f"- Event rate: {result['event_rate']}",
            f"- Baseline event rate: {result['baseline_event_rate']}",
            f"- Lift: {result['lift']}",
            f"- Mean return: {result['mean_return']}",
            f"- Rest mean return: {result['rest_mean_return']}",
            f"- Return difference: {result['return_difference']}",
            "",
            "## Hypothesis",
            "",
            config.question,
            "",
        ]
    )

    (
        run_dir / "report.md"
    ).write_text(
        report,
        encoding="utf-8",
    )

    latest = ROOT / "latest"
    latest.mkdir(
        parents=True,
        exist_ok=True,
    )

    _write_json(
        latest / "result.json",
        result,
    )

    _write_json(
        latest / "metadata.json",
        metadata,
    )

    (
        latest / "report.md"
    ).write_text(
        report,
        encoding="utf-8",
    )

    print(
        "Frozen hypothesis OOS complete.",
        flush=True,
    )

    print(
        f"Result: {result}",
        flush=True,
    )


if __name__ == "__main__":
    run()
