from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

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
from .null_test import run_frozen_null_test


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


def _build_report(
    config,
    candidate: Candidate,
    result: dict,
    null_result: dict,
) -> str:
    lines = [
        "# Blankdiss Frozen Hypothesis",
        "",
        "## Hypothesis",
        "",
        config.question,
        "",
        "## Frozen candidate",
        "",
        f"- Candidate: `{candidate.candidate_id}`",
        f"- Target: `{candidate.target_name}`",
        f"- Signal: `{candidate.signal_name}`",
        f"- Signal tail: `{candidate.signal_tail}`",
        f"- Stress feature: `{candidate.stress_feature}`",
        f"- Stress tail: `{candidate.stress_tail}`",
        f"- Stress direction: `{candidate.stress_direction}`",
        "",
        "## Discovery boundary",
        "",
        f"- Discovery run: `{config.discovery_run}`",
        f"- Discovery end: `{config.discovery_end_date}`",
        "",
        "## OOS evaluation",
        "",
        f"- Start: `{config.evaluation.start_date}`",
        f"- End: `{config.evaluation.end_date}`",
        f"- Rows: {result['n']}",
        f"- Events: {result['events']}",
        f"- Event rate: {result['event_rate']}",
        f"- Baseline event rate: {result['baseline_event_rate']}",
        f"- Lift: {result['lift']}",
        f"- Mean return: {result['mean_return']}",
        f"- Rest mean return: {result['rest_mean_return']}",
        f"- Return difference: {result['return_difference']}",
        "",
        "## Frozen null test",
        "",
        f"- Metric: `{null_result['metric']}`",
        f"- Permutations: {null_result['permutations_requested']:,}",
        f"- Valid permutations: {null_result['permutations_valid']:,}",
        f"- Seed: {null_result['seed']}",
        f"- Observed: {null_result['observed']}",
        f"- Null mean: {null_result['null_mean']}",
        f"- Null std: {null_result['null_std']}",
        f"- Null 95th percentile: {null_result['null_percentile_95']}",
        f"- Null 99th percentile: {null_result['null_percentile_99']}",
        f"- Empirical p-value: {null_result['p_value']}",
        "",
        "## Method",
        "",
        (
            "The candidate was frozen before the OOS evaluation. "
            "The null test keeps the candidate, signal and stress "
            "definition fixed and permutes only the outcome within "
            "the OOS period."
        ),
        "",
        (
            "No candidate search or parameter selection is performed "
            "during the frozen null test."
        ),
        "",
    ]

    return "\n".join(lines)


def run() -> None:
    config = load_config()

    discovery_config = _build_discovery_config(config)

    data = prepare_data(
        discovery_config
    )

    dates = pd.to_datetime(
        data.frame["snapshot_date"],
        errors="coerce",
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

    candidate = _build_candidate(config)

    result = evaluate_candidate_on_mask(
        data=data,
        candidate=candidate,
        base_mask=evaluation_mask,
        split="oos",
    )

    null_result = run_frozen_null_test(
        data=data,
        candidate=candidate,
        oos_mask=evaluation_mask,
        observed_result=result,
        permutations=config.null_test.permutations,
        seed=config.null_test.seed,
        metric=config.null_test.metric,
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
        "null_test": {
            "metric": config.null_test.metric,
            "permutations": config.null_test.permutations,
            "seed": config.null_test.seed,
        },
    }

    report = _build_report(
        config=config,
        candidate=candidate,
        result=result,
        null_result=null_result,
    )

    _write_json(
        run_dir / "result.json",
        result,
    )

    _write_json(
        run_dir / "null_test.json",
        null_result,
    )

    _write_json(
        run_dir / "metadata.json",
        metadata,
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
        latest / "null_test.json",
        null_result,
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


if __name__ == "__main__":
    run()
