from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import load_discovery_config
from .permutation import (
    build_candidate_masks,
    empirical_p_value,
    evaluate_observed,
    run_permutations,
)


ROOT = Path(
    "data/processed/ml/research/discovery_null"
)


def _write_json(
    path: Path,
    payload: Any,
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
        )


def _write_jsonl(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            )


def _build_report(
    metadata: dict[str, Any],
    observed: dict[str, Any],
    null_distribution: list[dict[str, Any]],
) -> str:
    observed_score = observed[
        "max_discovery_score"
    ]

    p_value = metadata[
        "empirical_p_value"
    ]

    best = observed[
        "best_candidate"
    ]

    lines = [
        "# Blankdiss Discovery Null Test",
        "",
        (
            f"Generated: "
            f"{metadata['created_at_utc']}"
        ),
        "",
        "## Summary",
        "",
        (
            f"- Candidates per run: "
            f"{metadata['candidate_count']:,}"
        ),
        (
            f"- Permutations: "
            f"{metadata['permutations']:,}"
        ),
        (
            f"- Seed: "
            f"{metadata['seed']}"
        ),
        (
            f"- Observed max discovery score: "
            f"{observed_score}"
        ),
        (
            f"- Empirical p-value: "
            f"{p_value}"
        ),
        "",
        "## Observed best candidate",
        "",
    ]

    if best is None:
        lines.append(
            "No candidate passed the configured discovery thresholds."
        )
    else:
        lines.extend(
            [
                f"- Candidate: {best['candidate_id']}",
                f"- Target: {best['target_name']}",
                f"- Signal: {best['signal_name']}",
                f"- Signal tail: {best['signal_tail']}",
                f"- Stress feature: {best['stress_feature']}",
                f"- Stress tail: {best['stress_tail']}",
                f"- Stress direction: {best['stress_direction']}",
                f"- Lift: {best.get('lift')}",
                (
                    "- Return difference: "
                    f"{best.get('return_difference')}"
                ),
                (
                    "- Valid windows: "
                    f"{best.get('valid_windows')}"
                ),
                (
                    "- Discovery score: "
                    f"{best.get('discovery_score')}"
                ),
            ]
        )

    values = [
        row["max_discovery_score"]
        for row in null_distribution
        if row["max_discovery_score"] is not None
    ]

    lines.extend(
        [
            "",
            "## Null distribution",
            "",
            (
                f"- Valid null scores: "
                f"{len(values):,}"
            ),
        ]
    )

    if values:
        values_sorted = sorted(
            values
        )

        def percentile(
            fraction: float,
        ) -> float:
            index = min(
                int(
                    fraction
                    * (len(values_sorted) - 1)
                ),
                len(values_sorted) - 1,
            )

            return float(
                values_sorted[index]
            )

        lines.extend(
            [
                (
                    f"- Median: "
                    f"{percentile(0.50)}"
                ),
                (
                    f"- 95th percentile: "
                    f"{percentile(0.95)}"
                ),
                (
                    f"- 99th percentile: "
                    f"{percentile(0.99)}"
                ),
                (
                    f"- Maximum: "
                    f"{max(values)}"
                ),
            ]
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "The null test repeats the complete discovery "
                "candidate search after randomly permuting the "
                "outcome series while keeping signals, stress "
                "features and candidate definitions unchanged."
            ),
            "",
            (
                "For each permutation, the maximum discovery "
                "score across the full candidate space is retained. "
                "The empirical p-value therefore measures how often "
                "a discovery at least as strong as the observed one "
                "appears under the permutation null."
            ),
            "",
            (
                "This is a multiple-testing/null-distribution "
                "diagnostic, not confirmation that a discovered "
                "candidate is a causal or tradable signal."
            ),
            "",
        ]
    )

    return "\n".join(
        lines
    )


def _metadata(
    *,
    timestamp: str,
    feature_rows: int,
    candidate_count: int,
    permutations: int,
    seed: int,
    config: Any,
    observed: dict[str, Any],
    p_value: float | None,
) -> dict[str, Any]:
    return {
        "created_at_utc": timestamp,
        "feature_rows": feature_rows,
        "candidate_count": candidate_count,
        "permutations": permutations,
        "seed": seed,
        "observed_max_discovery_score": (
            observed[
                "max_discovery_score"
            ]
        ),
        "empirical_p_value": p_value,
        "targets": list(
            config.targets
        ),
        "signals": list(
            config.signals
        ),
        "stress_features": list(
            config.stress_features
        ),
        "tails": list(
            config.tails
        ),
        "stress_directions": dict(
            config.stress_directions
        ),
        "validation": {
            "min_rows_per_window": (
                config.validation
                .min_rows_per_window
            ),
            "min_positive_windows": (
                config.validation
                .min_positive_windows
            ),
            "min_lift": (
                config.validation
                .min_lift
            ),
            "max_return_difference": (
                config.validation
                .max_return_difference
            ),
            "max_findings": (
                config.validation
                .max_findings
            ),
        },
    }


def run() -> None:
    config = load_discovery_config()

    if not config.enabled:
        print(
            "Discovery disabled.",
            flush=True,
        )
        return

    permutations = int(
        os.environ.get(
            "DISCOVERY_NULL_PERMUTATIONS",
            "1000",
        )
    )

    seed = int(
        os.environ.get(
            "DISCOVERY_NULL_SEED",
            "42",
        )
    )

    print(
        "=== Blankdiss Discovery Null Test ===",
        flush=True,
    )

    print(
        f"Permutations: {permutations:,}",
        flush=True,
    )

    print(
        f"Seed: {seed}",
        flush=True,
    )

    # Importeras här så att null-runnern använder exakt samma
    # feature loading och walk-forward data som Discovery V1.
    from .engine import prepare_data

    print(
        "Preparing discovery data...",
        flush=True,
    )

    data = prepare_data(
        config
    )

    (
        candidates,
        candidate_masks,
    ) = build_candidate_masks(
        data,
        config,
    )

    print(
        f"Discovery candidates: "
        f"{len(candidates):,}",
        flush=True,
    )

    expected_candidates = (
        len(config.targets)
        * len(config.signals)
        * len(config.stress_features)
        * len(config.tails)
        * len(config.tails)
    )

    if len(candidates) != expected_candidates:
        raise RuntimeError(
            "Unexpected candidate count: "
            f"expected {expected_candidates}, "
            f"got {len(candidates)}"
        )

    observed = evaluate_observed(
        data,
        config,
        candidate_masks,
    )

    print(
        "Observed discovery score: "
        f"{observed['max_discovery_score']}",
        flush=True,
    )

    null_distribution = run_permutations(
        data=data,
        config=config,
        candidate_masks=candidate_masks,
        permutations=permutations,
        seed=seed,
    )

    p_value = empirical_p_value(
        observed[
            "max_discovery_score"
        ],
        null_distribution,
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

    metadata = _metadata(
        timestamp=timestamp,
        feature_rows=len(
            data.frame
        ),
        candidate_count=len(
            candidates
        ),
        permutations=permutations,
        seed=seed,
        config=config,
        observed=observed,
        p_value=p_value,
    )

    report = _build_report(
        metadata,
        observed,
        null_distribution,
    )

    _write_json(
        run_dir / "observed_pooled.json",
        observed["pooled"],
    )

    _write_json(
        run_dir / "observed_findings.json",
        observed["findings"],
    )

    _write_json(
        run_dir / "observed_best.json",
        observed["best_candidate"],
    )

    _write_jsonl(
        run_dir / "permutations.jsonl",
        null_distribution,
    )

    _write_json(
        run_dir / "null_distribution.json",
        {
            "scores": [
                row["max_discovery_score"]
                for row in null_distribution
                if row["max_discovery_score"]
                is not None
            ],
            "valid_permutations": sum(
                1
                for row in null_distribution
                if row["max_discovery_score"]
                is not None
            ),
            "permutations": permutations,
        },
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

    latest_dir = ROOT / "latest"

    if latest_dir.exists():
        shutil.rmtree(
            latest_dir
        )

    latest_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for filename in (
        "observed_pooled.json",
        "observed_findings.json",
        "observed_best.json",
        "permutations.jsonl",
        "null_distribution.json",
        "metadata.json",
        "report.md",
    ):
        source = (
            run_dir / filename
        )
        target = (
            latest_dir / filename
        )

        shutil.copy2(
            source,
            target,
        )

    print(
        "Null test complete.",
        flush=True,
    )

    print(
        f"Run directory: {run_dir}",
        flush=True,
    )

    print(
        f"Empirical p-value: {p_value}",
        flush=True,
    )


if __name__ == "__main__":
    run()
