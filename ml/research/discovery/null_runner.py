from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .config import DiscoveryConfig, load_discovery_config
from .engine import (
    DiscoveryData,
    build_candidates,
    evaluate_candidates,
    find_candidates,
    pool_results,
    prepare_data,
)
from .permutation import permute_discovery_data


DEFAULT_PERMUTATIONS = 1000
DEFAULT_SEED = 20260919
DEFAULT_BLOCK_SIZE = 1
DEFAULT_TOP_CANDIDATES = 25


def _env_int(
    name: str,
    default: int,
) -> int:
    value = os.environ.get(name)

    if value is None:
        return default

    try:
        result = int(value)
    except ValueError as exc:
        raise ValueError(
            f"{name} måste vara ett heltal."
        ) from exc

    if result < 1:
        raise ValueError(
            f"{name} måste vara >= 1."
        )

    return result


def _env_optional_int(
    name: str,
    default: int,
) -> int:
    value = os.environ.get(name)

    if value is None:
        return default

    try:
        result = int(value)
    except ValueError as exc:
        raise ValueError(
            f"{name} måste vara ett heltal."
        ) from exc

    if result < 0:
        raise ValueError(
            f"{name} måste vara >= 0."
        )

    return result


def _discovery_score(
    row: dict[str, Any],
) -> float:
    score = 0.0

    lift = row.get(
        "lift"
    )

    return_difference = row.get(
        "return_difference"
    )

    if lift is not None:
        score += (
            float(lift) - 1.0
        )

    if return_difference is not None:
        score += max(
            0.0,
            -float(return_difference),
        )

    return score


def _score_pooled(
    pooled: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    scored = []

    for row in pooled:
        scored.append(
            {
                **row,
                "discovery_score": (
                    _discovery_score(row)
                ),
            }
        )

    scored.sort(
        key=lambda row: (
            row["discovery_score"],
            row.get("lift") or 0.0,
        ),
        reverse=True,
    )

    return scored


def _compact_candidate(
    row: dict[str, Any],
) -> dict[str, Any]:
    return {
        "candidate_id": row[
            "candidate_id"
        ],
        "target_name": row[
            "target_name"
        ],
        "signal_name": row[
            "signal_name"
        ],
        "signal_tail": row[
            "signal_tail"
        ],
        "stress_feature": row[
            "stress_feature"
        ],
        "stress_tail": row[
            "stress_tail"
        ],
        "stress_direction": row[
            "stress_direction"
        ],
        "valid_windows": row[
            "valid_windows"
        ],
        "min_n": row[
            "min_n"
        ],
        "max_n": row[
            "max_n"
        ],
        "lift": row.get(
            "lift"
        ),
        "return_difference": row.get(
            "return_difference"
        ),
        "discovery_score": row[
            "discovery_score"
        ],
        "stable_lift_windows": row.get(
            "stable_lift_windows"
        ),
        "negative_return_windows": row.get(
            "negative_return_windows"
        ),
    }


def _top_candidates(
    pooled: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    scored = _score_pooled(
        pooled
    )

    return [
        _compact_candidate(row)
        for row in scored[:limit]
    ]


def _finding_candidates(
    pooled: list[dict[str, Any]],
    config: DiscoveryConfig,
) -> list[dict[str, Any]]:
    findings = find_candidates(
        pooled,
        config,
    )

    return [
        _compact_candidate(
            {
                **row,
                "discovery_score": (
                    _discovery_score(row)
                ),
            }
        )
        for row in findings
    ]


def _maximum_score(
    pooled: list[dict[str, Any]],
) -> float:
    if not pooled:
        raise ValueError(
            "Null-permutationen gav inga pooled candidates."
        )

    return max(
        _discovery_score(row)
        for row in pooled
    )


def _empirical_p_value(
    observed_score: float,
    null_scores: list[float],
) -> float:
    if not null_scores:
        return None

    exceedances = sum(
        score >= observed_score
        for score in null_scores
    )

    return (
        1.0 + exceedances
    ) / (
        1.0 + len(null_scores)
    )


def _percentile(
    values: list[float],
    percentile: float,
) -> float | None:
    if not values:
        return None

    return float(
        np.percentile(
            np.asarray(
                values,
                dtype=float,
            ),
            percentile,
        )
    )


def _write_json(
    path: Path,
    payload: Any,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ),
        encoding="utf-8",
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
    observed_top: list[dict[str, Any]],
    null_summaries: list[dict[str, Any]],
) -> str:
    lines = [
        "# Blankdiss Discovery Null Test",
        "",
        (
            f"Generated: "
            f"{metadata['created_at_utc']}"
        ),
        "",
        "## Null-test design",
        "",
        (
            f"- Permutations: "
            f"{metadata['permutations']:,}"
        ),
        (
            f"- Candidate space: "
            f"{metadata['candidate_count']:,}"
        ),
        (
            f"- Walk-forward windows: "
            f"{metadata['walk_forward_window_count']}"
        ),
        (
            f"- Block size: "
            f"{metadata['block_size']}"
        ),
        (
            f"- Seed: "
            f"{metadata['seed']}"
        ),
        "",
        (
            "Each permutation re-runs the complete Discovery "
            "candidate space. The maximum discovery score across "
            "all candidates is retained for the permutation."
        ),
        "",
        "## Observed result",
        "",
        (
            f"- Observed maximum score: "
            f"{metadata['observed_max_score']:.6f}"
        ),
        (
            f"- Null 95th percentile: "
            f"{metadata['null_p95']:.6f}"
        ),
        (
            f"- Null 99th percentile: "
            f"{metadata['null_p99']:.6f}"
        ),
        (
            f"- Empirical maximum-statistic p-value: "
            f"{metadata['empirical_p_value']:.6f}"
            if metadata["empirical_p_value"]
            is not None
            else (
                "- Empirical maximum-statistic p-value: "
                "not available"
            )
        ),
        "",
        "## Observed top candidates",
        "",
        (
            "| Candidate | Target | FI signal | Stress | "
            "Lift | Return diff | Score |"
        ),
        (
            "|---|---|---|---|---:|---:|---:|"
        ),
    ]

    for row in observed_top:
        lines.append(
            "| "
            f"{row['candidate_id']} | "
            f"{row['target_name']} | "
            f"{row['signal_name']} | "
            f"{row['stress_feature']} | "
            f"{_fmt(row.get('lift'))} | "
            f"{_fmt(row.get('return_difference'))} | "
            f"{_fmt(row.get('discovery_score'))} |"
        )

    lines.extend(
        [
            "",
            "## Null permutations",
            "",
            (
                "| Permutation | Max score | "
                "Top candidate | Passed threshold |"
            ),
            (
                "|---:|---:|---|---|"
            ),
        ]
    )

    for row in null_summaries:
        lines.append(
            "| "
            f"{row['permutation']} | "
            f"{row['max_score']:.6f} | "
            f"{row['top_candidate_id']} | "
            f"{row['top_candidate_passed_threshold']} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "The empirical p-value is based on the maximum "
                "discovery score across the complete candidate "
                "space for each permutation. It therefore accounts "
                "for the fact that Discovery searches 900 candidates "
                "and retains the strongest-looking result."
            ),
            "",
            (
                "This is a null-model diagnostic, not a guarantee "
                "that the permutation scheme is valid for every "
                "possible form of temporal dependence. Block "
                "permutations can be used to make the null more "
                "conservative when appropriate."
            ),
            "",
        ]
    )

    return "\n".join(lines)


def _fmt(
    value: Any,
) -> str:
    if value is None:
        return ""

    if isinstance(
        value,
        float,
    ):
        return f"{value:.6f}"

    return str(value)


def run() -> None:
    config = load_discovery_config()

    if not config.enabled:
        print(
            "Discovery disabled.",
            flush=True,
        )
        return

    permutations = _env_int(
        "DISCOVERY_NULL_PERMUTATIONS",
        DEFAULT_PERMUTATIONS,
    )

    seed = _env_optional_int(
        "DISCOVERY_NULL_SEED",
        DEFAULT_SEED,
    )

    block_size = _env_int(
        "DISCOVERY_NULL_BLOCK_SIZE",
        DEFAULT_BLOCK_SIZE,
    )

    top_candidates = _env_int(
        "DISCOVERY_NULL_TOP_CANDIDATES",
        DEFAULT_TOP_CANDIDATES,
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
        f"Block size: {block_size}",
        flush=True,
    )

    print(
        f"Seed: {seed}",
        flush=True,
    )

    print(
        "Preparing discovery data...",
        flush=True,
    )

    data = prepare_data(
        config
    )

    candidates = build_candidates(
        config
    )

    print(
        f"Candidate space: "
        f"{len(candidates):,}",
        flush=True,
    )

    if not candidates:
        raise RuntimeError(
            "Discovery candidate space är tom."
        )

    print(
        "Running observed Discovery pass...",
        flush=True,
    )

    observed_results = evaluate_candidates(
        data,
        candidates,
    )

    observed_pooled = pool_results(
        observed_results,
        min_rows_per_window=(
            config.validation
            .min_rows_per_window
        ),
    )

    observed_top = _top_candidates(
        observed_pooled,
        top_candidates,
    )

    observed_findings = (
        _finding_candidates(
            observed_pooled,
            config,
        )
    )

    observed_max_score = _maximum_score(
        observed_pooled
    )

    print(
        f"Observed maximum score: "
        f"{observed_max_score:.6f}",
        flush=True,
    )

    rng = np.random.default_rng(
        seed
    )

    null_scores: list[float] = []
    null_summaries: list[
        dict[str, Any]
    ] = []

    for permutation_number in range(
        1,
        permutations + 1,
    ):
        print(
            (
                f"Null permutation "
                f"{permutation_number:,}/"
                f"{permutations:,}"
            ),
            flush=True,
        )

        null_data = (
            permute_discovery_data(
                data=data,
                rng=rng,
                block_size=block_size,
            )
        )

        null_results = evaluate_candidates(
            null_data,
            candidates,
        )

        null_pooled = pool_results(
            null_results,
            min_rows_per_window=(
                config.validation
                .min_rows_per_window
            ),
        )

        max_score = _maximum_score(
            null_pooled
        )

        top = _top_candidates(
            null_pooled,
            1,
        )

        findings = _finding_candidates(
            null_pooled,
            config,
        )

        top_candidate = (
            top[0]
            if top
            else None
        )

        top_passed_threshold = bool(
            findings
            and top_candidate is not None
            and findings[0][
                "candidate_id"
            ]
            == top_candidate[
                "candidate_id"
            ]
        )

        null_scores.append(
            max_score
        )

        null_summaries.append(
            {
                "permutation": (
                    permutation_number
                ),
                "max_score": max_score,
                "top_candidate_id": (
                    top_candidate[
                        "candidate_id"
                    ]
                    if top_candidate
                    else None
                ),
                "top_candidate_score": (
                    top_candidate[
                        "discovery_score"
                    ]
                    if top_candidate
                    else None
                ),
                "top_candidate_passed_threshold": (
                    top_passed_threshold
                ),
                "finding_count": len(
                    findings
                ),
            }
        )

    empirical_p = _empirical_p_value(
        observed_score=(
            observed_max_score
        ),
        null_scores=null_scores,
    )

    metadata = {
        "created_at_utc": datetime.now(
            timezone.utc
        ).strftime(
            "%Y%m%dT%H%M%SZ"
        ),
        "permutations": permutations,
        "seed": seed,
        "block_size": block_size,
        "top_candidates": top_candidates,
        "candidate_count": len(
            candidates
        ),
        "feature_rows": len(
            data.frame
        ),
        "walk_forward_window_count": len(
            data.windows
        ),
        "observed_result_count": len(
            observed_results
        ),
        "observed_pooled_count": len(
            observed_pooled
        ),
        "observed_finding_count": len(
            observed_findings
        ),
        "observed_max_score": (
            observed_max_score
        ),
        "null_p95": _percentile(
            null_scores,
            95,
        ),
        "null_p99": _percentile(
            null_scores,
            99,
        ),
        "null_mean": float(
            np.mean(null_scores)
        ),
        "null_std": float(
            np.std(null_scores)
        ),
        "null_min": float(
            np.min(null_scores)
        ),
        "null_max": float(
            np.max(null_scores)
        ),
        "empirical_p_value": (
            empirical_p
        ),
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

    root = Path(
        "data/processed/ml/research/discovery-null"
    )

    timestamp = metadata[
        "created_at_utc"
    ]

    run_dir = root / timestamp

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    report = _build_report(
        metadata=metadata,
        observed_top=observed_top,
        null_summaries=null_summaries,
    )

    _write_json(
        run_dir / "metadata.json",
        metadata,
    )

    _write_json(
        run_dir / "observed_top.json",
        observed_top,
    )

    _write_json(
        run_dir / "observed_findings.json",
        observed_findings,
    )

    _write_json(
        run_dir / "null_summaries.json",
        null_summaries,
    )

    _write_json(
        run_dir / "null_scores.json",
        null_scores,
    )

    (run_dir / "report.md").write_text(
        report,
        encoding="utf-8",
    )

    latest_dir = root / "latest"

    if latest_dir.exists():
        shutil.rmtree(
            latest_dir
        )

    latest_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for filename in (
        "metadata.json",
        "observed_top.json",
        "observed_findings.json",
        "null_summaries.json",
        "null_scores.json",
        "report.md",
    ):
        source = run_dir / filename
        destination = (
            latest_dir / filename
        )

        if source.suffix == ".md":
            destination.write_text(
                source.read_text(
                    encoding="utf-8"
                ),
                encoding="utf-8",
            )
        else:
            destination.write_text(
                source.read_text(
                    encoding="utf-8"
                ),
                encoding="utf-8",
            )

    print(
        "",
        flush=True,
    )

    print(
        "=== Null test complete ===",
        flush=True,
    )

    print(
        f"Observed max score: "
        f"{observed_max_score:.6f}",
        flush=True,
    )

    print(
        f"Null P95: "
        f"{metadata['null_p95']:.6f}",
        flush=True,
    )

    print(
        f"Null P99: "
        f"{metadata['null_p99']:.6f}",
        flush=True,
    )

    print(
        f"Empirical p-value: "
        f"{empirical_p:.6f}"
        if empirical_p is not None
        else "Empirical p-value: unavailable",
        flush=True,
    )

    print(
        f"Output: {run_dir}",
        flush=True,
    )


if __name__ == "__main__":
    run()
