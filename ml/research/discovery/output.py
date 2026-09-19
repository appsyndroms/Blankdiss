from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ml.config import WALK_FORWARD_WINDOWS

from .config import DiscoveryConfig
from .engine import Candidate


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


def _fmt(
    value: Any,
) -> str:
    if value is None:
        return ""

    if isinstance(
        value,
        float,
    ):
        return f"{value:.4f}"

    return str(value)


def _build_report(
    results: list[dict[str, Any]],
    pooled: list[dict[str, Any]],
    findings: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> str:
    lines = [
        "# Blankdiss Discovery",
        "",
        f"Generated: {metadata['created_at_utc']}",
        "",
        "## Summary",
        "",
        f"- Candidates: {metadata['candidate_count']:,}",
        f"- OOS results: {len(results):,}",
        f"- Pooled candidates: {len(pooled):,}",
        f"- Findings: {len(findings):,}",
        "",
        "## Findings",
        "",
    ]

    if not findings:
        lines.append(
            "No candidates passed the configured discovery thresholds."
        )
    else:
        lines.extend(
            [
                "| Candidate | Target | FI signal | FI tail | Stress | Stress tail | Lift | Return diff | Valid windows |",
                "|---|---|---|---:|---|---:|---:|---:|---:|",
            ]
        )

        for row in findings:
            lines.append(
                "| "
                f"{row['candidate_id']} | "
                f"{row['target_name']} | "
                f"{row['signal_name']} | "
                f"{row['signal_tail']} | "
                f"{row['stress_feature']} | "
                f"{row['stress_tail']} | "
                f"{_fmt(row.get('lift'))} | "
                f"{_fmt(row.get('return_difference'))} | "
                f"{row.get('valid_windows', '')} |"
            )

    lines.extend(
        [
            "",
            "## Candidate Space",
            "",
            "| Dimension | Values |",
            "|---|---|",
            (
                "| Targets | "
                f"{', '.join(metadata['targets'])} |"
            ),
            (
                "| FI signals | "
                f"{', '.join(metadata['signals'])} |"
            ),
            (
                "| Stress features | "
                f"{', '.join(metadata['stress_features'])} |"
            ),
            (
                "| Tail fractions | "
                f"{', '.join(map(str, metadata['tails']))} |"
            ),
            "",
            "## Walk-forward Windows",
            "",
        ]
    )

    for window in metadata[
        "walk_forward_windows"
    ]:
        lines.append(
            "- "
            f"{window['name']}: "
            f"train <= {window['train_end']}, "
            f"validation <= {window['validation_end']}, "
            f"test <= {window['test_end']}"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "Discovery is exploratory. Findings are candidates "
                "for subsequent diagnostics, not confirmed trading signals."
            ),
            "",
        ]
    )

    return "\n".join(lines)


def write_results(
    *,
    config: DiscoveryConfig,
    candidates: list[Candidate],
    results: list[dict[str, Any]],
    pooled: list[dict[str, Any]],
    findings: list[dict[str, Any]],
    feature_rows: int,
) -> Path:
    root = Path(
        "data/processed/ml/research/discovery"
    )

    timestamp = datetime.now(
        timezone.utc
    ).strftime(
        "%Y%m%dT%H%M%SZ"
    )

    run_dir = root / timestamp

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata = {
        "created_at_utc": timestamp,
        "feature_rows": feature_rows,
        "candidate_count": len(candidates),
        "result_count": len(results),
        "pooled_count": len(pooled),
        "finding_count": len(findings),
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
        "walk_forward_windows": [
            {
                "name": f"window_{index}",
                "train_end": str(
                    window.train_end
                ),
                "validation_end": str(
                    window.validation_end
                ),
                "test_end": str(
                    window.test_end
                ),
            }
            for index, window
            in enumerate(
                WALK_FORWARD_WINDOWS,
                start=1,
            )
        ],
    }

    report = _build_report(
        results,
        pooled,
        findings,
        metadata,
    )

    _write_jsonl(
        run_dir / "results.jsonl",
        results,
    )

    _write_json(
        run_dir / "pooled.json",
        pooled,
    )

    _write_json(
        run_dir / "findings.json",
        findings,
    )

    _write_json(
        run_dir / "metadata.json",
        metadata,
    )

    (run_dir / "report.md").write_text(
        report,
        encoding="utf-8",
    )

    latest_dir = root / "latest"

    _write_jsonl(
        latest_dir / "results.jsonl",
        results,
    )

    _write_json(
        latest_dir / "pooled.json",
        pooled,
    )

    _write_json(
        latest_dir / "findings.json",
        findings,
    )

    _write_json(
        latest_dir / "metadata.json",
        metadata,
    )

    (latest_dir / "report.md").write_text(
        report,
        encoding="utf-8",
    )

    return run_dir
