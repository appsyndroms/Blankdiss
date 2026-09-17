from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def write_jsonl(
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


def write_json(
    path: Path,
    payload: dict[str, Any] | list[Any],
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


def build_summary(
    results: list[dict[str, Any]],
    pooled_results: list[dict[str, Any]],
) -> dict[str, Any]:
    statuses: dict[str, int] = {}

    for result in pooled_results:
        status = result.get(
            "status",
            "UNKNOWN",
        )

        statuses[status] = (
            statuses.get(status, 0) + 1
        )

    return {
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "result_count": len(results),
        "pooled_result_count": len(
            pooled_results
        ),
        "statuses": statuses,
    }


def build_markdown_report(
    summary: dict[str, Any],
    pooled_results: list[dict[str, Any]],
) -> str:
    lines = [
        "# Blankdiss Research Report",
        "",
        f"Generated: {summary['generated_at']}",
        "",
        f"Split results: {summary['result_count']}",
        f"Pooled experiments: {summary['pooled_result_count']}",
        "",
        "## Status",
        "",
    ]

    for status, count in sorted(
        summary["statuses"].items()
    ):
        lines.append(
            f"- {status}: {count}"
        )

    lines.extend(
        [
            "",
            "## Pooled Results",
            "",
            "| Experiment | Target | Signal | Tail | Direction | AUC | Event rate | Baseline | Lift | Mean return | Median return | N | Status |",
            "|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )

    for result in pooled_results:
        lines.append(
            "| "
            f"{result.get('experiment_id', '')} | "
            f"{result.get('target_name', '')} | "
            f"{result.get('signal_name', '')} | "
            f"{result.get('tail_fraction', '')} | "
            f"{result.get('tail_direction', '')} | "
            f"{_fmt(result.get('auc'))} | "
            f"{_fmt(result.get('event_rate'))} | "
            f"{_fmt(result.get('baseline_event_rate'))} | "
            f"{_fmt(result.get('lift'))} | "
            f"{_fmt(result.get('mean_return'))} | "
            f"{_fmt(result.get('median_return'))} | "
            f"{result.get('n', '')} | "
            f"{result.get('status', '')} |"
        )

    return "\n".join(lines) + "\n"


def write_markdown_report(
    path: Path,
    pooled_results: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> None:
    summary = build_summary(
        metadata.get(
            "results",
            [],
        ),
        pooled_results,
    )

    report = build_markdown_report(
        summary,
        pooled_results,
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        report,
        encoding="utf-8",
    )


def _fmt(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, float):
        return f"{value:.4f}"

    return str(value)
