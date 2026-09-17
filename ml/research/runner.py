from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path

from ml.config import WALK_FORWARD_WINDOWS
from ml.dataset import load_features

from .cache import ResearchCache
from .evaluator import evaluate_experiment
from .experiments import (
    build_experiment_matrix,
)
from .reporting import (
    build_markdown_report,
    build_summary,
    write_json,
    write_jsonl,
)


ROOT = Path(__file__).resolve().parents[2]

RESEARCH_DIR = (
    ROOT
    / "data"
    / "processed"
    / "research"
)

RUNS_DIR = RESEARCH_DIR / "runs"
LATEST_DIR = RESEARCH_DIR / "latest"


def main() -> None:
    started_at = datetime.now(
        timezone.utc
    )

    print("Loading features...")
    frame = load_features()

    print(
        f"Loaded {len(frame):,} feature rows"
    )

    experiments = build_experiment_matrix()

    signal_names = sorted(
        {
            experiment.signal_name
            for experiment in experiments
        }
    )

    target_names = sorted(
        {
            experiment.target_name
            for experiment in experiments
        }
    )

    print(
        f"Experiments: {len(experiments):,}"
    )

    print(
        f"Signals: {len(signal_names):,}"
    )

    print(
        f"Targets: {len(target_names):,}"
    )

    # ------------------------------------------------------------
    # Expensive preprocessing happens once.
    # ------------------------------------------------------------

    print("Building research cache...")

    cache = ResearchCache.build(
        frame,
        signal_names,
        target_names,
    )

    print("Research cache ready.")

    # ------------------------------------------------------------
    # Experiment evaluation
    # ------------------------------------------------------------

    results = []

    total = (
        len(experiments)
        * len(WALK_FORWARD_WINDOWS)
    )

    completed = 0

    for window in WALK_FORWARD_WINDOWS:
        train_end = window["train_end"]
        validation_end = window[
            "validation_end"
        ]
        test_end = window["test_end"]

        print(
            f"Evaluating window "
            f"{train_end} → {test_end}"
        )

        for experiment in experiments:
            result = evaluate_experiment(
                frame,
                experiment,
                cache,
                train_end=train_end,
                validation_end=validation_end,
                test_end=test_end,
            )

            results.append(result)

            completed += 1

            if (
                completed % 100 == 0
                or completed == total
            ):
                print(
                    f"Progress: "
                    f"{completed:,}/{total:,}"
                )

    # ------------------------------------------------------------
    # Pool results across walk-forward windows.
    # ------------------------------------------------------------

    pooled_results = pool_results(
        results
    )

    summary = build_summary(
        results,
        pooled_results,
    )

    report = build_markdown_report(
        summary,
        pooled_results,
    )

    run_id = started_at.strftime(
        "%Y%m%dT%H%M%SZ"
    )

    run_dir = RUNS_DIR / run_id

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_jsonl(
        run_dir / "experiment_results.jsonl",
        results,
    )

    write_jsonl(
        run_dir / "pooled_results.jsonl",
        pooled_results,
    )

    write_json(
        run_dir / "research_summary.json",
        summary,
    )

    (run_dir / "research_report.md").write_text(
        report,
        encoding="utf-8",
    )

    # ------------------------------------------------------------
    # Latest
    # ------------------------------------------------------------

    if LATEST_DIR.exists():
        shutil.rmtree(LATEST_DIR)

    shutil.copytree(
        run_dir,
        LATEST_DIR,
    )

    print("")
    print("Research completed.")
    print(
        f"Results: {len(results):,}"
    )
    print(
        f"Pooled: {len(pooled_results):,}"
    )
    print(
        f"Output: {run_dir}"
    )


def pool_results(
    results: list[dict],
) -> list[dict]:
    """
    Pool walk-forward results by experiment_id.

    This intentionally keeps the pooling simple and deterministic.
    """

    grouped: dict[str, list[dict]] = {}

    for result in results:
        experiment_id = result[
            "experiment_id"
        ]

        grouped.setdefault(
            experiment_id,
            [],
        ).append(result)

    pooled = []

    for experiment_id, rows in grouped.items():
        pooled.append(
            _pool_experiment(
                experiment_id,
                rows,
            )
        )

    return pooled


def _pool_experiment(
    experiment_id: str,
    rows: list[dict],
) -> dict:
    first = rows[0]

    auc_values = [
        row["auc"]
        for row in rows
        if row.get("auc") is not None
    ]

    hit_values = [
        row["hit_rate"]
        for row in rows
        if row.get("hit_rate") is not None
    ]

    return_values = [
        row["return_difference"]
        for row in rows
        if row.get("return_difference")
        is not None
    ]

    return {
        "experiment_id": experiment_id,
        "signal_name": first[
            "signal_name"
        ],
        "target_name": first[
            "target_name"
        ],
        "tail_fraction": first[
            "tail_fraction"
        ],
        "tail_direction": first[
            "tail_direction"
        ],
        "windows": len(rows),
        "auc": _mean(auc_values),
        "hit_rate": _mean(hit_values),
        "return_difference": _mean(
            return_values
        ),
        "status": _pooled_status(rows),
    }


def _pooled_status(
    rows: list[dict],
) -> str:
    statuses = {
        row.get("status")
        for row in rows
    }

    if (
        "STRONG RESEARCH CANDIDATE"
        in statuses
    ):
        return "STRONG RESEARCH CANDIDATE"

    if "INTERESTING" in statuses:
        return "INTERESTING"

    if "INSUFFICIENT_DATA" in statuses:
        return "INSUFFICIENT_DATA"

    return "NO SIGNAL"


def _mean(
    values: list[float],
) -> float | None:
    if not values:
        return None

    return sum(values) / len(values)


if __name__ == "__main__":
    main()
