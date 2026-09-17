"""Entry point for the Blankdiss deterministic research matrix."""

from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path

from ml.config import WALK_FORWARD_WINDOWS
from ml.dataset import load_features

from .aggregation import pool_results
from .cache import ResearchCache
from .evaluator import evaluate_experiment
from .experiments import build_experiment_matrix
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

RUNS_DIR = (
    RESEARCH_DIR
    / "runs"
)

LATEST_DIR = (
    RESEARCH_DIR
    / "latest"
)


def main() -> None:
    started_at = datetime.now(
        timezone.utc
    )

    # ---------------------------------------------------------
    # Load
    # ---------------------------------------------------------

    print(
        "Loading features..."
    )

    frame = load_features()

    print(
        f"Loaded {len(frame):,} feature rows"
    )

    # ---------------------------------------------------------
    # Experiment matrix
    # ---------------------------------------------------------

    experiments = (
        build_experiment_matrix()
    )

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

    # ---------------------------------------------------------
    # Precompute
    #
    # This is where the expensive pandas work happens.
    # It happens once.
    # ---------------------------------------------------------

    print(
        "Building research cache..."
    )

    cache = ResearchCache.build(
        frame,
        signal_names,
        target_names,
    )

    print(
        "Research cache ready."
    )

    # ---------------------------------------------------------
    # Evaluate
    # ---------------------------------------------------------

    results = []

    window_count = len(
        WALK_FORWARD_WINDOWS
    )

    total = (
        len(experiments)
        * window_count
    )

    completed = 0

    for window in WALK_FORWARD_WINDOWS:
        train_end = window[
            "train_end"
        ]

        validation_end = window[
            "validation_end"
        ]

        test_end = window[
            "test_end"
        ]

        print(
            ""
        )

        print(
            "Evaluating window "
            f"{train_end} → {test_end}"
        )

        for experiment in experiments:
            results.append(
                evaluate_experiment(
                    frame,
                    experiment,
                    cache,
                    train_end=train_end,
                    validation_end=validation_end,
                    test_end=test_end,
                )
            )

            completed += 1

            if (
                completed % 100 == 0
                or completed == total
            ):
                print(
                    "Progress: "
                    f"{completed:,}/"
                    f"{total:,}"
                )

    # ---------------------------------------------------------
    # Aggregate
    # ---------------------------------------------------------

    print(
        ""
    )

    print(
        "Pooling walk-forward results..."
    )

    pooled_results = pool_results(
        results
    )

    # ---------------------------------------------------------
    # Reports
    # ---------------------------------------------------------

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
        run_dir
        / "experiment_results.jsonl",
        results,
    )

    write_jsonl(
        run_dir
        / "pooled_results.jsonl",
        pooled_results,
    )

    write_json(
        run_dir
        / "research_summary.json",
        summary,
    )

    (
        run_dir
        / "research_report.md"
    ).write_text(
        report,
        encoding="utf-8",
    )

    # ---------------------------------------------------------
    # Latest
    # ---------------------------------------------------------

    if LATEST_DIR.exists():
        shutil.rmtree(
            LATEST_DIR
        )

    shutil.copytree(
        run_dir,
        LATEST_DIR,
    )

    # ---------------------------------------------------------
    # Done
    # ---------------------------------------------------------

    print(
        ""
    )

    print(
        "Research completed."
    )

    print(
        f"Window results: "
        f"{len(results):,}"
    )

    print(
        f"Pooled experiments: "
        f"{len(pooled_results):,}"
    )

    print(
        f"Output: {run_dir}"
    )


if __name__ == "__main__":
    main()
