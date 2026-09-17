from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from ml.config import WALK_FORWARD_WINDOWS
from ml.dataset import load_features
from ml.research.aggregation import aggregate_results
from ml.research.cache import build_research_cache
from ml.research.evaluator import evaluate_experiment
from ml.research.experiments import build_experiment_matrix
from ml.research.reporting import (
    write_json,
    write_jsonl,
    write_markdown_report,
)


ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = ROOT / "data" / "processed" / "ml" / "research"


def _window_name(index: int) -> str:
    return f"window_{index + 1}"


def run() -> None:
    print("Loading features...", flush=True)

    frame = load_features()

    print(
        f"Loaded {len(frame):,} feature rows",
        flush=True,
    )

    experiments = build_experiment_matrix()

    print(
        f"Experiments: {len(experiments):,}",
        flush=True,
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
        f"Signals: {len(signal_names):,}",
        flush=True,
    )

    print(
        f"Targets: {len(target_names):,}",
        flush=True,
    )

    print(
        "Building research cache...",
        flush=True,
    )

    cache = build_research_cache(
        frame,
        experiments,
    )

    print(
        "Research cache ready.",
        flush=True,
    )

    results: list[dict] = []

    for window_index, _window in enumerate(WALK_FORWARD_WINDOWS):
        window_name = _window_name(window_index)

        # WalkForwardWindow is a dataclass/object.
        # Keep this explicit here so the runner never assumes dictionary access.
        train_end = _window.train_end
        validation_end = _window.validation_end
        test_end = _window.test_end

        print(
            f"\n{window_name}: "
            f"train <= {train_end}, "
            f"validation <= {validation_end}, "
            f"test <= {test_end}",
            flush=True,
        )

        for split_name in (
            "train",
            "validation",
            "test",
        ):
            split_results = []

            for experiment in experiments:
                result = evaluate_experiment(
                    frame=frame,
                    cache=cache,
                    experiment=experiment,
                    window_name=window_name,
                    split_name=split_name,
                )

                split_results.append(result)

            results.extend(split_results)

            print(
                f"  {split_name}: "
                f"{len(split_results):,} experiments",
                flush=True,
            )

    print(
        f"\nRaw results: {len(results):,}",
        flush=True,
    )

    pooled = aggregate_results(results)

    print(
        f"Pooled experiments: {len(pooled):,}",
        flush=True,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    run_timestamp = datetime.now(
        timezone.utc,
    ).strftime(
        "%Y%m%dT%H%M%SZ",
    )

    run_dir = OUTPUT_DIR / run_timestamp
    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata = {
        "created_at_utc": run_timestamp,
        "feature_rows": int(len(frame)),
        "experiments": int(len(experiments)),
        "signals": signal_names,
        "targets": target_names,
        "walk_forward_windows": [
            {
                "train_end": str(window.train_end),
                "validation_end": str(window.validation_end),
                "test_end": str(window.test_end),
            }
            for window in WALK_FORWARD_WINDOWS
        ],
    }

    write_jsonl(
        run_dir / "results.jsonl",
        results,
    )

    write_json(
        run_dir / "pooled.json",
        pooled,
    )

    write_json(
        run_dir / "metadata.json",
        metadata,
    )

    write_markdown_report(
        run_dir / "report.md",
        pooled,
        metadata,
    )

    latest_dir = OUTPUT_DIR / "latest"
    latest_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_jsonl(
        latest_dir / "results.jsonl",
        results,
    )

    write_json(
        latest_dir / "pooled.json",
        pooled,
    )

    write_json(
        latest_dir / "metadata.json",
        metadata,
    )

    write_markdown_report(
        latest_dir / "report.md",
        pooled,
        metadata,
    )

    print(
        f"\nResearch complete: {run_dir}",
        flush=True,
    )


if __name__ == "__main__":
    run()
