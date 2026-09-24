from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import pandas as pd

from ml.research.engine import run_spec
from ml.research.evaluator import evaluate_experiment
from ml.research.experiments import (
    TAIL_FRACTIONS,
    TARGET_NAMES,
    SIGNAL_SPECS,
    Experiment,
)
from ml.research.session import build_session
from ml.research.spec import (
    AnalysisSpec,
    ResearchSpec,
    SignalSpec,
)


ROOT = Path(__file__).resolve().parents[2]

DEFAULT_OUTPUT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "ml"
    / "research"
    / "migration"
)

COMMON_FIELDS = (
    "n",
    "events",
    "event_rate",
    "lift",
    "mean_return",
)


def _build_migration_spec() -> ResearchSpec:
    """
    Build one declarative spec representing exactly the old
    generic evaluator experiment matrix.

    The old matrix allows multiple directions for some signals,
    so the same signal name intentionally occurs more than once
    with different directions.
    """
    signals: list[SignalSpec] = []

    for signal_name, directions in SIGNAL_SPECS:
        for direction in directions:
            signals.append(
                SignalSpec(
                    name=signal_name,
                    direction=direction,
                    bins=TAIL_FRACTIONS,
                )
            )

    return ResearchSpec(
        id="generic_migration_comparison",
        question=(
            "Numerical migration comparison between the old "
            "generic evaluator and the new Research Engine."
        ),
        signals=tuple(signals),
        targets=TARGET_NAMES,
        analysis=AnalysisSpec(
            type="tail",
            bootstrap=False,
        ),
        mode="scan",
        windows=(
            "window_1",
            "window_2",
        ),
        splits=("test",),
        metadata={
            "purpose": "old_new_generic_evaluator_migration",
            "bootstrap_compared": False,
        },
    )


def _old_results(
    session,
    experiments: list[Experiment],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for experiment in experiments:
        for window_name in session.cache.window_masks:
            for split_name in session.cache.window_masks[
                window_name
            ]:
                result = evaluate_experiment(
                    session.frame,
                    session.cache,
                    experiment,
                    window_name,
                    split_name,
                )

                rows.append(
                    {
                        "experiment_id": (
                            experiment.experiment_id
                        ),
                        "signal": (
                            experiment.signal_name
                        ),
                        "direction": (
                            experiment.tail_direction
                        ),
                        "fraction": (
                            experiment.tail_fraction
                        ),
                        "target": (
                            experiment.target_name
                        ),
                        "window": window_name,
                        "split": split_name,
                        **{
                            field: result.get(field)
                            for field in COMMON_FIELDS
                        },
                    }
                )

    return rows


def _new_results(
    session,
    spec: ResearchSpec,
) -> list[dict[str, Any]]:
    result = run_spec(
        session.cache,
        spec,
    )

    rows: list[dict[str, Any]] = []

    for row in result["results"]:
        rows.append(
            {
                "experiment_id": (
                    _experiment_id(
                        row["signal"],
                        row["direction"],
                        row["fraction"],
                        row["target"],
                    )
                ),
                "signal": row["signal"],
                "direction": row["direction"],
                "fraction": row["fraction"],
                "target": row["target"],
                "window": row["window"],
                "split": row["split"],
                "n": row["n"],
                "events": row["events"],
                "event_rate": row["event_rate"],
                "lift": row["lift"],
                "mean_return": row["mean_return"],
            }
        )

    return rows


def _fraction_name(
    fraction: float,
) -> str:
    if fraction == 0.20:
        return "20pct"

    if fraction == 0.10:
        return "10pct"

    if fraction == 0.05:
        return "5pct"

    if fraction == 0.025:
        return "2_5pct"

    if fraction == 0.01:
        return "1pct"

    raise ValueError(
        f"Unknown tail fraction: {fraction}"
    )


def _experiment_id(
    signal: str,
    direction: str,
    fraction: float,
    target: str,
) -> str:
    return (
        f"{signal}"
        f"__{direction}"
        f"__{_fraction_name(fraction)}"
        f"__{target}"
    )


def _same_value(
    old: Any,
    new: Any,
) -> bool:
    if old is None or new is None:
        return old is None and new is None

    if isinstance(old, int) and isinstance(new, int):
        return old == new

    try:
        old_float = float(old)
        new_float = float(new)
    except (TypeError, ValueError):
        return old == new

    if math.isnan(old_float) or math.isnan(new_float):
        return math.isnan(old_float) and math.isnan(
            new_float
        )

    return math.isclose(
        old_float,
        new_float,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )


def _compare(
    old_rows: list[dict[str, Any]],
    new_rows: list[dict[str, Any]],
) -> pd.DataFrame:
    old_by_key = {
        (
            row["signal"],
            row["direction"],
            row["fraction"],
            row["target"],
            row["window"],
            row["split"],
        ): row
        for row in old_rows
    }

    new_by_key = {
        (
            row["signal"],
            row["direction"],
            row["fraction"],
            row["target"],
            row["window"],
            row["split"],
        ): row
        for row in new_rows
    }

    all_keys = sorted(
        set(old_by_key)
        | set(new_by_key),
        key=str,
    )

    rows: list[dict[str, Any]] = []

    for key in all_keys:
        (
            signal,
            direction,
            fraction,
            target,
            window,
            split,
        ) = key

        old = old_by_key.get(key)
        new = new_by_key.get(key)

        row: dict[str, Any] = {
            "experiment": _experiment_id(
                signal,
                direction,
                fraction,
                target,
            ),
            "window": window,
            "split": split,
            "signal": signal,
            "direction": direction,
            "target": target,
            "fraction": fraction,
        }

        failures: list[str] = []

        for field in COMMON_FIELDS:
            old_value = (
                old.get(field)
                if old is not None
                else None
            )
            new_value = (
                new.get(field)
                if new is not None
                else None
            )

            row[f"{field}_old"] = old_value
            row[f"{field}_new"] = new_value

            equal = _same_value(
                old_value,
                new_value,
            )

            row[f"{field}_match"] = equal

            if not equal:
                failures.append(field)

        row["status"] = (
            "PASS"
            if not failures
            and old is not None
            and new is not None
            else "FAIL"
        )

        row["differences"] = (
            ",".join(failures)
            if failures
            else ""
        )

        rows.append(row)

    return pd.DataFrame(rows)


def _print_summary(
    comparison: pd.DataFrame,
) -> None:
    total = len(comparison)

    passed = int(
        (comparison["status"] == "PASS").sum()
    )

    failed = total - passed

    print()
    print("=" * 80)
    print("OLD vs NEW GENERIC RESEARCH EVALUATOR")
    print("=" * 80)
    print(f"Rows compared: {total:,}")
    print(f"PASS:          {passed:,}")
    print(f"FAIL:          {failed:,}")

    if failed:
        print()
        print("Failures:")
        print(
            comparison.loc[
                comparison["status"] == "FAIL",
                [
                    "experiment",
                    "window",
                    "split",
                    "differences",
                ],
            ].to_string(
                index=False
            )
        )
    else:
        print()
        print(
            "PASS: All common metrics match exactly "
            "within numerical tolerance."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the old generic evaluator with "
            "the new declarative Research Engine."
        )
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Directory for the comparison CSV and report."
        ),
    )

    args = parser.parse_args()

    spec = _build_migration_spec()

    print(
        "Building shared research session...",
        flush=True,
    )

    session = build_session([spec])

    experiments = [
        Experiment(
            experiment_id=_experiment_id(
                signal_name,
                direction,
                fraction,
                target_name,
            ),
            signal_name=signal_name,
            target_name=target_name,
            tail_fraction=fraction,
            tail_direction=direction,
        )
        for signal_name, directions in SIGNAL_SPECS
        for direction in directions
        for fraction in TAIL_FRACTIONS
        for target_name in TARGET_NAMES
    ]

    print(
        f"Old generic experiments: "
        f"{len(experiments):,}",
        flush=True,
    )

    print(
        "Running old evaluator...",
        flush=True,
    )

    old_rows = _old_results(
        session,
        experiments,
    )

    print(
        f"Old evaluator rows: "
        f"{len(old_rows):,}",
        flush=True,
    )

    print(
        "Running new Engine...",
        flush=True,
    )

    new_rows = _new_results(
        session,
        spec,
    )

    print(
        f"New Engine rows: "
        f"{len(new_rows):,}",
        flush=True,
    )

    comparison = _compare(
        old_rows,
        new_rows,
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = (
        args.output_dir
        / "old_new_generic_comparison.csv"
    )

    comparison.to_csv(
        csv_path,
        index=False,
    )

    report_path = (
        args.output_dir
        / "old_new_generic_comparison.md"
    )

    total = len(comparison)
    passed = int(
        (comparison["status"] == "PASS").sum()
    )
    failed = total - passed

    report_lines = [
        "# Old vs New Generic Research Comparison",
        "",
        "## Summary",
        "",
        f"- Rows compared: {total:,}",
        f"- PASS: {passed:,}",
        f"- FAIL: {failed:,}",
        "",
        "## Compared metrics",
        "",
        "- n",
        "- events",
        "- event_rate",
        "- lift",
        "- mean_return",
        "",
        "Bootstrap confidence intervals are not compared.",
        "The old evaluator computes a CI for the selected",
        "group mean, while the new Engine computes a CI for",
        "selected minus rest.",
        "",
    ]

    if failed:
        report_lines.extend(
            [
                "## Failures",
                "",
                comparison.loc[
                    comparison["status"] == "FAIL",
                    [
                        "experiment",
                        "window",
                        "split",
                        "differences",
                    ],
                ].to_markdown(index=False),
                "",
            ]
        )
    else:
        report_lines.extend(
            [
                "## Result",
                "",
                "PASS — all common metrics match within the",
                "configured numerical tolerance.",
                "",
            ]
        )

    report_path.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    _print_summary(comparison)

    print()
    print(f"CSV:    {csv_path}")
    print(f"Report: {report_path}")

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
