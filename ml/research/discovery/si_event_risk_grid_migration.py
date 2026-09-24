from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pandas as pd

from ml.config import WALK_FORWARD_WINDOWS
from ml.dataset import load_features
from ml.diagnostics.experiments.si_event_risk_grid_diagnostic import (
    SIEventRiskGridExperiment,
)
from ml.diagnostics.framework.context import ExperimentContext
from ml.diagnostics.framework import event_risk_grid as old_event_risk_grid
from ml.research.discovery import si_event_risk_grid as new_event_risk_grid


ROOT = Path(__file__).resolve().parents[2]

DEFAULT_OUTPUT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "ml"
    / "research"
    / "migration"
)

# The migration check verifies that old and new implementations
# produce the same result using the same bootstrap algorithm.
#
# Production discovery still uses 2,000 iterations.
# The migration comparator deliberately uses fewer iterations
# because it runs the complete discovery grid twice per window.
MIGRATION_BOOTSTRAP_ITERATIONS = 200

TABLE_NAME = "interaction_grid"

COMPARE_COLUMNS = (
    "selection_stage",
    "is_discovery_grid",
    "horizon_days",
    "event_threshold",
    "downside_target",
    "risk_cutoff",
    "risk_threshold",
    "si_change_cutoff",
    "si_change_threshold",
    "event_model",
    "event_features",
    "event_validation_auc",
    "risk_high_si_n",
    "risk_low_si_n",
    "outside_high_si_n",
    "outside_low_si_n",
    "risk_high_si_rate",
    "risk_low_si_rate",
    "outside_high_si_rate",
    "outside_low_si_rate",
    "risk_si_effect",
    "outside_si_effect",
    "interaction",
    "interaction_ci_low",
    "interaction_ci_high",
    "interaction_p_positive",
)

METRIC_KEYS = (
    "analysis_type",
    "horizons",
    "event_thresholds",
    "risk_cutoffs",
    "si_change_cutoffs",
    "downside_targets",
    "result_rows",
)


def _same_value(
    old: Any,
    new: Any,
) -> bool:
    if old is None or new is None:
        return old is None and new is None

    if isinstance(old, (list, tuple)) or isinstance(
        new,
        (list, tuple),
    ):
        return list(old) == list(new)

    try:
        old_float = float(old)
        new_float = float(new)
    except (TypeError, ValueError):
        return old == new

    if math.isnan(old_float) or math.isnan(new_float):
        return (
            math.isnan(old_float)
            and math.isnan(new_float)
        )

    return math.isclose(
        old_float,
        new_float,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )


def _compare_tables(
    old: pd.DataFrame,
    new: pd.DataFrame,
) -> pd.DataFrame:
    old = old.reset_index(drop=True)
    new = new.reset_index(drop=True)

    rows: list[dict[str, Any]] = []

    max_rows = max(
        len(old),
        len(new),
    )

    for index in range(max_rows):
        old_row = (
            old.iloc[index]
            if index < len(old)
            else None
        )

        new_row = (
            new.iloc[index]
            if index < len(new)
            else None
        )

        differences: list[str] = []

        row: dict[str, Any] = {
            "row": index,
        }

        for column in COMPARE_COLUMNS:
            old_value = (
                old_row[column]
                if old_row is not None
                else None
            )

            new_value = (
                new_row[column]
                if new_row is not None
                else None
            )

            match = _same_value(
                old_value,
                new_value,
            )

            row[f"{column}_old"] = old_value
            row[f"{column}_new"] = new_value
            row[f"{column}_match"] = match

            if not match:
                differences.append(
                    column
                )

        row["status"] = (
            "PASS"
            if not differences
            else "FAIL"
        )

        row["differences"] = ",".join(
            differences
        )

        rows.append(row)

    return pd.DataFrame(rows)


def _compare_metrics(
    old: dict[str, Any],
    new: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for key in METRIC_KEYS:
        old_value = old.get(key)
        new_value = new.get(key)

        match = _same_value(
            old_value,
            new_value,
        )

        rows.append(
            {
                "metric": key,
                "old": old_value,
                "new": new_value,
                "match": match,
                "status": (
                    "PASS"
                    if match
                    else "FAIL"
                ),
            }
        )

    return pd.DataFrame(rows)


def _run_old(
    frame: pd.DataFrame,
    *,
    train_end: str,
    validation_end: str,
    test_end: str,
) -> tuple[
    pd.DataFrame,
    dict[str, Any],
]:
    context = ExperimentContext(
        frame,
        train_end=train_end,
        validation_end=validation_end,
        test_end=test_end,
    )

    experiment = SIEventRiskGridExperiment()

    result = experiment.execute(
        context
    )

    table = result.tables.get(
        TABLE_NAME,
        pd.DataFrame(),
    )

    metrics = {
        key: result.metrics.get(key)
        for key in METRIC_KEYS
    }

    return table, metrics


def _run_new(
    frame: pd.DataFrame,
    *,
    train_end: str,
    validation_end: str,
    test_end: str,
) -> tuple[
    pd.DataFrame,
    dict[str, Any],
]:
    return new_event_risk_grid.run(
        frame,
        train_end=train_end,
        validation_end=validation_end,
        test_end=test_end,
    )


def _compare_window(
    frame: pd.DataFrame,
    *,
    window_name: str,
    train_end: str,
    validation_end: str,
    test_end: str,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    print()
    print("=" * 80)
    print(f"WINDOW: {window_name}")
    print("=" * 80)

    print(
        "Bootstrap iterations: "
        f"{MIGRATION_BOOTSTRAP_ITERATIONS}"
    )

    print("Running old diagnostic...")

    old_table, old_metrics = _run_old(
        frame,
        train_end=train_end,
        validation_end=validation_end,
        test_end=test_end,
    )

    print("Running new discovery analysis...")

    new_table, new_metrics = _run_new(
        frame,
        train_end=train_end,
        validation_end=validation_end,
        test_end=test_end,
    )

    table_comparison = _compare_tables(
        old_table,
        new_table,
    )

    metric_comparison = _compare_metrics(
        old_metrics,
        new_metrics,
    )

    table_pass = (
        table_comparison["status"] == "PASS"
    ).all()

    metric_pass = (
        metric_comparison["status"] == "PASS"
    ).all()

    print(
        f"Analysis rows: "
        f"{len(old_table):,} old / "
        f"{len(new_table):,} new"
    )

    print(
        "Analysis:      "
        + (
            "PASS"
            if table_pass
            else "FAIL"
        )
    )

    print(
        f"Metric rows: "
        f"{len(metric_comparison):,}"
    )

    print(
        "Metrics:       "
        + (
            "PASS"
            if metric_pass
            else "FAIL"
        )
    )

    return (
        table_comparison,
        metric_comparison,
    )


def _dataframe_to_markdown(
    frame: pd.DataFrame,
) -> str:
    if frame.empty:
        return "None."

    columns = list(
        frame.columns
    )

    def _format(
        value: Any,
    ) -> str:
        if pd.isna(value):
            return ""

        return (
            str(value)
            .replace("\\", "\\\\")
            .replace("|", "\\|")
            .replace("\n", " ")
        )

    header = (
        "| "
        + " | ".join(columns)
        + " |"
    )

    separator = (
        "| "
        + " | ".join(
            "---"
            for _ in columns
        )
        + " |"
    )

    rows = [
        "| "
        + " | ".join(
            _format(value)
            for value in row
        )
        + " |"
        for row in frame.itertuples(
            index=False,
            name=None,
        )
    ]

    return "\n".join(
        [
            header,
            separator,
            *rows,
        ]
    )


def _write_report(
    output_dir: Path,
    comparisons: dict[
        str,
        tuple[
            pd.DataFrame,
            pd.DataFrame,
        ],
    ],
) -> bool:
    lines = [
        "# SI Event Risk Grid Migration",
        "",
        (
            "Numerical comparison between the old "
            "discovery-grid diagnostic and the new "
            "research/discovery implementation."
        ),
        "",
        (
            "Migration bootstrap iterations: "
            f"{MIGRATION_BOOTSTRAP_ITERATIONS}"
        ),
        "",
        (
            "Production discovery remains configured "
            "for 2,000 bootstrap iterations."
        ),
        "",
    ]

    overall_pass = True

    for window_name, (
        analysis,
        metrics,
    ) in comparisons.items():
        analysis_failed = int(
            (
                analysis["status"] == "FAIL"
            ).sum()
        )

        metric_failed = int(
            (
                metrics["status"] == "FAIL"
            ).sum()
        )

        window_pass = (
            analysis_failed == 0
            and metric_failed == 0
        )

        overall_pass &= window_pass

        lines.extend(
            [
                f"## {window_name}",
                "",
                (
                    f"- Analysis rows: "
                    f"{len(analysis):,}"
                ),
                (
                    f"- Analysis failures: "
                    f"{analysis_failed:,}"
                ),
                (
                    f"- Metric rows: "
                    f"{len(metrics):,}"
                ),
                (
                    f"- Metric failures: "
                    f"{metric_failed:,}"
                ),
                (
                    "- Status: "
                    + (
                        "PASS"
                        if window_pass
                        else "FAIL"
                    )
                ),
                "",
            ]
        )

        if not window_pass:
            analysis_failures = analysis.loc[
                analysis["status"] == "FAIL"
            ]

            metric_failures = metrics.loc[
                metrics["status"] == "FAIL"
            ]

            lines.extend(
                [
                    "### Analysis failures",
                    "",
                    _dataframe_to_markdown(
                        analysis_failures
                    ),
                    "",
                    "### Metric failures",
                    "",
                    _dataframe_to_markdown(
                        metric_failures
                    ),
                    "",
                ]
            )

    lines.extend(
        [
            "## Overall result",
            "",
            (
                "PASS"
                if overall_pass
                else "FAIL"
            ),
            "",
        ]
    )

    path = (
        output_dir
        / "si_event_risk_grid_migration.md"
    )

    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    return overall_pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the old SI event-risk discovery "
            "grid with the new research/discovery "
            "implementation."
        )
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    args = parser.parse_args()

    print("Loading features...")
    frame = load_features()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparisons: dict[
        str,
        tuple[
            pd.DataFrame,
            pd.DataFrame,
        ],
    ] = {}

    # The old and new implementations both look up
    # BOOTSTRAP_ITERATIONS at call time.
    #
    # Patch both modules for this migration run only.
    # The production value remains 2,000.
    with (
        patch.object(
            old_event_risk_grid,
            "BOOTSTRAP_ITERATIONS",
            MIGRATION_BOOTSTRAP_ITERATIONS,
        ),
        patch.object(
            new_event_risk_grid,
            "BOOTSTRAP_ITERATIONS",
            MIGRATION_BOOTSTRAP_ITERATIONS,
        ),
    ):
        for index, window in enumerate(
            WALK_FORWARD_WINDOWS,
            start=1,
        ):
            window_name = (
                f"window_{index}"
            )

            comparisons[window_name] = (
                _compare_window(
                    frame,
                    window_name=window_name,
                    train_end=window.train_end,
                    validation_end=window.validation_end,
                    test_end=window.test_end,
                )
            )

            analysis, metrics = (
                comparisons[window_name]
            )

            analysis.to_csv(
                args.output_dir
                / (
                    "si_event_risk_grid_"
                    f"{window_name}.csv"
                ),
                index=False,
            )

            metrics.to_csv(
                args.output_dir
                / (
                    "si_event_risk_grid_"
                    f"{window_name}_metrics.csv"
                ),
                index=False,
            )

    overall_pass = _write_report(
        args.output_dir,
        comparisons,
    )

    print()
    print("=" * 80)
    print("SI EVENT RISK GRID MIGRATION")
    print("=" * 80)

    if overall_pass:
        print(
            "PASS: old and new implementations "
            "match within numerical tolerance."
        )
    else:
        print(
            "FAIL: old and new implementations "
            "differ."
        )

        raise SystemExit(1)


if __name__ == "__main__":
    main()
