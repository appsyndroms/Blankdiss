from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import pandas as pd

from ml.config import WALK_FORWARD_WINDOWS
from ml.dataset import load_features
from ml.diagnostics.experiments.fi_short_interest_event_risk_interaction_diagnostic import (
    FIShortInterestEventRiskInteractionExperiment,
)
from ml.diagnostics.framework.context import ExperimentContext
from ml.research.custom.fi_short_interest_event_risk_interaction import (
    run as run_new,
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

TABLE_NAME = "risk_tail_results"

COMPARE_COLUMNS = (
    "risk_cutoff",
    "risk_threshold",
    "positive_change_cutoff",
    "positive_change_threshold",
    "risk_low_n",
    "risk_high_n",
    "outside_low_n",
    "outside_high_n",
    "risk_low_down_rate",
    "risk_high_down_rate",
    "outside_low_down_rate",
    "outside_high_down_rate",
    "risk_change_effect",
    "outside_change_effect",
    "change_event_interaction",
    "interaction_ci_low",
    "interaction_ci_high",
    "interaction_p_positive",
)


def _same_value(
    old: Any,
    new: Any,
) -> bool:
    if old is None or new is None:
        return old is None and new is None

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
    columns: tuple[str, ...],
) -> pd.DataFrame:
    old = old.reset_index(
        drop=True
    )

    new = new.reset_index(
        drop=True
    )

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

        for column in columns:
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

            row[
                f"{column}_old"
            ] = old_value

            row[
                f"{column}_new"
            ] = new_value

            row[
                f"{column}_match"
            ] = match

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

    experiment = (
        FIShortInterestEventRiskInteractionExperiment()
    )

    result = experiment.execute(
        context
    )

    table = result.tables.get(
        TABLE_NAME,
        pd.DataFrame(),
    )

    metrics = {
        key: value
        for key, value in result.metrics.items()
        if key in {
            "event_model",
            "event_features",
            "validation_auc",
            "event_threshold",
            "positive_change_cutoff",
        }
    }

    return (
        table,
        metrics,
    )


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
    return run_new(
        frame,
        train_end=train_end,
        validation_end=validation_end,
        test_end=test_end,
    )


def _compare_metrics(
    old: dict[str, Any],
    new: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    keys = (
        "event_model",
        "event_features",
        "validation_auc",
        "event_threshold",
        "positive_change_cutoff",
    )

    for key in keys:
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

    print("Running old diagnostic...")

    old_table, old_metrics = _run_old(
        frame,
        train_end=train_end,
        validation_end=validation_end,
        test_end=test_end,
    )

    print("Running new custom analysis...")

    new_table, new_metrics = _run_new(
        frame,
        train_end=train_end,
        validation_end=validation_end,
        test_end=test_end,
    )

    table_comparison = _compare_tables(
        old_table,
        new_table,
        COMPARE_COLUMNS,
    )

    metric_comparison = _compare_metrics(
        old_metrics,
        new_metrics,
    )

    table_pass = (
        table_comparison["status"]
        == "PASS"
    ).all()

    metric_pass = (
        metric_comparison["status"]
        == "PASS"
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
        "# FI Short Interest Event Risk Interaction Migration",
        "",
        "Numerical comparison between the old diagnostic",
        "and the new research/custom implementation.",
        "",
    ]

    overall_pass = True

    for window_name, (
        analysis,
        metrics,
    ) in comparisons.items():
        analysis_failed = int(
            (
                analysis["status"]
                == "FAIL"
            ).sum()
        )

        metric_failed = int(
            (
                metrics["status"]
                == "FAIL"
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
            analysis_failures = (
                analysis.loc[
                    analysis["status"]
                    == "FAIL"
                ]
            )

            metric_failures = (
                metrics.loc[
                    metrics["status"]
                    == "FAIL"
                ]
            )

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
        / "fi_short_interest_event_risk_interaction_migration.md"
    )

    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    return overall_pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the old FI short-interest/event-risk "
            "interaction diagnostic with the new custom "
            "research implementation."
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

    comparisons = {}

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
                "fi_short_interest_event_risk_"
                f"interaction_{window_name}.csv"
            ),
            index=False,
        )

        metrics.to_csv(
            args.output_dir
            / (
                "fi_short_interest_event_risk_"
                f"interaction_{window_name}_metrics.csv"
            ),
            index=False,
        )

    overall_pass = _write_report(
        args.output_dir,
        comparisons,
    )

    print()
    print("=" * 80)
    print(
        "FI SHORT INTEREST EVENT RISK INTERACTION MIGRATION"
    )
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
