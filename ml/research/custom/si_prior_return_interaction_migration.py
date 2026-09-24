from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import pandas as pd

from ml.config import WALK_FORWARD_WINDOWS
from ml.dataset import load_features
from ml.diagnostics.experiments.si_prior_return_interaction_diagnostic import (
    SIPriorReturnInteractionExperiment,
)
from ml.diagnostics.framework.context import ExperimentContext
from ml.research.custom.si_prior_return_interaction import (
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

TABLE_NAME = "prior_return_price_return_5d"
THRESHOLD_TABLE_NAME = "thresholds"

COMPARE_COLUMNS = (
    "prior_return_column",
    "prior_group",
    "prior_moderate_threshold",
    "prior_strong_threshold",
    "si_change_cutoff",
    "si_change_threshold",
    "horizon_days",
    "high_si_n",
    "other_positive_n",
    "high_si_mean_return",
    "other_positive_mean_return",
    "mean_return_delta",
    "mean_return_ci_low",
    "mean_return_ci_high",
    "high_si_median_return",
    "other_positive_median_return",
    "high_si_positive_rate",
    "other_positive_rate",
)

THRESHOLD_COLUMNS = (
    "prior_return_column",
    "positive_prior_median",
    "positive_prior_p80",
    "si_change_top10_threshold",
    "si_change_top20_threshold",
    "si_change_top30_threshold",
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


def _run_old(
    frame: pd.DataFrame,
    *,
    train_end: str,
    validation_end: str,
    test_end: str,
) -> dict[str, pd.DataFrame]:
    context = ExperimentContext(
        frame,
        train_end=train_end,
        validation_end=validation_end,
        test_end=test_end,
    )

    experiment = (
        SIPriorReturnInteractionExperiment()
    )

    result = experiment.execute(
        context
    )

    return result.tables


def _run_new(
    frame: pd.DataFrame,
    *,
    validation_end: str,
    test_end: str,
) -> dict[str, pd.DataFrame]:
    result = run_new(
        frame,
        pretest_end=validation_end,
        test_end=test_end,
    )

    # The new run() returns the main analysis table.
    # Reconstruct the threshold table exactly as the old
    # implementation does, so it can also be compared.
    from ml.research.custom.si_prior_return_interaction import (
        _build_results,
        add_si_change,
    )

    data = add_si_change(
        frame
    )

    data["snapshot_date"] = pd.to_datetime(
        data["snapshot_date"],
        errors="coerce",
    )

    pretest = data[
        data["snapshot_date"]
        <= pd.Timestamp(validation_end)
    ].copy()

    test = data[
        data["snapshot_date"]
        > pd.Timestamp(validation_end)
    ].copy()

    test = test[
        test["snapshot_date"]
        <= pd.Timestamp(test_end)
    ].copy()

    _, thresholds = _build_results(
        pretest,
        test,
        prior_column="price_return_5d",
        si_change_cutoffs=(
            0.10,
            0.20,
            0.30,
        ),
        horizons=(
            1,
            3,
            5,
            10,
            20,
        ),
    )

    return {
        TABLE_NAME: result,
        THRESHOLD_TABLE_NAME: thresholds,
    }


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
    old_tables = _run_old(
        frame,
        train_end=train_end,
        validation_end=validation_end,
        test_end=test_end,
    )

    print("Running new custom analysis...")
    new_tables = _run_new(
        frame,
        validation_end=validation_end,
        test_end=test_end,
    )

    old_analysis = old_tables.get(
        TABLE_NAME,
        pd.DataFrame(),
    )

    new_analysis = new_tables.get(
        TABLE_NAME,
        pd.DataFrame(),
    )

    old_thresholds = old_tables.get(
        THRESHOLD_TABLE_NAME,
        pd.DataFrame(),
    )

    new_thresholds = new_tables.get(
        THRESHOLD_TABLE_NAME,
        pd.DataFrame(),
    )

    analysis_comparison = _compare_tables(
        old_analysis,
        new_analysis,
        COMPARE_COLUMNS,
    )

    threshold_comparison = _compare_tables(
        old_thresholds,
        new_thresholds,
        THRESHOLD_COLUMNS,
    )

    analysis_pass = (
        analysis_comparison["status"]
        == "PASS"
    ).all()

    threshold_pass = (
        threshold_comparison["status"]
        == "PASS"
    ).all()

    print(
        f"Analysis rows: "
        f"{len(old_analysis):,} old / "
        f"{len(new_analysis):,} new"
    )

    print(
        "Analysis:      "
        + ("PASS" if analysis_pass else "FAIL")
    )

    print(
        f"Threshold rows: "
        f"{len(old_thresholds):,} old / "
        f"{len(new_thresholds):,} new"
    )

    print(
        "Thresholds:     "
        + (
            "PASS"
            if threshold_pass
            else "FAIL"
        )
    )

    return (
        analysis_comparison,
        threshold_comparison,
    )


def _write_report(
    output_dir: Path,
    comparisons: dict[
        str,
        tuple[pd.DataFrame, pd.DataFrame],
    ],
) -> bool:
    report_lines = [
        "# SI Prior Return Interaction Migration",
        "",
        "Numerical comparison between the old diagnostic",
        "and the new research/custom implementation.",
        "",
    ]

    overall_pass = True

    for window_name, (
        analysis,
        thresholds,
    ) in comparisons.items():
        analysis_failed = int(
            (
                analysis["status"]
                == "FAIL"
            ).sum()
        )

        threshold_failed = int(
            (
                thresholds["status"]
                == "FAIL"
            ).sum()
        )

        window_pass = (
            analysis_failed == 0
            and threshold_failed == 0
        )

        overall_pass &= window_pass

        report_lines.extend(
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
                    f"- Threshold rows: "
                    f"{len(thresholds):,}"
                ),
                (
                    f"- Threshold failures: "
                    f"{threshold_failed:,}"
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
            report_lines.extend(
                [
                    "### Analysis failures",
                    "",
                    analysis.loc[
                        analysis["status"] == "FAIL"
                    ].to_markdown(
                        index=False
                    ),
                    "",
                    "### Threshold failures",
                    "",
                    thresholds.loc[
                        thresholds["status"] == "FAIL"
                    ].to_markdown(
                        index=False
                    ),
                    "",
                ]
            )

    report_lines.extend(
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
        / "si_prior_return_interaction_migration.md"
    )

    path.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    return overall_pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the old SI/prior-return "
            "diagnostic with the new custom "
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

        analysis, thresholds = (
            comparisons[window_name]
        )

        analysis.to_csv(
            args.output_dir
            / (
                "si_prior_return_interaction_"
                f"{window_name}_analysis.csv"
            ),
            index=False,
        )

        thresholds.to_csv(
            args.output_dir
            / (
                "si_prior_return_interaction_"
                f"{window_name}_thresholds.csv"
            ),
            index=False,
        )

    overall_pass = _write_report(
        args.output_dir,
        comparisons,
    )

    print()
    print("=" * 80)
    print("SI PRIOR RETURN INTERACTION MIGRATION")
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
