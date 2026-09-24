from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


NUMERIC_TOLERANCE = 1e-9


@dataclass
class ComparisonResult:
    analysis: str
    status: str
    differences: list[dict[str, Any]]
    old: dict[str, Any]
    new: dict[str, Any]

    @property
    def equivalent(self) -> bool:
        return self.status == "equivalent"

    def to_dict(self) -> dict[str, Any]:
        return {
            "analysis": self.analysis,
            "status": self.status,
            "equivalent": self.equivalent,
            "differences": self.differences,
            "old": self.old,
            "new": self.new,
        }


def compare_scalar(
    name: str,
    old: Any,
    new: Any,
    *,
    tolerance: float = NUMERIC_TOLERANCE,
) -> dict[str, Any] | None:
    if old is None or new is None:
        if old == new:
            return None

        return {
            "field": name,
            "old": old,
            "new": new,
            "reason": "value_changed",
        }

    if isinstance(old, (int, float)) and isinstance(new, (int, float)):
        difference = abs(float(old) - float(new))

        if difference <= tolerance:
            return None

        return {
            "field": name,
            "old": old,
            "new": new,
            "reason": "numeric_difference",
            "absolute_difference": difference,
        }

    if old == new:
        return None

    return {
        "field": name,
        "old": old,
        "new": new,
        "reason": "value_changed",
    }


def compare_mapping(
    old: dict[str, Any],
    new: dict[str, Any],
    *,
    fields: list[str] | None = None,
    tolerance: float = NUMERIC_TOLERANCE,
) -> list[dict[str, Any]]:
    if fields is None:
        fields = sorted(set(old) | set(new))

    differences: list[dict[str, Any]] = []

    for field in fields:
        difference = compare_scalar(
            field,
            old.get(field),
            new.get(field),
            tolerance=tolerance,
        )

        if difference is not None:
            differences.append(difference)

    return differences


def _population_fields() -> list[str]:
    return [
        "n",
        "rows",
        "feature_rows",
        "input_rows",
        "analysis_rows",
        "event_count",
        "event_rate",
        "baseline_event_rate",
        "discovery_end",
        "target",
        "target_name",
        "signal",
        "signals",
        "tail_fraction",
        "tail_fractions",
        "direction",
        "window",
        "windows",
    ]


def compare_population(
    old: dict[str, Any],
    new: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Compare the population represented by the two analyses.

    This is intentionally public because migration tests should be able
    to verify population equivalence independently from metric
    equivalence.
    """
    return compare_mapping(
        old,
        new,
        fields=_population_fields(),
        tolerance=NUMERIC_TOLERANCE,
    )


def _metric_fields() -> list[str]:
    return [
        "baseline",
        "baseline_rate",
        "event_rate",
        "lift",
        "mean_return",
        "median_return",
        "return_difference",
        "mean_return_difference",
        "roc_auc",
        "brier",
        "log_loss",
        "spearman",
    ]


def compare_metrics(
    old: dict[str, Any],
    new: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Compare the common research metrics.
    """
    return compare_mapping(
        old,
        new,
        fields=_metric_fields(),
        tolerance=NUMERIC_TOLERANCE,
    )


def compare_sequence(
    name: str,
    old: Any,
    new: Any,
) -> dict[str, Any] | None:
    if old == new:
        return None

    return {
        "field": name,
        "old": old,
        "new": new,
        "reason": "sequence_changed",
    }


def compare_interaction_cells(
    old: dict[str, Any],
    new: dict[str, Any],
) -> list[dict[str, Any]]:
    old_cells = old.get("cells")
    new_cells = new.get("cells")

    if old_cells is None and new_cells is None:
        return []

    if not isinstance(old_cells, dict) or not isinstance(new_cells, dict):
        return [
            {
                "field": "cells",
                "old": old_cells,
                "new": new_cells,
                "reason": "interaction_cells_structure_changed",
            }
        ]

    differences: list[dict[str, Any]] = []

    for cell in ("00", "01", "10", "11"):
        old_cell = old_cells.get(cell)
        new_cell = new_cells.get(cell)

        if old_cell is None or new_cell is None:
            if old_cell != new_cell:
                differences.append(
                    {
                        "field": f"cells.{cell}",
                        "old": old_cell,
                        "new": new_cell,
                        "reason": "cell_added_or_removed",
                    }
                )
            continue

        if isinstance(old_cell, dict) and isinstance(new_cell, dict):
            for difference in compare_mapping(
                old_cell,
                new_cell,
                tolerance=NUMERIC_TOLERANCE,
            ):
                differences.append(
                    {
                        **difference,
                        "field": (
                            f"cells.{cell}."
                            f"{difference['field']}"
                        ),
                    }
                )
        else:
            difference = compare_scalar(
                f"cells.{cell}",
                old_cell,
                new_cell,
            )

            if difference is not None:
                differences.append(difference)

    return differences


def compare_interaction_analysis(
    old: dict[str, Any],
    new: dict[str, Any],
) -> ComparisonResult:
    differences: list[dict[str, Any]] = []

    differences.extend(compare_population(old, new))
    differences.extend(compare_metrics(old, new))
    differences.extend(compare_interaction_cells(old, new))

    for field in (
        "interaction",
        "interaction_effect",
        "risk_ratio_interaction",
        "additive_interaction",
        "multiplicative_interaction",
    ):
        difference = compare_scalar(
            field,
            old.get(field),
            new.get(field),
        )

        if difference is not None:
            differences.append(difference)

    return ComparisonResult(
        analysis="interaction",
        status="equivalent" if not differences else "different",
        differences=differences,
        old=old,
        new=new,
    )


def _compare_nested_analysis(
    differences: list[dict[str, Any]],
    old: dict[str, Any],
    new: dict[str, Any],
    field: str,
) -> None:
    old_value = old.get(field)
    new_value = new.get(field)

    if old_value is None and new_value is None:
        return

    if old_value is None or new_value is None:
        differences.append(
            {
                "field": field,
                "old": old_value,
                "new": new_value,
                "reason": "analysis_added_or_removed",
            }
        )
        return

    if isinstance(old_value, dict) and isinstance(new_value, dict):
        for difference in compare_mapping(
            old_value,
            new_value,
            tolerance=NUMERIC_TOLERANCE,
        ):
            differences.append(
                {
                    **difference,
                    "field": (
                        f"{field}.{difference['field']}"
                    ),
                }
            )
        return

    if isinstance(old_value, list) and isinstance(new_value, list):
        if len(old_value) != len(new_value):
            differences.append(
                {
                    "field": f"{field}.length",
                    "old": len(old_value),
                    "new": len(new_value),
                    "reason": "row_count_changed",
                }
            )

        for index, (old_row, new_row) in enumerate(
            zip(old_value, new_value)
        ):
            if isinstance(old_row, dict) and isinstance(new_row, dict):
                for difference in compare_mapping(
                    old_row,
                    new_row,
                    tolerance=NUMERIC_TOLERANCE,
                ):
                    differences.append(
                        {
                            **difference,
                            "field": (
                                f"{field}[{index}]."
                                f"{difference['field']}"
                            ),
                        }
                    )
            elif old_row != new_row:
                differences.append(
                    {
                        "field": f"{field}[{index}]",
                        "old": old_row,
                        "new": new_row,
                        "reason": "row_changed",
                    }
                )

        return

    if old_value != new_value:
        differences.append(
            {
                "field": field,
                "old": old_value,
                "new": new_value,
                "reason": "analysis_changed",
            }
        )


def compare_incremental_si_analysis(
    old: dict[str, Any],
    new: dict[str, Any],
) -> ComparisonResult:
    differences: list[dict[str, Any]] = []

    differences.extend(compare_population(old, new))
    differences.extend(compare_metrics(old, new))

    for field in (
        "target",
        "target_name",
        "discovery_end",
        "min_model_rows",
        "volatility_groups",
        "si_deciles",
        "model_features",
        "models",
    ):
        old_value = old.get(field)
        new_value = new.get(field)

        if isinstance(old_value, list) and isinstance(new_value, list):
            difference = compare_sequence(
                field,
                old_value,
                new_value,
            )
        else:
            difference = compare_scalar(
                field,
                old_value,
                new_value,
            )

        if difference is not None:
            differences.append(difference)

    for field in (
        "decile_analysis",
        "conditional_rank_analysis",
        "walk_forward_models",
    ):
        _compare_nested_analysis(
            differences,
            old,
            new,
            field,
        )

    return ComparisonResult(
        analysis="incremental_si_analysis",
        status="equivalent" if not differences else "different",
        differences=differences,
        old=old,
        new=new,
    )


def compare_analysis(
    analysis: str,
    old: dict[str, Any],
    new: dict[str, Any],
) -> ComparisonResult:
    normalized = analysis.lower().strip()

    if normalized in {
        "interaction",
        "si_volatility_interaction",
        "si_volatility_downside_interaction",
    }:
        return compare_interaction_analysis(old, new)

    if normalized in {
        "incremental_si_analysis",
        "incremental_si",
        "momentum_si_incremental_locked_oos",
    }:
        return compare_incremental_si_analysis(old, new)

    differences = []

    differences.extend(compare_population(old, new))
    differences.extend(compare_metrics(old, new))

    return ComparisonResult(
        analysis=analysis,
        status="equivalent" if not differences else "different",
        differences=differences,
        old=old,
        new=new,
    )


def _load_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)

    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)

    if not isinstance(value, dict):
        raise ValueError(
            f"Expected JSON object in {path}"
        )

    return value


def compare_files(
    analysis: str,
    old_path: str | Path,
    new_path: str | Path,
) -> ComparisonResult:
    old = _load_json(old_path)
    new = _load_json(new_path)

    return compare_analysis(
        analysis,
        old,
        new,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare legacy and Research Engine "
            "migration results."
        )
    )

    parser.add_argument(
        "--analysis",
        required=True,
        help="Analysis identifier.",
    )

    parser.add_argument(
        "--old",
        required=True,
        help="Path to legacy result JSON.",
    )

    parser.add_argument(
        "--new",
        required=True,
        help="Path to migrated result JSON.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Path for comparison JSON.",
    )

    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    result = compare_files(
        analysis=args.analysis,
        old_path=args.old,
        new_path=args.new,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            result.to_dict(),
            handle,
            indent=2,
            ensure_ascii=False,
            default=str,
        )

    print(
        f"Migration comparison: "
        f"{args.analysis} -> {result.status}"
    )

    if result.differences:
        print(
            f"Differences found: "
            f"{len(result.differences)}"
        )

        for difference in result.differences:
            print(
                f"- {difference.get('field')}: "
                f"{difference.get('reason')}"
            )
    else:
        print("No semantic differences found.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
