from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


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
    tolerance: float = 1e-12,
) -> dict[str, Any] | None:
    """
    Compare two scalar values.

    Numeric values are compared using an absolute tolerance.
    Other values require exact equality.
    """
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
        try:
            difference = abs(float(old) - float(new))
        except (TypeError, ValueError):
            difference = None

        if difference is not None and difference <= tolerance:
            return None

        return {
            "field": name,
            "old": old,
            "new": new,
            "reason": "numeric_difference",
            "absolute_difference": difference,
        }

    if old != new:
        return {
            "field": name,
            "old": old,
            "new": new,
            "reason": "value_changed",
        }

    return None


def compare_mapping(
    old: dict[str, Any],
    new: dict[str, Any],
    *,
    fields: list[str] | None = None,
    tolerance: float = 1e-12,
) -> list[dict[str, Any]]:
    """
    Compare selected fields from two mappings.
    """
    if fields is None:
        fields = sorted(set(old) | set(new))

    differences: list[dict[str, Any]] = []

    for field in fields:
        old_value = old.get(field)
        new_value = new.get(field)

        difference = compare_scalar(
            field,
            old_value,
            new_value,
            tolerance=tolerance,
        )

        if difference is not None:
            differences.append(difference)

    return differences


def _load_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)

    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)

    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")

    return value


def _compare_population(
    old: dict[str, Any],
    new: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Compare fields describing the analyzed population.

    These fields are particularly important during a migration because
    identical-looking metrics can still represent different populations.
    """
    fields = [
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

    return compare_mapping(old, new, fields=fields)


def _compare_metrics(
    old: dict[str, Any],
    new: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Compare common research metrics.

    Small floating-point differences are tolerated.
    """
    fields = [
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

    return compare_mapping(
        old,
        new,
        fields=fields,
        tolerance=1e-9,
    )


def _compare_sequence(
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


def _compare_interaction_cells(
    old: dict[str, Any],
    new: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Compare 2x2 interaction cells where available.

    Expected structure:

        {
            "cells": {
                "00": {...},
                "01": {...},
                "10": {...},
                "11": {...}
            }
        }
    """
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
            differences.extend(
                {
                    **difference,
                    "field": f"cells.{cell}.{difference['field']}",
                }
                for difference in compare_mapping(
                    old_cell,
                    new_cell,
                    tolerance=1e-9,
                )
            )
        else:
            difference = compare_scalar(
                f"cells.{cell}",
                old_cell,
                new_cell,
                tolerance=1e-9,
            )

            if difference is not None:
                differences.append(difference)

    return differences


def compare_interaction_analysis(
    old: dict[str, Any],
    new: dict[str, Any],
) -> ComparisonResult:
    """
    Compare an old and new interaction analysis.

    The comparison intentionally focuses on semantics:
    population, target, tails, metrics and 2x2 cells.
    """
    differences: list[dict[str, Any]] = []

    differences.extend(_compare_population(old, new))
    differences.extend(_compare_metrics(old, new))
    differences.extend(_compare_interaction_cells(old, new))

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
            tolerance=1e-9,
        )

        if difference is not None:
            differences.append(difference)

    status = "equivalent" if not differences else "different"

    return ComparisonResult(
        analysis="interaction",
        status=status,
        differences=differences,
        old=old,
        new=new,
    )


def compare_incremental_si_analysis(
    old: dict[str, Any],
    new: dict[str, Any],
) -> ComparisonResult:
    """
    Compare the legacy and migrated incremental SI analysis.

    This analysis is intentionally treated as custom research rather
    than as a simple declarative tail comparison because it contains:

    - SI decile analysis
    - volatility conditioning
    - rank/monotonicity analysis
    - walk-forward model comparison
    - volatility-only model
    - volatility + SI model
    - volatility + SI interaction model
    """
    differences: list[dict[str, Any]] = []

    differences.extend(_compare_population(old, new))
    differences.extend(_compare_metrics(old, new))

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
            difference = _compare_sequence(
                field,
                old_value,
                new_value,
            )
        else:
            difference = compare_scalar(
                field,
                old_value,
                new_value,
                tolerance=1e-9,
            )

        if difference is not None:
            differences.append(difference)

    _compare_nested_analysis(
        differences,
        old,
        new,
        "decile_analysis",
    )

    _compare_nested_analysis(
        differences,
        old,
        new,
        "conditional_rank_analysis",
    )

    _compare_nested_analysis(
        differences,
        old,
        new,
        "walk_forward_models",
    )

    status = "equivalent" if not differences else "different"

    return ComparisonResult(
        analysis="incremental_si_analysis",
        status=status,
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
    """
    Compare nested analysis structures when present.

    The function deliberately does not require a single rigid schema,
    because migration artifacts may represent tables either as mappings
    or lists of records.
    """
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
        differences.extend(
            {
                **difference,
                "field": f"{field}.{difference['field']}",
            }
            for difference in compare_mapping(
                old_value,
                new_value,
                tolerance=1e-9,
            )
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
                differences.extend(
                    {
                        **difference,
                        "field": (
                            f"{field}[{index}]."
                            f"{difference['field']}"
                        ),
                    }
                    for difference in compare_mapping(
                        old_row,
                        new_row,
                        tolerance=1e-9,
                    )
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


def compare_analysis(
    analysis: str,
    old: dict[str, Any],
    new: dict[str, Any],
) -> ComparisonResult:
    """
    Dispatch to the analysis-specific comparator.
    """
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
    differences.extend(_compare_population(old, new))
    differences.extend(_compare_metrics(old, new))

    status = "equivalent" if not differences else "different"

    return ComparisonResult(
        analysis=analysis,
        status=status,
        differences=differences,
        old=old,
        new=new,
    )


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
            "Compare legacy and Research Engine migration results."
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

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(
            result.to_dict(),
            handle,
            indent=2,
            ensure_ascii=False,
            default=str,
        )

    print(
        f"Migration comparison: {args.analysis} "
        f"-> {result.status}"
    )

    if result.differences:
        print(
            f"Differences found: {len(result.differences)}"
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
