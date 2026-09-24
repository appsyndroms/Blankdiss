from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_TOLERANCE = 1e-9


@dataclass(frozen=True)
class Comparison:
    field: str
    old: Any
    new: Any
    equal: bool
    reason: str | None = None


def _values_equal(
    old: Any,
    new: Any,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> bool:
    if old is None or new is None:
        return old == new

    if isinstance(old, bool) or isinstance(new, bool):
        return old == new

    if isinstance(old, (int, float)) and isinstance(new, (int, float)):
        if isinstance(old, float) and math.isnan(old):
            return isinstance(new, float) and math.isnan(new)

        if isinstance(new, float) and math.isnan(new):
            return False

        return abs(float(old) - float(new)) <= tolerance

    if isinstance(old, dict) and isinstance(new, dict):
        return old == new

    if isinstance(old, (list, tuple)) and isinstance(
        new,
        (list, tuple),
    ):
        return old == new

    return old == new


def _comparison(
    field: str,
    old: Any,
    new: Any,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> Comparison:
    equal = _values_equal(
        old,
        new,
        tolerance=tolerance,
    )

    if equal:
        reason = None
    elif old is None or new is None:
        reason = "missing_or_added_value"
    elif isinstance(old, (int, float)) and isinstance(
        new,
        (int, float),
    ):
        reason = "numeric_difference"
    else:
        reason = "value_difference"

    return Comparison(
        field=field,
        old=old,
        new=new,
        equal=equal,
        reason=reason,
    )


def compare_mapping(
    old: dict[str, Any],
    new: dict[str, Any],
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> list[Comparison]:
    """
    Compare two mappings field by field.

    The result deliberately includes equal comparisons. This makes the
    comparator useful for migration verification: a successful
    comparison is explicit rather than represented by an empty list.
    """
    fields = sorted(set(old) | set(new))

    return [
        _comparison(
            field,
            old.get(field),
            new.get(field),
            tolerance=tolerance,
        )
        for field in fields
    ]


def compare_semantics(
    old: dict[str, Any],
    new: dict[str, Any],
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> list[Comparison]:
    """
    Compare fields that define what an analysis means.

    Supports both top-level result fields and the metadata structure
    used by the migration artifacts.
    """
    fields = [
        "target",
        "target_name",
        "signal",
        "signals",
        "direction",
        "tail_fraction",
        "tail_fractions",
        "window",
        "windows",
        "discovery_end",
        "analysis_type",
        "mode",
        "walk_forward",
        "walk_forward_windows",
        "grouping",
        "bins",
        "bin_edges",
        "volatility_groups",
        "si_deciles",
        "min_model_rows",
        "model_features",
        "models",
    ]

    comparisons: list[Comparison] = []

    old_metadata = old.get("metadata", {})
    new_metadata = new.get("metadata", {})

    if not isinstance(old_metadata, dict):
        old_metadata = {}

    if not isinstance(new_metadata, dict):
        new_metadata = {}

    for field in fields:
        old_value = (
            old[field]
            if field in old
            else old_metadata.get(field)
        )

        new_value = (
            new[field]
            if field in new
            else new_metadata.get(field)
        )

        if field in old or field in new or (
            field in old_metadata or field in new_metadata
        ):
            comparisons.append(
                _comparison(
                    field,
                    old_value,
                    new_value,
                    tolerance=tolerance,
                )
            )

    return comparisons


def compare_population(
    old: dict[str, Any],
    new: dict[str, Any],
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> list[Comparison]:
    """
    Compare the analyzed population.

    Population fields are normally stored under metadata, but
    top-level fields are also supported.
    """
    fields = [
        "feature_rows",
        "eligible_rows",
        "analysis_rows",
        "rows",
        "n",
        "event_count",
        "event_rate",
        "baseline_event_rate",
    ]

    comparisons: list[Comparison] = []

    old_metadata = old.get("metadata", {})
    new_metadata = new.get("metadata", {})

    if not isinstance(old_metadata, dict):
        old_metadata = {}

    if not isinstance(new_metadata, dict):
        new_metadata = {}

    for field in fields:
        old_value = (
            old[field]
            if field in old
            else old_metadata.get(field)
        )

        new_value = (
            new[field]
            if field in new
            else new_metadata.get(field)
        )

        if field in old or field in new or (
            field in old_metadata or field in new_metadata
        ):
            comparisons.append(
                _comparison(
                    field,
                    old_value,
                    new_value,
                    tolerance=tolerance,
                )
            )

    return comparisons


def compare_metrics(
    old: dict[str, Any],
    new: dict[str, Any],
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> list[Comparison]:
    """
    Compare numerical research metrics.
    """
    fields = [
        "event_rate",
        "baseline",
        "baseline_rate",
        "lift",
        "mean_return",
        "median_return",
        "return_difference",
        "mean_return_difference",
        "roc_auc",
        "auc",
        "brier",
        "log_loss",
        "spearman",
        "additive_interaction",
        "relative_risk_interaction",
        "risk_ratio_interaction",
        "multiplicative_interaction",
    ]

    comparisons: list[Comparison] = []

    old_metrics = old.get("metrics", {})
    new_metrics = new.get("metrics", {})

    if not isinstance(old_metrics, dict):
        old_metrics = {}

    if not isinstance(new_metrics, dict):
        new_metrics = {}

    for field in fields:
        old_value = (
            old[field]
            if field in old
            else old_metrics.get(field)
        )

        new_value = (
            new[field]
            if field in new
            else new_metrics.get(field)
        )

        if field in old or field in new or (
            field in old_metrics or field in new_metrics
        ):
            comparisons.append(
                _comparison(
                    field,
                    old_value,
                    new_value,
                    tolerance=tolerance,
                )
            )

    return comparisons


def compare_interaction_cells(
    old: dict[str, Any],
    new: dict[str, Any],
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> list[Comparison]:
    """
    Compare the four cells of a 2x2 interaction analysis.
    """
    old_cells = old.get("cells", {})
    new_cells = new.get("cells", {})

    if not isinstance(old_cells, dict):
        old_cells = {}

    if not isinstance(new_cells, dict):
        new_cells = {}

    comparisons: list[Comparison] = []

    for cell in ("11", "10", "01", "00"):
        old_cell = old_cells.get(cell)
        new_cell = new_cells.get(cell)

        if isinstance(old_cell, dict) and isinstance(
            new_cell,
            dict,
        ):
            fields = sorted(
                set(old_cell) | set(new_cell)
            )

            for field in fields:
                comparisons.append(
                    _comparison(
                        f"cells.{cell}.{field}",
                        old_cell.get(field),
                        new_cell.get(field),
                        tolerance=tolerance,
                    )
                )
        else:
            comparisons.append(
                _comparison(
                    f"cells.{cell}",
                    old_cell,
                    new_cell,
                    tolerance=tolerance,
                )
            )

    return comparisons


def compare_interaction_analysis(
    old: dict[str, Any],
    new: dict[str, Any],
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> list[Comparison]:
    """
    Compare a complete interaction analysis.
    """
    comparisons: list[Comparison] = []

    comparisons.extend(
        compare_semantics(
            old,
            new,
            tolerance=tolerance,
        )
    )

    comparisons.extend(
        compare_population(
            old,
            new,
            tolerance=tolerance,
        )
    )

    comparisons.extend(
        compare_metrics(
            old,
            new,
            tolerance=tolerance,
        )
    )

    comparisons.extend(
        compare_interaction_cells(
            old,
            new,
            tolerance=tolerance,
        )
    )

    old_bootstrap = old.get("bootstrap", {})
    new_bootstrap = new.get("bootstrap", {})

    if not isinstance(old_bootstrap, dict):
        old_bootstrap = {}

    if not isinstance(new_bootstrap, dict):
        new_bootstrap = {}

    for field in (
        "iterations",
        "seed",
    ):
        if field in old_bootstrap or field in new_bootstrap:
            comparisons.append(
                _comparison(
                    f"bootstrap.{field}",
                    old_bootstrap.get(field),
                    new_bootstrap.get(field),
                    tolerance=tolerance,
                )
            )

    return comparisons


def _compare_nested(
    old: Any,
    new: Any,
    prefix: str,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> list[Comparison]:
    """
    Recursively compare nested dictionaries and lists.
    """
    comparisons: list[Comparison] = []

    if isinstance(old, dict) and isinstance(new, dict):
        fields = sorted(set(old) | set(new))

        for field in fields:
            child_prefix = (
                f"{prefix}.{field}"
                if prefix
                else field
            )

            comparisons.extend(
                _compare_nested(
                    old.get(field),
                    new.get(field),
                    child_prefix,
                    tolerance=tolerance,
                )
            )

        return comparisons

    if isinstance(old, list) and isinstance(new, list):
        max_length = max(
            len(old),
            len(new),
        )

        for index in range(max_length):
            child_prefix = f"{prefix}[{index}]"

            old_value = (
                old[index]
                if index < len(old)
                else None
            )

            new_value = (
                new[index]
                if index < len(new)
                else None
            )

            comparisons.extend(
                _compare_nested(
                    old_value,
                    new_value,
                    child_prefix,
                    tolerance=tolerance,
                )
            )

        return comparisons

    comparisons.append(
        _comparison(
            prefix,
            old,
            new,
            tolerance=tolerance,
        )
    )

    return comparisons


def compare_incremental_si_analysis(
    old: dict[str, Any],
    new: dict[str, Any],
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> list[Comparison]:
    """
    Compare the incremental SI analysis.

    This analysis contains several nested result structures, so the
    comparison deliberately walks the complete relevant result tree.
    """
    comparisons: list[Comparison] = []

    comparisons.extend(
        compare_semantics(
            old,
            new,
            tolerance=tolerance,
        )
    )

    comparisons.extend(
        compare_population(
            old,
            new,
            tolerance=tolerance,
        )
    )

    comparisons.extend(
        compare_metrics(
            old,
            new,
            tolerance=tolerance,
        )
    )

    for field in (
        "decile_analysis",
        "conditional_rank_analysis",
        "walk_forward_models",
    ):
        old_value = old.get(field)
        new_value = new.get(field)

        if old_value is not None or new_value is not None:
            comparisons.extend(
                _compare_nested(
                    old_value,
                    new_value,
                    field,
                    tolerance=tolerance,
                )
            )

    return comparisons


def compare_analysis(
    analysis: str,
    old: dict[str, Any],
    new: dict[str, Any],
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> list[Comparison]:
    """
    Dispatch comparison according to analysis type.
    """
    normalized = analysis.lower().strip()

    if normalized in {
        "interaction",
        "si_volatility_interaction",
        "si_volatility_downside_interaction",
    }:
        return compare_interaction_analysis(
            old,
            new,
            tolerance=tolerance,
        )

    if normalized in {
        "incremental_si_analysis",
        "incremental_si",
        "momentum_si_incremental_locked_oos",
    }:
        return compare_incremental_si_analysis(
            old,
            new,
            tolerance=tolerance,
        )

    comparisons: list[Comparison] = []

    comparisons.extend(
        compare_semantics(
            old,
            new,
            tolerance=tolerance,
        )
    )

    comparisons.extend(
        compare_population(
            old,
            new,
            tolerance=tolerance,
        )
    )

    comparisons.extend(
        compare_metrics(
            old,
            new,
            tolerance=tolerance,
        )
    )

    return comparisons


def _load_json(
    path: str | Path,
) -> dict[str, Any]:
    path = Path(path)

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
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
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> list[Comparison]:
    old = _load_json(old_path)
    new = _load_json(new_path)

    return compare_analysis(
        analysis,
        old,
        new,
        tolerance=tolerance,
    )


def _comparison_to_dict(
    comparison: Comparison,
) -> dict[str, Any]:
    return {
        "field": comparison.field,
        "old": comparison.old,
        "new": comparison.new,
        "equal": comparison.equal,
        "reason": comparison.reason,
    }


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

    parser.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_TOLERANCE,
        help=(
            "Absolute numeric comparison tolerance. "
            f"Default: {DEFAULT_TOLERANCE}"
        ),
    )

    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    comparisons = compare_files(
        analysis=args.analysis,
        old_path=args.old,
        new_path=args.new,
        tolerance=args.tolerance,
    )

    output_path = Path(args.output)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = {
        "analysis": args.analysis,
        "equivalent": all(
            comparison.equal
            for comparison in comparisons
        ),
        "comparisons": [
            _comparison_to_dict(comparison)
            for comparison in comparisons
        ],
    }

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            output,
            handle,
            indent=2,
            ensure_ascii=False,
            default=str,
        )

    equal_count = sum(
        comparison.equal
        for comparison in comparisons
    )

    difference_count = len(comparisons) - equal_count

    print(
        f"Migration comparison: {args.analysis}"
    )
    print(
        f"Comparisons: {len(comparisons)}"
    )
    print(
        f"Equal: {equal_count}"
    )
    print(
        f"Different: {difference_count}"
    )

    return 0 if difference_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
