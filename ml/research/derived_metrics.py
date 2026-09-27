from __future__ import annotations
import ast
from collections import defaultdict
from copy import deepcopy
from typing import Any
from .spec import ResearchSpec
def _safe_divide(
    numerator: int | float | None,
    denominator: int | float | None,
) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return float(numerator) / float(denominator)
def _legacy_definitions(
    metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Translate the original dictionary-shaped YAML definition into
    the executable derived-metric representation.
    The formulas themselves remain authoritative. This adapter only
    converts the existing YAML shape into a list of named formulas.
    Example:
        derived_metrics:
          p_down_10_given_down_7:
            formula: down_10pct_5d_events / down_7pct_5d_events
    becomes:
        [
            {
                "name": "p_down_10_given_down_7",
                "formula": "...",
            }
        ]
    This keeps existing YAML files backwards compatible.
    """
    raw = metadata.get("derived_metrics")
    if not isinstance(raw, dict):
        return []
    definitions: list[dict[str, Any]] = []
    for name, definition in raw.items():
        if not isinstance(definition, dict):
            raise ValueError(
                f"Derived metric '{name}' måste vara ett objekt."
            )
        formula = definition.get("formula")
        if not isinstance(formula, str) or not formula.strip():
            raise ValueError(
                f"Derived metric '{name}' saknar en giltig formula."
            )
        definitions.append(
            {
                "name": str(name),
                "formula": formula.strip(),
            }
        )
    return definitions
def _definitions(
    metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Return normalized derived-metric definitions.
    Supported YAML forms:
    1. Existing dictionary form:
        derived_metrics:
          metric_name:
            formula: ...
    2. Structured list form:
        derived_metrics:
          - name: metric_name
            formula: ...
    Existing YAML files therefore continue to work unchanged.
    """
    raw = metadata.get("derived_metrics")
    if raw is None:
        return []
    if isinstance(raw, dict):
        return _legacy_definitions(metadata)
    if isinstance(raw, list):
        definitions: list[dict[str, Any]] = []
        for item in raw:
            if not isinstance(item, dict):
                raise ValueError(
                    "Varje derived metric måste vara ett objekt."
                )
            name = item.get("name")
            formula = item.get("formula")
            if not isinstance(name, str) or not name.strip():
                raise ValueError(
                    "Derived metric saknar name."
                )
            if not isinstance(formula, str) or not formula.strip():
                raise ValueError(
                    f"Derived metric '{name}' saknar en giltig formula."
                )
            definitions.append(
                {
                    "name": name.strip(),
                    "formula": formula.strip(),
                }
            )
        return definitions
    raise ValueError(
        "metadata.derived_metrics måste vara en lista "
        "eller ett objekt."
    )
def _target_counts(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, int | float | None]]:
    """
    Extract exact event counts from the nested-regime result.
    The returned structure is:
        {
            "down_7pct_5d": {
                "baseline_events": ...,
                "incremental_events": ...,
                "comparator_events": ...,
            },
            ...
        }
    No event rates are used here.
    """
    counts: dict[
        str,
        dict[str, int | float | None],
    ] = {}
    for row in rows:
        target = row.get("target")
        if target is None:
            continue
        target_name = str(target)
        counts[target_name] = {
            "baseline_events": row.get(
                "baseline_events"
            ),
            "incremental_events": row.get(
                "incremental_events"
            ),
            "comparator_events": row.get(
                "comparator_events"
            ),
        }
    return counts
def _generic_counts(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, int | float | None]]:
    """
    Extract target/event counts from result formats where each regime
    is represented as its own row.
    This is the compatibility path for non-nested analyses.
    """
    counts: dict[
        str,
        dict[str, int | float | None],
    ] = defaultdict(dict)
    for row in rows:
        target = row.get("target")
        if target is None:
            continue
        if "events" not in row:
            continue
        target_name = str(target)
        if "incremental_signal" in row:
            regime = "incremental"
        elif "baseline_signals" in row:
            regime = "baseline"
        elif "regime" in row:
            regime = str(row["regime"])
        else:
            regime = "result"
        counts[target_name][
            f"{regime}_events"
        ] = row.get("events")
    return dict(counts)
def _regime_from_name(
    name: str,
) -> tuple[str | None, str]:
    """
    Resolve optional regime suffixes used by formulas.
    Examples:
        p_x_incremental -> ("incremental", "p_x")
        p_x_baseline    -> ("baseline", "p_x")
        p_x_comparator  -> ("comparator", "p_x")
        p_x             -> (None, "p_x")
    """
    for regime in (
        "incremental",
        "baseline",
        "comparator",
    ):
        suffix = f"_{regime}"
        if name.endswith(suffix):
            return regime, name[: -len(suffix)]
    return None, name
def _safe_numeric_operation(
    operator: ast.operator,
    left: float | None,
    right: float | None,
) -> float | None:
    if left is None or right is None:
        return None
    if isinstance(operator, ast.Add):
        return left + right
    if isinstance(operator, ast.Sub):
        return left - right
    if isinstance(operator, ast.Mult):
        return left * right
    if isinstance(operator, ast.Div):
        if right == 0:
            return None
        return left / right
    raise ValueError(
        f"Operatorn '{type(operator).__name__}' "
        "stöds inte i derived metrics."
    )
def _evaluate_expression(
    expression: str,
    values: dict[str, float | None],
) -> float | None:
    """
    Safely evaluate a small arithmetic expression.
    Only numeric constants, named values, parentheses and
    + - * / are supported.
    No Python function calls, attributes, indexing or arbitrary
    code execution are permitted.
    """
    try:
        tree = ast.parse(
            expression,
            mode="eval",
        )
    except SyntaxError as exc:
        raise ValueError(
            f"Ogiltigt derived metric-uttryck: {expression!r}"
        ) from exc
    def evaluate(
        node: ast.AST,
    ) -> float | None:
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(
                node.value,
                (int, float),
            ):
                return float(node.value)
            raise ValueError(
                "Derived metric-formler får bara innehålla "
                "numeriska konstanter."
            )
        if isinstance(node, ast.Name):
            if node.id not in values:
                raise ValueError(
                    f"Derived metric refererar till okänt värde "
                    f"'{node.id}'."
                )
            return values[node.id]
        if isinstance(node, ast.UnaryOp):
            value = evaluate(node.operand)
            if value is None:
                return None
            if isinstance(node.op, ast.UAdd):
                return value
            if isinstance(node.op, ast.USub):
                return -value
            raise ValueError(
                f"Unary operator '{type(node.op).__name__}' "
                "stöds inte."
            )
        if isinstance(node, ast.BinOp):
            return _safe_numeric_operation(
                node.op,
                evaluate(node.left),
                evaluate(node.right),
            )
        raise ValueError(
            "Derived metric-formler får bara innehålla "
            "variabler, tal och + - * /."
        )
    return evaluate(tree)
def _base_values_for_regime(
    counts: dict[str, dict[str, int | float | None]],
    regime: str,
) -> dict[str, float | None]:
    """
    Expose exact event counts as formula variables.
    For example, with regime='baseline':
        down_10pct_5d_events
        down_7pct_5d_events
    """
    values: dict[str, float | None] = {}
    suffix = f"{regime}_events"
    for target, target_counts in counts.items():
        value = target_counts.get(suffix)
        values[
            f"{target}_events"
        ] = (
            None
            if value is None
            else float(value)
        )
    return values
def _result_values(
    regime_values: dict[
        str,
        dict[str, float | None],
    ],
) -> dict[str, float | None]:
    """
    Build formula variables for cross-regime expressions.
    Example:
        p_down_10_given_down_7_incremental
        p_down_10_given_down_7_baseline
    """
    values: dict[str, float | None] = {}
    for regime, metrics in regime_values.items():
        if regime == "result":
            continue
        for name, value in metrics.items():
            values[
                f"{name}_{regime}"
            ] = value
    return values
def _calculate_metrics(
    definitions: list[dict[str, Any]],
    counts: dict[str, dict[str, int | float | None]],
) -> dict[str, dict[str, float | None]]:
    """
    Calculate all derived metrics for one window/split group.
    Metrics whose formulas use raw event-count variables are
    calculated independently for each regime.
    Metrics whose formulas reference '<metric>_baseline',
    '<metric>_incremental' or '<metric>_comparator' are calculated
    once as cross-regime result metrics.
    """
    values: dict[
        str,
        dict[str, float | None],
    ] = {
        "baseline": {},
        "incremental": {},
        "comparator": {},
        "result": {},
    }
    regimes = (
        "baseline",
        "incremental",
        "comparator",
    )
    # First calculate metrics from exact event counts.
    for definition in definitions:
        name = str(definition["name"])
        formula = str(definition["formula"])
        regime_references = any(
            f"_{regime}" in formula
            for regime in regimes
        )
        if regime_references:
            continue
        for regime in regimes:
            formula_values = _base_values_for_regime(
                counts,
                regime,
            )
            values[regime][name] = (
                _evaluate_expression(
                    formula,
                    formula_values,
                )
            )
    # Then calculate metrics that compare regimes.
    for definition in definitions:
        name = str(definition["name"])
        formula = str(definition["formula"])
        regime_references = any(
            f"_{regime}" in formula
            for regime in regimes
        )
        if not regime_references:
            continue
        formula_values = _result_values(
            values
        )
        values["result"][name] = (
            _evaluate_expression(
                formula,
                formula_values,
            )
        )
    return values
def _apply_nested_metrics(
    definitions: list[dict[str, Any]],
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Apply derived metrics independently to each window/split.
    This is important because validation/test and window_1/window_2
    must never be mixed.
    The raw nested-regime rows remain untouched apart from the
    additional derived_metrics field.
    """
    grouped: dict[
        tuple[str, str],
        list[dict[str, Any]],
    ] = defaultdict(list)
    for row in results:
        grouped[
            (
                str(row.get("window", "")),
                str(row.get("split", "")),
            )
        ].append(row)
    metrics_by_group: dict[
        tuple[str, str],
        dict[str, dict[str, float | None]],
    ] = {}
    for key, rows in grouped.items():
        counts = _target_counts(rows)
        metrics_by_group[key] = _calculate_metrics(
            definitions,
            counts,
        )
    enriched: list[dict[str, Any]] = []
    for row in results:
        key = (
            str(row.get("window", "")),
            str(row.get("split", "")),
        )
        values = metrics_by_group[key]
        updated = deepcopy(row)
        updated["derived_metrics"] = {
            "baseline": dict(
                values.get("baseline", {})
            ),
            "incremental": dict(
                values.get("incremental", {})
            ),
            "comparator": dict(
                values.get("comparator", {})
            ),
            "result": dict(
                values.get("result", {})
            ),
        }
        enriched.append(updated)
    return enriched
def _apply_generic_metrics(
    definitions: list[dict[str, Any]],
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Compatibility path for result formats where each regime is
    represented as its own row.
    """
    grouped: dict[
        tuple[str, str],
        list[dict[str, Any]],
    ] = defaultdict(list)
    for row in results:
        grouped[
            (
                str(row.get("window", "")),
                str(row.get("split", "")),
            )
        ].append(row)
    enriched: list[dict[str, Any]] = []
    for key, rows in grouped.items():
        counts = _generic_counts(rows)
        values = _calculate_metrics(
            definitions,
            counts,
        )
        for row in rows:
            updated = deepcopy(row)
            updated["derived_metrics"] = {
                "baseline": dict(
                    values.get("baseline", {})
                ),
                "incremental": dict(
                    values.get("incremental", {})
                ),
                "comparator": dict(
                    values.get("comparator", {})
                ),
                "result": dict(
                    values.get("result", {})
                ),
            }
            enriched.append(updated)
    return enriched
def apply_derived_metrics(
    spec: ResearchSpec,
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Add declarative derived metrics to research results.
    Specs without metadata.derived_metrics are returned unchanged.
    The YAML formula is the source of truth. This module only
    supplies the exact event counts and safely evaluates arithmetic
    expressions.
    For nested_regime_comparison, formulas are calculated separately
    for every window/split combination and use the exact event
    counts from the corresponding target rows.
    Example:
        p_down_10_given_down_7:
            formula: >
                down_10pct_5d_events / down_7pct_5d_events
    is evaluated independently for baseline, incremental and
    comparator.
    A cross-regime formula such as:
        conditional_severity_difference:
            formula: >
                p_down_10_given_down_7_incremental -
                p_down_10_given_down_7_baseline
    is evaluated after the regime-specific metrics exist.
    """
    definitions = _definitions(
        spec.metadata
    )
    if not definitions:
        return results
    if not results:
        return results
    if (
        spec.analysis.type
        == "nested_regime_comparison"
        and all(
            "baseline_events" in row
            and "incremental_events" in row
            and "comparator_events" in row
            for row in results
        )
    ):
        return _apply_nested_metrics(
            definitions,
            results,
        )
    return _apply_generic_metrics(
        definitions,
        results,
    )
