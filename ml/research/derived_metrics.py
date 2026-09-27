from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from typing import Any

from .spec import ResearchSpec


def _regime_name(result: dict[str, Any]) -> str:
    if "incremental_signal" in result:
        return "incremental"

    if "baseline_signals" in result:
        return "baseline"

    if "regime" in result:
        return str(result["regime"])

    return "result"


def _result_groups(
    results: list[dict[str, Any]],
) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    groups: dict[
        tuple[str, str, str],
        list[dict[str, Any]],
    ] = defaultdict(list)

    for result in results:
        key = (
            str(result.get("window", "")),
            str(result.get("split", "")),
            _regime_name(result),
        )
        groups[key].append(result)

    return groups


def _target_event_counts(
    rows: list[dict[str, Any]],
) -> dict[str, int]:
    counts: dict[str, int] = {}

    for row in rows:
        target = row.get("target")
        events = row.get("events")

        if target is None or events is None:
            continue

        counts[str(target)] = int(events)

    return counts


def _safe_rate(
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
    Translate the original severity YAML shape into the executable
    derived-metric DSL.

    This keeps existing YAML files valid while giving new specs a
    structured representation.
    """
    derived = metadata.get("derived_metrics")
    if not isinstance(derived, dict):
        return []

    evaluation = metadata.get("evaluation")
    if not isinstance(evaluation, dict):
        return []

    severity = evaluation.get("severity")
    if not isinstance(severity, dict):
        return []

    numerator_target = severity.get("target")
    denominator_target = severity.get("conditioning_target")

    if not numerator_target or not denominator_target:
        return []

    definitions: list[dict[str, Any]] = []

    if "p_down_10_given_down_7" in derived:
        definitions.append(
            {
                "name": "p_down_10_given_down_7",
                "type": "conditional_event_rate",
                "numerator_target": str(numerator_target),
                "denominator_target": str(denominator_target),
            }
        )

    if "conditional_severity_difference" in derived:
        definitions.append(
            {
                "name": "conditional_severity_difference",
                "type": "rate_difference",
                "left": "incremental.p_down_10_given_down_7",
                "right": "baseline.p_down_10_given_down_7",
            }
        )

    if "conditional_severity_lift" in derived:
        definitions.append(
            {
                "name": "conditional_severity_lift",
                "type": "rate_ratio",
                "numerator": "incremental.p_down_10_given_down_7",
                "denominator": "baseline.p_down_10_given_down_7",
            }
        )

    return definitions


def _definitions(
    metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    raw = metadata.get("derived_metrics")

    if raw is None:
        return []

    if isinstance(raw, list):
        return [
            dict(item)
            for item in raw
            if isinstance(item, dict)
        ]

    if isinstance(raw, dict):
        return _legacy_definitions(metadata)

    raise ValueError(
        "metadata.derived_metrics måste vara en lista "
        "eller ett objekt."
    )


def _conditional_event_rate(
    definition: dict[str, Any],
    rows: list[dict[str, Any]],
) -> float | None:
    counts = _target_event_counts(rows)

    numerator_target = definition.get(
        "numerator_target"
    )
    denominator_target = definition.get(
        "denominator_target"
    )

    if not numerator_target or not denominator_target:
        raise ValueError(
            "conditional_event_rate kräver "
            "numerator_target och denominator_target."
        )

    return _safe_rate(
        counts.get(str(numerator_target)),
        counts.get(str(denominator_target)),
    )


def _resolve_reference(
    reference: str,
    values: dict[str, dict[str, float | None]],
) -> float | None:
    parts = str(reference).split(".", 1)

    if len(parts) == 1:
        return values.get("result", {}).get(parts[0])

    regime, metric = parts
    return values.get(regime, {}).get(metric)


def _calculate_group_metrics(
    definitions: list[dict[str, Any]],
    grouped_rows: dict[
        str,
        list[dict[str, Any]],
    ],
) -> dict[str, dict[str, float | None]]:
    values: dict[
        str,
        dict[str, float | None],
    ] = defaultdict(dict)

    for regime, rows in grouped_rows.items():
        for definition in definitions:
            if definition.get("type") != "conditional_event_rate":
                continue

            name = definition.get("name")
            if not name:
                raise ValueError(
                    "Derived metric saknar name."
                )

            values[regime][str(name)] = (
                _conditional_event_rate(
                    definition,
                    rows,
                )
            )

    for definition in definitions:
        name = definition.get("name")
        metric_type = definition.get("type")

        if not name:
            raise ValueError(
                "Derived metric saknar name."
            )

        if metric_type == "rate_difference":
            left = _resolve_reference(
                str(definition.get("left")),
                values,
            )
            right = _resolve_reference(
                str(definition.get("right")),
                values,
            )

            values["result"][str(name)] = (
                None
                if left is None or right is None
                else left - right
            )

        elif metric_type == "rate_ratio":
            numerator = _resolve_reference(
                str(definition.get("numerator")),
                values,
            )
            denominator = _resolve_reference(
                str(definition.get("denominator")),
                values,
            )

            values["result"][str(name)] = _safe_rate(
                numerator,
                denominator,
            )

        elif metric_type != "conditional_event_rate":
            raise ValueError(
                f"Okänd derived metric type "
                f"'{metric_type}'."
            )

    return values


def apply_derived_metrics(
    spec: ResearchSpec,
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Add declarative derived metrics without changing legacy results.

    Specs without metadata.derived_metrics are returned unchanged.
    All conditional probabilities are calculated from exact event
    counts in the raw result rows.
    """
    definitions = _definitions(spec.metadata)

    if not definitions:
        return results

    groups = _result_groups(results)

    by_window_split: dict[
        tuple[str, str],
        dict[str, list[dict[str, Any]]],
    ] = defaultdict(lambda: defaultdict(list))

    for (window, split, regime), rows in groups.items():
        by_window_split[(window, split)][regime].extend(rows)

    derived_by_row: dict[int, dict[str, float | None]] = {}

    for (window, split), regime_rows in by_window_split.items():
        values = _calculate_group_metrics(
            definitions,
            regime_rows,
        )

        for (group_window, group_split, regime), rows in groups.items():
            if (
                group_window != window
                or group_split != split
            ):
                continue

            metrics = dict(values.get(regime, {}))
            metrics.update(values.get("result", {}))

            for row in rows:
                derived_by_row[id(row)] = metrics

    enriched: list[dict[str, Any]] = []

    for row in results:
        updated = deepcopy(row)
        metrics = derived_by_row.get(id(row))

        if metrics:
            updated["regime"] = _regime_name(row)
            updated["derived_metrics"] = metrics

        enriched.append(updated)

    return enriched
