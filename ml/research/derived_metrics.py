from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from typing import Any

from .spec import ResearchSpec


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
                "left": (
                    "incremental."
                    "p_down_10_given_down_7"
                ),
                "right": (
                    "baseline."
                    "p_down_10_given_down_7"
                ),
            }
        )

    if "conditional_severity_lift" in derived:
        definitions.append(
            {
                "name": "conditional_severity_lift",
                "type": "rate_ratio",
                "numerator": (
                    "incremental."
                    "p_down_10_given_down_7"
                ),
                "denominator": (
                    "baseline."
                    "p_down_10_given_down_7"
                ),
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


def _target_events(
    rows: list[dict[str, Any]],
    target: str,
) -> dict[str, int | None]:
    """
    Extract exact regime event counts for one target from the
    current nested_regime_comparison result shape.
    """
    for row in rows:
        if row.get("target") != target:
            continue

        return {
            "baseline": row.get("baseline_events"),
            "incremental": row.get(
                "incremental_events"
            ),
            "comparator": row.get(
                "comparator_events"
            ),
        }

    return {
        "baseline": None,
        "incremental": None,
        "comparator": None,
    }


def _resolve_reference(
    reference: str,
    values: dict[str, dict[str, float | None]],
) -> float | None:
    parts = str(reference).split(".", 1)

    if len(parts) == 1:
        return values.get(
            "result",
            {},
        ).get(parts[0])

    regime, metric = parts

    return values.get(
        regime,
        {},
    ).get(metric)


def _nested_severity_metrics(
    definitions: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, float | None]]:
    """
    Calculate derived metrics from the actual nested-regime output.

    A nested result contains all three regime counts on every target
    row. Conditional probabilities must therefore be calculated by
    matching numerator/denominator targets, not by treating rows as
    separate regimes.
    """
    values: dict[
        str,
        dict[str, float | None],
    ] = defaultdict(dict)

    # First calculate metrics that depend directly on event counts.
    for definition in definitions:
        metric_type = definition.get("type")
        name = definition.get("name")

        if not name:
            raise ValueError(
                "Derived metric saknar name."
            )

        if metric_type != "conditional_event_rate":
            continue

        numerator_target = definition.get(
            "numerator_target"
        )
        denominator_target = definition.get(
            "denominator_target"
        )

        if (
            not numerator_target
            or not denominator_target
        ):
            raise ValueError(
                "conditional_event_rate kräver "
                "numerator_target och "
                "denominator_target."
            )

        numerator = _target_events(
            rows,
            str(numerator_target),
        )
        denominator = _target_events(
            rows,
            str(denominator_target),
        )

        for regime in (
            "baseline",
            "incremental",
            "comparator",
        ):
            values[regime][str(name)] = _safe_rate(
                numerator[regime],
                denominator[regime],
            )

    # Then calculate metrics that depend on derived values.
    for definition in definitions:
        metric_type = definition.get("type")
        name = definition.get("name")

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


def _apply_nested_metrics(
    definitions: list[dict[str, Any]],
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Apply derived metrics independently for every window/split.

    This prevents validation/test or window_1/window_2 observations
    from being mixed together.
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
        metrics_by_group[key] = _nested_severity_metrics(
            definitions,
            rows,
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
                values.get(
                    "baseline",
                    {},
                )
            ),
            "incremental": dict(
                values.get(
                    "incremental",
                    {},
                )
            ),
            "comparator": dict(
                values.get(
                    "comparator",
                    {},
                )
            ),
            "result": dict(
                values.get(
                    "result",
                    {},
                )
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
    groups: dict[
        tuple[str, str, str],
        list[dict[str, Any]],
    ] = defaultdict(list)

    for result in results:
        if "incremental_signal" in result:
            regime = "incremental"
        elif "baseline_signals" in result:
            regime = "baseline"
        elif "regime" in result:
            regime = str(result["regime"])
        else:
            regime = "result"

        groups[
            (
                str(result.get("window", "")),
                str(result.get("split", "")),
                regime,
            )
        ].append(result)

    by_window_split: dict[
        tuple[str, str],
        dict[str, list[dict[str, Any]]],
    ] = defaultdict(
        lambda: defaultdict(list)
    )

    for (
        window,
        split,
        regime,
    ), rows in groups.items():
        by_window_split[
            (window, split)
        ][regime].extend(rows)

    enriched: list[dict[str, Any]] = []

    for row in results:
        key = (
            str(row.get("window", "")),
            str(row.get("split", "")),
        )

        regime_rows = by_window_split[key]

        values: dict[
            str,
            dict[str, float | None],
        ] = defaultdict(dict)

        for regime, rows in regime_rows.items():
            for definition in definitions:
                if (
                    definition.get("type")
                    != "conditional_event_rate"
                ):
                    continue

                name = definition.get("name")

                if not name:
                    raise ValueError(
                        "Derived metric saknar name."
                    )

                counts = {
                    str(item.get("target")): item.get(
                        "events"
                    )
                    for item in rows
                    if item.get("target") is not None
                }

                values[regime][str(name)] = _safe_rate(
                    counts.get(
                        str(
                            definition.get(
                                "numerator_target"
                            )
                        )
                    ),
                    counts.get(
                        str(
                            definition.get(
                                "denominator_target"
                            )
                        )
                    ),
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
                values["result"][str(name)] = _safe_rate(
                    _resolve_reference(
                        str(
                            definition.get(
                                "numerator"
                            )
                        ),
                        values,
                    ),
                    _resolve_reference(
                        str(
                            definition.get(
                                "denominator"
                            )
                        ),
                        values,
                    ),
                )

            elif metric_type != "conditional_event_rate":
                raise ValueError(
                    f"Okänd derived metric type "
                    f"'{metric_type}'."
                )

        updated = deepcopy(row)

        updated["derived_metrics"] = {
            **dict(
                values.get(
                    "baseline",
                    {},
                )
            ),
            **dict(
                values.get(
                    "incremental",
                    {},
                )
            ),
            **dict(
                values.get(
                    "result",
                    {},
                )
            ),
        }

        enriched.append(updated)

    return enriched


def apply_derived_metrics(
    spec: ResearchSpec,
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Add declarative derived metrics.

    Specs without metadata.derived_metrics are returned unchanged.

    For nested_regime_comparison, calculations use the exact
    baseline/incremental/comparator event counts from the raw result
    rows. No rounded event rates are used.
    """
    definitions = _definitions(
        spec.metadata
    )

    if not definitions:
        return results

    if (
        spec.analysis.type
        == "nested_regime_comparison"
        and results
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
