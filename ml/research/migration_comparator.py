from __future__ import annotations

from dataclasses import dataclass
from math import isclose
from typing import Any


DEFAULT_ABS_TOL = 1e-10
DEFAULT_REL_TOL = 1e-8


@dataclass(frozen=True)
class ComparisonResult:
    name: str
    old_value: Any
    new_value: Any
    equal: bool
    reason: str = ""


def compare_scalar(
    name: str,
    old_value: Any,
    new_value: Any,
    *,
    abs_tol: float = DEFAULT_ABS_TOL,
    rel_tol: float = DEFAULT_REL_TOL,
) -> ComparisonResult:
    if old_value is None or new_value is None:
        equal = old_value == new_value
        return ComparisonResult(
            name=name,
            old_value=old_value,
            new_value=new_value,
            equal=equal,
            reason="exact comparison",
        )

    if isinstance(old_value, (int, float)) and isinstance(new_value, (int, float)):
        equal = isclose(
            float(old_value),
            float(new_value),
            abs_tol=abs_tol,
            rel_tol=rel_tol,
        )
        return ComparisonResult(
            name=name,
            old_value=old_value,
            new_value=new_value,
            equal=equal,
            reason="numeric comparison",
        )

    equal = old_value == new_value

    return ComparisonResult(
        name=name,
        old_value=old_value,
        new_value=new_value,
        equal=equal,
        reason="exact comparison",
    )


def compare_mapping(
    old: dict[str, Any],
    new: dict[str, Any],
    *,
    prefix: str = "",
) -> list[ComparisonResult]:
    results: list[ComparisonResult] = []

    keys = sorted(set(old) | set(new))

    for key in keys:
        name = f"{prefix}.{key}" if prefix else key

        if key not in old:
            results.append(
                ComparisonResult(
                    name=name,
                    old_value=None,
                    new_value=new[key],
                    equal=False,
                    reason="missing from legacy result",
                )
            )
            continue

        if key not in new:
            results.append(
                ComparisonResult(
                    name=name,
                    old_value=old[key],
                    new_value=None,
                    equal=False,
                    reason="missing from new result",
                )
            )
            continue

        old_value = old[key]
        new_value = new[key]

        if isinstance(old_value, dict) and isinstance(new_value, dict):
            results.extend(
                compare_mapping(
                    old_value,
                    new_value,
                    prefix=name,
                )
            )
            continue

        results.append(
            compare_scalar(
                name,
                old_value,
                new_value,
            )
        )

    return results
