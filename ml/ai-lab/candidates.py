from __future__ import annotations

import itertools
from typing import Any

from config import (
    ADAPTIVE_FRACTIONS,
    ADAPTIVE_PREFIX,
    RUNS_DIR,
    SPEC_DIR,
)


def load_yaml(
    path,
) -> dict[str, Any]:
    import yaml

    payload = yaml.safe_load(
        path.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(
        payload,
        dict,
    ):
        raise ValueError(
            f"Research spec must be an object: {path}"
        )

    return payload


def candidate_grid(
    source: dict[str, Any],
) -> list[
    tuple[
        float,
        float,
        str,
    ]
]:
    targets = source.get(
        "targets",
        [],
    )

    if not targets:
        raise ValueError(
            "Source spec has no targets."
        )

    return [
        (
            float(baseline),
            float(incremental),
            str(target),
        )
        for baseline, incremental, target
        in itertools.product(
            ADAPTIVE_FRACTIONS,
            ADAPTIVE_FRACTIONS,
            [
                str(target)
                for target in targets
            ],
        )
    ]


def candidate_key(
    baseline: float,
    incremental: float,
    target: str,
) -> str:
    return (
        f"{baseline:.4f}|"
        f"{incremental:.4f}|"
        f"{target}"
    )


def existing_adaptive_specs() -> dict[
    str,
    dict[str, Any],
]:
    result: dict[
        str,
        dict[str, Any],
    ] = {}

    for path in SPEC_DIR.glob(
        f"{ADAPTIVE_PREFIX}*.yaml"
    ):
        try:
            payload = load_yaml(
                path
            )
        except Exception:
            continue

        spec_id = payload.get(
            "id"
        )

        if spec_id:
            result[
                str(spec_id)
            ] = payload

    return result


def spec_candidate(
    spec: dict[str, Any],
) -> tuple[
    float,
    float,
    str,
] | None:
    try:
        signals = spec[
            "signals"
        ]

        baseline = float(
            signals[0][
                "bins"
            ][0]
        )

        incremental = float(
            signals[1][
                "bins"
            ][0]
        )

        target = str(
            spec[
                "targets"
            ][0]
        )

        return (
            baseline,
            incremental,
            target,
        )

    except (
        KeyError,
        IndexError,
        TypeError,
        ValueError,
    ):
        return None


def generated_candidate_ids() -> dict[
    str,
    str,
]:
    mapping: dict[
        str,
        str,
    ] = {}

    for (
        spec_id,
        spec,
    ) in existing_adaptive_specs().items():
        candidate = spec_candidate(
            spec
        )

        if candidate is None:
            continue

        mapping[
            candidate_key(
                *candidate
            )
        ] = spec_id

    return mapping


def source_candidate_keys(
    source: dict[str, Any],
) -> set[str]:
    signals = source.get(
        "signals",
        [],
    )

    if len(signals) != 2:
        return set()

    baseline_bins = [
        float(value)
        for value in signals[0].get(
            "bins",
            [],
        )
    ]

    incremental_bins = [
        float(value)
        for value in signals[1].get(
            "bins",
            [],
        )
    ]

    targets = [
        str(value)
        for value in source.get(
            "targets",
            [],
        )
    ]

    return {
        candidate_key(
            baseline,
            incremental,
            target,
        )
        for (
            baseline,
            incremental,
            target,
        ) in itertools.product(
            baseline_bins,
            incremental_bins,
            targets,
        )
    }


def completed_spec_ids() -> set[str]:
    completed: set[str] = set()

    if not RUNS_DIR.exists():
        return completed

    import json

    for manifest_path in RUNS_DIR.glob(
        "*/manifest.json"
    ):
        try:
            payload = json.loads(
                manifest_path.read_text(
                    encoding="utf-8"
                )
            )
        except (
            OSError,
            json.JSONDecodeError,
        ):
            continue

        if not isinstance(
            payload,
            dict,
        ):
            continue

        for item in payload.get(
            "specs",
            [],
        ):
            if isinstance(
                item,
                str,
            ):
                completed.add(item)
                continue

            if not isinstance(
                item,
                dict,
            ):
                continue

            value = (
                item.get("id")
                or item.get("spec_id")
            )

            if value is not None:
                completed.add(
                    str(value)
                )

    return completed


def choose_next_candidate(
    source: dict[str, Any],
) -> tuple[
    float,
    float,
    str,
] | None:
    completed = completed_spec_ids()
    generated = generated_candidate_ids()
    source_keys = source_candidate_keys(
        source
    )

    for (
        baseline,
        incremental,
        target,
    ) in candidate_grid(
        source
    ):
        key = candidate_key(
            baseline,
            incremental,
            target,
        )

        # Already covered by the original research.
        if key in source_keys:
            continue

        spec_id = generated.get(
            key
        )

        if (
            spec_id is not None
            and spec_id in completed
        ):
            continue

        return (
            baseline,
            incremental,
            target,
        )

    return None
