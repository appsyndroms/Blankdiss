from __future__ import annotations

from typing import Any

from ml.dataset import load_features
from ml.research.cache import (
    ResearchRequirement,
    build_research_cache,
)
from ml.research.session import ResearchSession

from config import ADAPTIVE_FRACTIONS


def build_shared_session(
    source: dict[str, Any],
) -> ResearchSession:
    source_signals = source.get(
        "signals",
        [],
    )

    if not source_signals:
        raise ValueError(
            "Source spec has no signals."
        )

    source_targets = [
        str(value)
        for value in source.get(
            "targets",
            [],
        )
    ]

    if not source_targets:
        raise ValueError(
            "Source spec has no targets."
        )

    requirements: list[
        ResearchRequirement
    ] = []

    seen: set[tuple] = set()

    fractions = {
        float(value)
        for signal in source_signals
        for value in signal.get(
            "bins",
            [],
        )
    }

    fractions.update(
        float(value)
        for value in ADAPTIVE_FRACTIONS
    )

    for signal in source_signals:
        signal_name = str(
            signal["name"]
        )

        direction = str(
            signal.get(
                "direction",
                "lower",
            )
        )

        for target_name in source_targets:
            for fraction in sorted(
                fractions
            ):
                key = (
                    signal_name,
                    target_name,
                    fraction,
                    direction,
                )

                if key in seen:
                    continue

                seen.add(key)

                requirements.append(
                    ResearchRequirement(
                        signal_name=signal_name,
                        target_name=target_name,
                        tail_fraction=fraction,
                        tail_direction=direction,
                    )
                )

    print(
        "Loading features once for the complete AI Lab cycle...",
        flush=True,
    )

    frame = load_features()

    print(
        f"Loaded {len(frame):,} feature rows",
        flush=True,
    )

    print(
        "AI Lab cache requirements: "
        f"{len(requirements):,}",
        flush=True,
    )

    print(
        "Building AI Lab shared research cache once...",
        flush=True,
    )

    cache = build_research_cache(
        frame,
        requirements,
    )

    print(
        "AI Lab shared research cache ready.",
        flush=True,
    )

    return ResearchSession(
        frame=frame,
        cache=cache,
    )
