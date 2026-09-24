from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ml.dataset import load_features
from ml.research.cache import (
    ResearchCache,
    ResearchRequirement,
    build_research_cache,
)


@dataclass
class ResearchSession:
    """
    Gemensam context för en hel research-körning.

    Alla specs som körs tillsammans delar samma DataFrame
    och samma ResearchCache.
    """

    frame: pd.DataFrame
    cache: ResearchCache


def _required_requirements(
    specs,
) -> list[ResearchRequirement]:
    requirements: list[ResearchRequirement] = []

    seen: set[tuple] = set()

    for spec in specs:
        for signal in spec.signals:
            for target_name in spec.targets:
                for fraction in signal.bins:
                    key = (
                        signal.name,
                        target_name,
                        fraction,
                        signal.direction,
                    )

                    if key in seen:
                        continue

                    seen.add(key)

                    requirements.append(
                        ResearchRequirement(
                            signal_name=signal.name,
                            target_name=target_name,
                            tail_fraction=fraction,
                            tail_direction=signal.direction,
                        )
                    )

    return requirements


def build_session(
    specs,
) -> ResearchSession:
    print(
        "Loading features...",
        flush=True,
    )

    frame = load_features()

    print(
        f"Loaded {len(frame):,} feature rows",
        flush=True,
    )

    requirements = _required_requirements(
        specs
    )

    unique_signals = {
        signal.name
        for spec in specs
        for signal in spec.signals
    }

    unique_targets = {
        target
        for spec in specs
        for target in spec.targets
    }

    unique_tails = {
        (
            signal.name,
            signal.direction,
            fraction,
        )
        for spec in specs
        for signal in spec.signals
        for fraction in signal.bins
    }

    print(
        "Cache requirements: "
        f"{len(requirements):,} logical combinations",
        flush=True,
    )

    print(
        "Unique cache work: "
        f"{len(unique_signals):,} signals, "
        f"{len(unique_targets):,} targets, "
        f"{len(unique_tails):,} tails",
        flush=True,
    )

    print(
        "Building shared research cache...",
        flush=True,
    )

    cache = build_research_cache(
        frame,
        requirements,
    )

    print(
        "Shared research cache ready.",
        flush=True,
    )

    return ResearchSession(
        frame=frame,
        cache=cache,
    )
