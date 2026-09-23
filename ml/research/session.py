from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ml.dataset import load_features
from ml.research.cache import (
    ResearchCache,
    build_research_cache,
)
from ml.research.experiments import Experiment


@dataclass
class ResearchSession:
    frame: pd.DataFrame
    cache: ResearchCache


def _required_experiments(
    specs,
) -> list[Experiment]:
    experiments: list[Experiment] = []
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

                    experiments.append(
                        Experiment(
                            experiment_id=(
                                f"spec_cache__"
                                f"{signal.name}__"
                                f"{signal.direction}__"
                                f"{fraction}__"
                                f"{target_name}"
                            ),
                            signal_name=signal.name,
                            target_name=target_name,
                            tail_fraction=fraction,
                            tail_direction=signal.direction,
                        )
                    )

    return experiments


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

    experiments = _required_experiments(
        specs
    )

    print(
        f"Cache requirements: "
        f"{len(experiments):,}",
        flush=True,
    )

    print(
        "Building shared research cache...",
        flush=True,
    )

    cache = build_research_cache(
        frame,
        experiments,
    )

    print(
        "Shared research cache ready.",
        flush=True,
    )

    return ResearchSession(
        frame=frame,
        cache=cache,
    )
