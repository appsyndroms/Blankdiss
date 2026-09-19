from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ml.config import TARGETS
from ml.dataset import build_target

from .config import DiscoveryConfig
from .engine import (
    Candidate,
    DiscoveryData,
    _target_return_column,
    build_candidates,
    find_candidates,
    pool_results,
)
from ml.research.signals import tail_mask


@dataclass(frozen=True)
class CandidateMasks:
    candidate: Candidate
    window_masks: dict[str, np.ndarray]
    selected_masks: dict[str, np.ndarray]
    rest_masks: dict[str, np.ndarray]
    baseline_masks: dict[str, np.ndarray]


def _target_map() -> dict[str, Any]:
    return {
        target.name: target
        for target in TARGETS
    }


def _build_candidate_masks(
    data: DiscoveryData,
    candidates: list[Candidate],
) -> list[CandidateMasks]:
    """
    Bygger alla signal/stress-selectioner en gång.

    Dessa förändras inte mellan permutationerna eftersom endast
    utfallet permuteras.
    """
    frame = data.frame

    result: list[CandidateMasks] = []

    for candidate in candidates:
        signal = data.signals[
            candidate.signal_name
        ]

        stress = data.stress_signals[
            candidate.stress_feature
        ]

        signal_mask = tail_mask(
            frame,
            pd.Series(
                signal,
                index=frame.index,
            ),
            candidate.signal_tail,
            direction="upper",
        ).to_numpy()

        stress_mask = tail_mask(
            frame,
            pd.Series(
                stress,
                index=frame.index,
            ),
            candidate.stress_tail,
            direction=candidate.stress_direction,
        ).to_numpy()

        window_masks: dict[str, np.ndarray] = {}
        selected_masks: dict[str, np.ndarray] = {}
        rest_masks: dict[str, np.ndarray] = {}
        baseline_masks: dict[str, np.ndarray] = {}

        for window_name, masks in data.windows.items():
            test_mask = masks["test"]

            selected = (
                test_mask
                & signal_mask
                & stress_mask
            )

            window_masks[window_name] = test_mask
            selected_masks[window_name] = selected
            rest_masks[window_name] = (
                test_mask
                & ~selected
            )
            baseline_masks[window_name] = test_mask

        result.append(
            CandidateMasks(
                candidate=candidate,
                window_masks=window_masks,
                selected_masks=selected_masks,
                rest_masks=rest_masks,
                baseline_masks=baseline_masks,
            )
        )

    return result


def _evaluate_candidate(
    candidate_masks: CandidateMasks,
    targets: dict[str, np.ndarray],
    returns: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    """
    Utvärderar en kandidat mot ett givet target/return-dataset.

    Samma grundmått som Discovery V1 används:
      - event rate
      - baseline event rate
      - lift
      - mean return
      - rest mean return
      - return difference

    Bootstrap utelämnas här eftersom null-testet körs tusentals gånger.
    Bootstrap påverkar inte Discovery V1:s discovery_score.
    """
    candidate = candidate_masks.candidate

    target = targets[
        candidate.target_name
    ]

    return_column = _target_return_column(
        candidate.target_name
    )

    returns_array = returns.get(
        return_column
    )

    rows: list[dict[str, Any]] = []

    for window_name in candidate_masks.window_masks:
        window_mask = candidate_masks.window_masks[
            window_name
        ]

        selected_mask = (
            candidate_masks.selected_masks[
                window_name
            ]
            & np.isfinite(target)
        )

        rest_mask = (
            candidate_masks.rest_masks[
                window_name
            ]
            & np.isfinite(target)
        )

        baseline_mask = (
            candidate_masks.baseline_masks[
                window_name
            ]
            & np.isfinite(target)
        )

        selected_target = target[
            selected_mask
        ]

        baseline_target = target[
            baseline_mask
        ]

        n = int(
            selected_target.size
        )

        events = (
            selected_target > 0
        )

        event_count = int(
            events.sum()
        )

        event_rate = (
            float(events.mean())
            if n
            else None
        )

        baseline_events = (
            baseline_target > 0
        )

        baseline_rate = (
            float(baseline_events.mean())
            if baseline_target.size
            else None
        )

        lift = (
            event_rate / baseline_rate
            if (
                event_rate is not None
                and baseline_rate is not None
                and baseline_rate > 0
            )
            else None
        )

        mean_return = None
        rest_mean_return = None
        return_difference = None

        if returns_array is not None:
            selected_returns = returns_array[
                selected_mask
                & np.isfinite(returns_array)
            ]

            rest_returns = returns_array[
                rest_mask
                & np.isfinite(returns_array)
            ]

            if selected_returns.size:
                mean_return = float(
                    selected_returns.mean()
                )

            if rest_returns.size:
                rest_mean_return = float(
                    rest_returns.mean()
                )

            if (
                mean_return is not None
                and rest_mean_return is not None
            ):
                return_difference = (
                    mean_return
                    - rest_mean_return
                )

        rows.append(
            {
                "candidate_id": candidate.candidate_id,
                "target_name": candidate.target_name,
                "signal_name": candidate.signal_name,
                "signal_tail": candidate.signal_tail,
                "stress_feature": candidate.stress_feature,
                "stress_tail": candidate.stress_tail,
                "stress_direction": candidate.stress_direction,
                "window": window_name,
                "split": "test",
                "n": n,
                "events": event_count,
                "event_rate": event_rate,
                "baseline_event_rate": baseline_rate,
                "lift": lift,
                "mean_return": mean_return,
                "rest_mean_return": rest_mean_return,
                "return_difference": return_difference,
                "bootstrap_ci_low": None,
                "bootstrap_ci_high": None,
                "selected_fraction": (
                    float(
                        selected_mask[
                            window_mask
                        ].mean()
                    )
                    if window_mask.any()
                    else None
                ),
            }
        )

    return rows


def _build_permuted_outcomes(
    frame: pd.DataFrame,
    config: DiscoveryConfig,
    rng: np.random.Generator,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
]:
    """
    Permuterar utfallet.

    Alla targets som använder samma return-kolumn får samma permutation.
    Det bevarar exempelvis relationen mellan down_5pct_5d,
    down_7pct_5d och down_10pct_5d.

    Signaler och stressvariabler lämnas helt orörda.
    """
    target_map = _target_map()

    unique_return_columns = {
        target_map[name].return_column
        for name in config.targets
    }

    permuted_returns: dict[
        str,
        np.ndarray,
    ] = {}

    for column in unique_return_columns:
        if column not in frame.columns:
            raise ValueError(
                f"Saknar return-kolumn: {column}"
            )

        values = pd.to_numeric(
            frame[column],
            errors="coerce",
        ).to_numpy(dtype=float)

        permuted = values.copy()

        valid = np.isfinite(
            permuted
        )

        valid_values = permuted[
            valid
        ].copy()

        rng.shuffle(
            valid_values
        )

        permuted[
            valid
        ] = valid_values

        permuted_returns[column] = permuted

    permuted_frame = frame.copy()

    for column, values in permuted_returns.items():
        permuted_frame[column] = values

    permuted_targets: dict[
        str,
        np.ndarray,
    ] = {}

    for target_name in config.targets:
        target_config = target_map[
            target_name
        ]

        permuted_targets[
            target_name
        ] = build_target(
            permuted_frame,
            target_config,
        ).to_numpy(
            dtype=float
        )

    return (
        permuted_targets,
        permuted_returns,
    )


def _max_discovery_score(
    pooled: list[dict[str, Any]],
    config: DiscoveryConfig,
) -> tuple[
    float | None,
    dict[str, Any] | None,
]:
    findings = find_candidates(
        pooled,
        config,
    )

    if not findings:
        return (
            None,
            None,
        )

    best = findings[0]

    return (
        float(
            best["discovery_score"]
        ),
        best,
    )


def evaluate_observed(
    data: DiscoveryData,
    config: DiscoveryConfig,
    candidate_masks: list[CandidateMasks],
) -> dict[str, Any]:
    """
    Kör samma discovery-logik som V1 utan bootstrap.

    Detta används för att få ett observed discovery_score som är
    direkt jämförbart med permutationerna.
    """
    results: list[dict[str, Any]] = []

    for candidate_masks_item in candidate_masks:
        results.extend(
            _evaluate_candidate(
                candidate_masks_item,
                data.targets,
                data.returns,
            )
        )

    pooled = pool_results(
        results,
        min_rows_per_window=(
            config.validation
            .min_rows_per_window
        ),
    )

    score, best = _max_discovery_score(
        pooled,
        config,
    )

    return {
        "results": results,
        "pooled": pooled,
        "findings": find_candidates(
            pooled,
            config,
        ),
        "max_discovery_score": score,
        "best_candidate": best,
    }


def run_permutations(
    data: DiscoveryData,
    config: DiscoveryConfig,
    candidate_masks: list[CandidateMasks],
    permutations: int,
    seed: int,
) -> list[dict[str, Any]]:
    if permutations < 1:
        raise ValueError(
            "Antalet permutationer måste vara >= 1."
        )

    rng = np.random.default_rng(
        seed
    )

    distribution: list[
        dict[str, Any]
    ] = []

    for permutation_index in range(
        1,
        permutations + 1,
    ):
        (
            permuted_targets,
            permuted_returns,
        ) = _build_permuted_outcomes(
            data.frame,
            config,
            rng,
        )

        results: list[
            dict[str, Any]
        ] = []

        for candidate_masks_item in candidate_masks:
            results.extend(
                _evaluate_candidate(
                    candidate_masks_item,
                    permuted_targets,
                    permuted_returns,
                )
            )

        pooled = pool_results(
            results,
            min_rows_per_window=(
                config.validation
                .min_rows_per_window
            ),
        )

        (
            max_score,
            best_candidate,
        ) = _max_discovery_score(
            pooled,
            config,
        )

        distribution.append(
            {
                "permutation": permutation_index,
                "max_discovery_score": max_score,
                "best_candidate_id": (
                    best_candidate[
                        "candidate_id"
                    ]
                    if best_candidate
                    else None
                ),
                "best_target": (
                    best_candidate[
                        "target_name"
                    ]
                    if best_candidate
                    else None
                ),
                "best_lift": (
                    best_candidate.get(
                        "lift"
                    )
                    if best_candidate
                    else None
                ),
                "best_return_difference": (
                    best_candidate.get(
                        "return_difference"
                    )
                    if best_candidate
                    else None
                ),
            }
        )

        if (
            permutation_index == 1
            or permutation_index % 10 == 0
            or permutation_index == permutations
        ):
            print(
                "Null permutation progress: "
                f"{permutation_index:,}/{permutations:,}",
                flush=True,
            )

    return distribution


def empirical_p_value(
    observed_score: float | None,
    null_distribution: list[dict[str, Any]],
) -> float | None:
    if observed_score is None:
        return None

    values = [
        row["max_discovery_score"]
        for row in null_distribution
        if row["max_discovery_score"] is not None
    ]

    if not values:
        return None

    exceedances = sum(
        value >= observed_score
        for value in values
    )

    return float(
        (exceedances + 1)
        / (len(values) + 1)
    )


def build_candidate_masks(
    data: DiscoveryData,
    config: DiscoveryConfig,
) -> tuple[
    list[Candidate],
    list[CandidateMasks],
]:
    candidates = build_candidates(
        config
    )

    candidate_masks = _build_candidate_masks(
        data,
        candidates,
    )

    return (
        candidates,
        candidate_masks,
    )
