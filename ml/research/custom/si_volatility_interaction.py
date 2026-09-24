from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml.research.cache import (
    ResearchRequirement,
    build_research_cache,
)
from ml.research.session import (
    ResearchSession,
)
from ml.research.spec import (
    load_spec,
)


DISCOVERY_END = pd.Timestamp(
    "2025-12-19"
)

TARGET_NAME = "down_10pct_5d"

TAILS = (
    0.20,
    0.10,
    0.05,
    0.025,
    0.01,
)

BOOTSTRAP_ITERATIONS = 2000
BOOTSTRAP_SEED = 20260920

SPEC_PATH = Path(
    "ml/research/specs/"
    "si_volatility_downside_interaction.yaml"
)

OUTPUT_DIR = Path(
    "data/processed/ml/research/"
    "custom/si_volatility_interaction"
)


def _tail_key(
    signal_name: str,
    direction: str,
    fraction: float,
) -> str:
    return (
        f"{signal_name}|"
        f"{direction}|"
        f"{fraction}"
    )


def _evaluate_cell(
    *,
    target: np.ndarray,
    returns: np.ndarray,
    mask: np.ndarray,
    label: str,
) -> dict:
    valid = (
        mask
        & np.isfinite(target)
    )

    selected_target = target[valid]

    selected_returns = returns[
        valid
        & np.isfinite(returns)
    ]

    n = int(valid.sum())

    events = int(
        (selected_target > 0).sum()
    )

    event_rate = (
        events / n
        if n
        else None
    )

    mean_return = (
        float(selected_returns.mean())
        if len(selected_returns)
        else None
    )

    return {
        "label": label,
        "n": n,
        "events": events,
        "event_rate": event_rate,
        "mean_return": mean_return,
    }


def _bootstrap_interaction(
    *,
    rates: dict[str, float],
    counts: dict[str, int],
) -> dict:
    rng = np.random.default_rng(
        BOOTSTRAP_SEED
    )

    keys = (
        "11",
        "10",
        "01",
        "00",
    )

    draws = {}

    for key in keys:
        draws[key] = (
            rng.binomial(
                counts[key],
                rates[key],
                size=BOOTSTRAP_ITERATIONS,
            )
            / counts[key]
        )

    interaction = (
        (draws["11"] - draws["10"])
        -
        (draws["01"] - draws["00"])
    )

    valid_rr = (
        (draws["10"] > 0)
        & (draws["01"] > 0)
        & (draws["00"] > 0)
    )

    rr_interaction = np.full(
        BOOTSTRAP_ITERATIONS,
        np.nan,
        dtype=float,
    )

    rr_interaction[valid_rr] = (
        (
            draws["11"][valid_rr]
            / draws["10"][valid_rr]
        )
        /
        (
            draws["01"][valid_rr]
            / draws["00"][valid_rr]
        )
    )

    interaction_ci = np.percentile(
        interaction,
        [2.5, 97.5],
    )

    valid_rr_values = (
        rr_interaction[
            np.isfinite(rr_interaction)
        ]
    )

    if len(valid_rr_values):
        rr_ci = np.percentile(
            valid_rr_values,
            [2.5, 97.5],
        )

        rr_median = float(
            np.median(valid_rr_values)
        )
    else:
        rr_ci = (
            None,
            None,
        )

        rr_median = None

    return {
        "bootstrap_iterations": (
            BOOTSTRAP_ITERATIONS
        ),
        "bootstrap_seed": (
            BOOTSTRAP_SEED
        ),
        "interaction_ci_low": float(
            interaction_ci[0]
        ),
        "interaction_ci_high": float(
            interaction_ci[1]
        ),
        "rr_interaction_median": (
            rr_median
        ),
        "rr_interaction_ci_low": (
            float(rr_ci[0])
            if rr_ci[0] is not None
            else None
        ),
        "rr_interaction_ci_high": (
            float(rr_ci[1])
            if rr_ci[1] is not None
            else None
        ),
        "rr_bootstrap_valid_fraction": (
            float(
                len(valid_rr_values)
                / BOOTSTRAP_ITERATIONS
            )
        ),
    }


def _calculate_2x2(
    *,
    target: np.ndarray,
    returns: np.ndarray,
    si_mask: np.ndarray,
    vol_mask: np.ndarray,
    si_tail: float,
    vol_tail: float,
) -> dict:
    masks = {
        "11": (
            si_mask
            & vol_mask
        ),
        "10": (
            si_mask
            & ~vol_mask
        ),
        "01": (
            ~si_mask
            & vol_mask
        ),
        "00": (
            ~si_mask
            & ~vol_mask
        ),
    }

    cells = {}

    for key, mask in masks.items():
        cells[key] = _evaluate_cell(
            target=target,
            returns=returns,
            mask=mask,
            label=key,
        )

    rates = {
        key: cells[key]["event_rate"]
        for key in masks
    }

    counts = {
        key: cells[key]["n"]
        for key in masks
    }

    if any(
        rates[key] is None
        for key in rates
    ):
        return {
            "si_tail": si_tail,
            "vol_tail": vol_tail,
            "status": "insufficient_data",
            "cells": cells,
        }

    vol_effect_high_si = (
        rates["11"]
        - rates["10"]
    )

    vol_effect_low_si = (
        rates["01"]
        - rates["00"]
    )

    risk_difference_interaction = (
        vol_effect_high_si
        - vol_effect_low_si
    )

    si_effect_high_vol = (
        rates["11"]
        - rates["01"]
    )

    si_effect_low_vol = (
        rates["10"]
        - rates["00"]
    )

    risk_difference_interaction_check = (
        si_effect_high_vol
        - si_effect_low_vol
    )

    if not np.isclose(
        risk_difference_interaction,
        risk_difference_interaction_check,
    ):
        raise RuntimeError(
            "2x2 interaction identity failed."
        )

    risk_ratio_interaction = None

    if (
        rates["10"] > 0
        and rates["01"] > 0
        and rates["00"] > 0
    ):
        risk_ratio_interaction = (
            (
                rates["11"]
                / rates["10"]
            )
            /
            (
                rates["01"]
                / rates["00"]
            )
        )

    bootstrap = _bootstrap_interaction(
        rates=rates,
        counts=counts,
    )

    return_values = {}

    for key, mask in masks.items():
        selected = returns[
            mask
            & np.isfinite(returns)
        ]

        return_values[key] = (
            float(selected.mean())
            if len(selected)
            else None
        )

    return_interaction = None

    if all(
        return_values[key] is not None
        for key in return_values
    ):
        return_interaction = (
            (
                return_values["11"]
                - return_values["10"]
            )
            -
            (
                return_values["01"]
                - return_values["00"]
            )
        )

    return {
        "si_tail": si_tail,
        "vol_tail": vol_tail,
        "status": "ok",

        "cell_11_si_high_vol_high":
            cells["11"],

        "cell_10_si_high_vol_low":
            cells["10"],

        "cell_01_si_low_vol_high":
            cells["01"],

        "cell_00_si_low_vol_low":
            cells["00"],

        "vol_effect_when_si_high":
            vol_effect_high_si,

        "vol_effect_when_si_low":
            vol_effect_low_si,

        "si_effect_when_vol_high":
            si_effect_high_vol,

        "si_effect_when_vol_low":
            si_effect_low_vol,

        "risk_difference_interaction":
            risk_difference_interaction,

        "risk_difference_interaction_ci_low":
            bootstrap[
                "interaction_ci_low"
            ],

        "risk_difference_interaction_ci_high":
            bootstrap[
                "interaction_ci_high"
            ],

        "risk_ratio_interaction":
            risk_ratio_interaction,

        "rr_interaction_bootstrap_median":
            bootstrap[
                "rr_interaction_median"
            ],

        "rr_interaction_ci_low":
            bootstrap[
                "rr_interaction_ci_low"
            ],

        "rr_interaction_ci_high":
            bootstrap[
                "rr_interaction_ci_high"
            ],

        "rr_bootstrap_valid_fraction":
            bootstrap[
                "rr_bootstrap_valid_fraction"
            ],

        "return_interaction":
            return_interaction,

        "bootstrap_iterations":
            bootstrap[
                "bootstrap_iterations"
            ],

        "bootstrap_seed":
            bootstrap[
                "bootstrap_seed"
            ],
    }


def _build_session() -> ResearchSession:
    spec = load_spec(
        SPEC_PATH
    )

    requirements = []

    for signal in spec.signals:
        for target in spec.targets:
            for fraction in signal.bins:
                requirements.append(
                    ResearchRequirement(
                        signal_name=signal.name,
                        target_name=target,
                        tail_fraction=fraction,
                        tail_direction=signal.direction,
                    )
                )

    from ml.dataset import load_features

    frame = load_features()

    cache = build_research_cache(
        frame,
        requirements,
    )

    return ResearchSession(
        frame=frame,
        cache=cache,
    )


def run() -> dict:
    session = _build_session()

    frame = session.frame.copy()

    frame["snapshot_date"] = (
        pd.to_datetime(
            frame["snapshot_date"],
            errors="coerce",
        )
    )

    discovery_mask = (
        frame["snapshot_date"]
        <= DISCOVERY_END
    ).to_numpy()

    target = session.cache.targets[
        TARGET_NAME
    ][discovery_mask]

    target_config = (
        session.cache.target_configs[
            TARGET_NAME
        ]
    )

    returns = session.cache.returns[
        target_config.return_column
    ][discovery_mask]

    si_name = "short_interest_change"
    vol_name = "price_volatility_20d"

    interactions = []

    for si_tail in TAILS:
        si_mask = (
            session.cache.tail_masks[
                _tail_key(
                    si_name,
                    "upper",
                    si_tail,
                )
            ][discovery_mask]
        )

        for vol_tail in TAILS:
            vol_mask = (
                session.cache.tail_masks[
                    _tail_key(
                        vol_name,
                        "upper",
                        vol_tail,
                    )
                ][discovery_mask]
            )

            interactions.append(
                _calculate_2x2(
                    target=target,
                    returns=returns,
                    si_mask=si_mask,
                    vol_mask=vol_mask,
                    si_tail=si_tail,
                    vol_tail=vol_tail,
                )
            )

    valid_target = np.isfinite(
        target
    )

    baseline_event_rate = (
        float(
            (target[valid_target] > 0).mean()
        )
        if valid_target.any()
        else None
    )

    return {
        "discovery_end": str(
            DISCOVERY_END.date()
        ),
        "target": TARGET_NAME,
        "rows": int(
            discovery_mask.sum()
        ),
        "baseline_event_rate":
            baseline_event_rate,
        "tails": list(TAILS),
        "bootstrap": {
            "iterations":
                BOOTSTRAP_ITERATIONS,
            "seed":
                BOOTSTRAP_SEED,
            "method":
                (
                    "parametric binomial bootstrap "
                    "of four 2x2 cell event rates"
                ),
        },
        "interactions": interactions,
    }


def main() -> None:
    result = run()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_DIR
        / "results.json"
    )

    output_path.write_text(
        json.dumps(
            result,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"Written to {output_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
