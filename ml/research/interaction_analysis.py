from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from ml.config import TARGETS
from ml.dataset import build_target, load_features
from ml.research.signals import build_signal, tail_mask
DISCOVERY_END = pd.Timestamp("2025-12-19")
TARGET_NAME = "down_10pct_5d"
TAILS = [0.01, 0.025, 0.05, 0.10]
BOOTSTRAP_ITERATIONS = 2000
BOOTSTRAP_SEED = 20260920
OUTPUT_DIR = Path(
    "data/processed/ml/research/interaction"
)
def target_config():
    for target in TARGETS:
        if target.name == TARGET_NAME:
            return target
    raise ValueError(
        f"Unknown target: {TARGET_NAME}"
    )
def evaluate(
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
def bootstrap_interaction(
    rates: dict[str, float],
    counts: dict[str, int],
    *,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    rng = np.random.default_rng(seed)
    n_boot = {
        key: counts[key]
        for key in (
            "11",
            "10",
            "01",
            "00",
        )
    }
    p = {
        key: rates[key]
        for key in (
            "11",
            "10",
            "01",
            "00",
        )
    }
    draws = {}
    for key in (
        "11",
        "10",
        "01",
        "00",
    ):
        draws[key] = (
            rng.binomial(
                n_boot[key],
                p[key],
                size=iterations,
            )
            / n_boot[key]
        )
    # Additive interaction:
    #
    # (risk11 - risk10)
    # -
    # (risk01 - risk00)
    #
    # = effect of volatility when SI is high
    #   minus
    #   effect of volatility when SI is low
    #
    interaction = (
        (draws["11"] - draws["10"])
        - (draws["01"] - draws["00"])
    )
    # Multiplicative interaction:
    #
    # (risk11 / risk10)
    # /
    # (risk01 / risk00)
    #
    # Values > 1 indicate that the volatility
    # effect is larger when SI is high.
    #
    # Guard against zero bootstrap rates.
    valid_rr = (
        (draws["10"] > 0)
        & (draws["01"] > 0)
        & (draws["00"] > 0)
    )
    rr_interaction = np.full(
        iterations,
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
        rr_ci = [
            None,
            None,
        ]
        rr_median = None
    return {
        "bootstrap_iterations": iterations,
        "bootstrap_seed": seed,
        "interaction_ci_low": float(
            interaction_ci[0]
        ),
        "interaction_ci_high": float(
            interaction_ci[1]
        ),
        "rr_interaction_median": rr_median,
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
                / iterations
            )
        ),
    }
def calculate_2x2_interaction(
    *,
    target: np.ndarray,
    returns: np.ndarray,
    si_mask: np.ndarray,
    vol_mask: np.ndarray,
    si_tail: float,
    vol_tail: float,
) -> dict:
    base = np.ones(
        len(target),
        dtype=bool,
    )
    # Four cells:
    #
    # 11 = SI high + VOL high
    # 10 = SI high + VOL low
    # 01 = SI low  + VOL high
    # 00 = SI low  + VOL low
    masks = {
        "11": (
            base
            & si_mask
            & vol_mask
        ),
        "10": (
            base
            & si_mask
            & ~vol_mask
        ),
        "01": (
            base
            & ~si_mask
            & vol_mask
        ),
        "00": (
            base
            & ~si_mask
            & ~vol_mask
        ),
    }
    cells = {}
    for key, mask in masks.items():
        cells[key] = evaluate(
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
    # Need all cells populated.
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
    # ---------------------------------------------------------
    # Additive interaction
    #
    # VOL effect when SI is high
    # minus
    # VOL effect when SI is low
    # ---------------------------------------------------------
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
    # Equivalent formulation:
    #
    # (SI effect under high VOL)
    # -
    # (SI effect under low VOL)
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
    # ---------------------------------------------------------
    # Multiplicative interaction
    # ---------------------------------------------------------
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
    bootstrap = bootstrap_interaction(
        rates=rates,
        counts=counts,
    )
    # ---------------------------------------------------------
    # Return interaction
    #
    # Same additive 2x2 structure, using mean 5d return.
    #
    # This is exploratory because return distributions are
    # not independent observations in the strict sense.
    # ---------------------------------------------------------
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
        "cell_11_si_high_vol_high": cells["11"],
        "cell_10_si_high_vol_low": cells["10"],
        "cell_01_si_low_vol_high": cells["01"],
        "cell_00_si_low_vol_low": cells["00"],
        "vol_effect_when_si_high": (
            vol_effect_high_si
        ),
        "vol_effect_when_si_low": (
            vol_effect_low_si
        ),
        "si_effect_when_vol_high": (
            si_effect_high_vol
        ),
        "si_effect_when_vol_low": (
            si_effect_low_vol
        ),
        "risk_difference_interaction": (
            risk_difference_interaction
        ),
        "risk_difference_interaction_ci_low": (
            bootstrap[
                "interaction_ci_low"
            ]
        ),
        "risk_difference_interaction_ci_high": (
            bootstrap[
                "interaction_ci_high"
            ]
        ),
        "risk_ratio_interaction": (
            risk_ratio_interaction
        ),
        "rr_interaction_bootstrap_median": (
            bootstrap[
                "rr_interaction_median"
            ]
        ),
        "rr_interaction_ci_low": (
            bootstrap[
                "rr_interaction_ci_low"
            ]
        ),
        "rr_interaction_ci_high": (
            bootstrap[
                "rr_interaction_ci_high"
            ]
        ),
        "rr_bootstrap_valid_fraction": (
            bootstrap[
                "rr_bootstrap_valid_fraction"
            ]
        ),
        "return_interaction": (
            return_interaction
        ),
        "bootstrap_iterations": (
            bootstrap[
                "bootstrap_iterations"
            ]
        ),
        "bootstrap_seed": (
            bootstrap[
                "bootstrap_seed"
            ]
        ),
    }
def main() -> None:
    print(
        "Loading features...",
        flush=True,
    )
    frame = load_features().copy()
    frame["snapshot_date"] = pd.to_datetime(
        frame["snapshot_date"],
        errors="coerce",
    )
    frame = frame.loc[
        frame["snapshot_date"]
        <= DISCOVERY_END
    ].copy()
    frame.reset_index(
        drop=True,
        inplace=True,
    )
    print(
        f"Rows through "
        f"{DISCOVERY_END.date()}: "
        f"{len(frame):,}",
        flush=True,
    )
    target_cfg = target_config()
    target = build_target(
        frame,
        target_cfg,
    ).to_numpy(
        dtype=float
    )
    returns = pd.to_numeric(
        frame[target_cfg.return_column],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )
    si = build_signal(
        frame,
        "short_interest_change",
    )
    volatility = build_signal(
        frame,
        "price_volatility_20d",
    )
    baseline_target = target[
        np.isfinite(target)
    ]
    baseline_event_rate = (
        float(
            (baseline_target > 0).mean()
        )
        if len(baseline_target)
        else None
    )
    results = []
    # ---------------------------------------------------------
    # 1. SI ONLY
    # ---------------------------------------------------------
    for tail in TAILS:
        si_mask = tail_mask(
            frame,
            si,
            tail,
            direction="upper",
        ).to_numpy()
        result = evaluate(
            target=target,
            returns=returns,
            mask=si_mask,
            label=f"SI {tail:g}",
        )
        result.update(
            {
                "analysis": "si_only",
                "si_tail": tail,
                "vol_tail": None,
                "baseline_event_rate": (
                    baseline_event_rate
                ),
                "lift": (
                    result["event_rate"]
                    / baseline_event_rate
                    if (
                        result["event_rate"]
                        is not None
                        and baseline_event_rate
                        and baseline_event_rate > 0
                    )
                    else None
                ),
            }
        )
        results.append(result)
    # ---------------------------------------------------------
    # 2. VOLATILITY ONLY
    # ---------------------------------------------------------
    for tail in TAILS:
        vol_mask = tail_mask(
            frame,
            volatility,
            tail,
            direction="upper",
        ).to_numpy()
        result = evaluate(
            target=target,
            returns=returns,
            mask=vol_mask,
            label=f"VOL {tail:g}",
        )
        result.update(
            {
                "analysis": "volatility_only",
                "si_tail": None,
                "vol_tail": tail,
                "baseline_event_rate": (
                    baseline_event_rate
                ),
                "lift": (
                    result["event_rate"]
                    / baseline_event_rate
                    if (
                        result["event_rate"]
                        is not None
                        and baseline_event_rate
                        and baseline_event_rate > 0
                    )
                    else None
                ),
            }
        )
        results.append(result)
    # ---------------------------------------------------------
    # 3. SI × VOL
    # ---------------------------------------------------------
    for si_tail in TAILS:
        si_mask = tail_mask(
            frame,
            si,
            si_tail,
            direction="upper",
        ).to_numpy()
        for vol_tail in TAILS:
            vol_mask = tail_mask(
                frame,
                volatility,
                vol_tail,
                direction="upper",
            ).to_numpy()
            mask = (
                si_mask
                & vol_mask
            )
            result = evaluate(
                target=target,
                returns=returns,
                mask=mask,
                label=(
                    f"SI {si_tail:g} × "
                    f"VOL {vol_tail:g}"
                ),
            )
            result.update(
                {
                    "analysis": "interaction",
                    "si_tail": si_tail,
                    "vol_tail": vol_tail,
                    "baseline_event_rate": (
                        baseline_event_rate
                    ),
                    "lift": (
                        result["event_rate"]
                        / baseline_event_rate
                        if (
                            result["event_rate"]
                            is not None
                            and baseline_event_rate
                            and baseline_event_rate > 0
                        )
                        else None
                    ),
                }
            )
            results.append(result)
    # ---------------------------------------------------------
    # 4. FORMAL 2×2 INTERACTION
    # ---------------------------------------------------------
    interactions = []
    for si_tail in TAILS:
        si_mask = tail_mask(
            frame,
            si,
            si_tail,
            direction="upper",
        ).to_numpy()
        for vol_tail in TAILS:
            vol_mask = tail_mask(
                frame,
                volatility,
                vol_tail,
                direction="upper",
            ).to_numpy()
            result = calculate_2x2_interaction(
                target=target,
                returns=returns,
                si_mask=si_mask,
                vol_mask=vol_mask,
                si_tail=si_tail,
                vol_tail=vol_tail,
            )
            interactions.append(result)
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    document = {
        "discovery_end": str(
            DISCOVERY_END.date()
        ),
        "target": TARGET_NAME,
        "rows": len(frame),
        "baseline_event_rate": (
            baseline_event_rate
        ),
        "tails": TAILS,
        "bootstrap": {
            "iterations": (
                BOOTSTRAP_ITERATIONS
            ),
            "seed": BOOTSTRAP_SEED,
            "method": (
                "parametric binomial bootstrap "
                "of four 2x2 cell event rates"
            ),
        },
        "results": results,
        "interactions": interactions,
    }
    with (
        OUTPUT_DIR / "results.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as fh:
        json.dump(
            document,
            fh,
            indent=2,
        )
    pd.DataFrame(results).to_csv(
        OUTPUT_DIR / "results.csv",
        index=False,
    )
    pd.DataFrame(
        interactions
    ).to_csv(
        OUTPUT_DIR / "interactions.csv",
        index=False,
    )
    print()
    print(
        "=== FORMAL 2×2 INTERACTION ==="
    )
    interaction_frame = (
        pd.DataFrame(interactions)
    )
    print(
        interaction_frame[
            [
                "si_tail",
                "vol_tail",
                "risk_difference_interaction",
                "risk_difference_interaction_ci_low",
                "risk_difference_interaction_ci_high",
                "risk_ratio_interaction",
                "rr_interaction_ci_low",
                "rr_interaction_ci_high",
                "return_interaction",
            ]
        ].to_string(
            index=False
        )
    )
    print()
    print(
        f"Written to {OUTPUT_DIR}"
    )
if __name__ == "__main__":
    main()
