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

OUTPUT_DIR = Path(
    "data/processed/ml/research/interaction"
)


def target_config():
    for target in TARGETS:
        if target.name == TARGET_NAME:
            return target
    raise ValueError(f"Unknown target: {TARGET_NAME}")


def evaluate(
    *,
    frame: pd.DataFrame,
    target: np.ndarray,
    returns: np.ndarray,
    mask: np.ndarray,
    label: str,
) -> dict:

    valid = mask & np.isfinite(target)

    selected_target = target[valid]
    selected_returns = returns[
        valid & np.isfinite(returns)
    ]

    n = int(valid.sum())
    events = int((selected_target > 0).sum())

    event_rate = (
        events / n
        if n
        else None
    )

    baseline_target = target[
        np.isfinite(target)
    ]

    baseline_rate = (
        float((baseline_target > 0).mean())
        if len(baseline_target)
        else None
    )

    lift = (
        event_rate / baseline_rate
        if event_rate is not None
        and baseline_rate
        and baseline_rate > 0
        else None
    )

    mean_return = (
        float(selected_returns.mean())
        if len(selected_returns)
        else None
    )

    rest_mask = (
        ~valid
        & np.isfinite(returns)
    )

    rest_returns = returns[rest_mask]

    rest_mean_return = (
        float(rest_returns.mean())
        if len(rest_returns)
        else None
    )

    return_difference = (
        mean_return - rest_mean_return
        if mean_return is not None
        and rest_mean_return is not None
        else None
    )

    return {
        "label": label,
        "n": n,
        "events": events,
        "event_rate": event_rate,
        "baseline_event_rate": baseline_rate,
        "lift": lift,
        "mean_return": mean_return,
        "rest_mean_return": rest_mean_return,
        "return_difference": return_difference,
    }


def main() -> None:

    print("Loading features...", flush=True)

    frame = load_features().copy()

    frame["snapshot_date"] = pd.to_datetime(
        frame["snapshot_date"],
        errors="coerce",
    )

    frame = frame.loc[
        frame["snapshot_date"] <= DISCOVERY_END
    ].copy()

    frame.reset_index(drop=True, inplace=True)

    print(
        f"Rows through {DISCOVERY_END.date()}: "
        f"{len(frame):,}",
        flush=True,
    )

    target_cfg = target_config()

    target = build_target(
        frame,
        target_cfg,
    ).to_numpy(dtype=float)

    returns = pd.to_numeric(
        frame[target_cfg.return_column],
        errors="coerce",
    ).to_numpy(dtype=float)

    si = build_signal(
        frame,
        "short_interest_change",
    )

    volatility = build_signal(
        frame,
        "price_volatility_20d",
    )

    # Base universe.
    base = np.ones(
        len(frame),
        dtype=bool,
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
            frame=frame,
            target=target,
            returns=returns,
            mask=base & si_mask,
            label=f"SI {tail:g}",
        )

        result.update(
            {
                "analysis": "si_only",
                "si_tail": tail,
                "vol_tail": None,
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
            frame=frame,
            target=target,
            returns=returns,
            mask=base & vol_mask,
            label=f"VOL {tail:g}",
        )

        result.update(
            {
                "analysis": "volatility_only",
                "si_tail": None,
                "vol_tail": tail,
            }
        )

        results.append(result)

    # ---------------------------------------------------------
    # 3. SI × VOL INTERACTION
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

            result = evaluate(
                frame=frame,
                target=target,
                returns=returns,
                mask=(
                    base
                    & si_mask
                    & vol_mask
                ),
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
                }
            )

            results.append(result)

    # ---------------------------------------------------------
    # 4. DIRECT INTERACTION MEASURE
    #
    # For each SI threshold:
    #
    # effect in high volatility
    # minus
    # effect in non-high volatility
    #
    # This is the key test.
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

            si_high_vol = (
                base
                & si_mask
                & vol_mask
            )

            si_low_vol = (
                base
                & si_mask
                & ~vol_mask
            )

            high = evaluate(
                frame=frame,
                target=target,
                returns=returns,
                mask=si_high_vol,
                label="SI high / VOL high",
            )

            low = evaluate(
                frame=frame,
                target=target,
                returns=returns,
                mask=si_low_vol,
                label="SI high / VOL low",
            )

            event_rate_difference = None

            if (
                high["event_rate"] is not None
                and low["event_rate"] is not None
            ):
                event_rate_difference = (
                    high["event_rate"]
                    - low["event_rate"]
                )

            lift_ratio = None

            if (
                high["lift"] is not None
                and low["lift"] is not None
                and low["lift"] > 0
            ):
                lift_ratio = (
                    high["lift"]
                    / low["lift"]
                )

            interactions.append(
                {
                    "si_tail": si_tail,
                    "vol_tail": vol_tail,
                    "si_high_vol_n": high["n"],
                    "si_low_vol_n": low["n"],
                    "si_high_vol_event_rate": (
                        high["event_rate"]
                    ),
                    "si_low_vol_event_rate": (
                        low["event_rate"]
                    ),
                    "event_rate_difference": (
                        event_rate_difference
                    ),
                    "si_high_vol_lift": (
                        high["lift"]
                    ),
                    "si_low_vol_lift": (
                        low["lift"]
                    ),
                    "lift_ratio_high_vs_low_vol": (
                        lift_ratio
                    ),
                    "return_difference_high_vol": (
                        high["return_difference"]
                    ),
                    "return_difference_low_vol": (
                        low["return_difference"]
                    ),
                }
            )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with (
        OUTPUT_DIR / "results.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as fh:
        json.dump(
            {
                "discovery_end": str(
                    DISCOVERY_END.date()
                ),
                "target": TARGET_NAME,
                "rows": len(frame),
                "results": results,
                "interactions": interactions,
            },
            fh,
            indent=2,
        )

    pd.DataFrame(results).to_csv(
        OUTPUT_DIR / "results.csv",
        index=False,
    )

    pd.DataFrame(interactions).to_csv(
        OUTPUT_DIR / "interactions.csv",
        index=False,
    )

    print()
    print("=== SI ONLY ===")
    print(
        pd.DataFrame(
            [
                r
                for r in results
                if r["analysis"] == "si_only"
            ]
        )[
            [
                "si_tail",
                "n",
                "events",
                "event_rate",
                "lift",
                "return_difference",
            ]
        ].to_string(index=False)
    )

    print()
    print("=== VOLATILITY ONLY ===")
    print(
        pd.DataFrame(
            [
                r
                for r in results
                if r["analysis"]
                == "volatility_only"
            ]
        )[
            [
                "vol_tail",
                "n",
                "events",
                "event_rate",
                "lift",
                "return_difference",
            ]
        ].to_string(index=False)
    )

    print()
    print("=== SI × VOL ===")
    print(
        pd.DataFrame(
            [
                r
                for r in results
                if r["analysis"]
                == "interaction"
            ]
        )[
            [
                "si_tail",
                "vol_tail",
                "n",
                "events",
                "event_rate",
                "lift",
                "return_difference",
            ]
        ].to_string(index=False)
    )

    print()
    print("=== DIRECT INTERACTION ===")
    print(
        pd.DataFrame(
            interactions
        ).to_string(index=False)
    )

    print()
    print(
        f"Written to {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
