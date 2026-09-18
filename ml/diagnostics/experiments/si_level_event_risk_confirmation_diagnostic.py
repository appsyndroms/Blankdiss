"""
Confirmation diagnostic:
Short-interest level x extreme event-risk.

Primary question:
    Does short-interest level contain additional information about
    extreme downside risk when event risk is already very high?

Design:
    1. Select the event-risk model using train/validation only.
    2. Refit the selected model on all pre-test data.
    3. Calculate event-risk scores for OOS observations.
    4. Define event-risk tails from PRE-TEST scores only:
           primary:   top 10%
           secondary: top 5%
           secondary: top 2%
    5. Define HIGH short interest as the top 20% of pre-test
       short-interest levels.
    6. Compare HIGH SI vs LOW SI:
           - inside the event-risk tail
           - outside the event-risk tail
    7. Calculate the interaction:
           (HIGH SI - LOW SI) inside risk tail
           -
           (HIGH SI - LOW SI) outside risk tail
    8. Report both:
           - extreme-downside event rate
           - actual 5-day forward return
    9. Also evaluate short interest as a continuous variable
       inside each risk tail.

All thresholds are derived without using OOS outcomes.

No repository files are modified by this script.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ml.config import WALK_FORWARD_WINDOWS
from ml.diagnostics.fi_direction_short_dynamics_diagnostic import (
    EVENT_FEATURE_SETS,
    EVENT_THRESHOLD,
    RANDOM_STATE,
    fit_event_model,
    predict_event_score,
    prepare_data,
    select_event_model,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PRIMARY_RISK_CUTOFF = 0.10

SECONDARY_RISK_CUTOFFS = (
    0.05,
    0.02,
)

RISK_CUTOFFS = (
    PRIMARY_RISK_CUTOFF,
    *SECONDARY_RISK_CUTOFFS,
)

SHORT_INTEREST_CUTOFF = 0.20

BOOTSTRAP_ITERATIONS = 2_000


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class InteractionResult:
    risk_cutoff: float

    risk_low_n: int
    risk_high_n: int

    risk_low_down_rate: float
    risk_high_down_rate: float

    risk_low_return: float
    risk_high_return: float

    risk_down_effect: float
    risk_return_effect: float

    other_low_n: int
    other_high_n: int

    other_low_down_rate: float
    other_high_down_rate: float

    other_low_return: float
    other_high_return: float

    other_down_effect: float
    other_return_effect: float

    down_interaction: float
    down_ci_low: float
    down_ci_high: float
    down_p_positive: float

    return_interaction: float
    return_ci_low: float
    return_ci_high: float
    return_p_positive: float


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

def risk_label(
    cutoff: float,
) -> str:
    if cutoff >= 0.01:
        return f"top_{cutoff:.0%}"

    return f"top_{cutoff * 100:.1f}%"


def other_label(
    cutoff: float,
) -> str:
    return f"other_{(1.0 - cutoff):.0%}"


# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------

def numeric_series(
    data: pd.DataFrame,
    column: str,
) -> pd.Series:
    values = data[column]

    if isinstance(values, pd.DataFrame):
        values = values.iloc[:, 0]

    return pd.to_numeric(
        values,
        errors="coerce",
    )


def safe_mean(
    values: pd.Series,
) -> float:
    values = pd.to_numeric(
        values,
        errors="coerce",
    ).dropna()

    if values.empty:
        return float("nan")

    return float(
        values.mean()
    )


def safe_spearman(
    x: pd.Series,
    y: pd.Series,
) -> tuple[float, float]:
    frame = pd.DataFrame(
        {
            "x": pd.to_numeric(
                x,
                errors="coerce",
            ),
            "y": pd.to_numeric(
                y,
                errors="coerce",
            ),
        }
    ).dropna()

    if len(frame) < 3:
        return (
            float("nan"),
            float("nan"),
        )

    if frame["x"].nunique() < 2:
        return (
            float("nan"),
            float("nan"),
        )

    if frame["y"].nunique() < 2:
        return (
            float("nan"),
            float("nan"),
        )

    correlation, p_value = spearmanr(
        frame["x"],
        frame["y"],
    )

    return (
        float(correlation),
        float(p_value),
    )


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def bootstrap_interaction(
    risk_low: pd.Series,
    risk_high: pd.Series,
    other_low: pd.Series,
    other_high: pd.Series,
    iterations: int = BOOTSTRAP_ITERATIONS,
    random_state: int = RANDOM_STATE,
) -> tuple[float, float, float, float]:
    """
    Bootstrap:

        (HIGH - LOW) in risk tail
        -
        (HIGH - LOW) outside risk tail

    Returns:

        observed,
        CI low,
        CI high,
        P(interaction > 0)
    """

    groups = [
        pd.to_numeric(
            values,
            errors="coerce",
        )
        .dropna()
        .to_numpy()
        for values in (
            risk_low,
            risk_high,
            other_low,
            other_high,
        )
    ]

    if any(
        len(values) == 0
        for values in groups
    ):
        return (
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
        )

    (
        risk_low_values,
        risk_high_values,
        other_low_values,
        other_high_values,
    ) = groups

    observed = (
        risk_high_values.mean()
        - risk_low_values.mean()
        - other_high_values.mean()
        + other_low_values.mean()
    )

    rng = np.random.default_rng(
        random_state
    )

    interactions = np.empty(
        iterations,
        dtype=float,
    )

    for index in range(iterations):
        risk_low_sample = rng.choice(
            risk_low_values,
            size=len(risk_low_values),
            replace=True,
        )

        risk_high_sample = rng.choice(
            risk_high_values,
            size=len(risk_high_values),
            replace=True,
        )

        other_low_sample = rng.choice(
            other_low_values,
            size=len(other_low_values),
            replace=True,
        )

        other_high_sample = rng.choice(
            other_high_values,
            size=len(other_high_values),
            replace=True,
        )

        interactions[index] = (
            risk_high_sample.mean()
            - risk_low_sample.mean()
            - other_high_sample.mean()
            + other_low_sample.mean()
        )

    return (
        float(observed),
        float(
            np.quantile(
                interactions,
                0.025,
            )
        ),
        float(
            np.quantile(
                interactions,
                0.975,
            )
        ),
        float(
            (interactions > 0).mean()
        ),
    )


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

def percentile_threshold(
    values: pd.Series,
    upper_tail: float,
) -> float:
    values = pd.to_numeric(
        values,
        errors="coerce",
    ).dropna()

    if values.empty:
        return float("nan")

    return float(
        values.quantile(
            1.0 - upper_tail
        )
    )


# ---------------------------------------------------------------------------
# Event-risk assignment
# ---------------------------------------------------------------------------

def assign_risk_group(
    scores: pd.Series,
    threshold: float,
    cutoff: float,
) -> pd.Series:
    numeric_scores = pd.to_numeric(
        scores,
        errors="coerce",
    )

    result = pd.Series(
        pd.NA,
        index=scores.index,
        dtype="string",
    )

    valid = numeric_scores.notna()

    result.loc[
        valid
        & (numeric_scores >= threshold)
    ] = risk_label(cutoff)

    result.loc[
        valid
        & (numeric_scores < threshold)
    ] = other_label(cutoff)

    return result


# ---------------------------------------------------------------------------
# Interaction analysis
# ---------------------------------------------------------------------------

def interaction_analysis(
    frame: pd.DataFrame,
    risk_cutoff: float,
    title: str,
) -> InteractionResult | None:
    selected = risk_label(
        risk_cutoff
    )

    outside = other_label(
        risk_cutoff
    )

    subset = frame.dropna(
        subset=[
            "event_risk_group",
            "short_interest_pct",
            "forward_return_5d",
            "down",
        ]
    ).copy()

    if subset.empty:
        return None

    subset["high_si"] = (
        subset["short_interest_pct"]
        >= subset["training_si_threshold"]
    )

    risk_low = subset[
        (subset["event_risk_group"] == selected)
        & (~subset["high_si"])
    ]

    risk_high = subset[
        (subset["event_risk_group"] == selected)
        & (subset["high_si"])
    ]

    other_low = subset[
        (subset["event_risk_group"] == outside)
        & (~subset["high_si"])
    ]

    other_high = subset[
        (subset["event_risk_group"] == outside)
        & (subset["high_si"])
    ]

    if any(
        len(group) == 0
        for group in (
            risk_low,
            risk_high,
            other_low,
            other_high,
        )
    ):
        return None

    risk_low_down_rate = safe_mean(
        risk_low["down"]
    )

    risk_high_down_rate = safe_mean(
        risk_high["down"]
    )

    other_low_down_rate = safe_mean(
        other_low["down"]
    )

    other_high_down_rate = safe_mean(
        other_high["down"]
    )

    risk_low_return = safe_mean(
        risk_low["forward_return_5d"]
    )

    risk_high_return = safe_mean(
        risk_high["forward_return_5d"]
    )

    other_low_return = safe_mean(
        other_low["forward_return_5d"]
    )

    other_high_return = safe_mean(
        other_high["forward_return_5d"]
    )

    risk_down_effect = (
        risk_high_down_rate
        - risk_low_down_rate
    )

    other_down_effect = (
        other_high_down_rate
        - other_low_down_rate
    )

    risk_return_effect = (
        risk_high_return
        - risk_low_return
    )

    other_return_effect = (
        other_high_return
        - other_low_return
    )

    (
        down_interaction,
        down_ci_low,
        down_ci_high,
        down_p_positive,
    ) = bootstrap_interaction(
        risk_low=risk_low["down"],
        risk_high=risk_high["down"],
        other_low=other_low["down"],
        other_high=other_high["down"],
    )

    (
        return_interaction,
        return_ci_low,
        return_ci_high,
        return_p_positive,
    ) = bootstrap_interaction(
        risk_low=risk_low["forward_return_5d"],
        risk_high=risk_high["forward_return_5d"],
        other_low=other_low["forward_return_5d"],
        other_high=other_high["forward_return_5d"],
    )

    result = InteractionResult(
        risk_cutoff=risk_cutoff,

        risk_low_n=len(risk_low),
        risk_high_n=len(risk_high),

        risk_low_down_rate=risk_low_down_rate,
        risk_high_down_rate=risk_high_down_rate,

        risk_low_return=risk_low_return,
        risk_high_return=risk_high_return,

        risk_down_effect=risk_down_effect,
        risk_return_effect=risk_return_effect,

        other_low_n=len(other_low),
        other_high_n=len(other_high),

        other_low_down_rate=other_low_down_rate,
        other_high_down_rate=other_high_down_rate,

        other_low_return=other_low_return,
        other_high_return=other_high_return,

        other_down_effect=other_down_effect,
        other_return_effect=other_return_effect,

        down_interaction=down_interaction,
        down_ci_low=down_ci_low,
        down_ci_high=down_ci_high,
        down_p_positive=down_p_positive,

        return_interaction=return_interaction,
        return_ci_low=return_ci_low,
        return_ci_high=return_ci_high,
        return_p_positive=return_p_positive,
    )

    print()
    print(
        f"{title} [{risk_label(risk_cutoff)}]"
    )

    print(
        "  "
        "                         LOW SI          HIGH SI"
    )

    print(
        f"  {selected:<20}"
        f"{risk_low_down_rate:.4f} "
        f"(n={len(risk_low):4d})       "
        f"{risk_high_down_rate:.4f} "
        f"(n={len(risk_high):4d})"
    )

    print(
        f"  {outside:<20}"
        f"{other_low_down_rate:.4f} "
        f"(n={len(other_low):4d})       "
        f"{other_high_down_rate:.4f} "
        f"(n={len(other_high):4d})"
    )

    print()
    print(
        "  Extreme-downside rate:"
    )

    print(
        f"    Risk tail HIGH-LOW: "
        f"{risk_down_effect:+.4f}"
    )

    print(
        f"    Outside HIGH-LOW:   "
        f"{other_down_effect:+.4f}"
    )

    print(
        f"    INTERACTION:        "
        f"{down_interaction:+.4f}"
    )

    print(
        f"    Bootstrap CI:       "
        f"[{down_ci_low:+.4f}, "
        f"{down_ci_high:+.4f}]"
    )

    print(
        f"    P(interaction > 0): "
        f"{down_p_positive:.4f}"
    )

    print()
    print(
        "  Actual 5-day return:"
    )

    print(
        f"    Risk tail HIGH-LOW: "
        f"{risk_return_effect:+.4%}"
    )

    print(
        f"    Outside HIGH-LOW:   "
        f"{other_return_effect:+.4%}"
    )

    print(
        f"    INTERACTION:        "
        f"{return_interaction:+.4%}"
    )

    print(
        f"    Bootstrap CI:       "
        f"[{return_ci_low:+.4%}, "
        f"{return_ci_high:+.4%}]"
    )

    print(
        f"    P(interaction > 0): "
        f"{return_p_positive:.4f}"
    )

    return result


# ---------------------------------------------------------------------------
# Continuous SI analysis
# ---------------------------------------------------------------------------

def continuous_si_analysis(
    frame: pd.DataFrame,
    risk_cutoff: float,
) -> None:
    selected = risk_label(
        risk_cutoff
    )

    subset = frame[
        frame["event_risk_group"] == selected
    ].copy()

    subset = subset.dropna(
        subset=[
            "short_interest_pct",
            "down",
            "forward_return_5d",
        ]
    )

    if len(subset) < 20:
        print(
            f"  Continuous SI analysis "
            f"[{selected}]: insufficient rows "
            f"(n={len(subset)})"
        )
        return

    correlation, p_value = safe_spearman(
        subset["short_interest_pct"],
        subset["down"],
    )

    return_correlation, return_p_value = (
        safe_spearman(
            subset["short_interest_pct"],
            subset["forward_return_5d"],
        )
    )

    print()
    print(
        f"  Continuous SI [{selected}]"
    )

    print(
        f"    Spearman SI vs DOWN: "
        f"{correlation:+.4f} "
        f"(p={p_value:.4f})"
    )

    print(
        f"    Spearman SI vs 5d return: "
        f"{return_correlation:+.4f} "
        f"(p={return_p_value:.4f})"
    )


# ---------------------------------------------------------------------------
# Window
# ---------------------------------------------------------------------------

def run_window(
    data: pd.DataFrame,
    train_end: str,
    validation_end: str,
    test_end: str,
) -> tuple[pd.DataFrame, list[InteractionResult]]:
    train = data[
        data["price_date"]
        <= train_end
    ].copy()

    validation = data[
        (data["price_date"] > train_end)
        & (data["price_date"] <= validation_end)
    ].copy()

    test = data[
        (data["price_date"] > validation_end)
        & (data["price_date"] <= test_end)
    ].copy()

    print()
    print("=" * 80)
    print(
        f"WINDOW: "
        f"{train_end} -> "
        f"{validation_end} -> "
        f"{test_end}"
    )
    print("=" * 80)

    print(
        f"Train:      {len(train):,}"
    )

    print(
        f"Validation: {len(validation):,}"
    )

    print(
        f"Test:       {len(test):,}"
    )

    (
        model_name,
        features,
        validation_auc,
    ) = select_event_model(
        train,
        validation,
    )

    print()
    print(
        f"Event-risk model: {model_name}"
    )

    print(
        f"Validation AUC:   "
        f"{validation_auc:.4f}"
    )

    train_validation = pd.concat(
        [
            train,
            validation,
        ],
        ignore_index=True,
    )

    final_model = fit_event_model(
        train_validation,
        features,
    )

    if final_model is None:
        raise RuntimeError(
            "Could not fit final event model."
        )

    pretest_scores = predict_event_score(
        final_model,
        train_validation,
        features,
    )

    risk_thresholds: dict[
        float,
        float,
    ] = {}

    print()
    print(
        "PRE-TEST EVENT-RISK THRESHOLDS"
    )

    for cutoff in RISK_CUTOFFS:
        threshold = percentile_threshold(
            pretest_scores,
            cutoff,
        )

        risk_thresholds[cutoff] = threshold

        print(
            f"  {risk_label(cutoff):<10}"
            f" >= {threshold:.6f}"
        )

    si_threshold = percentile_threshold(
        train_validation[
            "short_interest_pct"
        ],
        SHORT_INTEREST_CUTOFF,
    )

    if not np.isfinite(
        si_threshold
    ):
        raise RuntimeError(
            "Could not calculate "
            "short-interest threshold."
        )

    print()
    print(
        "PRE-TEST SHORT-INTEREST THRESHOLD"
    )

    print(
        f"  HIGH SI = top "
        f"{SHORT_INTEREST_CUTOFF:.0%}"
    )

    print(
        f"  threshold = "
        f"{si_threshold:.6f}"
    )

    # ---------------------------------------------------------------
    # OOS scoring
    # ---------------------------------------------------------------

    test = test.copy()

    test["event_score"] = predict_event_score(
        final_model,
        test,
        features,
    )

    test["training_si_threshold"] = (
        si_threshold
    )

    test["down"] = (
        test["forward_return_5d"]
        <= -EVENT_THRESHOLD
    ).astype(float)

    test = test.dropna(
        subset=[
            "event_score",
            "short_interest_pct",
            "forward_return_5d",
        ]
    )

    print()
    print(
        f"OOS rows: {len(test):,}"
    )

    frames: list[pd.DataFrame] = []
    results: list[InteractionResult] = []

    for cutoff in RISK_CUTOFFS:
        local = test.copy()

        local["event_risk_group"] = (
            assign_risk_group(
                local["event_score"],
                risk_thresholds[cutoff],
                cutoff,
            )
        )

        result = interaction_analysis(
            local,
            cutoff,
            title="OOS interaction",
        )

        if result is not None:
            results.append(
                result
            )

        continuous_si_analysis(
            local,
            cutoff,
        )

        local["window"] = (
            f"{train_end}_"
            f"{validation_end}_"
            f"{test_end}"
        )

        local[
            f"risk_threshold_{cutoff}"
        ] = risk_thresholds[cutoff]

        frames.append(
            local
        )

    if not frames:
        return (
            pd.DataFrame(),
            results,
        )

    # One copy is enough for pooled analysis.
    frame = frames[0].copy()

    for cutoff in RISK_CUTOFFS:
        frame[
            f"risk_threshold_{cutoff}"
        ] = risk_thresholds[cutoff]

    return (
        frame,
        results,
    )


# ---------------------------------------------------------------------------
# Pooled analysis
# ---------------------------------------------------------------------------

def pooled_analysis(
    frames: list[pd.DataFrame],
) -> None:
    if not frames:
        print(
            "No OOS frames available "
            "for pooled confirmation."
        )
        return

    pooled = pd.concat(
        frames,
        ignore_index=True,
    )

    print()
    print("=" * 80)
    print(
        "POOLED OOS CONFIRMATION"
    )
    print("=" * 80)

    print(
        f"Pooled OOS rows: "
        f"{len(pooled):,}"
    )

    for cutoff in RISK_CUTOFFS:
        threshold_column = (
            f"risk_threshold_{cutoff}"
        )

        if threshold_column not in pooled.columns:
            continue

        local = pooled.copy()

        threshold = pd.to_numeric(
            local[threshold_column],
            errors="coerce",
        )

        local["event_risk_group"] = pd.Series(
            pd.NA,
            index=local.index,
            dtype="string",
        )

        valid = (
            local["event_score"].notna()
            & threshold.notna()
        )

        local.loc[
            valid
            & (
                local["event_score"]
                >= threshold
            ),
            "event_risk_group",
        ] = risk_label(cutoff)

        local.loc[
            valid
            & (
                local["event_score"]
                < threshold
            ),
            "event_risk_group",
        ] = other_label(cutoff)

        result = interaction_analysis(
            local,
            cutoff,
            title="POOLED",
        )

        continuous_si_analysis(
            local,
            cutoff,
        )

        if result is None:
            print(
                f"No usable pooled result "
                f"for {risk_label(cutoff)}."
            )


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(
    results: list[InteractionResult],
) -> None:
    if not results:
        return

    print()
    print("=" * 80)
    print(
        "CONFIRMATION SUMMARY"
    )
    print("=" * 80)

    print(
        "Risk tail    "
        "N-tail   "
        "N-other   "
        "SI effect   "
        "Interaction   "
        "CI low      "
        "CI high     "
        "P(>0)"
    )

    for result in results:
        n_tail = (
            result.risk_low_n
            + result.risk_high_n
        )

        n_other = (
            result.other_low_n
            + result.other_high_n
        )

        print(
            f"{risk_label(result.risk_cutoff):<11}"
            f"{n_tail:7d}   "
            f"{n_other:7d}   "
            f"{result.risk_down_effect:+.4f}     "
            f"{result.down_interaction:+.4f}       "
            f"{result.down_ci_low:+.4f}    "
            f"{result.down_ci_high:+.4f}    "
            f"{result.down_p_positive:.4f}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(
        "Blankdiss "
        "Short-Interest Level / "
        "Event-Risk Confirmation Diagnostic"
    )

    print()
    print(
        "Primary question:"
    )

    print(
        "Does high short-interest level add "
        "information when event risk is already high?"
    )

    print()
    print(
        f"Event threshold: "
        f"{EVENT_THRESHOLD:.0%}"
    )

    print(
        f"PRIMARY risk tail: "
        f"top {PRIMARY_RISK_CUTOFF:.0%}"
    )

    print(
        "Secondary risk tails:"
    )

    for cutoff in SECONDARY_RISK_CUTOFFS:
        print(
            f"  top {cutoff:.0%}"
        )

    print()
    print(
        f"HIGH SI: top "
        f"{SHORT_INTEREST_CUTOFF:.0%} "
        f"of pre-test SI level"
    )

    data = prepare_data()

    print()
    print(
        f"Prepared rows: "
        f"{len(data):,}"
    )

    print(
        f"Date range: "
        f"{data['price_date'].min()} "
        f"-> "
        f"{data['price_date'].max()}"
    )

    all_frames: list[pd.DataFrame] = []
    all_results: list[InteractionResult] = []

    for window in WALK_FORWARD_WINDOWS:
        (
            frame,
            results,
        ) = run_window(
            data=data,
            train_end=window.train_end,
            validation_end=window.validation_end,
            test_end=window.test_end,
        )

        if not frame.empty:
            all_frames.append(
                frame
            )

        all_results.extend(
            results
        )

    pooled_analysis(
        all_frames
    )

    print_summary(
        all_results
    )

    print()
    print("=" * 80)
    print(
        "DIAGNOSTIC COMPLETE"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
