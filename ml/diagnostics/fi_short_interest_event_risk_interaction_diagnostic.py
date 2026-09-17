"""
Conditional interaction diagnostic.
Question:
    Is the positive short-interest-change effect stronger when event risk
    is extremely high?
Design:
    1. Fit/select the event-risk model using train/validation only.
    2. Calculate OOS event-risk scores.
    3. Define the extreme event-risk group from the training distribution:
           - top 5%
           - remaining 95%
    4. Within each OOS risk group, compare:
           - top 20% of positive short-interest changes
           - all other positive changes
    5. Only observations with an extreme direction are included:
           DOWN = forward return <= -10%
           UP   = forward return >= +10%
    6. Test the interaction directly:
           (HIGH SI - LOW SI) in top 5%
           minus
           (HIGH SI - LOW SI) in remaining 95%
The positive-change threshold is fixed at the training 80th percentile
among positive changes. It is NOT searched over multiple cutoffs.
No repository files are modified by this script.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from sklearn.metrics import roc_auc_score
from ml.config import WALK_FORWARD_WINDOWS
from ml.diagnostics.fi_direction_short_dynamics_diagnostic import (
    EVENT_FEATURE_SETS,
    EVENT_THRESHOLD,
    EVENT_TAIL,
    RANDOM_STATE,
    fit_event_model,
    predict_event_score,
    prepare_data,
)
# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
POSITIVE_CHANGE_CUTOFF = 0.20
EVENT_RISK_CUTOFF = 0.05
BOOTSTRAP_ITERATIONS = 2_000
# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CellResult:
    group: str
    change_group: str
    n: int
    down: int
    down_rate: float
@dataclass(frozen=True)
class InteractionResult:
    top5_low_n: int
    top5_high_n: int
    top5_low_rate: float
    top5_high_rate: float
    top5_effect: float
    other_low_n: int
    other_high_n: int
    other_low_rate: float
    other_high_rate: float
    other_effect: float
    interaction: float
    ci_low: float
    ci_high: float
    p_positive: float
def safe_rate(
    values: pd.Series,
) -> float:
    if len(values) == 0:
        return float("nan")
    return float(
        values.mean()
    )
def bootstrap_interaction(
    top5_low: pd.Series,
    top5_high: pd.Series,
    other_low: pd.Series,
    other_high: pd.Series,
    iterations: int = BOOTSTRAP_ITERATIONS,
    random_state: int = RANDOM_STATE,
) -> tuple[float, float, float, float]:
    """
    Bootstrap the interaction:
        (HIGH - LOW) in TOP5
        -
        (HIGH - LOW) in OTHER95
    Returns:
        observed interaction,
        2.5% CI,
        97.5% CI,
        P(interaction > 0)
    """
    groups = [
        pd.to_numeric(
            values,
            errors="coerce",
        ).dropna().to_numpy()
        for values in (
            top5_low,
            top5_high,
            other_low,
            other_high,
        )
    ]
    if any(len(values) == 0 for values in groups):
        return (
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
        )
    top5_low_values, top5_high_values, other_low_values, other_high_values = groups
    observed = (
        top5_high_values.mean()
        - top5_low_values.mean()
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
        top5_low_sample = rng.choice(
            top5_low_values,
            size=len(top5_low_values),
            replace=True,
        )
        top5_high_sample = rng.choice(
            top5_high_values,
            size=len(top5_high_values),
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
            top5_high_sample.mean()
            - top5_low_sample.mean()
            - other_high_sample.mean()
            + other_low_sample.mean()
        )
    return (
        float(observed),
        float(np.quantile(interactions, 0.025)),
        float(np.quantile(interactions, 0.975)),
        float((interactions > 0).mean()),
    )
def safe_fisher_p(
    low_down: int,
    low_up: int,
    high_down: int,
    high_up: int,
) -> float:
    if (
        low_down + low_up == 0
        or high_down + high_up == 0
    ):
        return float("nan")
    table = [
        [low_down, low_up],
        [high_down, high_up],
    ]
    _, p_value = fisher_exact(
        table,
        alternative="two-sided",
    )
    return float(p_value)
# ---------------------------------------------------------------------------
# Event model selection
# ---------------------------------------------------------------------------
def select_event_model(
    train: pd.DataFrame,
    validation: pd.DataFrame,
) -> tuple[str | None, object | None, float]:
    best_name: str | None = None
    best_model = None
    best_auc = float("-inf")
    for name, features in EVENT_FEATURE_SETS.items():
        missing = [
            column
            for column in features
            if column not in train.columns
            or column not in validation.columns
        ]
        if missing:
            continue
        model = fit_event_model(
            train,
            list(features),
        )
        if model is None:
            continue
        validation_scores = predict_event_score(
            model,
            validation,
            list(features),
        )
        frame = pd.DataFrame(
            {
                "event": validation["event"],
                "score": validation_scores,
            }
        ).dropna()
        if len(frame) < 100:
            continue
        if frame["event"].nunique() < 2:
            continue
        auc = float(
            roc_auc_score(
                frame["event"],
                frame["score"],
            )
        )
        if auc > best_auc:
            best_auc = auc
            best_name = name
            best_model = model
    return (
        best_name,
        best_model,
        best_auc,
    )
# ---------------------------------------------------------------------------
# Training thresholds
# ---------------------------------------------------------------------------
def training_event_risk_threshold(
    scores: pd.Series,
) -> float:
    """
    Extreme-risk threshold corresponding to the top 5%.
    The threshold is calculated before the OOS test period.
    """
    scores = pd.to_numeric(
        scores,
        errors="coerce",
    ).dropna()
    if scores.empty:
        return float("nan")
    return float(
        scores.quantile(
            1.0 - EVENT_RISK_CUTOFF
        )
    )
def positive_change_threshold(
    train: pd.DataFrame,
) -> float:
    """
    Fixed top-20% threshold among positive short-interest changes.
    Learned from pre-test data only.
    """
    values = pd.to_numeric(
        train["short_interest_pct_change"],
        errors="coerce",
    )
    positive = values[
        values > 0
    ].dropna()
    if positive.empty:
        return float("nan")
    return float(
        positive.quantile(
            1.0 - POSITIVE_CHANGE_CUTOFF
        )
    )
# ---------------------------------------------------------------------------
# OOS group assignment
# ---------------------------------------------------------------------------
def assign_event_risk_group(
    scores: pd.Series,
    threshold: float,
) -> pd.Series:
    """
    Assign OOS observations to:
        top_5pct
        other_95pct
    using a threshold calculated from pre-test data only.
    """
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
    ] = "top_5pct"
    result.loc[
        valid
        & (numeric_scores < threshold)
    ] = "other_95pct"
    return result
# ---------------------------------------------------------------------------
# 2x2 interaction
# ---------------------------------------------------------------------------
def interaction_analysis(
    frame: pd.DataFrame,
    title: str = "Interaction",
) -> InteractionResult | None:
    """
    Direct 2x2 interaction analysis.
    Rows:
        top 5% event risk
        other 95% event risk
    Columns:
        LOW positive SI change
        HIGH positive SI change
    Target:
        DOWN = 1
        UP   = 0
    """
    subset = frame.copy()
    subset = subset.dropna(
        subset=[
            "event_risk_group",
            "short_interest_pct_change",
            "direction",
            "training_positive_threshold",
        ]
    )
    # Only positive short-interest changes.
    subset = subset[
        subset[
            "short_interest_pct_change"
        ] > 0
    ].copy()
    if subset.empty:
        return None
    subset["high_change"] = (
        subset[
            "short_interest_pct_change"
        ]
        >= subset[
            "training_positive_threshold"
        ]
    )
    subset["down"] = (
        subset["direction"] == 1
    ).astype(int)
    top5_low = subset[
        (subset["event_risk_group"] == "top_5pct")
        & (~subset["high_change"])
    ]["down"]
    top5_high = subset[
        (subset["event_risk_group"] == "top_5pct")
        & (subset["high_change"])
    ]["down"]
    other_low = subset[
        (subset["event_risk_group"] == "other_95pct")
        & (~subset["high_change"])
    ]["down"]
    other_high = subset[
        (subset["event_risk_group"] == "other_95pct")
        & (subset["high_change"])
    ]["down"]
    if (
        len(top5_low) == 0
        or len(top5_high) == 0
        or len(other_low) == 0
        or len(other_high) == 0
    ):
        return None
    top5_low_rate = safe_rate(top5_low)
    top5_high_rate = safe_rate(top5_high)
    other_low_rate = safe_rate(other_low)
    other_high_rate = safe_rate(other_high)
    top5_effect = (
        top5_high_rate
        - top5_low_rate
    )
    other_effect = (
        other_high_rate
        - other_low_rate
    )
    (
        interaction,
        ci_low,
        ci_high,
        p_positive,
    ) = bootstrap_interaction(
        top5_low=top5_low,
        top5_high=top5_high,
        other_low=other_low,
        other_high=other_high,
    )
    result = InteractionResult(
        top5_low_n=len(top5_low),
        top5_high_n=len(top5_high),
        top5_low_rate=top5_low_rate,
        top5_high_rate=top5_high_rate,
        top5_effect=top5_effect,
        other_low_n=len(other_low),
        other_high_n=len(other_high),
        other_low_rate=other_low_rate,
        other_high_rate=other_high_rate,
        other_effect=other_effect,
        interaction=interaction,
        ci_low=ci_low,
        ci_high=ci_high,
        p_positive=p_positive,
    )
    print()
    print(
        f"{title}:"
    )
    print(
        "  "
        "                    LOW SI       HIGH SI"
    )
    print(
        f"  top 5% risk        "
        f"{top5_low_rate:.4f} "
        f"(n={len(top5_low):3d})    "
        f"{top5_high_rate:.4f} "
        f"(n={len(top5_high):3d})"
    )
    print(
        f"  other 95%         "
        f"{other_low_rate:.4f} "
        f"(n={len(other_low):3d})    "
        f"{other_high_rate:.4f} "
        f"(n={len(other_high):3d})"
    )
    print()
    print(
        "  HIGH - LOW effect:"
    )
    print(
        f"    top 5%:   "
        f"{top5_effect:+.4f}"
    )
    print(
        f"    other 95%:"
        f"{other_effect:+.4f}"
    )
    print()
    print(
        "  Interaction "
        "(top5 effect - other95 effect):"
    )
    print(
        f"    {interaction:+.4f}"
    )
    print(
        "  Bootstrap CI:"
        f" [{ci_low:+.4f}, {ci_high:+.4f}]"
    )
    print(
        "  P(interaction > 0):"
        f" {p_positive:.4f}"
    )
    return result
# ---------------------------------------------------------------------------
# Walk-forward window
# ---------------------------------------------------------------------------
def run_window(
    data: pd.DataFrame,
    train_end: str,
    validation_end: str,
    test_end: str,
) -> pd.DataFrame:
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
        f"Window: "
        f"{train_end} -> "
        f"{validation_end} -> "
        f"{test_end}"
    )
    print(
        f"Train rows:      {len(train):,}"
    )
    print(
        f"Validation rows: {len(validation):,}"
    )
    print(
        f"Test rows:       {len(test):,}"
    )
    (
        model_name,
        event_model,
        validation_auc,
    ) = select_event_model(
        train,
        validation,
    )
    if event_model is None:
        print(
            "No usable event model."
        )
        return pd.DataFrame()
    print(
        f"Event model: {model_name}"
    )
    print(
        f"Validation AUC: "
        f"{validation_auc:.4f}"
    )
    features = list(
        EVENT_FEATURE_SETS[
            model_name
        ]
    )
    # ---------------------------------------------------------------
    # Refit selected model on all pre-test data.
    # ---------------------------------------------------------------
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
        print(
            "Could not fit final event model."
        )
        return pd.DataFrame()
    # ---------------------------------------------------------------
    # Pre-test event-risk scores.
    #
    # These are used only to establish the top-5% threshold before
    # looking at test outcomes.
    # ---------------------------------------------------------------
    train_validation_scores = (
        predict_event_score(
            final_model,
            train_validation,
            features,
        )
    )
    event_risk_threshold = (
        training_event_risk_threshold(
            train_validation_scores
        )
    )
    if not np.isfinite(
        event_risk_threshold
    ):
        print(
            "Could not calculate event-risk threshold."
        )
        return pd.DataFrame()
    positive_threshold = (
        positive_change_threshold(
            train_validation
        )
    )
    if not np.isfinite(
        positive_threshold
    ):
        print(
            "Could not calculate positive-change threshold."
        )
        return pd.DataFrame()
    print(
        "Extreme event-risk threshold "
        "(pre-test derived):"
    )
    print(
        f"  top 5%: score >= "
        f"{event_risk_threshold:.6f}"
    )
    print(
        "Positive short-interest "
        "top-20% threshold:"
    )
    print(
        f"  threshold = "
        f"{positive_threshold:.6f}"
    )
    # ---------------------------------------------------------------
    # OOS scoring.
    # ---------------------------------------------------------------
    test = test.copy()
    test["event_score"] = (
        predict_event_score(
            final_model,
            test,
            features,
        )
    )
    test["event_risk_group"] = (
        assign_event_risk_group(
            test["event_score"],
            event_risk_threshold,
        )
    )
    test[
        "training_positive_threshold"
    ] = positive_threshold
    # Direction only:
    # 1 = DOWN <= -10%
    # 0 = UP >= +10%
    test = test.dropna(
        subset=[
            "direction",
            "event_risk_group",
            "short_interest_pct_change",
        ]
    )
    print(
        f"Directional OOS rows: "
        f"{len(test):,}"
    )
    interaction_analysis(
        test,
        title="2x2 OOS interaction",
    )
    test["window"] = (
        f"{train_end}_"
        f"{validation_end}_"
        f"{test_end}"
    )
    return test
# ---------------------------------------------------------------------------
# Pooled analysis
# ---------------------------------------------------------------------------
def pooled_analysis(
    frames: list[pd.DataFrame],
) -> None:
    if not frames:
        return
    pooled = pd.concat(
        frames,
        ignore_index=True,
    )
    print()
    print("=" * 80)
    print("POOLED OOS 2x2 INTERACTION")
    print("=" * 80)
    print(
        f"Directional OOS rows: "
        f"{len(pooled):,}"
    )
    interaction_analysis(
        pooled,
        title="Pooled interaction",
    )
    # ---------------------------------------------------------------
    # Per-window interaction results.
    # ---------------------------------------------------------------
    print()
    print("=" * 80)
    print("PER-WINDOW INTERACTION")
    print("=" * 80)
    for window, frame in pooled.groupby(
        "window",
        sort=True,
    ):
        interaction_analysis(
            frame,
            title=window,
        )
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print(
        "Blankdiss Short-Interest / "
        "Event-Risk Interaction Diagnostic"
    )
    print(
        f"Event threshold: "
        f"{EVENT_THRESHOLD:.0%}"
    )
    print(
        f"Event tail reference: "
        f"{EVENT_TAIL:.0%}"
    )
    print(
        "Positive-change cutoff: "
        f"top {POSITIVE_CHANGE_CUTOFF:.0%}"
    )
    print(
        "Extreme event-risk cutoff: "
        f"top {EVENT_RISK_CUTOFF:.0%}"
    )
    print(
        "Risk groups:"
    )
    print(
        "  top_5pct:    0% -> 5%"
    )
    print(
        "  other_95pct: 5% -> 100%"
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
    frames: list[pd.DataFrame] = []
    for window in WALK_FORWARD_WINDOWS:
        frame = run_window(
            data=data,
            train_end=window.train_end,
            validation_end=window.validation_end,
            test_end=window.test_end,
        )
        if not frame.empty:
            frames.append(frame)
    pooled_analysis(
        frames
    )
if __name__ == "__main__":
    main()
