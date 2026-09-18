"""
Conditional interaction diagnostic.
Question:
    Is the positive short-interest-change effect stronger when event risk
    is extremely high?
Design:
    1. Fit/select the event-risk model using train/validation only.
    2. Refit the selected model on all pre-test data.
    3. Calculate OOS event-risk scores.
    4. Define several extreme event-risk groups from the pre-test
       score distribution:
           - top 20%
           - top 10%
           - top 5%
           - top 2.5%
           - top 1%
    5. Within each OOS risk group, compare:
           - top 20% of positive short-interest changes
           - all other positive changes
    6. Only observations with an extreme direction are included:
           DOWN = forward return <= -10%
           UP   = forward return >= +10%
    7. Test the interaction directly for every risk tail:
           (HIGH SI - LOW SI) in risk tail
           minus
           (HIGH SI - LOW SI) in all observations outside that tail
The positive-change threshold is fixed at the training 80th percentile
among positive changes. It is NOT searched over multiple cutoffs.
Risk-tail thresholds are calculated from pre-test event-risk scores only.
They are never derived from test outcomes.
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
# Sweep the event-risk tail rather than testing only top 5%.
EVENT_RISK_CUTOFFS = (
    0.20,
    0.10,
    0.05,
    0.025,
    0.01,
)
BOOTSTRAP_ITERATIONS = 2_000
# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class InteractionResult:
    risk_cutoff: float
    risk_low_n: int
    risk_high_n: int
    risk_low_rate: float
    risk_high_rate: float
    risk_effect: float
    other_low_n: int
    other_high_n: int
    other_low_rate: float
    other_high_rate: float
    other_effect: float
    interaction: float
    ci_low: float
    ci_high: float
    p_positive: float
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def safe_rate(
    values: pd.Series,
) -> float:
    if len(values) == 0:
        return float("nan")
    return float(
        values.mean()
    )
def risk_label(
    cutoff: float,
) -> str:
    if cutoff >= 0.01:
        return f"top_{cutoff:.0%}"
    return f"top_{cutoff * 100:.1f}%"
def bootstrap_interaction(
    risk_low: pd.Series,
    risk_high: pd.Series,
    other_low: pd.Series,
    other_high: pd.Series,
    iterations: int = BOOTSTRAP_ITERATIONS,
    random_state: int = RANDOM_STATE,
) -> tuple[float, float, float, float]:
    """
    Bootstrap the interaction:
        (HIGH - LOW) in selected risk tail
        -
        (HIGH - LOW) outside selected risk tail
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
        float(np.quantile(
            interactions,
            0.025,
        )),
        float(np.quantile(
            interactions,
            0.975,
        )),
        float(
            (interactions > 0).mean()
        ),
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
    risk_cutoff: float,
) -> float:
    """
    Calculate the event-risk threshold corresponding to the selected
    upper tail.
    Example:
        risk_cutoff=0.05 -> top 5%
        risk_cutoff=0.01 -> top 1%
    Threshold is calculated from pre-test scores only.
    """
    scores = pd.to_numeric(
        scores,
        errors="coerce",
    ).dropna()
    if scores.empty:
        return float("nan")
    return float(
        scores.quantile(
            1.0 - risk_cutoff
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
# OOS risk-tail assignment
# ---------------------------------------------------------------------------
def assign_event_risk_group(
    scores: pd.Series,
    threshold: float,
    cutoff: float,
) -> pd.Series:
    """
    Assign OOS observations to:
        selected risk tail
        other observations
    using a threshold calculated from pre-test data only.
    """
    numeric_scores = pd.to_numeric(
        scores,
        errors="coerce",
    )
    label = risk_label(
        cutoff
    )
    other_label = (
        f"other_{(1.0 - cutoff):.1%}"
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
    ] = label
    result.loc[
        valid
        & (numeric_scores < threshold)
    ] = other_label
    return result
# ---------------------------------------------------------------------------
# Single risk-tail interaction
# ---------------------------------------------------------------------------
def interaction_analysis(
    frame: pd.DataFrame,
    risk_cutoff: float,
    title: str = "Interaction",
) -> InteractionResult | None:
    """
    Direct interaction analysis for one event-risk tail.
    Rows:
        selected event-risk tail
        all observations outside selected tail
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
    high_change = (
        subset[
            "short_interest_pct_change"
        ]
        >= subset[
            "training_positive_threshold"
        ]
    )
    subset["high_change"] = high_change
    subset["down"] = (
        subset["direction"] == 1
    ).astype(int)
    selected_label = risk_label(
        risk_cutoff
    )
    other_label = (
        f"other_{(1.0 - risk_cutoff):.1%}"
    )
    risk_low = subset[
        (subset["event_risk_group"] == selected_label)
        & (~subset["high_change"])
    ]["down"]
    risk_high = subset[
        (subset["event_risk_group"] == selected_label)
        & (subset["high_change"])
    ]["down"]
    other_low = subset[
        (subset["event_risk_group"] == other_label)
        & (~subset["high_change"])
    ]["down"]
    other_high = subset[
        (subset["event_risk_group"] == other_label)
        & (subset["high_change"])
    ]["down"]
    if (
        len(risk_low) == 0
        or len(risk_high) == 0
        or len(other_low) == 0
        or len(other_high) == 0
    ):
        return None
    risk_low_rate = safe_rate(
        risk_low
    )
    risk_high_rate = safe_rate(
        risk_high
    )
    other_low_rate = safe_rate(
        other_low
    )
    other_high_rate = safe_rate(
        other_high
    )
    risk_effect = (
        risk_high_rate
        - risk_low_rate
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
        risk_low=risk_low,
        risk_high=risk_high,
        other_low=other_low,
        other_high=other_high,
    )
    result = InteractionResult(
        risk_cutoff=risk_cutoff,
        risk_low_n=len(risk_low),
        risk_high_n=len(risk_high),
        risk_low_rate=risk_low_rate,
        risk_high_rate=risk_high_rate,
        risk_effect=risk_effect,
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
        f"{title} "
        f"[{risk_label(risk_cutoff)}]:"
    )
    print(
        "  "
        "                         LOW SI       HIGH SI"
    )
    print(
        f"  {selected_label:<20}"
        f"{risk_low_rate:.4f} "
        f"(n={len(risk_low):3d})    "
        f"{risk_high_rate:.4f} "
        f"(n={len(risk_high):3d})"
    )
    print(
        f"  {other_label:<20}"
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
        f"    {risk_label(risk_cutoff):<12}"
        f"{risk_effect:+.4f}"
    )
    print(
        f"    {other_label:<12}"
        f"{other_effect:+.4f}"
    )
    print()
    print(
        "  Interaction "
        "(risk-tail effect - outside effect):"
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
# Risk-tail sweep
# ---------------------------------------------------------------------------
def risk_tail_sweep(
    frame: pd.DataFrame,
    thresholds: dict[float, float],
    title: str = "Risk-tail sweep",
) -> list[InteractionResult]:
    results: list[InteractionResult] = []
    for cutoff in EVENT_RISK_CUTOFFS:
        threshold = thresholds.get(
            cutoff
        )
        if threshold is None:
            continue
        scored = frame.copy()
        scored["event_risk_group"] = (
            assign_event_risk_group(
                scored["event_score"],
                threshold,
                cutoff,
            )
        )
        result = interaction_analysis(
            scored,
            risk_cutoff=cutoff,
            title=title,
        )
        if result is not None:
            results.append(
                result
            )
    return results
def print_risk_tail_summary(
    results: list[InteractionResult],
    title: str,
) -> None:
    if not results:
        return
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)
    print(
        "Risk tail     "
        "n_tail    n_other    "
        "tail effect    "
        "other effect    "
        "interaction    "
        "CI low       CI high      "
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
            f"{risk_label(result.risk_cutoff):<12}"
            f"{n_tail:7d}    "
            f"{n_other:7d}    "
            f"{result.risk_effect:+.4f}        "
            f"{result.other_effect:+.4f}        "
            f"{result.interaction:+.4f}        "
            f"{result.ci_low:+.4f}      "
            f"{result.ci_high:+.4f}      "
            f"{result.p_positive:.4f}"
        )
# ---------------------------------------------------------------------------
# Walk-forward window
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
        return (
            pd.DataFrame(),
            [],
        )
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
        return (
            pd.DataFrame(),
            [],
        )
    # ---------------------------------------------------------------
    # Pre-test event-risk scores.
    #
    # These are used only to establish the risk-tail thresholds before
    # looking at test outcomes.
    # ---------------------------------------------------------------
    train_validation_scores = (
        predict_event_score(
            final_model,
            train_validation,
            features,
        )
    )
    thresholds: dict[float, float] = {}
    print()
    print(
        "Extreme event-risk thresholds "
        "(pre-test derived):"
    )
    for cutoff in EVENT_RISK_CUTOFFS:
        threshold = (
            training_event_risk_threshold(
                train_validation_scores,
                cutoff,
            )
        )
        thresholds[cutoff] = threshold
        print(
            f"  {risk_label(cutoff):<12}"
            f"score >= {threshold:.6f}"
        )
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
        return (
            pd.DataFrame(),
            [],
        )
    print()
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
    test[
        "training_positive_threshold"
    ] = positive_threshold
    # Direction only:
    # 1 = DOWN <= -10%
    # 0 = UP >= +10%
    test = test.dropna(
        subset=[
            "direction",
            "event_score",
            "short_interest_pct_change",
        ]
    )
    print(
        f"Directional OOS rows: "
        f"{len(test):,}"
    )
    # ---------------------------------------------------------------
    # Risk-tail sweep.
    # ---------------------------------------------------------------
    results = risk_tail_sweep(
        frame=test,
        thresholds=thresholds,
        title="OOS interaction",
    )
    print_risk_tail_summary(
        results,
        title="RISK-TAIL SWEEP",
    )
    test["window"] = (
        f"{train_end}_"
        f"{validation_end}_"
        f"{test_end}"
    )
    return (
        test,
        results,
    )
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
    print("POOLED OOS RISK-TAIL INTERACTION")
    print("=" * 80)
    print(
        f"Directional OOS rows: "
        f"{len(pooled):,}"
    )
    # ---------------------------------------------------------------
    # Reconstruct the pre-test thresholds for each window from the
    # stored OOS event scores.
    #
    # The window-specific threshold information is stored by rerunning
    # the percentile on the OOS score would be wrong, so thresholds
    # are recovered from the group labels already assigned below.
    #
    # Therefore pooled analysis uses each window's existing
    # event-risk score together with thresholds reconstructed from
    # the window-specific pre-test score distribution stored in
    # `event_risk_thresholds`.
    # ---------------------------------------------------------------
    if "event_risk_thresholds" not in pooled.columns:
        print(
            "No window-specific risk thresholds available "
            "for pooled risk-tail analysis."
        )
        return
    pooled_results: list[InteractionResult] = []
    for cutoff in EVENT_RISK_CUTOFFS:
        threshold_column = (
            f"risk_threshold_{cutoff}"
        )
        if threshold_column not in pooled.columns:
            continue
        frame = pooled.copy()
        threshold = pd.to_numeric(
            frame[threshold_column],
            errors="coerce",
        )
        label = risk_label(
            cutoff
        )
        other_label = (
            f"other_{(1.0 - cutoff):.1%}"
        )
        frame["event_risk_group"] = pd.Series(
            pd.NA,
            index=frame.index,
            dtype="string",
        )
        valid = (
            frame["event_score"].notna()
            & threshold.notna()
        )
        frame.loc[
            valid
            & (
                frame["event_score"]
                >= threshold
            ),
            "event_risk_group",
        ] = label
        frame.loc[
            valid
            & (
                frame["event_score"]
                < threshold
            ),
            "event_risk_group",
        ] = other_label
        result = interaction_analysis(
            frame,
            risk_cutoff=cutoff,
            title="Pooled interaction",
        )
        if result is not None:
            pooled_results.append(
                result
            )
    print_risk_tail_summary(
        pooled_results,
        title="POOLED RISK-TAIL SWEEP SUMMARY",
    )
    # ---------------------------------------------------------------
    # Per-window interaction results.
    # ---------------------------------------------------------------
    print()
    print("=" * 80)
    print("PER-WINDOW RISK-TAIL INTERACTION")
    print("=" * 80)
    for window, frame in pooled.groupby(
        "window",
        sort=True,
    ):
        window_results: list[
            InteractionResult
        ] = []
        for cutoff in EVENT_RISK_CUTOFFS:
            threshold_column = (
                f"risk_threshold_{cutoff}"
            )
            if threshold_column not in frame.columns:
                continue
            local = frame.copy()
            threshold = pd.to_numeric(
                local[threshold_column],
                errors="coerce",
            )
            label = risk_label(
                cutoff
            )
            other_label = (
                f"other_{(1.0 - cutoff):.1%}"
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
            ] = label
            local.loc[
                valid
                & (
                    local["event_score"]
                    < threshold
                ),
                "event_risk_group",
            ] = other_label
            result = interaction_analysis(
                local,
                risk_cutoff=cutoff,
                title=window,
            )
            if result is not None:
                window_results.append(
                    result
                )
        print_risk_tail_summary(
            window_results,
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
        "Extreme event-risk cutoffs:"
    )
    for cutoff in EVENT_RISK_CUTOFFS:
        print(
            f"  {risk_label(cutoff)}"
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
        frame, results = run_window(
            data=data,
            train_end=window.train_end,
            validation_end=window.validation_end,
            test_end=window.test_end,
        )
        if frame.empty:
            continue
        # Store the exact pre-test threshold used for every risk tail
        # so pooled and per-window analyses never recalculate a
        # threshold from OOS/test observations.
        train = data[
            data["price_date"]
            <= window.train_end
        ].copy()
        validation = data[
            (data["price_date"] > window.train_end)
            & (
                data["price_date"]
                <= window.validation_end
            )
        ].copy()
        (
            model_name,
            event_model,
            _,
        ) = select_event_model(
            train,
            validation,
        )
        if event_model is None:
            continue
        features = list(
            EVENT_FEATURE_SETS[
                model_name
            ]
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
            continue
        pretest_scores = predict_event_score(
            final_model,
            train_validation,
            features,
        )
        for cutoff in EVENT_RISK_CUTOFFS:
            threshold = (
                training_event_risk_threshold(
                    pretest_scores,
                    cutoff,
                )
            )
            frame[
                f"risk_threshold_{cutoff}"
            ] = threshold
        frame[
            "event_risk_thresholds"
        ] = True
        frames.append(
            frame
        )
    pooled_analysis(
        frames
    )
if __name__ == "__main__":
    main()
