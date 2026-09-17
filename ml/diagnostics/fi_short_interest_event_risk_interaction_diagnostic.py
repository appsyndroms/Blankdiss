"""
Conditional interaction diagnostic.

Question:
    Is the positive short-interest-change effect stronger when event risk
    is higher?

Design:
    1. Fit/select the event-risk model using train/validation only.
    2. Calculate OOS event-risk scores.
    3. Define event-risk bands from TRAINING score distribution:
           - top 5%
           - 5-20%
           - 20-50%
           - bottom 50%
    4. Within every OOS band, compare:
           - top 20% of positive short-interest changes
           - all other positive changes
    5. Only observations with an extreme direction are included:
           DOWN = forward return <= -10%
           UP   = forward return >= +10%

The positive-change threshold is fixed at the training 80th percentile
among positive changes. It is NOT searched over multiple cutoffs.

No repository files are modified by this script.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

from analysis.feature_config import PRICE_DIR
from analysis.feature_prices import find_price_files, load_prices
from ml.config import WALK_FORWARD_WINDOWS
from ml.dataset import load_features

from ml.diagnostics.fi_direction_short_dynamics_diagnostic import (
    EVENT_FEATURE_SETS,
    EVENT_THRESHOLD,
    EVENT_TAIL,
    RANDOM_STATE,
    build_event_model,
    deduplicate_columns,
    first_existing_column,
    fit_event_model,
    load_price_data,
    normalize_date_column,
    numeric_series,
    predict_event_score,
    prepare_data,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

POSITIVE_CHANGE_CUTOFF = 0.20

RISK_BANDS = (
    ("top_5pct", 0.00, 0.05),
    ("5_to_20pct", 0.05, 0.20),
    ("20_to_50pct", 0.20, 0.50),
    ("bottom_50pct", 0.50, 1.00),
)

# Number of bootstrap iterations for confidence intervals.
BOOTSTRAP_ITERATIONS = 2_000


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ComparisonResult:
    band: str
    low_n: int
    low_down: int
    low_down_rate: float
    high_n: int
    high_down: int
    high_down_rate: float
    delta: float
    ci_low: float
    ci_high: float
    p_positive: float
    fisher_p: float


def bootstrap_difference(
    low: pd.Series,
    high: pd.Series,
    iterations: int = BOOTSTRAP_ITERATIONS,
    random_state: int = RANDOM_STATE,
) -> tuple[float, float, float, float]:
    """
    Bootstrap:

        mean(high) - mean(low)

    Returns:
        delta,
        2.5% CI,
        97.5% CI,
        P(delta > 0)
    """

    low = pd.to_numeric(
        low,
        errors="coerce",
    ).dropna().to_numpy()

    high = pd.to_numeric(
        high,
        errors="coerce",
    ).dropna().to_numpy()

    if len(low) == 0 or len(high) == 0:
        return (
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
        )

    rng = np.random.default_rng(
        random_state
    )

    deltas = np.empty(
        iterations,
        dtype=float,
    )

    for index in range(iterations):
        low_sample = rng.choice(
            low,
            size=len(low),
            replace=True,
        )

        high_sample = rng.choice(
            high,
            size=len(high),
            replace=True,
        )

        deltas[index] = (
            high_sample.mean()
            - low_sample.mean()
        )

    return (
        float(deltas.mean()),
        float(np.quantile(deltas, 0.025)),
        float(np.quantile(deltas, 0.975)),
        float((deltas > 0).mean()),
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

        from sklearn.metrics import roc_auc_score

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

def training_event_score_boundaries(
    train_scores: pd.Series,
) -> dict[str, float]:
    """
    Returns score boundaries corresponding to:

        top 5%
        top 20%
        top 50%

    The values are calculated from TRAINING data only.
    """

    scores = pd.to_numeric(
        train_scores,
        errors="coerce",
    ).dropna()

    if scores.empty:
        return {}

    return {
        "q95": float(
            scores.quantile(0.95)
        ),
        "q80": float(
            scores.quantile(0.80)
        ),
        "q50": float(
            scores.quantile(0.50)
        ),
    }


def positive_change_threshold(
    train: pd.DataFrame,
) -> float:
    """
    Fixed top-20% threshold among positive absolute FI changes.

    Learned from TRAINING data only.
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
# Risk bands
# ---------------------------------------------------------------------------

def assign_risk_band(
    scores: pd.Series,
    boundaries: dict[str, float],
) -> pd.Series:
    """
    Assign each OOS score to a training-defined event-risk band.
    """

    q95 = boundaries.get("q95")
    q80 = boundaries.get("q80")
    q50 = boundaries.get("q50")

    result = pd.Series(
        pd.NA,
        index=scores.index,
        dtype="string",
    )

    valid = pd.to_numeric(
        scores,
        errors="coerce",
    ).notna()

    result.loc[
        valid
        & (scores >= q95)
    ] = "top_5pct"

    result.loc[
        valid
        & (scores < q95)
        & (scores >= q80)
    ] = "5_to_20pct"

    result.loc[
        valid
        & (scores < q80)
        & (scores >= q50)
    ] = "20_to_50pct"

    result.loc[
        valid
        & (scores < q50)
    ] = "bottom_50pct"

    return result


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze_band(
    frame: pd.DataFrame,
    band: str,
) -> ComparisonResult:
    subset = frame[
        frame["risk_band"] == band
    ].copy()

    subset = subset.dropna(
        subset=[
            "direction",
            "short_interest_pct_change",
        ]
    )

    subset = subset[
        subset[
            "short_interest_pct_change"
        ] > 0
    ].copy()

    if subset.empty:
        return ComparisonResult(
            band=band,
            low_n=0,
            low_down=0,
            low_down_rate=float("nan"),
            high_n=0,
            high_down=0,
            high_down_rate=float("nan"),
            delta=float("nan"),
            ci_low=float("nan"),
            ci_high=float("nan"),
            p_positive=float("nan"),
            fisher_p=float("nan"),
        )

    threshold = float(
        subset["training_positive_threshold"]
        .iloc[0]
    )

    subset["high_change"] = (
        subset[
            "short_interest_pct_change"
        ]
        >= threshold
    )

    low = subset[
        ~subset["high_change"]
    ]["direction"]

    high = subset[
        subset["high_change"]
    ]["direction"]

    low_n = len(low)
    high_n = len(high)

    low_down = int(
        (low == 1).sum()
    )

    high_down = int(
        (high == 1).sum()
    )

    low_rate = (
        low_down / low_n
        if low_n
        else float("nan")
    )

    high_rate = (
        high_down / high_n
        if high_n
        else float("nan")
    )

    if low_n and high_n:
        delta, ci_low, ci_high, p_positive = (
            bootstrap_difference(
                low,
                high,
            )
        )

        fisher_p = safe_fisher_p(
            low_down=low_down,
            low_up=low_n - low_down,
            high_down=high_down,
            high_up=high_n - high_down,
        )
    else:
        delta = float("nan")
        ci_low = float("nan")
        ci_high = float("nan")
        p_positive = float("nan")
        fisher_p = float("nan")

    return ComparisonResult(
        band=band,
        low_n=low_n,
        low_down=low_down,
        low_down_rate=low_rate,
        high_n=high_n,
        high_down=high_down,
        high_down_rate=high_rate,
        delta=delta,
        ci_low=ci_low,
        ci_high=ci_high,
        p_positive=p_positive,
        fisher_p=fisher_p,
    )


def print_result(
    result: ComparisonResult,
) -> None:
    print()
    print(
        f"  {result.band}"
    )

    print(
        f"    LOW  n={result.low_n:4d} "
        f"DOWN={result.low_down:4d} "
        f"rate={result.low_down_rate:.4f}"
    )

    print(
        f"    HIGH n={result.high_n:4d} "
        f"DOWN={result.high_down:4d} "
        f"rate={result.high_down_rate:.4f}"
    )

    print(
        f"    delta HIGH-LOW: "
        f"{result.delta:+.4f}"
    )

    print(
        f"    bootstrap CI: "
        f"[{result.ci_low:+.4f}, "
        f"{result.ci_high:+.4f}]"
    )

    print(
        f"    P(delta > 0): "
        f"{result.p_positive:.4f}"
    )

    print(
        f"    Fisher p: "
        f"{result.fisher_p:.4f}"
    )


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
    # Refit the selected model on train + validation.
    #
    # Selection is based only on validation AUC.
    # The final model then gets all historical information available
    # before the test period.
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

    # Training+validation scores are used only to define the
    # event-risk bands before looking at test outcomes.
    train_validation_scores = (
        predict_event_score(
            final_model,
            train_validation,
            features,
        )
    )

    boundaries = (
        training_event_score_boundaries(
            train_validation_scores
        )
    )

    if not boundaries:
        print(
            "Could not calculate event-risk boundaries."
        )
        return pd.DataFrame()

    positive_threshold = (
        positive_change_threshold(
            train_validation
        )
    )

    print(
        "Event-risk boundaries "
        "(training-derived):"
    )
    print(
        f"  top 5%:    score >= "
        f"{boundaries['q95']:.6f}"
    )
    print(
        f"  top 20%:   score >= "
        f"{boundaries['q80']:.6f}"
    )
    print(
        f"  top 50%:   score >= "
        f"{boundaries['q50']:.6f}"
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
    # OOS scoring
    # ---------------------------------------------------------------

    test = test.copy()

    test["event_score"] = (
        predict_event_score(
            final_model,
            test,
            features,
        )
    )

    test["risk_band"] = assign_risk_band(
        test["event_score"],
        boundaries,
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
            "risk_band",
            "short_interest_pct_change",
        ]
    )

    print(
        f"Directional OOS rows: "
        f"{len(test):,}"
    )

    print()
    print(
        "Positive short-interest "
        "change interaction:"
    )

    results: list[ComparisonResult] = []

    for band, _, _ in RISK_BANDS:
        result = analyze_band(
            test,
            band,
        )

        results.append(result)
        print_result(result)

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
    print("POOLED OOS INTERACTION")
    print("=" * 80)

    print(
        f"Directional OOS rows: "
        f"{len(pooled):,}"
    )

    for band, _, _ in RISK_BANDS:
        result = analyze_band(
            pooled,
            band,
        )

        print_result(result)

    # ---------------------------------------------------------------
    # Direct interaction view
    # ---------------------------------------------------------------

    pooled = pooled.copy()

    pooled["high_change"] = (
        pooled[
            "short_interest_pct_change"
        ]
        >= pooled[
            "training_positive_threshold"
        ]
    )

    pooled["down"] = (
        pooled["direction"] == 1
    ).astype(int)

    print()
    print(
        "Interaction summary:"
    )

    for band, _, _ in RISK_BANDS:
        subset = pooled[
            pooled["risk_band"] == band
        ].copy()

        if subset.empty:
            continue

        low = subset[
            ~subset["high_change"]
        ]

        high = subset[
            subset["high_change"]
        ]

        print()
        print(
            f"{band}:"
        )
        print(
            f"  LOW : n={len(low):4d}, "
            f"DOWN={low['down'].mean():.4f}"
            if len(low)
            else "  LOW : n=0"
        )
        print(
            f"  HIGH: n={len(high):4d}, "
            f"DOWN={high['down'].mean():.4f}"
            if len(high)
            else "  HIGH: n=0"
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
        "Risk bands:"
    )

    for name, low, high in RISK_BANDS:
        print(
            f"  {name}: "
            f"{low:.0%} -> {high:.0%}"
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
