"""
Volatility x short-interest-level interaction diagnostic.

Question:
    Within an already identified extreme event-risk group, does a high
    short-interest level add information when volatility is high?

Primary hypothesis:
    The probability of a large downward move is particularly high when
    BOTH:
        - event risk is high
        - short interest is high
        - volatility is high

Design for each walk-forward window:

1. Select the event-risk model using TRAIN -> VALIDATION.
2. Refit the selected model on TRAIN + VALIDATION.
3. Calculate event-risk scores.
4. Define the extreme event-risk tail from PRE-TEST scores only:
       top 5%
5. Define HIGH volatility from PRE-TEST volatility distribution:
       top 20%
6. Define HIGH short interest from PRE-TEST short-interest distribution:
       top 20%
7. Restrict the primary analysis to the OOS 5% event-risk tail.
8. Build the 2x2 interaction table:

                    LOW SI       HIGH SI
    LOW VOL           A            B
    HIGH VOL          C            D

9. Test the interaction directly:

       (D - C) - (B - A)

   where each cell contains the event rate.

10. Evaluate:
       - down_10pct_5d
       - down_7pct_5d
       - down_5pct_5d

11. Repeat the same analysis for each walk-forward test window
    and for pooled OOS observations.

All thresholds are learned from PRE-TEST data only.
No test outcomes are used to define groups.

No repository files are modified by this script.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.config import WALK_FORWARD_WINDOWS
from ml.dataset import load_features


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RANDOM_STATE = 42
BOOTSTRAP_ITERATIONS = 2_000

EVENT_THRESHOLD = 0.10
EVENT_TAIL = 0.05

VOLATILITY_TAIL = 0.20
SHORT_INTEREST_TAIL = 0.20

PRIMARY_TARGET = "down_10pct_5d"

TARGETS = (
    "down_10pct_5d",
    "down_7pct_5d",
    "down_5pct_5d",
)


EVENT_FEATURE_SETS = {
    "volatility_20d": (
        "price_volatility_20d",
    ),
    "volatility_60d": (
        "volatility_60d",
    ),
    "volatility_20d_plus_60d": (
        "price_volatility_20d",
        "volatility_60d",
    ),
    "volatility_20d_plus_60d_plus_term_structure": (
        "price_volatility_20d",
        "volatility_60d",
        "volatility_term_structure",
    ),
}


# ---------------------------------------------------------------------------
# Result structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CellResult:
    volatility_group: str
    short_interest_group: str
    n: int
    events: int
    event_rate: float


@dataclass(frozen=True)
class InteractionResult:
    window: str
    target: str

    event_risk_tail_n: int

    low_vol_low_si_n: int
    low_vol_high_si_n: int
    high_vol_low_si_n: int
    high_vol_high_si_n: int

    low_vol_low_si_rate: float
    low_vol_high_si_rate: float
    high_vol_low_si_rate: float
    high_vol_high_si_rate: float

    low_vol_si_effect: float
    high_vol_si_effect: float

    interaction: float
    ci_low: float
    ci_high: float
    p_positive: float


# ---------------------------------------------------------------------------
# Basic helpers
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


def safe_auc(
    y_true: pd.Series,
    scores: pd.Series,
) -> float:
    frame = pd.DataFrame(
        {
            "y": pd.to_numeric(
                y_true,
                errors="coerce",
            ),
            "score": pd.to_numeric(
                scores,
                errors="coerce",
            ),
        }
    ).dropna()

    if len(frame) < 2:
        return float("nan")

    if frame["y"].nunique() < 2:
        return float("nan")

    if frame["score"].nunique() < 2:
        return 0.5

    return float(
        roc_auc_score(
            frame["y"],
            frame["score"],
        )
    )


def safe_rate(
    values: pd.Series,
) -> float:
    if len(values) == 0:
        return float("nan")

    return float(
        values.mean()
    )


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def bootstrap_interaction(
    low_vol_low_si: pd.Series,
    low_vol_high_si: pd.Series,
    high_vol_low_si: pd.Series,
    high_vol_high_si: pd.Series,
    iterations: int = BOOTSTRAP_ITERATIONS,
    random_state: int = RANDOM_STATE,
) -> tuple[
    float,
    float,
    float,
    float,
]:
    """
    Bootstrap:

        (HIGH SI - LOW SI) at HIGH VOL
        -
        (HIGH SI - LOW SI) at LOW VOL
    """

    groups = []

    for values in (
        low_vol_low_si,
        low_vol_high_si,
        high_vol_low_si,
        high_vol_high_si,
    ):
        numeric = pd.to_numeric(
            values,
            errors="coerce",
        ).dropna().to_numpy()

        groups.append(numeric)

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
        low_low,
        low_high,
        high_low,
        high_high,
    ) = groups

    observed = (
        high_high.mean()
        - high_low.mean()
        - low_high.mean()
        + low_low.mean()
    )

    rng = np.random.default_rng(
        random_state
    )

    interactions = np.empty(
        iterations,
        dtype=float,
    )

    for index in range(iterations):
        low_low_sample = rng.choice(
            low_low,
            size=len(low_low),
            replace=True,
        )

        low_high_sample = rng.choice(
            low_high,
            size=len(low_high),
            replace=True,
        )

        high_low_sample = rng.choice(
            high_low,
            size=len(high_low),
            replace=True,
        )

        high_high_sample = rng.choice(
            high_high,
            size=len(high_high),
            replace=True,
        )

        interactions[index] = (
            high_high_sample.mean()
            - high_low_sample.mean()
            - low_high_sample.mean()
            + low_low_sample.mean()
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
# Event model
# ---------------------------------------------------------------------------

def build_event_model() -> Pipeline:
    return Pipeline(
        [
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=2_000,
                    C=1.0,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def fit_event_model(
    train: pd.DataFrame,
    features: list[str],
) -> Pipeline | None:
    frame = train[
        features + ["event"]
    ].copy()

    for column in features:
        frame[column] = numeric_series(
            frame,
            column,
        )

    frame["event"] = numeric_series(
        frame,
        "event",
    )

    frame = frame.dropna()

    if len(frame) < 100:
        return None

    if frame["event"].nunique() < 2:
        return None

    model = build_event_model()

    model.fit(
        frame[features],
        frame["event"],
    )

    return model


def predict_event_score(
    model: Pipeline,
    data: pd.DataFrame,
    features: list[str],
) -> pd.Series:
    result = pd.Series(
        np.nan,
        index=data.index,
        dtype=float,
    )

    frame = data[
        features
    ].copy()

    for column in features:
        frame[column] = numeric_series(
            frame,
            column,
        )

    valid = frame.notna().all(
        axis=1
    )

    if not valid.any():
        return result

    result.loc[valid] = (
        model.predict_proba(
            frame.loc[
                valid,
                features,
            ]
        )[:, 1]
    )

    return result


def select_event_model(
    train: pd.DataFrame,
    validation: pd.DataFrame,
) -> tuple[
    str,
    list[str],
    float,
]:
    best_name = ""
    best_features: list[str] = []
    best_auc = float("-inf")

    for name, candidates in (
        EVENT_FEATURE_SETS.items()
    ):
        features = [
            column
            for column in candidates
            if (
                column in train.columns
                and column in validation.columns
            )
        ]

        if not features:
            continue

        model = fit_event_model(
            train,
            features,
        )

        if model is None:
            continue

        validation_score = (
            predict_event_score(
                model,
                validation,
                features,
            )
        )

        auc = safe_auc(
            validation["event"],
            validation_score,
        )

        if (
            not np.isnan(auc)
            and auc > best_auc
        ):
            best_name = name
            best_features = features
            best_auc = auc

    if not best_features:
        raise RuntimeError(
            "Could not select event-risk model."
        )

    return (
        best_name,
        best_features,
        float(best_auc),
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
# Data preparation
# ---------------------------------------------------------------------------

def prepare_data() -> pd.DataFrame:
    data = load_features()

    if data is None or data.empty:
        raise RuntimeError(
            "Feature dataset is empty."
        )

    required = [
        "snapshot_date",
        "security_key",
        "forward_return_5d",
        "short_interest_pct",
        "price_volatility_20d",
    ]

    missing = [
        column
        for column in required
        if column not in data.columns
    ]

    if missing:
        raise RuntimeError(
            "Missing required columns: "
            f"{missing}"
        )

    data = data.copy()

    data["snapshot_date"] = pd.to_datetime(
        data["snapshot_date"],
        errors="coerce",
    )

    data["short_interest_pct"] = (
        numeric_series(
            data,
            "short_interest_pct",
        )
    )

    data["price_volatility_20d"] = (
        numeric_series(
            data,
            "price_volatility_20d",
        )
    )

    data["forward_return_5d"] = (
        numeric_series(
            data,
            "forward_return_5d",
        )
    )

    data = data.dropna(
        subset=[
            "snapshot_date",
            "security_key",
            "forward_return_5d",
        ]
    )

    # Primary event model target:
    # absolute 5-day movement >= 10%.
    data["event"] = (
        data["forward_return_5d"]
        .abs()
        >= EVENT_THRESHOLD
    ).astype(int)

    # Direction:
    # 1 = DOWN >= 10%
    # 0 = UP >= 10%
    data["direction"] = np.where(
        data["forward_return_5d"]
        <= -EVENT_THRESHOLD,
        1,
        np.where(
            data["forward_return_5d"]
            >= EVENT_THRESHOLD,
            0,
            np.nan,
        ),
    )

    # Explicit targets.
    data["down_10pct_5d"] = (
        data["forward_return_5d"]
        <= -0.10
    ).astype(int)

    data["down_7pct_5d"] = (
        data["forward_return_5d"]
        <= -0.07
    ).astype(int)

    data["down_5pct_5d"] = (
        data["forward_return_5d"]
        <= -0.05
    ).astype(int)

    return data.sort_values(
        [
            "snapshot_date",
            "security_key",
        ],
        kind="mergesort",
    ).reset_index(
        drop=True
    )


# ---------------------------------------------------------------------------
# Event-risk tail
# ---------------------------------------------------------------------------

def assign_event_risk_tail(
    scores: pd.Series,
    threshold: float,
) -> pd.Series:
    numeric = pd.to_numeric(
        scores,
        errors="coerce",
    )

    result = pd.Series(
        pd.NA,
        index=scores.index,
        dtype="boolean",
    )

    valid = numeric.notna()

    result.loc[
        valid
    ] = (
        numeric.loc[valid]
        >= threshold
    )

    return result


# ---------------------------------------------------------------------------
# 2x2 interaction
# ---------------------------------------------------------------------------

def interaction_analysis(
    frame: pd.DataFrame,
    target: str,
    window_name: str,
) -> InteractionResult | None:
    required = [
        "volatility_group",
        "short_interest_group",
        target,
    ]

    frame = frame.dropna(
        subset=required
    ).copy()

    if frame.empty:
        return None

    low_vol_low_si = frame[
        (
            frame["volatility_group"]
            == "LOW"
        )
        & (
            frame["short_interest_group"]
            == "LOW"
        )
    ][target]

    low_vol_high_si = frame[
        (
            frame["volatility_group"]
            == "LOW"
        )
        & (
            frame["short_interest_group"]
            == "HIGH"
        )
    ][target]

    high_vol_low_si = frame[
        (
            frame["volatility_group"]
            == "HIGH"
        )
        & (
            frame["short_interest_group"]
            == "LOW"
        )
    ][target]

    high_vol_high_si = frame[
        (
            frame["volatility_group"]
            == "HIGH"
        )
        & (
            frame["short_interest_group"]
            == "HIGH"
        )
    ][target]

    groups = [
        low_vol_low_si,
        low_vol_high_si,
        high_vol_low_si,
        high_vol_high_si,
    ]

    if any(
        len(group) == 0
        for group in groups
    ):
        print(
            f"  {target}: "
            "insufficient cells for interaction."
        )
        return None

    rates = [
        safe_rate(group)
        for group in groups
    ]

    (
        low_low_rate,
        low_high_rate,
        high_low_rate,
        high_high_rate,
    ) = rates

    low_vol_si_effect = (
        low_high_rate
        - low_low_rate
    )

    high_vol_si_effect = (
        high_high_rate
        - high_low_rate
    )

    (
        interaction,
        ci_low,
        ci_high,
        p_positive,
    ) = bootstrap_interaction(
        low_vol_low_si,
        low_vol_high_si,
        high_vol_low_si,
        high_vol_high_si,
    )

    print()
    print(
        f"  Target: {target}"
    )
    print(
        "                         "
        "LOW SI        HIGH SI"
    )
    print(
        f"  LOW VOL                "
        f"{low_low_rate:.4f} "
        f"(n={len(low_vol_low_si):4d})    "
        f"{low_high_rate:.4f} "
        f"(n={len(low_vol_high_si):4d})"
    )
    print(
        f"  HIGH VOL               "
        f"{high_low_rate:.4f} "
        f"(n={len(high_vol_low_si):4d})    "
        f"{high_high_rate:.4f} "
        f"(n={len(high_vol_high_si):4d})"
    )

    print()
    print(
        "  SI effect at LOW volatility:  "
        f"{low_vol_si_effect:+.4f}"
    )

    print(
        "  SI effect at HIGH volatility: "
        f"{high_vol_si_effect:+.4f}"
    )

    print(
        "  INTERACTION:                 "
        f"{interaction:+.4f}"
    )

    print(
        "  Bootstrap 95% CI:            "
        f"[{ci_low:+.4f}, {ci_high:+.4f}]"
    )

    print(
        "  P(interaction > 0):          "
        f"{p_positive:.4f}"
    )

    return InteractionResult(
        window=window_name,
        target=target,
        event_risk_tail_n=len(frame),
        low_vol_low_si_n=len(
            low_vol_low_si
        ),
        low_vol_high_si_n=len(
            low_vol_high_si
        ),
        high_vol_low_si_n=len(
            high_vol_low_si
        ),
        high_vol_high_si_n=len(
            high_vol_high_si
        ),
        low_vol_low_si_rate=low_low_rate,
        low_vol_high_si_rate=low_high_rate,
        high_vol_low_si_rate=high_low_rate,
        high_vol_high_si_rate=high_high_rate,
        low_vol_si_effect=low_vol_si_effect,
        high_vol_si_effect=high_vol_si_effect,
        interaction=interaction,
        ci_low=ci_low,
        ci_high=ci_high,
        p_positive=p_positive,
    )


# ---------------------------------------------------------------------------
# Single walk-forward window
# ---------------------------------------------------------------------------

def run_window(
    data: pd.DataFrame,
    train_end: str,
    validation_end: str,
    test_end: str,
) -> tuple[
    pd.DataFrame,
    list[InteractionResult],
]:
    train = data[
        data["snapshot_date"]
        <= train_end
    ].copy()

    validation = data[
        (
            data["snapshot_date"]
            > train_end
        )
        & (
            data["snapshot_date"]
            <= validation_end
        )
    ].copy()

    test = data[
        (
            data["snapshot_date"]
            > validation_end
        )
        & (
            data["snapshot_date"]
            <= test_end
        )
    ].copy()

    window_name = (
        f"{train_end}_"
        f"{validation_end}_"
        f"{test_end}"
    )

    print()
    print("=" * 80)
    print(
        f"WINDOW: {window_name}"
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

    # ---------------------------------------------------------------
    # Select event-risk model.
    # ---------------------------------------------------------------

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

    # ---------------------------------------------------------------
    # Refit on all pre-test data.
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
        raise RuntimeError(
            "Could not fit final event model."
        )

    pretest_scores = predict_event_score(
        final_model,
        train_validation,
        features,
    )

    # ---------------------------------------------------------------
    # Pre-test thresholds.
    # ---------------------------------------------------------------

    event_risk_threshold = (
        percentile_threshold(
            pretest_scores,
            EVENT_TAIL,
        )
    )

    volatility_threshold = (
        percentile_threshold(
            train_validation[
                "price_volatility_20d"
            ],
            VOLATILITY_TAIL,
        )
    )

    short_interest_threshold = (
        percentile_threshold(
            train_validation[
                "short_interest_pct"
            ],
            SHORT_INTEREST_TAIL,
        )
    )

    print()
    print(
        "PRE-TEST THRESHOLDS"
    )
    print(
        f"  Event-risk top "
        f"{EVENT_TAIL:.0%}: "
        f"{event_risk_threshold:.6f}"
    )
    print(
        f"  Volatility top "
        f"{VOLATILITY_TAIL:.0%}: "
        f"{volatility_threshold:.6f}"
    )
    print(
        f"  Short interest top "
        f"{SHORT_INTEREST_TAIL:.0%}: "
        f"{short_interest_threshold:.6f}"
    )

    # ---------------------------------------------------------------
    # Score OOS test data.
    # ---------------------------------------------------------------

    test = test.copy()

    test["event_score"] = (
        predict_event_score(
            final_model,
            test,
            features,
        )
    )

    test = test.dropna(
        subset=[
            "event_score",
            "price_volatility_20d",
            "short_interest_pct",
        ]
    )

    # ---------------------------------------------------------------
    # Restrict to OOS extreme event-risk tail.
    # ---------------------------------------------------------------

    test["event_risk_tail"] = (
        test["event_score"]
        >= event_risk_threshold
    )

    event_tail = test[
        test["event_risk_tail"]
    ].copy()

    print()
    print(
        f"OOS rows: "
        f"{len(test):,}"
    )
    print(
        f"OOS 5% event-risk tail: "
        f"{len(event_tail):,}"
    )

    if event_tail.empty:
        return (
            event_tail,
            [],
        )

    # ---------------------------------------------------------------
    # Assign HIGH / LOW groups using PRE-TEST thresholds.
    # ---------------------------------------------------------------

    event_tail["volatility_group"] = (
        np.where(
            event_tail[
                "price_volatility_20d"
            ]
            >= volatility_threshold,
            "HIGH",
            "LOW",
        )
    )

    event_tail["short_interest_group"] = (
        np.where(
            event_tail[
                "short_interest_pct"
            ]
            >= short_interest_threshold,
            "HIGH",
            "LOW",
        )
    )

    # ---------------------------------------------------------------
    # Print group sizes.
    # ---------------------------------------------------------------

    print()
    print(
        "OOS EVENT-RISK TAIL 2x2:"
    )

    counts = (
        event_tail
        .groupby(
            [
                "volatility_group",
                "short_interest_group",
            ],
            dropna=False,
        )
        .size()
    )

    for volatility_group in (
        "LOW",
        "HIGH",
    ):
        for short_interest_group in (
            "LOW",
            "HIGH",
        ):
            count = int(
                counts.get(
                    (
                        volatility_group,
                        short_interest_group,
                    ),
                    0,
                )
            )

            print(
                f"  {volatility_group:<4} VOL / "
                f"{short_interest_group:<4} SI: "
                f"{count:,}"
            )

    # ---------------------------------------------------------------
    # Run interaction for every predefined target.
    # ---------------------------------------------------------------

    results: list[
        InteractionResult
    ] = []

    for target in TARGETS:
        result = interaction_analysis(
            event_tail,
            target,
            window_name,
        )

        if result is not None:
            results.append(
                result
            )

    return (
        event_tail,
        results,
    )


# ---------------------------------------------------------------------------
# Pooled analysis
# ---------------------------------------------------------------------------

def pooled_analysis(
    frames: list[pd.DataFrame],
) -> list[InteractionResult]:
    if not frames:
        print(
            "No OOS frames available "
            "for pooled analysis."
        )
        return []

    pooled = pd.concat(
        frames,
        ignore_index=True,
    )

    print()
    print("=" * 80)
    print(
        "POOLED OOS INTERACTION"
    )
    print("=" * 80)

    print(
        f"Pooled OOS event-risk rows: "
        f"{len(pooled):,}"
    )

    results: list[
        InteractionResult
    ] = []

    for target in TARGETS:
        result = interaction_analysis(
            pooled,
            target,
            "POOLED",
        )

        if result is not None:
            results.append(
                result
            )

    return results


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
        "INTERACTION SUMMARY"
    )
    print("=" * 80)

    print(
        "Window              "
        "Target             "
        "N       "
        "LOW-VOL SI Δ   "
        "HIGH-VOL SI Δ  "
        "Interaction   "
        "CI low       "
        "CI high      "
        "P(>0)"
    )

    for result in results:
        print(
            f"{result.window[:18]:<18} "
            f"{result.target:<18} "
            f"{result.event_risk_tail_n:6d} "
            f"{result.low_vol_si_effect:+.4f}       "
            f"{result.high_vol_si_effect:+.4f}       "
            f"{result.interaction:+.4f}      "
            f"{result.ci_low:+.4f}    "
            f"{result.ci_high:+.4f}    "
            f"{result.p_positive:.4f}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(
        "Blankdiss "
        "Volatility x Short-Interest-Level "
        "Interaction Diagnostic"
    )

    print()
    print(
        "Primary question:"
    )
    print(
        "Does high short interest add information "
        "when volatility is already high?"
    )

    print()
    print(
        f"Event threshold:       "
        f"{EVENT_THRESHOLD:.0%}"
    )
    print(
        f"Event-risk tail:       "
        f"top {EVENT_TAIL:.0%}"
    )
    print(
        f"Volatility tail:       "
        f"top {VOLATILITY_TAIL:.0%}"
    )
    print(
        f"Short-interest tail:   "
        f"top {SHORT_INTEREST_TAIL:.0%}"
    )

    data = prepare_data()

    print()
    print(
        f"Prepared rows: "
        f"{len(data):,}"
    )
    print(
        f"Date range: "
        f"{data['snapshot_date'].min().date()} "
        f"-> "
        f"{data['snapshot_date'].max().date()}"
    )

    all_frames: list[pd.DataFrame] = []
    all_results: list[
        InteractionResult
    ] = []

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

    pooled_results = pooled_analysis(
        all_frames
    )

    all_results.extend(
        pooled_results
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
