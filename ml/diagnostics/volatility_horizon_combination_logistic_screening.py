"""
Controlled comparison of volatility horizon combinations.
Question:
Does volatility_change_20d_60d add information to VOL60,
and is that different from adding it to VOL20?
Compared feature sets:
    1. volatility_60d
    2. volatility_60d_plus_change
    3. volatility_20d
    4. volatility_20d_plus_change
All experiments use:
    - logistic_regression only
    - identical walk-forward windows
    - identical economic target
    - identical top-fraction evaluation
"""
from __future__ import annotations
import time
import pandas as pd
import ml.walk_forward as walk_forward
from analysis.feature_config import PRICE_DIR
from ml.config import TARGETS, WalkForwardWindow
from ml.dataset import build_target, load_features
from ml.models import build_models
from prices import load_prices
TARGET_NAME = "down_5pct_5d"
TOP_FRACTIONS = (0.001, 0.005, 0.01, 0.02, 0.05)
WINDOWS = (
    WalkForwardWindow(
        train_end="2023-12-31",
        validation_end="2024-12-31",
        test_end="2025-12-31",
    ),
    WalkForwardWindow(
        train_end="2024-12-31",
        validation_end="2025-12-31",
        test_end="2026-12-31",
    ),
)
FEATURE_SETS = {
    "volatility_60d": [
        "volatility_60d",
    ],
    "volatility_60d_plus_change": [
        "volatility_60d",
        "volatility_change_20d_60d",
    ],
    "volatility_20d": [
        "price_volatility_20d",
    ],
    "volatility_20d_plus_change": [
        "price_volatility_20d",
        "volatility_change_20d_60d",
    ],
}
def resolve_target():
    for target in TARGETS:
        if target.name == TARGET_NAME:
            return target
    raise ValueError(
        f"Target not found in ml.config.TARGETS: {TARGET_NAME}"
    )
def build_volatility_features(features: pd.DataFrame) -> pd.DataFrame:
    print("Loading raw price data for 60d volatility...")
    prices = load_prices(PRICE_DIR)
    print(f"Prisrader: {len(prices):,}")
    prices = prices.copy()
    prices["price_date"] = pd.to_datetime(prices["price_date"])
    prices = prices.sort_values(
        ["yahoo_symbol", "price_date"]
    )
    prices["daily_return"] = (
        prices.groupby("yahoo_symbol")["close"]
        .pct_change()
    )
    prices["volatility_60d"] = (
        prices.groupby("yahoo_symbol")["daily_return"]
        .transform(
            lambda s: s.rolling(
                window=59,
                min_periods=59,
            ).std()
        )
    )
    volatility = prices[
        [
            "yahoo_symbol",
            "price_date",
            "volatility_60d",
        ]
    ].copy()
    volatility["price_date"] = pd.to_datetime(
        volatility["price_date"]
    )
    result = features.copy()
    result["snapshot_date"] = pd.to_datetime(
        result["snapshot_date"]
    )
    result = result.merge(
        volatility,
        left_on=["yahoo_symbol", "snapshot_date"],
        right_on=["yahoo_symbol", "price_date"],
        how="left",
    )
    result["volatility_change_20d_60d"] = (
        result["price_volatility_20d"]
        - result["volatility_60d"]
    )
    return result
def evaluate_economic(
    oos_frames: list[pd.DataFrame],
) -> dict:
    oos = pd.concat(
        oos_frames,
        ignore_index=True,
    )
    probability_column = "prediction"
    target_return_column = "target_return"
    oos = oos.dropna(
        subset=[
            probability_column,
            target_return_column,
        ]
    ).copy()
    oos = oos.sort_values(
        probability_column,
        ascending=False,
    )
    baseline_event_rate = oos["target"].mean()
    baseline_mean_return = oos[target_return_column].mean()
    result = {
        "rows": len(oos),
        "baseline_event_rate": baseline_event_rate,
        "baseline_mean_return": baseline_mean_return,
    }
    for fraction in TOP_FRACTIONS:
        n = max(
            1,
            int(len(oos) * fraction),
        )
        top = oos.head(n)
        event_rate = top["target"].mean()
        lift = (
            event_rate / baseline_event_rate
            if baseline_event_rate > 0
            else float("nan")
        )
        mean_return = top[target_return_column].mean()
        median_return = top[target_return_column].median()
        key = f"{fraction:.4f}"
        result[key] = {
            "n": n,
            "event_rate": event_rate,
            "lift": lift,
            "mean_return": mean_return,
            "median_return": median_return,
        }
    return result
def run_feature_set(
    name: str,
    features: pd.DataFrame,
    target,
    feature_columns: list[str],
) -> dict:
    print(f"Preparing feature set: {name}")
    print(f"  Features: {len(feature_columns)}")
    print(
        "  Feature names: "
        + ", ".join(feature_columns)
    )
    subset = features[
        [
            "snapshot_date",
            "security_key",
            "forward_return_5d",
            *feature_columns,
        ]
    ].copy()
    subset = subset.dropna(
        subset=feature_columns
    )
    print(f"  Rows: {len(subset):,}")
    target_data = build_target(
        subset,
        target,
    )
    data = target_data.copy()
    data = data.rename(
        columns={
            "forward_return_5d": "target_return",
        }
    )
    y = data["target"]
    print("=" * 58)
    print(f"RUNNING: {name}")
    print("=" * 58)
    start = time.perf_counter()
    oos_frames = []
    validation_aucs = []
    fit_time_total = 0.0
    for index, window in enumerate(WINDOWS, start=1):
        print(f"  Window {index}")
        print(f"    Train <= {window.train_end}")
        print(f"    Validation <= {window.validation_end}")
        print(f"    Test <= {window.test_end}")
        print("    Model: logistic_regression")
        window_start = time.perf_counter()
        result = walk_forward.train_window(
            data=data,
            y=y,
            feature_columns=feature_columns,
            window=window,
            task="classification",
            direction="below",
        )
        elapsed = time.perf_counter() - window_start
        validation_auc = result["validation_auc"]
        validation_aucs.append(validation_auc)
        fit_time = result.get(
            "fit_time",
            elapsed,
        )
        fit_time_total += fit_time
        print(
            f"    Validation AUC: {validation_auc:.6f}"
        )
        print(
            f"    Fit: {fit_time:.2f}s"
        )
        oos = result["oos_predictions"].copy()
        oos_frames.append(oos)
        economic = evaluate_economic([oos])
        print("    Economic OOS:")
        print(
            f"      Rows: {economic['rows']:,}"
        )
        print(
            "      Baseline event rate: "
            f"{economic['baseline_event_rate']:.4f}"
        )
        print(
            "      Baseline mean return: "
            f"{economic['baseline_mean_return']:+.4%}"
        )
        for fraction in TOP_FRACTIONS:
            stats = economic[f"{fraction:.4f}"]
            print(
                f"      Top {fraction * 100:4.1f}%: "
                f"n={stats['n']:,} "
                f"event={stats['event_rate']:.4f} "
                f"lift={stats['lift']:5.2f}x "
                f"mean={stats['mean_return']:+.4%} "
                f"median={stats['median_return']:+.4%}"
            )
    total_time = time.perf_counter() - start
    combined_economic = evaluate_economic(
        oos_frames
    )
    return {
        "name": name,
        "feature_columns": feature_columns,
        "rows": len(data),
        "validation_aucs": validation_aucs,
        "average_validation_auc": sum(validation_aucs)
        / len(validation_aucs),
        "total_time": total_time,
        "fit_time": fit_time_total,
        "economic": combined_economic,
    }
def print_result(result: dict) -> None:
    print(result["name"])
    print(
        f"  Features: {len(result['feature_columns'])}"
    )
    print(
        "  Feature names: "
        + ", ".join(result["feature_columns"])
    )
    print(
        f"  Rows: {result['rows']:,}"
    )
    print(
        f"  Walk-forward windows: {len(WINDOWS)}"
    )
    print(
        f"  Total: {result['total_time']:.2f}s"
    )
    print(
        f"  Fit: {result['fit_time']:.2f}s"
    )
    print(
        "  Average validation AUC: "
        f"{result['average_validation_auc']:.6f}"
    )
    print(
        "  Selected models: "
        + ", ".join(
            ["logistic_regression"] * len(WINDOWS)
        )
    )
    economic = result["economic"]
    print("  Combined economic OOS:")
    print(
        f"      Rows: {economic['rows']:,}"
    )
    print(
        "      Baseline event rate: "
        f"{economic['baseline_event_rate']:.4f}"
    )
    print(
        "      Baseline mean return: "
        f"{economic['baseline_mean_return']:+.4%}"
    )
    for fraction in TOP_FRACTIONS:
        stats = economic[f"{fraction:.4f}"]
        print(
            f"      Top {fraction * 100:4.1f}%: "
            f"n={stats['n']:,} "
            f"event={stats['event_rate']:.4f} "
            f"lift={stats['lift']:5.2f}x "
            f"mean={stats['mean_return']:+.4%} "
            f"median={stats['median_return']:+.4%}"
        )
def print_comparison(results: dict[str, dict]) -> None:
    print("=" * 90)
    print("CONTROLLED LOGISTIC COMPARISON")
    print("=" * 90)
    print(
        "Feature set | AUC | top1 event | lift | "
        "top1 mean | total | fit"
    )
    print("-" * 90)
    for name, result in results.items():
        top1 = result["economic"]["0.0100"]
        print(
            f"{name:<34}"
            f"{result['average_validation_auc']:.6f}   "
            f"{top1['event_rate']:.4f}   "
            f"{top1['lift']:.2f}x   "
            f"{top1['mean_return']:+.4%}   "
            f"{result['total_time']:.2f}s   "
            f"{result['fit_time']:.2f}s"
        )
def print_delta(
    base: dict,
    other: dict,
    label: str,
) -> None:
    base_top1 = base["economic"]["0.0100"]
    other_top1 = other["economic"]["0.0100"]
    print(label)
    print(
        "  AUC delta: "
        f"{other['average_validation_auc'] - base['average_validation_auc']:+.6f}"
    )
    print(
        "  Top1 event delta: "
        f"{other_top1['event_rate'] - base_top1['event_rate']:+.4f}"
    )
    print(
        "  Top1 lift delta: "
        f"{other_top1['lift'] - base_top1['lift']:+.2f}x"
    )
    print(
        "  Top1 mean return delta: "
        f"{other_top1['mean_return'] - base_top1['mean_return']:+.4%}"
    )
    print(
        "  Total time delta: "
        f"{other['total_time'] - base['total_time']:+.2f}s"
    )
    print(
        "  Fit time delta: "
        f"{other['fit_time'] - base['fit_time']:+.2f}s"
    )
def main() -> None:
    print(
        "BLANKDISS VOLATILITY HORIZON COMBINATION "
        "LOGISTIC SCREENING"
    )
    print("Model: logistic_regression only")
    print(
        f"Walk-forward windows: {len(WINDOWS)}"
    )
    print(
        f"Economic target: {TARGET_NAME}"
    )
    print(
        "Question: "
        "Ger förändringen VOL20-VOL60 extra signal "
        "ovanpå VOL60 jämfört med ovanpå VOL20?"
    )
    print("Loading feature data...")
    features = load_features()
    print(
        f"Feature-rader: {len(features):,}"
    )
    print(
        f"Feature rows: {len(features):,}"
    )
    print("Loading raw price data...")
    print("Building volatility horizon features...")
    features = build_volatility_features(
        features
    )
    print("=" * 58)
    print("VOLATILITY HORIZON COMBINATION QC")
    print("=" * 58)
    print(
        f"Feature rows: {len(features):,}"
    )
    with_60d = features["volatility_60d"].notna()
    print(
        f"Rows with 60d volatility: "
        f"{with_60d.sum():,}"
    )
    print(
        f"Rows without 60d volatility: "
        f"{(~with_60d).sum():,}"
    )
    print("Volatility statistics:")
    for column in (
        "price_volatility_20d",
        "volatility_60d",
        "volatility_change_20d_60d",
    ):
        series = features[column].dropna()
        print(
            f"  {column}: "
            f"median={series.median():.6f} "
            f"p20={series.quantile(0.20):.6f} "
            f"p80={series.quantile(0.80):.6f}"
        )
    target = resolve_target()
    print(
        f"Target resolved: {target.name}"
    )
    # Force walk_forward to use exactly one model:
    # logistic_regression.
    original_build_models = walk_forward.build_models
    def logistic_only(
        random_state,
        task="classification",
    ):
        models = build_models(
            random_state=random_state,
            task=task,
        )
        return {
            "logistic_regression": models[
                "logistic_regression"
            ]
        }
    walk_forward.build_models = logistic_only
    try:
        results = {}
        for name, feature_columns in FEATURE_SETS.items():
            results[name] = run_feature_set(
                name=name,
                features=features,
                target=target,
                feature_columns=feature_columns,
            )
        print("=" * 90)
        print("RESULTAT")
        print("=" * 90)
        for result in results.values():
            print_result(result)
        print_comparison(results)
        print("=" * 90)
        print("DELTA VS VOLATILITY_60D")
        print("=" * 90)
        print_delta(
            results["volatility_60d"],
            results["volatility_60d_plus_change"],
            "volatility_60d_plus_change",
        )
        print_delta(
            results["volatility_60d"],
            results["volatility_20d"],
            "volatility_20d",
        )
        print_delta(
            results["volatility_60d"],
            results["volatility_20d_plus_change"],
            "volatility_20d_plus_change",
        )
        print("=" * 90)
        print("DELTA: CHANGE ABOVE EACH VOLATILITY BASE")
        print("=" * 90)
        print_delta(
            results["volatility_60d"],
            results["volatility_60d_plus_change"],
            "VOL60 -> VOL60 + CHANGE",
        )
        print_delta(
            results["volatility_20d"],
            results["volatility_20d_plus_change"],
            "VOL20 -> VOL20 + CHANGE",
        )
    finally:
        walk_forward.build_models = original_build_models
if __name__ == "__main__":
    main()
