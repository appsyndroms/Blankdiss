from __future__ import annotations
import time
from typing import Any
import numpy as np
import pandas as pd
from analysis.feature_config import PRICE_DIR
from analysis.feature_prices import load_prices
from ml.config import TARGETS, WALK_FORWARD_WINDOWS
from ml.dataset import load_feature_data, prepare_feature_set
from ml.walk_forward import train_window
BENCHMARK_TREES = 100
RF_N_JOBS = 1
ECONOMIC_TARGET = "down_5pct_5d"
ECONOMIC_TOP_FRACTIONS = (
    0.001,
    0.005,
    0.01,
    0.02,
    0.05,
)
FEATURE_SETS = {
    "volatility_20d": {
        "price_features": {
            "price_volatility_20d",
        },
    },
    "volatility_20d_plus_change": {
        "price_features": {
            "price_volatility_20d",
            "volatility_change_20d_60d",
        },
    },
}
def find_target(target_name: str):
    for target in TARGETS:
        if target.name == target_name:
            return target
    available = ", ".join(target.name for target in TARGETS)
    raise ValueError(
        f"Unknown target: {target_name}. "
        f"Available targets: {available}"
    )
def load_price_data() -> pd.DataFrame:
    print("Loading raw price data for 60d volatility...")
    price_data = load_prices(PRICE_DIR)
    if price_data.empty:
        raise ValueError("No raw price data found.")
    required = {
        "yahoo_symbol",
        "date",
        "close",
    }
    missing = required - set(price_data.columns)
    if missing:
        raise ValueError(
            "Raw price data is missing required columns: "
            f"{sorted(missing)}"
        )
    price_data = price_data.copy()
    price_data["date"] = pd.to_datetime(
        price_data["date"]
    )
    price_data = price_data.sort_values(
        ["yahoo_symbol", "date"]
    ).reset_index(drop=True)
    print(f"Prisrader: {len(price_data):,}")
    return price_data
def add_volatility_change(
    features: pd.DataFrame,
    price_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Reconstruct 60d volatility using the same conceptual
    definition as price_volatility_20d:
      - last 60 close observations
      - daily pct_change()
      - standard deviation of those returns
    60 close observations produce 59 daily returns.
    """
    prices = price_data[
        [
            "yahoo_symbol",
            "date",
            "close",
        ]
    ].copy()
    prices = prices.sort_values(
        [
            "yahoo_symbol",
            "date",
        ]
    ).reset_index(drop=True)
    prices["daily_return"] = (
        prices
        .groupby("yahoo_symbol")["close"]
        .pct_change()
    )
    prices["volatility_60d"] = (
        prices
        .groupby("yahoo_symbol")["daily_return"]
        .transform(
            lambda series: series.rolling(
                window=59,
                min_periods=59,
            ).std()
        )
    )
    prices = prices.rename(
        columns={
            "date": "price_date",
        }
    )
    lookup = prices[
        [
            "yahoo_symbol",
            "price_date",
            "volatility_60d",
        ]
    ]
    result = features.copy()
    result["price_date"] = pd.to_datetime(
        result["price_date"]
    )
    result = result.merge(
        lookup,
        on=[
            "yahoo_symbol",
            "price_date",
        ],
        how="left",
        validate="many_to_one",
    )
    result["volatility_change_20d_60d"] = (
        result["price_volatility_20d"]
        - result["volatility_60d"]
    )
    return result
def print_feature_qc(
    features: pd.DataFrame,
) -> None:
    print("=" * 40)
    print("VOLATILITY CHANGE QC")
    print("=" * 40)
    print(
        f"Feature rows: {len(features):,}"
    )
    valid_60d = features[
        "volatility_60d"
    ].notna()
    print(
        "Rows with 60d volatility: "
        f"{valid_60d.sum():,}"
    )
    print(
        "Rows without 60d volatility: "
        f"{(~valid_60d).sum():,}"
    )
    print("Volatility statistics:")
    columns = (
        "price_volatility_20d",
        "volatility_60d",
        "volatility_change_20d_60d",
    )
    for column in columns:
        series = features[column].dropna()
        print(
            f"  {column}: "
            f"median={series.median():.6f} "
            f"p20={series.quantile(0.20):.6f} "
            f"p80={series.quantile(0.80):.6f}"
        )
def build_feature_sets(
    features: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    print("Building feature sets...")
    result = {}
    for name, config in FEATURE_SETS.items():
        price_features = config[
            "price_features"
        ]
        prepared = prepare_feature_set(
            features,
            include_price_features=True,
            price_features=price_features,
        )
        result[name] = prepared
    print("=" * 40)
    print("FEATURE SET QC")
    print("=" * 40)
    for name, data in result.items():
        feature_columns = [
            column
            for column in data.columns
            if column in {
                "price_volatility_20d",
                "volatility_change_20d_60d",
            }
        ]
        print(
            f"{name}: "
            f"{len(feature_columns)} features"
        )
        print(
            "  Features: "
            f"{', '.join(feature_columns)}"
        )
    return result
def logistic_only_models(
    random_state: int,
):
    """
    Use only logistic regression so that both
    feature sets are evaluated with exactly the
    same model family.
    """
    from ml.models import build_models
    models = build_models(
        random_state=random_state,
        task="classification",
    )
    if "logistic_regression" not in models:
        raise ValueError(
            "logistic_regression not found "
            "in build_models()."
        )
    return {
        "logistic_regression": models[
            "logistic_regression"
        ],
    }
def evaluate_economic_oos(
    rows: list[dict[str, Any]],
    target_column: str,
) -> dict[str, Any]:
    if not rows:
        raise ValueError(
            "No OOS rows available."
        )
    data = pd.DataFrame(rows)
    if "prediction" not in data.columns:
        raise ValueError(
            "OOS result rows do not contain "
            "'prediction'."
        )
    if target_column not in data.columns:
        raise ValueError(
            "OOS result rows do not contain "
            f"target column '{target_column}'."
        )
    data = data.dropna(
        subset=[
            "prediction",
            target_column,
        ]
    ).copy()
    if data.empty:
        raise ValueError(
            "No valid OOS rows after filtering."
        )
    event_rate = data[
        target_column
    ].mean()
    if "return_5d" in data.columns:
        baseline_return = data[
            "return_5d"
        ].mean()
    elif "target_return" in data.columns:
        baseline_return = data[
            "target_return"
        ].mean()
    else:
        baseline_return = np.nan
    result = {
        "rows": len(data),
        "baseline_event_rate": event_rate,
        "baseline_mean_return": baseline_return,
        "fractions": {},
    }
    for fraction in ECONOMIC_TOP_FRACTIONS:
        n = max(
            1,
            int(round(len(data) * fraction)),
        )
        top = data.nlargest(
            n,
            "prediction",
        )
        top_event_rate = top[
            target_column
        ].mean()
        if "return_5d" in top.columns:
            mean_return = top[
                "return_5d"
            ].mean()
            median_return = top[
                "return_5d"
            ].median()
        elif "target_return" in top.columns:
            mean_return = top[
                "target_return"
            ].mean()
            median_return = top[
                "target_return"
            ].median()
        else:
            mean_return = np.nan
            median_return = np.nan
        result["fractions"][fraction] = {
            "n": n,
            "event_rate": top_event_rate,
            "lift": (
                top_event_rate / event_rate
                if event_rate
                else np.nan
            ),
            "mean_return": mean_return,
            "median_return": median_return,
        }
    return result
def print_economic_result(
    result: dict[str, Any],
) -> None:
    print(
        f"      Rows: {result['rows']:,}"
    )
    print(
        "      Baseline event rate: "
        f"{result['baseline_event_rate']:.4f}"
    )
    baseline_return = (
        result["baseline_mean_return"]
    )
    if np.isfinite(baseline_return):
        print(
            "      Baseline mean return: "
            f"{baseline_return:+.4%}"
        )
    for fraction, values in result[
        "fractions"
    ].items():
        print(
            f"      Top {fraction * 100:4.1f}%: "
            f"n={values['n']:,} "
            f"event={values['event_rate']:.4f} "
            f"lift={values['lift']:5.2f}x "
            f"mean={values['mean_return']:+.4%} "
            f"median={values['median_return']:+.4%}"
        )
def run_feature_set(
    name: str,
    data: pd.DataFrame,
    target,
) -> dict[str, Any]:
    print("=" * 32)
    print(f"RUNNING: {name}")
    print("=" * 32)
    started = time.perf_counter()
    from ml import walk_forward
    original_build_models = (
        walk_forward.build_models
    )
    def controlled_build_models(
        random_state: int,
        task: str = "classification",
    ):
        return logistic_only_models(
            random_state
        )
    walk_forward.build_models = (
        controlled_build_models
    )
    try:
        window_results = []
        for index, window in enumerate(
            WALK_FORWARD_WINDOWS,
            start=1,
        ):
            result = train_window(
                data=data,
                window=window,
                target=target,
                random_state=42,
            )
            window_results.append(result)
            print(f"  Window {index}")
            print(
                f"    Train <= "
                f"{window.train_end}"
            )
            print(
                f"    Validation <= "
                f"{window.validation_end}"
            )
            print(
                f"    Test <= "
                f"{window.test_end}"
            )
            print(
                "    Selected model: "
                f"{result['model']}"
            )
            print(
                "    Selected validation AUC: "
                f"{result['validation_score']:.6f}"
            )
            economic = evaluate_economic_oos(
                result["oos_rows"],
                target_column=target.name,
            )
            print("    Economic OOS:")
            print_economic_result(
                economic
            )
        elapsed = (
            time.perf_counter()
            - started
        )
    finally:
        walk_forward.build_models = (
            original_build_models
        )
    oos_rows = []
    for result in window_results:
        oos_rows.extend(
            result["oos_rows"]
        )
    combined_economic = (
        evaluate_economic_oos(
            oos_rows,
            target_column=target.name,
        )
    )
    average_auc = float(
        np.mean(
            [
                result[
                    "validation_score"
                ]
                for result in window_results
            ]
        )
    )
    total_fit_time = sum(
        result.get(
            "fit_seconds",
            0.0,
        )
        for result in window_results
    )
    return {
        "name": name,
        "feature_count": len(
            FEATURE_SETS[name][
                "price_features"
            ]
        ),
        "rows": len(data),
        "windows": len(window_results),
        "total_seconds": elapsed,
        "fit_seconds": total_fit_time,
        "average_auc": average_auc,
        "selected_models": [
            result["model"]
            for result in window_results
        ],
        "economic": combined_economic,
    }
def print_summary(
    results: list[dict[str, Any]],
) -> None:
    print("=" * 40)
    print("RESULTAT")
    print("=" * 40)
    for result in results:
        print(result["name"])
        print(
            f"  Features: "
            f"{result['feature_count']}"
        )
        print(
            f"  Rows: "
            f"{result['rows']:,}"
        )
        print(
            "  Walk-forward windows: "
            f"{result['windows']}"
        )
        print(
            f"  Total: "
            f"{result['total_seconds']:.2f}s"
        )
        print(
            f"  Fit: "
            f"{result['fit_seconds']:.2f}s"
        )
        print(
            "  Average validation AUC: "
            f"{result['average_auc']:.6f}"
        )
        print(
            "  Selected models: "
            + ", ".join(
                result["selected_models"]
            )
        )
        print("  Combined economic OOS:")
        print_economic_result(
            result["economic"]
        )
def print_comparison(
    results: list[dict[str, Any]],
) -> None:
    print("=" * 80)
    print("CONTROLLED LOGISTIC COMPARISON")
    print("=" * 80)
    print(
        "Feature set | AUC | top1 event | lift | "
        "top1 mean | total | fit"
    )
    print("-" * 80)
    for result in results:
        top1 = result[
            "economic"
        ]["fractions"][0.01]
        print(
            f"{result['name']:<32}"
            f"{result['average_auc']:.6f}   "
            f"{top1['event_rate']:.4f}   "
            f"{top1['lift']:.2f}x   "
            f"{top1['mean_return']:+.4%}   "
            f"{result['total_seconds']:.2f}s   "
            f"{result['fit_seconds']:.2f}s"
        )
    print("=" * 80)
    print("DELTA VS VOLATILITY_20D")
    print("=" * 80)
    baseline = results[0]
    baseline_auc = (
        baseline["average_auc"]
    )
    baseline_top1 = baseline[
        "economic"
    ]["fractions"][0.01]
    for result in results[1:]:
        top1 = result[
            "economic"
        ]["fractions"][0.01]
        print(result["name"])
        print(
            "  AUC delta: "
            f"{result['average_auc'] - baseline_auc:+.6f}"
        )
        print(
            "  Top1 event delta: "
            f"{top1['event_rate'] - baseline_top1['event_rate']:+.4f}"
        )
        print(
            "  Top1 mean return delta: "
            f"{top1['mean_return'] - baseline_top1['mean_return']:+.4%}"
        )
    print("=" * 80)
    print("QUESTION")
    print("=" * 80)
    print(
        "Är volatility_change_20d_60d en faktisk "
        "extra signal när modelltypen hålls konstant?"
    )
def main() -> None:
    print()
    print(
        "BLANKDISS VOLATILITY CHANGE "
        "LOGISTIC SCREENING"
    )
    print(
        f"RF trees: {BENCHMARK_TREES}"
    )
    print(
        f"RF n_jobs: {RF_N_JOBS}"
    )
    print(
        "Walk-forward windows: "
        f"{len(WALK_FORWARD_WINDOWS)}"
    )
    print(
        f"Economic target: "
        f"{ECONOMIC_TARGET}"
    )
    print(
        "Question: Ger förändringen mellan "
        "20d- och 60d-volatilitet extra information?"
    )
    print(
        "Loading feature data..."
    )
    features = load_feature_data()
    print(
        f"Feature rows: "
        f"{len(features):,}"
    )
    price_data = load_price_data()
    features = add_volatility_change(
        features,
        price_data,
    )
    print_feature_qc(
        features
    )
    feature_sets = build_feature_sets(
        features
    )
    target = find_target(
        ECONOMIC_TARGET
    )
    print(
        f"Target resolved: "
        f"{target.name}"
    )
    results = []
    for name, data in feature_sets.items():
        result = run_feature_set(
            name=name,
            data=data,
            target=target,
        )
        results.append(result)
    print_summary(results)
    print_comparison(results)
    print("=" * 40)
    print("BENCHMARK COMPLETE")
    print("=" * 40)
if __name__ == "__main__":
    main()
