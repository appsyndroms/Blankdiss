"""Kontrollerad jämförelse av FI-information ovanpå volatilitet."""
from __future__ import annotations
import time
from typing import Any
import numpy as np
import pandas as pd
import ml.walk_forward as walk_forward
from analysis.feature_config import PRICE_DIR
from analysis.feature_prices import (
    find_price_files,
    load_prices,
)
from ml.config import TARGETS, WALK_FORWARD_WINDOWS
from ml.dataset import build_target, load_features
from ml.models import build_models
from ml.walk_forward import train_window
ECONOMIC_TARGET = "down_5pct_5d"
ECONOMIC_TOP_FRACTIONS = (
    0.001,
    0.005,
    0.01,
    0.02,
    0.05,
)
FI_FEATURES = [
    "short_interest_pct",
    "active_holders",
    "max_individual_position_pct",
    "max_position_share_pct",
    "previous_short_interest_pct",
    "previous_active_holders",
    "previous_max_individual_position_pct",
    "previous_max_position_share_pct",
    "fi_observation_gap_days",
    "short_interest_delta_pp",
    "holder_delta",
    "max_position_delta_pp",
    "concentration_delta_pp",
    "short_interest_relative_change",
    "short_interest_acceleration_pp",
    "above_1_0pct",
    "entered_above_1_0pct",
    "exited_below_1_0pct",
    "above_2_0pct",
    "entered_above_2_0pct",
    "exited_below_2_0pct",
    "above_3_0pct",
    "entered_above_3_0pct",
    "exited_below_3_0pct",
    "above_5_0pct",
    "entered_above_5_0pct",
    "exited_below_5_0pct",
    "new_visible_observation",
]
FEATURE_SETS = {
    "volatility_60d": [
        "volatility_60d",
    ],
    "volatility_20d_plus_60d": [
        "price_volatility_20d",
        "volatility_60d",
    ],
    "fi_plus_volatility_60d": [
        *FI_FEATURES,
        "volatility_60d",
    ],
    "fi_plus_volatility_20d_plus_60d": [
        *FI_FEATURES,
        "price_volatility_20d",
        "volatility_60d",
    ],
}
def find_target(target_name: str):
    for target in TARGETS:
        if target.name == target_name:
            return target
    available = ", ".join(
        target.name
        for target in TARGETS
    )
    raise ValueError(
        f"Unknown target: {target_name}. "
        f"Available targets: {available}"
    )
def load_price_data() -> pd.DataFrame:
    print(
        "Loading raw price data for 60d volatility..."
    )
    price_data = load_prices(
        find_price_files(
            PRICE_DIR
        )
    )
    if price_data.empty:
        raise ValueError(
            "No raw price data found."
        )
    required = {
        "yahoo_symbol",
        "date",
        "close",
    }
    missing = (
        required
        - set(price_data.columns)
    )
    if missing:
        raise ValueError(
            "Raw price data is missing "
            "required columns: "
            f"{sorted(missing)}"
        )
    price_data = price_data.copy()
    price_data["date"] = pd.to_datetime(
        price_data["date"],
        errors="coerce",
    )
    price_data["close"] = pd.to_numeric(
        price_data["close"],
        errors="coerce",
    )
    price_data = price_data.dropna(
        subset=[
            "yahoo_symbol",
            "date",
            "close",
        ]
    )
    price_data = price_data.sort_values(
        [
            "yahoo_symbol",
            "date",
        ]
    ).reset_index(
        drop=True
    )
    print(
        f"Prisrader: {len(price_data):,}"
    )
    return price_data
def add_volatility_features(
    features: pd.DataFrame,
    price_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Beräknar 60d-volatilitet med samma princip
    som price_volatility_20d:
      - 60 close-observationer
      - pct_change()
      - standardavvikelse av de 59 dagliga
        avkastningarna
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
    ).reset_index(
        drop=True
    )
    prices["daily_return"] = (
        prices
        .groupby(
            "yahoo_symbol"
        )["close"]
        .pct_change()
    )
    prices["volatility_60d"] = (
        prices
        .groupby(
            "yahoo_symbol"
        )["daily_return"]
        .transform(
            lambda series: (
                series
                .rolling(
                    window=59,
                    min_periods=59,
                )
                .std()
            )
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
        result["price_date"],
        errors="coerce",
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
    return result
def print_feature_qc(
    features: pd.DataFrame,
) -> None:
    print("=" * 60)
    print(
        "FI + VOLATILITY COMBINATION QC"
    )
    print("=" * 60)
    print(
        f"Feature rows: "
        f"{len(features):,}"
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
    print(
        "FI features: "
        f"{len(FI_FEATURES)}"
    )
    missing_fi = [
        column
        for column in FI_FEATURES
        if column not in features.columns
    ]
    if missing_fi:
        raise ValueError(
            "Missing FI feature columns: "
            f"{missing_fi}"
        )
    print(
        "Volatility statistics:"
    )
    for column in (
        "price_volatility_20d",
        "volatility_60d",
    ):
        series = features[
            column
        ].dropna()
        print(
            f"  {column}: "
            f"median={series.median():.6f} "
            f"p20={series.quantile(0.20):.6f} "
            f"p80={series.quantile(0.80):.6f}"
        )
def build_ml_dataset(
    features: pd.DataFrame,
    feature_columns: list[str],
    target,
) -> tuple[
    pd.DataFrame,
    pd.Series,
]:
    required_columns = [
        "snapshot_date",
        "security_key",
        "forward_return_5d",
        *feature_columns,
    ]
    missing = [
        column
        for column in required_columns
        if column not in features.columns
    ]
    if missing:
        raise ValueError(
            "Missing required columns: "
            f"{missing}"
        )
    feature_data = features.dropna(
        subset=feature_columns
    ).copy()
    target_values = build_target(
        feature_data,
        target,
    )
    valid = target_values.notna()
    data = feature_data.loc[
        valid,
        required_columns,
    ].copy()
    y = target_values.loc[
        valid
    ].copy()
    data["target_return"] = pd.to_numeric(
        data["forward_return_5d"],
        errors="coerce",
    )
    data = data.drop(
        columns=[
            "forward_return_5d",
        ]
    )
    valid_returns = data[
        "target_return"
    ].notna()
    data = data.loc[
        valid_returns
    ].copy()
    y = y.loc[
        data.index
    ].copy()
    data = data.reset_index(
        drop=True
    )
    y = y.reset_index(
        drop=True
    )
    return data, y
def logistic_only_models(
    random_state: int,
    task: str = "classification",
):
    models = build_models(
        random_state=random_state,
        task=task,
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
) -> dict[str, Any]:
    if not rows:
        raise ValueError(
            "No OOS rows available."
        )
    data = pd.DataFrame(
        rows
    )
    required = [
        "prediction",
        "target_return",
    ]
    missing = [
        column
        for column in required
        if column not in data.columns
    ]
    if missing:
        raise ValueError(
            "OOS rows are missing columns: "
            f"{missing}"
        )
    data = data.dropna(
        subset=required
    ).copy()
    if data.empty:
        raise ValueError(
            "No valid OOS rows after filtering."
        )
    data["event"] = (
        data["target_return"] <= -0.05
    ).astype(int)
    baseline_event_rate = (
        data["event"].mean()
    )
    baseline_mean_return = (
        data["target_return"].mean()
    )
    result = {
        "rows": len(data),
        "baseline_event_rate": (
            baseline_event_rate
        ),
        "baseline_mean_return": (
            baseline_mean_return
        ),
        "fractions": {},
    }
    for fraction in (
        ECONOMIC_TOP_FRACTIONS
    ):
        n = max(
            1,
            int(
                round(
                    len(data)
                    * fraction
                )
            ),
        )
        top = data.nlargest(
            n,
            "prediction",
        )
        event_rate = (
            top["event"].mean()
        )
        result[
            "fractions"
        ][fraction] = {
            "n": n,
            "event_rate": event_rate,
            "lift": (
                event_rate
                / baseline_event_rate
                if baseline_event_rate
                else np.nan
            ),
            "mean_return": (
                top[
                    "target_return"
                ].mean()
            ),
            "median_return": (
                top[
                    "target_return"
                ].median()
            ),
        }
    return result
def print_economic_result(
    result: dict[str, Any],
) -> None:
    print(
        "      Rows: "
        f"{result['rows']:,}"
    )
    print(
        "      Baseline event rate: "
        f"{result['baseline_event_rate']:.4f}"
    )
    print(
        "      Baseline mean return: "
        f"{result['baseline_mean_return']:+.4%}"
    )
    for (
        fraction,
        values,
    ) in result[
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
    y: pd.Series,
    feature_columns: list[str],
) -> dict[str, Any]:
    print(
        f"Preparing feature set: {name}"
    )
    print(
        f"  Features: "
        f"{len(feature_columns)}"
    )
    print(
        "  Feature names: "
        + ", ".join(
            feature_columns
        )
    )
    print(
        f"  Rows: {len(data):,}"
    )
    print("=" * 60)
    print(
        f"RUNNING: {name}"
    )
    print("=" * 60)
    started = time.perf_counter()
    original_build_models = (
        walk_forward.build_models
    )
    def controlled_build_models(
        random_state: int,
        task: str = "classification",
    ):
        return logistic_only_models(
            random_state=random_state,
            task=task,
        )
    walk_forward.build_models = (
        controlled_build_models
    )
    try:
        window_results = []
        oos_rows = []
        total_fit_seconds = 0.0
        for index, window in enumerate(
            WALK_FORWARD_WINDOWS,
            start=1,
        ):
            (
                model_results,
                window_oos_rows,
                timing,
            ) = train_window(
                data=data,
                y=y,
                feature_columns=feature_columns,
                window=window,
                task="classification",
                direction="below",
            )
            if not model_results:
                raise ValueError(
                    f"No model result for "
                    f"{name}, window {index}."
                )
            selected = next(
                (
                    result
                    for result in model_results
                    if result[
                        "selected_for_oos"
                    ]
                ),
                None,
            )
            if selected is None:
                raise ValueError(
                    "No selected OOS model "
                    f"for {name}, "
                    f"window {index}."
                )
            window_results.append(
                selected
            )
            oos_rows.extend(
                window_oos_rows
            )
            total_fit_seconds += (
                timing[
                    "fit_seconds"
                ]
            )
            print(
                f"  Window {index}"
            )
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
                "    Model: "
                f"{selected['model']}"
            )
            print(
                "    Validation AUC: "
                f"{selected['validation_score']:.6f}"
            )
            print(
                "    Fit: "
                f"{timing['fit_seconds']:.2f}s"
            )
            economic = (
                evaluate_economic_oos(
                    window_oos_rows
                )
            )
            print(
                "    Economic OOS:"
            )
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
    combined_economic = (
        evaluate_economic_oos(
            oos_rows
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
    return {
        "name": name,
        "feature_count": len(
            feature_columns
        ),
        "features": feature_columns,
        "rows": len(data),
        "windows": len(
            window_results
        ),
        "total_seconds": elapsed,
        "fit_seconds": (
            total_fit_seconds
        ),
        "average_auc": average_auc,
        "selected_models": [
            result["model"]
            for result in window_results
        ],
        "economic": combined_economic,
    }
def print_result(
    result: dict[str, Any],
) -> None:
    print(
        result["name"]
    )
    print(
        "  Features: "
        f"{result['feature_count']}"
    )
    print(
        "  Feature names: "
        + ", ".join(
            result["features"]
        )
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
        "  Total: "
        f"{result['total_seconds']:.2f}s"
    )
    print(
        "  Fit: "
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
    economic = result[
        "economic"
    ]
    print(
        "  Combined economic OOS:"
    )
    print(
        f"      Rows: "
        f"{economic['rows']:,}"
    )
    print(
        "      Baseline event rate: "
        f"{economic['baseline_event_rate']:.4f}"
    )
    print(
        "      Baseline mean return: "
        f"{economic['baseline_mean_return']:+.4%}"
    )
    for fraction in (
        ECONOMIC_TOP_FRACTIONS
    ):
        stats = economic[
            "fractions"
        ][fraction]
        print(
            f"      Top {fraction * 100:4.1f}%: "
            f"n={stats['n']:,} "
            f"event={stats['event_rate']:.4f} "
            f"lift={stats['lift']:5.2f}x "
            f"mean={stats['mean_return']:+.4%} "
            f"median={stats['median_return']:+.4%}"
        )
def print_comparison(
    results: list[dict[str, Any]],
) -> None:
    print("=" * 110)
    print(
        "CONTROLLED FI + VOLATILITY LOGISTIC COMPARISON"
    )
    print("=" * 110)
    print(
        "Feature set | AUC | top1 event | "
        "lift | top1 mean | total | fit"
    )
    print("-" * 110)
    for result in results:
        top1 = result[
            "economic"
        ][
            "fractions"
        ][0.01]
        print(
            f"{result['name']:<40}"
            f"{result['average_auc']:.6f}   "
            f"{top1['event_rate']:.4f}   "
            f"{top1['lift']:.2f}x   "
            f"{top1['mean_return']:+.4%}   "
            f"{result['total_seconds']:.2f}s   "
            f"{result['fit_seconds']:.2f}s"
        )
def print_delta(
    base: dict[str, Any],
    other: dict[str, Any],
    label: str,
) -> None:
    base_top1 = base[
        "economic"
    ][
        "fractions"
    ][0.01]
    other_top1 = other[
        "economic"
    ][
        "fractions"
    ][0.01]
    print(label)
    print(
        "  AUC delta: "
        f"{other['average_auc'] - base['average_auc']:+.6f}"
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
        f"{other['total_seconds'] - base['total_seconds']:+.2f}s"
    )
    print(
        "  Fit time delta: "
        f"{other['fit_seconds'] - base['fit_seconds']:+.2f}s"
    )
def main() -> None:
    print(
        "BLANKDISS FI + VOLATILITY "
        "COMBINATION LOGISTIC SCREENING"
    )
    print(
        "Model: logistic_regression only"
    )
    print(
        "Walk-forward windows: "
        f"{len(WALK_FORWARD_WINDOWS)}"
    )
    print(
        "Economic target: "
        f"{ECONOMIC_TARGET}"
    )
    print(
        "Question: "
        "Tillför FI-information prediktiv signal "
        "ovanpå volatilitet?"
    )
    print(
        "FI feature count: "
        f"{len(FI_FEATURES)}"
    )
    print(
        "Loading feature data..."
    )
    features = load_features()
    print(
        f"Feature-rader: "
        f"{len(features):,}"
    )
    print(
        f"Feature rows: "
        f"{len(features):,}"
    )
    print(
        "Loading raw price data..."
    )
    price_data = load_price_data()
    print(
        "Building volatility horizon features..."
    )
    features = add_volatility_features(
        features,
        price_data,
    )
    print_feature_qc(
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
    for (
        name,
        feature_columns,
    ) in FEATURE_SETS.items():
        data, y = build_ml_dataset(
            features=features,
            feature_columns=feature_columns,
            target=target,
        )
        result = run_feature_set(
            name=name,
            data=data,
            y=y,
            feature_columns=feature_columns,
        )
        results.append(
            result
        )
    print("=" * 110)
    print(
        "RESULTAT"
    )
    print("=" * 110)
    for result in results:
        print_result(
            result
        )
    print_comparison(
        results
    )
    by_name = {
        result["name"]: result
        for result in results
    }
    print("=" * 110)
    print(
        "DELTA: FI OVANPÅ VOL60"
    )
    print("=" * 110)
    print_delta(
        by_name[
            "volatility_60d"
        ],
        by_name[
            "fi_plus_volatility_60d"
        ],
        "VOL60 -> FI + VOL60",
    )
    print("=" * 110)
    print(
        "DELTA: FI OVANPÅ VOL20 + VOL60"
    )
    print("=" * 110)
    print_delta(
        by_name[
            "volatility_20d_plus_60d"
        ],
        by_name[
            "fi_plus_volatility_20d_plus_60d"
        ],
        "VOL20 + VOL60 -> FI + VOL20 + VOL60",
    )
    print("=" * 110)
    print(
        "DELTA: FI + VOL60 VS FI + VOL20 + VOL60"
    )
    print("=" * 110)
    print_delta(
        by_name[
            "fi_plus_volatility_60d"
        ],
        by_name[
            "fi_plus_volatility_20d_plus_60d"
        ],
        "FI + VOL60 -> FI + VOL20 + VOL60",
    )
if __name__ == "__main__":
    main()
