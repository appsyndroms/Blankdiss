"""Jämför absolut och relativ volatilitet över 20 och 60 dagar.
Syfte:
- testa om price_volatility_20d bär information som är specifik
  för kortare volatilitet
- testa om 20d-volatilitet relativt 60d-volatilitet är bättre
- testa om förändringen 20d - 60d är bättre
- jämföra validation och faktisk OOS-ekonomi
- hålla beräkningskostnaden låg
Testade feature sets:
- volatility_20d
- volatility_relative_20d_60d
- volatility_change_20d_60d
- volatility_20d_plus_relative
- volatility_20d_plus_change
- volatility_20d_relative_plus_change
60d-volatiliteten beräknas från samma råprisdefinition som den
befintliga price_volatility_20d:
    std(pct_change())
20d:
    20 dagliga avkastningar
60d:
    60 dagliga avkastningar
Ingen produktionsfil ändras.
"""
from __future__ import annotations
from time import perf_counter
import numpy as np
import pandas as pd
import ml.walk_forward as walk_forward
from analysis.feature_prices import (
    find_price_files,
    load_prices,
)
from analysis.feature_config import PRICE_DIR
from ml.config import (
    TARGETS,
    WALK_FORWARD_WINDOWS,
)
from ml.dataset import (
    load_features,
    prepare_ml_data_from_feature_set,
    prepare_feature_set,
)
from ml.models import build_models as original_build_models
BENCHMARK_TREES = 100
ECONOMIC_TARGET = "down_5pct_5d"
ECONOMIC_TOP_FRACTIONS = (
    0.001,
    0.005,
    0.01,
    0.02,
    0.05,
)
FEATURE_SET_CONFIG = (
    (
        "volatility_20d",
        (
            "price_volatility_20d",
        ),
    ),
    (
        "volatility_relative_20d_60d",
        (
            "volatility_relative_20d_60d",
        ),
    ),
    (
        "volatility_change_20d_60d",
        (
            "volatility_change_20d_60d",
        ),
    ),
    (
        "volatility_20d_plus_relative",
        (
            "price_volatility_20d",
            "volatility_relative_20d_60d",
        ),
    ),
    (
        "volatility_20d_plus_change",
        (
            "price_volatility_20d",
            "volatility_change_20d_60d",
        ),
    ),
    (
        "volatility_20d_relative_plus_change",
        (
            "price_volatility_20d",
            "volatility_relative_20d_60d",
            "volatility_change_20d_60d",
        ),
    ),
)
def build_benchmark_models(
    random_state: int,
    task: str = "classification",
):
    """Bygger benchmarkmodeller med 100 RF-träd."""
    models = original_build_models(
        random_state,
        task=task,
    )
    random_forest = models.get(
        "random_forest"
    )
    if random_forest is not None:
        random_forest.set_params(
            model__n_estimators=BENCHMARK_TREES,
            model__n_jobs=1,
        )
    random_forest_regression = models.get(
        "random_forest_regression"
    )
    if random_forest_regression is not None:
        random_forest_regression.set_params(
            model__n_estimators=BENCHMARK_TREES,
            model__n_jobs=1,
        )
    return models
def find_target(
    target_name: str,
):
    """Hämtar TargetConfig från TARGETS-tupeln."""
    for target in TARGETS:
        if target.name == target_name:
            return target
    raise ValueError(
        f"Okänd economic target: {target_name}"
    )
def add_volatility_horizon_features(
    features: pd.DataFrame,
) -> pd.DataFrame:
    """Lägger till 60d-volatilitet och relativa volatilitetmått.
    60d-volatiliteten beräknas direkt från råprisdata.
    Matchningen sker på:
        yahoo_symbol + price_date
    Det innebär att varje feature-rad får volatilitet baserad
    enbart på prisdata fram till och med signal-dagen.
    """
    result = features.copy()
    required_feature_columns = {
        "yahoo_symbol",
        "price_date",
        "price_volatility_20d",
    }
    missing = sorted(
        required_feature_columns
        - set(result.columns)
    )
    if missing:
        raise ValueError(
            "Feature-data saknar kolumner för "
            "volatilitetshorisont: "
            + ", ".join(missing)
        )
    print()
    print(
        "Loading raw price data for 60d volatility..."
    )
    price_files = find_price_files(
        PRICE_DIR
    )
    print(
        f"Prisfiler: {len(price_files)}"
    )
    prices = load_prices(
        price_files
    )
    print(
        f"Prisrader: {len(prices):,}"
    )
    prices = prices[
        [
            "yahoo_symbol",
            "date",
            "close",
        ]
    ].copy()
    prices["date"] = pd.to_datetime(
        prices["date"],
        errors="coerce",
    )
    prices["close"] = pd.to_numeric(
        prices["close"],
        errors="coerce",
    )
    prices = prices.loc[
        prices["date"].notna()
        & prices["close"].notna()
        & np.isfinite(prices["close"])
        & (prices["close"] > 0)
    ].copy()
    prices = prices.sort_values(
        [
            "yahoo_symbol",
            "date",
        ],
        kind="mergesort",
    )
    # Samma definition som price_volatility_20d:
    #
    #   pd.Series(history).pct_change().std()
    #
    # För 60d använder vi därför 60 dagliga returns.
    prices["daily_return"] = (
        prices.groupby(
            "yahoo_symbol",
            sort=False,
        )["close"]
        .pct_change()
    )
    prices["volatility_60d"] = (
        prices.groupby(
            "yahoo_symbol",
            sort=False,
        )["daily_return"]
        .transform(
            lambda values: (
                values
                .rolling(
                    window=60,
                    min_periods=60,
                )
                .std()
            )
        )
    )
    volatility_lookup = prices[
        [
            "yahoo_symbol",
            "date",
            "volatility_60d",
        ]
    ].rename(
        columns={
            "date": "price_date",
        }
    )
    result["price_date"] = pd.to_datetime(
        result["price_date"],
        errors="coerce",
    )
    result["yahoo_symbol"] = (
        result["yahoo_symbol"]
        .fillna("")
        .astype(str)
        .str.strip()
    )
    volatility_lookup["price_date"] = (
        pd.to_datetime(
            volatility_lookup["price_date"],
            errors="coerce",
        )
    )
    volatility_lookup["yahoo_symbol"] = (
        volatility_lookup["yahoo_symbol"]
        .fillna("")
        .astype(str)
        .str.strip()
    )
    result = result.merge(
        volatility_lookup,
        on=[
            "yahoo_symbol",
            "price_date",
        ],
        how="left",
        sort=False,
        validate="many_to_one",
    )
    result["volatility_relative_20d_60d"] = (
        result["price_volatility_20d"]
        / result["volatility_60d"]
    )
    result["volatility_change_20d_60d"] = (
        result["price_volatility_20d"]
        - result["volatility_60d"]
    )
    # Skydda ML-experimenten från division med noll,
    # oändliga värden och saknade 60d-historiker.
    derived_columns = [
        "volatility_60d",
        "volatility_relative_20d_60d",
        "volatility_change_20d_60d",
    ]
    for column in derived_columns:
        result[column] = result[column].replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
    matched_60d = result[
        "volatility_60d"
    ].notna()
    print()
    print(
        "VOLATILITY HORIZON QC"
    )
    print(
        f"Feature rows: {len(result):,}"
    )
    print(
        f"Rows with 60d volatility: "
        f"{matched_60d.sum():,}"
    )
    print(
        f"Rows without 60d volatility: "
        f"{(~matched_60d).sum():,}"
    )
    if matched_60d.any():
        print()
        print(
            "Volatility statistics:"
        )
        for column in (
            "price_volatility_20d",
            "volatility_60d",
            "volatility_relative_20d_60d",
            "volatility_change_20d_60d",
        ):
            values = pd.to_numeric(
                result.loc[
                    matched_60d,
                    column,
                ],
                errors="coerce",
            )
            values = values[
                np.isfinite(values)
            ]
            if values.empty:
                continue
            print(
                f"  {column}: "
                f"median={values.median():.6f} "
                f"p20={values.quantile(.20):.6f} "
                f"p80={values.quantile(.80):.6f}"
            )
    return result
def build_feature_sets(
    features: pd.DataFrame,
):
    """Förbereder exakt de sex feature sets som ska jämföras."""
    price_features = {
        "price_volatility_20d",
    }
    prepared_data, available_price_columns = (
        prepare_feature_set(
            features,
            include_price_features=True,
            price_features=price_features,
        )
    )
    required_columns = {
        feature
        for _, feature_columns
        in FEATURE_SET_CONFIG
        for feature in feature_columns
    }
    missing = sorted(
        required_columns
        - set(prepared_data.columns)
        - set(available_price_columns)
    )
    if missing:
        raise ValueError(
            "Följande volatilitetfeatures saknas: "
            + ", ".join(missing)
        )
    feature_sets = {}
    for (
        feature_set_name,
        feature_columns,
    ) in FEATURE_SET_CONFIG:
        feature_sets[
            feature_set_name
        ] = (
            prepared_data.copy(),
            list(feature_columns),
        )
    return feature_sets
def print_feature_set_qc(
    feature_sets,
):
    print()
    print(
        "================================"
    )
    print(
        "FEATURE SET QC"
    )
    print(
        "================================"
    )
    for (
        feature_set_name,
        _,
    ) in FEATURE_SET_CONFIG:
        _, feature_columns = feature_sets[
            feature_set_name
        ]
        print(
            f"{feature_set_name}: "
            f"{len(feature_columns)} features"
        )
        print(
            "  Features: "
            + ", ".join(
                feature_columns
            )
        )
def calculate_economic_metrics(
    oos_predictions,
):
    if not oos_predictions:
        return {
            "rows": 0,
            "baseline_event_rate": None,
            "baseline_mean_return": None,
            "top_fraction": [],
        }
    scores = np.asarray(
        [
            row["score"]
            for row in oos_predictions
        ],
        dtype=float,
    )
    returns = np.asarray(
        [
            row["target_return"]
            for row in oos_predictions
        ],
        dtype=float,
    )
    events = (
        returns <= -0.05
    ).astype(float)
    valid = (
        np.isfinite(scores)
        & np.isfinite(returns)
    )
    scores = scores[valid]
    returns = returns[valid]
    events = events[valid]
    rows = len(scores)
    if rows == 0:
        return {
            "rows": 0,
            "baseline_event_rate": None,
            "baseline_mean_return": None,
            "top_fraction": [],
        }
    order = np.argsort(
        -scores,
        kind="mergesort",
    )
    baseline_event_rate = float(
        events.mean()
    )
    baseline_mean_return = float(
        returns.mean()
    )
    top_fraction = []
    for fraction in ECONOMIC_TOP_FRACTIONS:
        count = max(
            1,
            int(
                np.ceil(
                    rows * fraction
                )
            ),
        )
        selected = order[:count]
        selected_returns = (
            returns[selected]
        )
        selected_events = (
            events[selected]
        )
        event_rate = float(
            selected_events.mean()
        )
        mean_return = float(
            selected_returns.mean()
        )
        median_return = float(
            np.median(
                selected_returns
            )
        )
        lift = (
            event_rate
            / baseline_event_rate
            if baseline_event_rate > 0
            else None
        )
        top_fraction.append(
            {
                "fraction": fraction,
                "rows": count,
                "event_rate": event_rate,
                "lift": lift,
                "mean_return": mean_return,
                "median_return": median_return,
            }
        )
    return {
        "rows": rows,
        "baseline_event_rate": baseline_event_rate,
        "baseline_mean_return": baseline_mean_return,
        "top_fraction": top_fraction,
    }
def print_economic_results(
    oos_predictions,
    indent="  ",
):
    metrics = calculate_economic_metrics(
        oos_predictions
    )
    print(
        f"{indent}Rows: "
        f"{metrics['rows']:,}"
    )
    if metrics[
        "baseline_event_rate"
    ] is not None:
        print(
            f"{indent}Baseline event rate: "
            f"{metrics['baseline_event_rate']:.4f}"
        )
    if metrics[
        "baseline_mean_return"
    ] is not None:
        print(
            f"{indent}Baseline mean return: "
            f"{metrics['baseline_mean_return']:+.4%}"
        )
    for item in metrics[
        "top_fraction"
    ]:
        fraction = item[
            "fraction"
        ]
        if fraction < 0.01:
            fraction_text = (
                f"{fraction:.1%}"
            )
        else:
            fraction_text = (
                f"{fraction:.0%}"
            )
        lift = item["lift"]
        lift_text = (
            f"{lift:.2f}x"
            if lift is not None
            else "n/a"
        )
        print(
            f"{indent}Top "
            f"{fraction_text:>5}: "
            f"n={item['rows']:,} "
            f"event={item['event_rate']:.4f} "
            f"lift={lift_text:>6} "
            f"mean={item['mean_return']:+.4%} "
            f"median={item['median_return']:+.4%}"
        )
def get_selected_validation_score(
    results,
):
    selected = [
        result
        for result in results
        if result.get(
            "selected_for_oos"
        )
    ]
    if not selected:
        return None, None
    selected_result = selected[0]
    return (
        float(
            selected_result[
                "validation_score"
            ]
        ),
        selected_result[
            "model"
        ],
    )
def run_one_experiment(
    feature_set_name,
    feature_set_data,
    feature_columns,
    target,
):
    experiment_start = perf_counter()
    data, y, feature_columns = (
        prepare_ml_data_from_feature_set(
            feature_set_data,
            feature_columns,
            target,
        )
    )
    validation_scores = []
    economic_oos = []
    window_results = []
    for (
        window_index,
        window,
    ) in enumerate(
        WALK_FORWARD_WINDOWS,
        start=1,
    ):
        (
            results,
            oos_predictions,
            timing,
        ) = walk_forward.train_window(
            data,
            y,
            feature_columns,
            window,
            task=target.task,
            direction=target.direction,
        )
        (
            validation_score,
            selected_model,
        ) = get_selected_validation_score(
            results
        )
        if validation_score is not None:
            validation_scores.append(
                validation_score
            )
        economic_oos.extend(
            oos_predictions
        )
        window_results.append(
            {
                "window_index": (
                    window_index
                ),
                "train_end": (
                    window.train_end
                ),
                "validation_end": (
                    window.validation_end
                ),
                "test_end": (
                    window.test_end
                ),
                "validation_score": (
                    validation_score
                ),
                "selected_model": (
                    selected_model
                ),
                "economic_oos": list(
                    oos_predictions
                ),
                "timing": dict(
                    timing
                ),
            }
        )
    total_seconds = (
        perf_counter()
        - experiment_start
    )
    return {
        "feature_set": (
            feature_set_name
        ),
        "feature_count": len(
            feature_columns
        ),
        "row_count": len(data),
        "validation_score": (
            sum(validation_scores)
            / len(validation_scores)
            if validation_scores
            else None
        ),
        "total_seconds": (
            total_seconds
        ),
        "fit_seconds": sum(
            result["timing"].get(
                "fit_seconds",
                0.0,
            )
            for result in window_results
        ),
        "economic_oos": (
            economic_oos
        ),
        "window_results": (
            window_results
        ),
    }
def print_window_results(
    results,
):
    print()
    print(
        "================================"
    )
    print(
        "RESULTAT PER WALK-FORWARD-FÖNSTER"
    )
    print(
        "================================"
    )
    for result in results:
        print()
        print(
            result["feature_set"]
        )
        for window in result[
            "window_results"
        ]:
            print()
            print(
                f"  Window "
                f"{window['window_index']}"
            )
            print(
                f"    Train <= "
                f"{window['train_end']}"
            )
            print(
                f"    Validation <= "
                f"{window['validation_end']}"
            )
            print(
                f"    Test <= "
                f"{window['test_end']}"
            )
            if window[
                "validation_score"
            ] is not None:
                print(
                    f"    Selected model: "
                    f"{window['selected_model']}"
                )
                print(
                    f"    Selected validation AUC: "
                    f"{window['validation_score']:.6f}"
                )
            else:
                print(
                    "    Selected model: n/a"
                )
            print(
                f"    Time: "
                f"{window['timing'].get('total_window_seconds', 0.0):.2f}s"
            )
            print(
                "    Economic OOS:"
            )
            print_economic_results(
                window[
                    "economic_oos"
                ],
                indent="      ",
            )
def print_combined_results(
    results,
):
    print()
    print(
        "================================"
    )
    print(
        "RESULTAT PER FEATURE SET"
    )
    print(
        "================================"
    )
    for result in results:
        print()
        print(
            result["feature_set"]
        )
        print(
            f"  Features: "
            f"{result['feature_count']}"
        )
        print(
            f"  Rows: "
            f"{result['row_count']:,}"
        )
        print(
            f"  Walk-forward windows: "
            f"{len(result['window_results'])}"
        )
        print(
            f"  Total: "
            f"{result['total_seconds']:.2f}s"
        )
        print(
            f"  Fit: "
            f"{result['fit_seconds']:.2f}s"
        )
        if result[
            "validation_score"
        ] is not None:
            print(
                f"  Average selected-model "
                f"validation AUC: "
                f"{result['validation_score']:.6f}"
            )
        else:
            print(
                "  Average selected-model "
                "validation AUC: n/a"
            )
        print(
            "  Selected models:"
        )
        for window in result[
            "window_results"
        ]:
            print(
                f"    Window "
                f"{window['window_index']}: "
                f"{window['selected_model']}"
            )
        print(
            "  Combined economic OOS:"
        )
        print_economic_results(
            result[
                "economic_oos"
            ],
            indent="    ",
        )
def get_top1(
    result,
):
    metrics = calculate_economic_metrics(
        result["economic_oos"]
    )
    return next(
        (
            item
            for item in metrics[
                "top_fraction"
            ]
            if item["fraction"] == 0.01
        ),
        None,
    )
def print_comparison(
    results,
):
    print()
    print(
        "================================"
    )
    print(
        "VOLATILITY HORIZON COMPARISON"
    )
    print(
        "================================"
    )
    print(
        "Feature set | val AUC | "
        "top1 event | lift | "
        "top1 mean | total | fit"
    )
    print(
        "-" * 115
    )
    for result in results:
        top1 = get_top1(
            result
        )
        if top1 is None:
            continue
        validation_score = (
            result[
                "validation_score"
            ]
            if result[
                "validation_score"
            ] is not None
            else float("nan")
        )
        print(
            f"{result['feature_set']:<38}"
            f"{validation_score:.6f}     "
            f"{top1['event_rate']:.4f}  "
            f"{top1['lift']:.2f}x   "
            f"{top1['mean_return']:+.4%}   "
            f"{result['total_seconds']:.2f}s  "
            f"{result['fit_seconds']:.2f}s"
        )
def print_deltas(
    results,
):
    """Visar skillnader mot ren 20d-volatilitet."""
    if not results:
        return
    baseline = next(
        (
            result
            for result in results
            if result["feature_set"]
            == "volatility_20d"
        ),
        None,
    )
    if baseline is None:
        return
    baseline_top1 = get_top1(
        baseline
    )
    if baseline_top1 is None:
        return
    print()
    print(
        "================================"
    )
    print(
        "DELTA VS VOLATILITY_20D"
    )
    print(
        "================================"
    )
    print(
        "Feature set | "
        "AUC delta | "
        "top1 event delta | "
        "top1 mean delta"
    )
    print(
        "-" * 95
    )
    for result in results:
        if result is baseline:
            continue
        top1 = get_top1(
            result
        )
        if top1 is None:
            continue
        auc_delta = (
            result[
                "validation_score"
            ]
            - baseline[
                "validation_score"
            ]
        )
        event_delta = (
            top1[
                "event_rate"
            ]
            - baseline_top1[
                "event_rate"
            ]
        )
        mean_delta = (
            top1[
                "mean_return"
            ]
            - baseline_top1[
                "mean_return"
            ]
        )
        print(
            f"{result['feature_set']:<38}"
            f"{auc_delta:+.6f}       "
            f"{event_delta:+.4f}          "
            f"{mean_delta:+.4%}"
        )
def main():
    print()
    print(
        "BLANKDISS VOLATILITY HORIZON SCREENING"
    )
    print(
        f"RF trees: "
        f"{BENCHMARK_TREES}"
    )
    print(
        "RF n_jobs: 1"
    )
    print(
        f"Walk-forward windows: "
        f"{len(WALK_FORWARD_WINDOWS)}"
    )
    print(
        f"Economic target: "
        f"{ECONOMIC_TARGET}"
    )
    print(
        "Question: Är 20d-volatilitet "
        "specifik, eller räcker relativ/förändrad "
        "volatilitet mot 60d?"
    )
    print()
    print(
        "Loading feature data..."
    )
    features = load_features()
    print(
        f"Feature rows: "
        f"{len(features):,}"
    )
    features = add_volatility_horizon_features(
        features
    )
    print()
    print(
        "Building feature sets..."
    )
    feature_sets = build_feature_sets(
        features
    )
    print_feature_set_qc(
        feature_sets
    )
    target = find_target(
        ECONOMIC_TARGET
    )
    print()
    print(
        f"Target resolved: "
        f"{target.name}"
    )
    # Begränsa benchmark-RF till 100 träd,
    # men behåll samma modellselektion som övriga
    # diagnostikexperiment.
    original_build_models_reference = (
        walk_forward.build_models
    )
    walk_forward.build_models = (
        build_benchmark_models
    )
    try:
        results = []
        for (
            feature_set_name,
            _,
        ) in FEATURE_SET_CONFIG:
            print()
            print(
                "================================"
            )
            print(
                f"RUNNING: "
                f"{feature_set_name}"
            )
            print(
                "================================"
            )
            (
                feature_data,
                feature_columns,
            ) = feature_sets[
                feature_set_name
            ]
            result = run_one_experiment(
                feature_set_name,
                feature_data,
                feature_columns,
                target,
            )
            results.append(
                result
            )
            print()
            print(
                f"Completed "
                f"{feature_set_name} "
                f"in "
                f"{result['total_seconds']:.2f}s"
            )
    finally:
        walk_forward.build_models = (
            original_build_models_reference
        )
    print_window_results(
        results
    )
    print_combined_results(
        results
    )
    print_comparison(
        results
    )
    print_deltas(
        results
    )
    total_runtime = sum(
        result[
            "total_seconds"
        ]
        for result in results
    )
    print()
    print(
        "================================"
    )
    print(
        "BENCHMARK COMPLETE"
    )
    print(
        "================================"
    )
    print(
        f"Total experiment time: "
        f"{total_runtime:.2f}s"
    )
if __name__ == "__main__":
    main()
