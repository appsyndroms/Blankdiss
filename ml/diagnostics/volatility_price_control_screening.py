"""Kontrollerar om volatility_20d är en proxy för annan prisinformation.

Syfte:
- avgöra om volatility_20d bär egen prediktiv information
- kontrollera om signalen försvinner när historiska prisrörelser läggs till
- kontrollera om signalen försvinner när avstånd från tidigare toppar läggs till
- jämföra validation och faktisk OOS-ekonomi
- hålla beräkningskostnaden låg

Testade feature sets:
- volatility_only
- volatility_plus_return_5d
- volatility_plus_return_20d
- volatility_plus_return_60d
- volatility_plus_all_returns
- volatility_plus_all_other_price

Ingen produktionsfil ändras.
"""

from __future__ import annotations

from time import perf_counter

import numpy as np

import ml.walk_forward as walk_forward
from ml.config import TARGETS, WALK_FORWARD_WINDOWS
from ml.dataset import (
    load_features,
    prepare_feature_set,
    prepare_ml_data_from_feature_set,
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

BASE_PRICE_FEATURE = "price_volatility_20d"

FEATURE_SET_CONFIG = (
    (
        "volatility_only",
        (
            "price_volatility_20d",
        ),
    ),
    (
        "volatility_plus_return_5d",
        (
            "price_volatility_20d",
            "price_return_5d",
        ),
    ),
    (
        "volatility_plus_return_20d",
        (
            "price_volatility_20d",
            "price_return_20d",
        ),
    ),
    (
        "volatility_plus_return_60d",
        (
            "price_volatility_20d",
            "price_return_60d",
        ),
    ),
    (
        "volatility_plus_all_returns",
        (
            "price_volatility_20d",
            "price_return_5d",
            "price_return_20d",
            "price_return_60d",
        ),
    ),
    (
        "volatility_plus_all_other_price",
        (
            "price_volatility_20d",
            "price_return_5d",
            "price_return_20d",
            "price_return_60d",
            "price_distance_from_20d_high",
            "price_distance_from_60d_high",
        ),
    ),
)


def build_benchmark_models(
    random_state: int,
    task: str = "classification",
):
    models = original_build_models(
        random_state,
        task=task,
    )

    random_forest = models.get("random_forest")

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


def build_feature_sets(features):
    price_data, available_price_columns = (
        prepare_feature_set(
            features,
            include_price_features=True,
            price_features={
                "price_volatility_20d",
                "price_return_5d",
                "price_return_20d",
                "price_return_60d",
                "price_distance_from_20d_high",
                "price_distance_from_60d_high",
            },
        )
    )

    required_columns = {
        feature
        for _, feature_columns in FEATURE_SET_CONFIG
        for feature in feature_columns
    }

    missing = sorted(
        required_columns
        - set(available_price_columns)
    )

    if missing:
        raise ValueError(
            "Följande prisfeatures saknas efter "
            "feature preparation: "
            + ", ".join(missing)
        )

    feature_sets = {}

    for feature_set_name, feature_columns in (
        FEATURE_SET_CONFIG
    ):
        feature_sets[feature_set_name] = (
            price_data.copy(),
            list(feature_columns),
        )

    return feature_sets


def print_feature_set_qc(feature_sets):
    print()
    print("================================")
    print("FEATURE SET QC")
    print("================================")

    for feature_set_name, _ in FEATURE_SET_CONFIG:
        _, feature_columns = feature_sets[
            feature_set_name
        ]

        print(
            f"{feature_set_name}: "
            f"{len(feature_columns)} features"
        )

        print(
            "  Features: "
            + ", ".join(feature_columns)
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

        selected_returns = returns[selected]
        selected_events = events[selected]

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
            event_rate / baseline_event_rate
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

    if metrics["baseline_event_rate"] is not None:
        print(
            f"{indent}Baseline event rate: "
            f"{metrics['baseline_event_rate']:.4f}"
        )

    if metrics["baseline_mean_return"] is not None:
        print(
            f"{indent}Baseline mean return: "
            f"{metrics['baseline_mean_return']:+.4%}"
        )

    for result in metrics["top_fraction"]:
        fraction = result["fraction"]

        if fraction < 0.01:
            fraction_text = f"{fraction:.1%}"
        else:
            fraction_text = f"{fraction:.0%}"

        lift = result["lift"]

        lift_text = (
            f"{lift:.2f}x"
            if lift is not None
            else "n/a"
        )

        print(
            f"{indent}Top {fraction_text:>5}: "
            f"n={result['rows']:,} "
            f"event={result['event_rate']:.4f} "
            f"lift={lift_text:>6} "
            f"mean={result['mean_return']:+.4%} "
            f"median={result['median_return']:+.4%}"
        )


def get_selected_validation_score(results):
    selected = [
        result
        for result in results
        if result.get("selected_for_oos")
    ]

    if not selected:
        return None, None

    selected_result = selected[0]

    return (
        float(
            selected_result["validation_score"]
        ),
        selected_result["model"],
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

    for window_index, window in enumerate(
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

        validation_score, selected_model = (
            get_selected_validation_score(
                results
            )
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
                "window_index": window_index,
                "train_end": window.train_end,
                "validation_end": (
                    window.validation_end
                ),
                "test_end": window.test_end,
                "validation_score": (
                    validation_score
                ),
                "selected_model": (
                    selected_model
                ),
                "economic_oos": list(
                    oos_predictions
                ),
                "timing": dict(timing),
            }
        )

    total_seconds = (
        perf_counter()
        - experiment_start
    )

    return {
        "feature_set": feature_set_name,
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
        "total_seconds": total_seconds,
        "fit_seconds": sum(
            result["timing"].get(
                "fit_seconds",
                0.0,
            )
            for result in window_results
        ),
        "economic_oos": economic_oos,
        "window_results": window_results,
    }


def print_window_results(results):
    print()
    print("================================")
    print("RESULTAT PER WALK-FORWARD-FÖNSTER")
    print("================================")

    for result in results:
        print()
        print(result["feature_set"])

        for window in result["window_results"]:
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

            if window["validation_score"] is not None:
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

            print("    Economic OOS:")

            print_economic_results(
                window["economic_oos"],
                indent="      ",
            )


def print_combined_results(results):
    print()
    print("================================")
    print("RESULTAT PER FEATURE SET")
    print("================================")

    for result in results:
        print()
        print(result["feature_set"])

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

        if result["validation_score"] is not None:
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

        print("  Selected models:")

        for window in result["window_results"]:
            print(
                f"    Window "
                f"{window['window_index']}: "
                f"{window['selected_model']}"
            )

        print("  Combined economic OOS:")

        print_economic_results(
            result["economic_oos"],
            indent="    ",
        )


def get_top1(result):
    metrics = calculate_economic_metrics(
        result["economic_oos"]
    )

    return next(
        (
            item
            for item in metrics["top_fraction"]
            if item["fraction"] == 0.01
        ),
        None,
    )


def print_comparison(results):
    print()
    print("================================")
    print("VOLATILITY PRICE CONTROL COMPARISON")
    print("================================")

    print(
        "Feature set | val AUC | "
        "top1 event | lift | "
        "top1 mean | total | fit"
    )

    print("-" * 115)

    for result in results:
        top1 = get_top1(result)

        if top1 is None:
            continue

        validation_score = (
            result["validation_score"]
            if result["validation_score"]
            is not None
            else float("nan")
        )

        print(
            f"{result['feature_set']:<38}"
            f"{validation_score:.6f}     "
            f"{top1['event_rate']:.4f}  "
            f"{top1['lift']:.2f}x   "
            f"{top1['mean_return']:+.4%}    "
            f"{result['total_seconds']:.2f}s    "
            f"{result['fit_seconds']:.2f}s"
        )


def print_key_deltas(results):
    lookup = {
        result["feature_set"]: result
        for result in results
    }

    baseline = lookup[
        "volatility_only"
    ]

    baseline_top1 = get_top1(
        baseline
    )

    print()
    print("================================")
    print("DELTA MOT VOLATILITY-ONLY")
    print("================================")

    if baseline_top1 is None:
        print("Ingen top-1%-data.")
        return

    for result in results:
        if (
            result["feature_set"]
            == "volatility_only"
        ):
            continue

        top1 = get_top1(result)

        if top1 is None:
            continue

        auc_delta = (
            result["validation_score"]
            - baseline["validation_score"]
        )

        event_delta = (
            top1["event_rate"]
            - baseline_top1["event_rate"]
        )

        return_delta = (
            top1["mean_return"]
            - baseline_top1["mean_return"]
        )

        print()
        print(result["feature_set"])

        print(
            f"  Validation AUC delta: "
            f"{auc_delta:+.6f}"
        )

        print(
            f"  Top1 event-rate delta: "
            f"{event_delta:+.4f}"
        )

        print(
            f"  Top1 mean-return delta: "
            f"{return_delta:+.4%}"
        )


def main():
    print("=" * 100)
    print("BLANKDISS VOLATILITY PRICE CONTROL SCREENING")
    print("=" * 100)
    print(
        f"RF trees: {BENCHMARK_TREES}"
    )
    print("RF n_jobs: 1")
    print(
        f"Walk-forward windows: "
        f"{len(WALK_FORWARD_WINDOWS)}"
    )
    print(
        f"Economic target: "
        f"{ECONOMIC_TARGET}"
    )
    print()
    print(
        "Question: "
        "Är volatility_20d en proxy för annan prisinformation?"
    )

    print()
    print("Loading feature data...")

    features = load_features()

    print(
        f"Feature rows: {len(features):,}"
    )

    print()
    print("Building feature sets...")

    feature_sets = build_feature_sets(
        features
    )

    print_feature_set_qc(
        feature_sets
    )

    # walk_forward använder build_models från
    # ml.models via modulens importerade symbol.
    walk_forward.build_models = (
        build_benchmark_models
    )

    target = TARGETS[
        ECONOMIC_TARGET
    ]

    results = []

    for (
        feature_set_name,
        _,
    ) in FEATURE_SET_CONFIG:
        print()
        print("=" * 96)
        print(
            f"RUN: {feature_set_name}"
        )
        print("=" * 96)

        feature_set_data, feature_columns = (
            feature_sets[
                feature_set_name
            ]
        )

        result = run_one_experiment(
            feature_set_name,
            feature_set_data,
            feature_columns,
            target,
        )

        results.append(result)

    print_window_results(
        results
    )

    print_combined_results(
        results
    )

    print_comparison(
        results
    )

    print_key_deltas(
        results
    )

    print()
    print("=" * 100)
    print("SCREENING COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()
