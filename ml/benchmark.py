"""Snabb feature-screening för Blankdiss ML-pipeline.

Syfte:
- jämföra de viktigaste feature-kombinationerna
- testa om resultaten håller över fler walk-forward-fönster
- mäta faktisk OOS-ekonomi
- hålla beräkningskostnaden låg

Detta test:
- Random Forest: 100 träd
- RF n_jobs=1
- 1 experiment-worker
- 4 walk-forward-fönster
- target: down_5pct_5d

Feature sets:
- FI-only
- FI + volatility_20d
- FI + volatility_20d + distance_from_20d_high

Produktionsfiler ändras inte.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from time import perf_counter

import numpy as np

import ml.walk_forward as walk_forward
from ml.config import (
    TARGETS,
    WALK_FORWARD_WINDOWS,
)
from ml.dataset import (
    load_features,
    prepare_feature_set,
    prepare_ml_data_from_feature_set,
)
from ml.models import build_models as original_build_models


MAX_PARALLEL_EXPERIMENTS = 1
BENCHMARK_TREES = 100
ECONOMIC_TARGET = "down_5pct_5d"

ECONOMIC_TOP_FRACTIONS = (
    0.001,
    0.005,
    0.01,
    0.02,
    0.05,
)

FEATURE_SETS = (
    (
        "fi_only",
        set(),
    ),
    (
        "fi_plus_volatility_20d",
        {
            "price_volatility_20d",
        },
    ),
    (
        "fi_plus_volatility_20d_distance_20d",
        {
            "price_volatility_20d",
            "price_distance_from_20d_high",
        },
    ),
)

# Använd fyra walk-forward-fönster för robusthetskontrollen.
# Produktionskonfigurationen ändras inte.
BENCHMARK_WALK_FORWARD_WINDOWS = (
    WALK_FORWARD_WINDOWS[:4]
)

PERFORMANCE_KEYS = (
    "fit_seconds",
    "validation_prediction_seconds",
    "oos_prediction_seconds",
    "evaluation_seconds",
    "oos_row_build_seconds",
    "split_seconds",
    "total_window_seconds",
)


def build_benchmark_models(
    random_state: int,
    task: str = "classification",
):
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


def build_experiment_list():
    experiments = []

    target = next(
        target
        for target in TARGETS
        if target.name == ECONOMIC_TARGET
    )

    for (
        feature_set_name,
        _price_features,
    ) in FEATURE_SETS:
        experiments.append(
            (
                feature_set_name,
                target,
            )
        )

    return experiments


def build_feature_set_cache(
    features,
):
    cache = {}

    for (
        feature_set_name,
        price_features,
    ) in FEATURE_SETS:
        (
            feature_set_data,
            feature_columns,
        ) = prepare_feature_set(
            features,
            include_price_features=bool(
                price_features
            ),
            price_features=price_features,
        )

        cache[feature_set_name] = (
            feature_set_data,
            feature_columns,
        )

    return cache


def print_feature_set_qc(
    feature_set_cache,
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
        _price_features,
    ) in FEATURE_SETS:
        (
            _feature_set_data,
            feature_columns,
        ) = feature_set_cache[
            feature_set_name
        ]

        print(
            f"{feature_set_name}: "
            f"{len(feature_columns)} features"
        )

        print(
            "  "
            + ", ".join(
                feature_columns
            )
        )


def empty_performance():
    return {
        key: 0.0
        for key in PERFORMANCE_KEYS
    }


def run_one_experiment(
    feature_set_cache,
    feature_set_name,
    target,
):
    experiment_start = perf_counter()

    (
        feature_set_data,
        feature_columns,
    ) = feature_set_cache[
        feature_set_name
    ]

    prepare_start = perf_counter()

    (
        data,
        y,
        feature_columns,
    ) = prepare_ml_data_from_feature_set(
        feature_set_data,
        feature_columns,
        target,
    )

    prepare_seconds = (
        perf_counter() - prepare_start
    )

    performance = empty_performance()
    validation_scores = []
    selected_models = {}
    economic_oos = []
    window_count = 0

    for window in BENCHMARK_WALK_FORWARD_WINDOWS:
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

        for key in PERFORMANCE_KEYS:
            performance[key] += float(
                timing.get(
                    key,
                    0.0,
                )
            )

        for result in results:
            model_name = result.get(
                "model"
            )

            validation_score = result.get(
                "validation_score"
            )

            if (
                validation_score is not None
                and model_name is not None
            ):
                validation_scores.append(
                    float(validation_score)
                )

                selected_models[
                    model_name
                ] = (
                    selected_models.get(
                        model_name,
                        0,
                    )
                    + 1
                )

        economic_oos.extend(
            oos_predictions
        )

        window_count += 1

    total_seconds = (
        perf_counter() - experiment_start
    )

    return {
        "feature_set": feature_set_name,
        "target": target.name,
        "feature_count": len(
            feature_columns
        ),
        "row_count": len(data),
        "prepare_seconds": (
            prepare_seconds
        ),
        "performance": performance,
        "total_seconds": total_seconds,
        "window_count": window_count,
        "validation_scores": validation_scores,
        "selected_models": selected_models,
        "economic_oos": economic_oos,
    }


def run_benchmark(
    feature_set_cache,
    experiments,
):
    def build_models_for_benchmark(
        random_state: int,
        task: str = "classification",
    ):
        return build_benchmark_models(
            random_state,
            task=task,
        )

    walk_forward.build_models = (
        build_models_for_benchmark
    )

    start = perf_counter()

    with ThreadPoolExecutor(
        max_workers=MAX_PARALLEL_EXPERIMENTS,
        thread_name_prefix="benchmark-feature",
    ) as executor:
        futures = [
            executor.submit(
                run_one_experiment,
                feature_set_cache,
                feature_set_name,
                target,
            )
            for (
                feature_set_name,
                target,
            ) in experiments
        ]

        results = [
            future.result()
            for future in futures
        ]

    wall_seconds = (
        perf_counter() - start
    )

    return {
        "wall_seconds": wall_seconds,
        "experiment_results": results,
    }


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

    probabilities = np.asarray(
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

    events = np.asarray(
        [
            1.0
            if row["target_return"] <= -0.05
            else 0.0
            for row in oos_predictions
        ],
        dtype=float,
    )

    valid = (
        np.isfinite(probabilities)
        & np.isfinite(returns)
        & np.isfinite(events)
    )

    probabilities = probabilities[
        valid
    ]

    returns = returns[
        valid
    ]

    events = events[
        valid
    ]

    rows = len(probabilities)

    if rows == 0:
        return {
            "rows": 0,
            "baseline_event_rate": None,
            "baseline_mean_return": None,
            "top_fraction": [],
        }

    order = np.argsort(
        -probabilities,
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

        selected = order[
            :count
        ]

        selected_returns = returns[
            selected
        ]

        selected_events = events[
            selected
        ]

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
                "event_rate_lift": lift,
                "mean_return": mean_return,
                "median_return": median_return,
            }
        )

    return {
        "rows": rows,
        "baseline_event_rate": (
            baseline_event_rate
        ),
        "baseline_mean_return": (
            baseline_mean_return
        ),
        "top_fraction": top_fraction,
    }


def print_economic_results(
    economic_oos,
):
    metrics = calculate_economic_metrics(
        economic_oos
    )

    print()
    print(
        f"Ekonomisk OOS: "
        f"{ECONOMIC_TARGET}"
    )

    print(
        f"  Rows: "
        f"{metrics['rows']:,}"
    )

    if metrics[
        "baseline_event_rate"
    ] is not None:
        print(
            f"  Baseline event rate: "
            f"{metrics['baseline_event_rate']:.4f}"
        )

    if metrics[
        "baseline_mean_return"
    ] is not None:
        print(
            f"  Baseline mean return: "
            f"{metrics['baseline_mean_return']:+.4%}"
        )

    print()

    for result in metrics[
        "top_fraction"
    ]:
        fraction = result[
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

        lift = result[
            "event_rate_lift"
        ]

        lift_text = (
            f"{lift:.2f}x"
            if lift is not None
            else "n/a"
        )

        print(
            f"  Top {fraction_text:>5}: "
            f"n={result['rows']:,} "
            f"event={result['event_rate']:.4f} "
            f"lift={lift_text:>6} "
            f"mean={result['mean_return']:+.4%} "
            f"median={result['median_return']:+.4%}"
        )

    return metrics


def print_feature_set_results(
    experiment_results,
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

    for result in sorted(
        experiment_results,
        key=lambda item: item[
            "feature_set"
        ],
    ):
        validation_scores = result[
            "validation_scores"
        ]

        average_validation_score = (
            sum(validation_scores)
            / len(validation_scores)
            if validation_scores
            else 0.0
        )

        performance = result[
            "performance"
        ]

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
            f"{result['window_count']}"
        )

        print(
            f"  Prepare: "
            f"{result['prepare_seconds']:.2f}s"
        )

        print(
            f"  Total: "
            f"{result['total_seconds']:.2f}s"
        )

        print(
            f"  Fit: "
            f"{performance['fit_seconds']:.2f}s"
        )

        print(
            f"  Validation prediction: "
            f"{performance['validation_prediction_seconds']:.2f}s"
        )

        print(
            f"  OOS prediction: "
            f"{performance['oos_prediction_seconds']:.2f}s"
        )

        print(
            f"  Evaluation: "
            f"{performance['evaluation_seconds']:.2f}s"
        )

        print(
            f"  OOS row build: "
            f"{performance['oos_row_build_seconds']:.2f}s"
        )

        print(
            f"  Average validation score: "
            f"{average_validation_score:.6f}"
        )

        print(
            "  Selected models:"
        )

        for (
            model_name,
            count,
        ) in sorted(
            result[
                "selected_models"
            ].items()
        ):
            print(
                f"    {model_name}: "
                f"{count}"
            )

        print_economic_results(
            result["economic_oos"]
        )


def print_comparison(
    experiment_results,
):
    print()
    print(
        "================================"
    )
    print(
        "FEATURE COMPARISON"
    )
    print(
        "================================"
    )

    rows = []

    for result in experiment_results:
        validation_scores = result[
            "validation_scores"
        ]

        average_validation_score = (
            sum(validation_scores)
            / len(validation_scores)
            if validation_scores
            else 0.0
        )

        metrics = calculate_economic_metrics(
            result["economic_oos"]
        )

        top_1 = None

        for item in metrics[
            "top_fraction"
        ]:
            if item["fraction"] == 0.01:
                top_1 = item
                break

        performance = result[
            "performance"
        ]

        rows.append(
            (
                result["feature_set"],
                average_validation_score,
                (
                    top_1["event_rate"]
                    if top_1 is not None
                    else None
                ),
                (
                    top_1["event_rate_lift"]
                    if top_1 is not None
                    else None
                ),
                (
                    top_1["mean_return"]
                    if top_1 is not None
                    else None
                ),
                result["total_seconds"],
                performance["fit_seconds"],
            )
        )

    rows.sort(
        key=lambda item: (
            item[2]
            if item[2] is not None
            else -np.inf
        ),
        reverse=True,
    )

    print()

    print(
        "Feature set"
        " | val"
        " | top1 event"
        " | lift"
        " | top1 mean"
        " | total"
        " | fit"
    )

    print("-" * 100)

    for row in rows:
        (
            feature_set,
            validation_score,
            event_rate,
            lift,
            mean_return,
            total_seconds,
            fit_seconds,
        ) = row

        event_text = (
            f"{event_rate:.4f}"
            if event_rate is not None
            else "n/a"
        )

        lift_text = (
            f"{lift:.2f}x"
            if lift is not None
            else "n/a"
        )

        mean_text = (
            f"{mean_return:+.4%}"
            if mean_return is not None
            else "n/a"
        )

        print(
            f"{feature_set:<45} "
            f"{validation_score:.6f} "
            f"{event_text:>10} "
            f"{lift_text:>6} "
            f"{mean_text:>10} "
            f"{total_seconds:>8.2f}s "
            f"{fit_seconds:>8.2f}s"
        )


def main() -> None:
    total_start = perf_counter()

    print(
        "================================"
    )
    print(
        "Blankdiss feature screening"
    )
    print(
        "================================"
    )

    print()

    print(
        f"RF trees: "
        f"{BENCHMARK_TREES}"
    )

    print(
        f"Experiment workers: "
        f"{MAX_PARALLEL_EXPERIMENTS}"
    )

    print(
        f"Walk-forward windows: "
        f"{len(BENCHMARK_WALK_FORWARD_WINDOWS)}"
    )

    print(
        "RF n_jobs: 1"
    )

    print(
        f"Economic target: "
        f"{ECONOMIC_TARGET}"
    )

    print(
        "Economic top fractions: "
        + ", ".join(
            (
                f"{fraction:.1%}"
                for fraction
                in ECONOMIC_TOP_FRACTIONS
            )
        )
    )

    print()

    print(
        "Laddar features..."
    )

    start = perf_counter()

    features = load_features()

    print(
        f"Feature-rader: "
        f"{len(features):,}"
    )

    print(
        f"Laddning: "
        f"{perf_counter() - start:.2f}s"
    )

    print()

    print(
        "Bygger feature-set cache..."
    )

    start = perf_counter()

    feature_set_cache = (
        build_feature_set_cache(
            features
        )
    )

    print(
        f"Feature cache: "
        f"{perf_counter() - start:.2f}s"
    )

    print_feature_set_qc(
        feature_set_cache
    )

    experiments = (
        build_experiment_list()
    )

    print()

    print(
        f"Experiments: "
        f"{len(experiments)}"
    )

    for (
        feature_set_name,
        target,
    ) in experiments:
        print(
            f"  {feature_set_name} "
            f"-> {target.name}"
        )

    print()

    print(
        "================================"
    )
    print(
        "RUN FEATURE SCREENING"
    )
    print(
        "================================"
    )

    benchmark_result = run_benchmark(
        feature_set_cache,
        experiments,
    )

    print()

    print(
        f"Wall time: "
        f"{benchmark_result['wall_seconds']:.2f}s"
    )

    print_feature_set_results(
        benchmark_result[
            "experiment_results"
        ]
    )

    print_comparison(
        benchmark_result[
            "experiment_results"
        ]
    )

    print()

    print(
        "================================"
    )

    print(
        f"Benchmark total: "
        f"{perf_counter() - total_start:.2f}s"
    )

    print(
        "================================"
    )


if __name__ == "__main__":
    main()
