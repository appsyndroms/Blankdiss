"""Benchmark av Random Forest-storlek i hela Blankdiss ML-pipelinen.

Jämför:
- Random Forest 100 träd
- Random Forest 200 träd
- Random Forest 300 träd

Produktionslik belastning:
- samma feature sets och targets som ordinarie benchmark
- 4 parallella experiment
- 2 walk-forward-fönster
- RF n_jobs=1

Utöver tids- och valideringsmått jämför benchmarken faktisk OOS-ekonomi
för down_5pct_5d.

Produktionsfiler ändras inte.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from time import perf_counter

import numpy as np

import ml.walk_forward as walk_forward

from ml.config import (
    RANDOM_STATE,
    TARGETS,
    WALK_FORWARD_WINDOWS,
)
from ml.dataset import (
    load_features,
    prepare_feature_set,
    prepare_ml_data_from_feature_set,
)
from ml.models import build_models as original_build_models


MAX_PARALLEL_EXPERIMENTS = 4

BENCHMARK_TREES = (
    100,
    200,
    300,
)

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
        None,
    ),
    (
        "fi_plus_price_return_5d",
        {"price_return_5d"},
    ),
    (
        "fi_plus_price_return_20d",
        {"price_return_20d"},
    ),
    (
        "fi_plus_price_return_60d",
        {"price_return_60d"},
    ),
    (
        "fi_plus_price_volatility_20d",
        {"price_volatility_20d"},
    ),
    (
        "fi_plus_price_distance_from_20d_high",
        {"price_distance_from_20d_high"},
    ),
    (
        "fi_plus_price_distance_from_60d_high",
        {"price_distance_from_60d_high"},
    ),
    (
        "fi_plus_all_price",
        {
            "price_return_5d",
            "price_return_20d",
            "price_return_60d",
            "price_volatility_20d",
            "price_distance_from_20d_high",
            "price_distance_from_60d_high",
        },
    ),
)


ABLATION_TARGETS = {
    "up_5pct_5d",
    "down_5pct_5d",
}


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
    n_estimators: int = 300,
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
            model__n_estimators=n_estimators,
            model__n_jobs=1,
        )

    random_forest_regression = models.get(
        "random_forest_regression"
    )

    if random_forest_regression is not None:
        random_forest_regression.set_params(
            model__n_estimators=n_estimators,
            model__n_jobs=1,
        )

    return models


def build_experiment_list():
    experiments = []

    for (
        feature_set_name,
        price_features,
    ) in FEATURE_SETS:
        for target in TARGETS:
            is_single_price_ablation = (
                price_features is not None
                and feature_set_name
                != "fi_plus_all_price"
            )

            if (
                is_single_price_ablation
                and target.name
                not in ABLATION_TARGETS
            ):
                continue

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
            include_price_features=(
                price_features is not None
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
    (
        feature_set_data,
        feature_columns,
    ) = feature_set_cache[
        feature_set_name
    ]

    (
        data,
        y,
        feature_columns,
    ) = prepare_ml_data_from_feature_set(
        feature_set_data,
        feature_columns,
        target,
    )

    performance = empty_performance()

    validation_scores = []
    selected_models = {}

    economic_oos = []

    window_count = 0

    for window in WALK_FORWARD_WINDOWS:
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

        if (
            target.name
            == ECONOMIC_TARGET
        ):
            economic_oos.extend(
                oos_predictions
            )

        window_count += 1

    return {
        "feature_set": feature_set_name,
        "target": target.name,
        "performance": performance,
        "window_count": window_count,
        "validation_scores": validation_scores,
        "selected_models": selected_models,
        "economic_oos": economic_oos,
    }


def run_benchmark(
    feature_set_cache,
    experiments,
    n_estimators,
):
    def build_models_for_benchmark(
        random_state: int,
        task: str = "classification",
    ):
        return build_benchmark_models(
            random_state,
            task=task,
            n_estimators=n_estimators,
        )

    walk_forward.build_models = (
        build_models_for_benchmark
    )

    start = perf_counter()

    with ThreadPoolExecutor(
        max_workers=MAX_PARALLEL_EXPERIMENTS,
        thread_name_prefix=(
            f"benchmark-rf-{n_estimators}"
        ),
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

    aggregate = empty_performance()

    validation_scores = []
    selected_models = {}

    economic_oos = []

    window_count = 0

    experiment_results = []

    for result in results:
        experiment_results.append(
            result
        )

        performance = result[
            "performance"
        ]

        for key in PERFORMANCE_KEYS:
            aggregate[key] += (
                performance[key]
            )

        validation_scores.extend(
            result[
                "validation_scores"
            ]
        )

        for (
            model_name,
            count,
        ) in result[
            "selected_models"
        ].items():
            selected_models[
                model_name
            ] = (
                selected_models.get(
                    model_name,
                    0,
                )
                + count
            )

        economic_oos.extend(
            result["economic_oos"]
        )

        window_count += result[
            "window_count"
        ]

    return {
        "wall_seconds": wall_seconds,
        "performance": aggregate,
        "window_count": window_count,
        "validation_scores": validation_scores,
        "selected_models": selected_models,
        "economic_oos": economic_oos,
        "experiment_results": experiment_results,
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
    n_estimators,
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
            f"{metrics['baseline_mean_return']:.4%}"
        )

    print()

    for result in metrics[
        "top_fraction"
    ]:
        fraction = (
            result["fraction"]
        )

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


def print_feature_set_economics(
    n_estimators,
    experiment_results,
):
    print()
    print(
        "================================"
    )
    print(
        f"OOS PER FEATURE SET — "
        f"{n_estimators} träd"
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
        if result[
            "target"
        ] != ECONOMIC_TARGET:
            continue

        oos = result[
            "economic_oos"
        ]

        if not oos:
            continue

        metrics = calculate_economic_metrics(
            oos
        )

        print()
        print(
            result["feature_set"]
        )

        for economic_result in metrics[
            "top_fraction"
        ]:
            fraction = economic_result[
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

            print(
                f"  top {fraction_text:>5}: "
                f"mean="
                f"{economic_result['mean_return']:+.4%} "
                f"median="
                f"{economic_result['median_return']:+.4%} "
                f"event="
                f"{economic_result['event_rate']:.4f} "
                f"n="
                f"{economic_result['rows']:,}"
            )


def print_benchmark_summary(
    benchmark_results,
):
    print()
    print(
        "================================"
    )
    print("TID + VALIDERING")
    print(
        "================================"
    )

    baseline = benchmark_results[
        300
    ]

    baseline_wall = baseline[
        "wall_seconds"
    ]

    baseline_scores = baseline[
        "validation_scores"
    ]

    baseline_score = (
        sum(baseline_scores)
        / len(baseline_scores)
        if baseline_scores
        else 0.0
    )

    for n_estimators in BENCHMARK_TREES:
        result = benchmark_results[
            n_estimators
        ]

        wall_seconds = result[
            "wall_seconds"
        ]

        validation_scores = result[
            "validation_scores"
        ]

        average_score = (
            sum(validation_scores)
            / len(validation_scores)
            if validation_scores
            else 0.0
        )

        time_change = (
            (
                wall_seconds
                / baseline_wall
            )
            - 1.0
        ) * 100.0

        score_change = (
            average_score
            - baseline_score
        )

        print(
            f"{n_estimators:>3} träd: "
            f"{wall_seconds:7.2f}s "
            f"({time_change:+.1f}% tid, "
            f"{score_change:+.6f} val-score)"
        )

    print()
    print(
        "================================"
    )
    print("VALDA MODELLER")
    print(
        "================================"
    )

    for n_estimators in BENCHMARK_TREES:
        print()
        print(
            f"{n_estimators} träd:"
        )

        selected_models = (
            benchmark_results[
                n_estimators
            ][
                "selected_models"
            ]
        )

        for (
            model_name,
            count,
        ) in sorted(
            selected_models.items()
        ):
            print(
                f"  {model_name}: "
                f"{count}"
            )


def print_economic_comparison(
    benchmark_results,
):
    print()
    print(
        "================================"
    )
    print(
        "EKONOMISK JÄMFÖRELSE"
    )
    print(
        f"Target: {ECONOMIC_TARGET}"
    )
    print(
        "================================"
    )

    all_metrics = {}

    for n_estimators in BENCHMARK_TREES:
        metrics = calculate_economic_metrics(
            benchmark_results[
                n_estimators
            ][
                "economic_oos"
            ]
        )

        all_metrics[
            n_estimators
        ] = metrics

        print()
        print(
            f"RF {n_estimators} träd"
        )
        print(
            "-" * 60
        )

        if metrics[
            "baseline_event_rate"
        ] is not None:
            print(
                f"Baseline event rate: "
                f"{metrics['baseline_event_rate']:.4f}"
            )

        if metrics[
            "baseline_mean_return"
        ] is not None:
            print(
                f"Baseline mean return: "
                f"{metrics['baseline_mean_return']:+.4%}"
            )

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

            print(
                f"Top {fraction_text:>5}: "
                f"mean="
                f"{result['mean_return']:+.4%}, "
                f"median="
                f"{result['median_return']:+.4%}, "
                f"event="
                f"{result['event_rate']:.4f}, "
                f"lift="
                f"{result['event_rate_lift']:.2f}x "
                if result[
                    "event_rate_lift"
                ] is not None
                else
                f"Top {fraction_text:>5}: "
                f"mean="
                f"{result['mean_return']:+.4%}, "
                f"median="
                f"{result['median_return']:+.4%}, "
                f"event="
                f"{result['event_rate']:.4f}, "
                f"lift=n/a"
            )

    print()
    print(
        "================================"
    )
    print(
        "DIREKT 100 vs 200 vs 300"
    )
    print(
        "================================"
    )

    for fraction_index, fraction in enumerate(
        ECONOMIC_TOP_FRACTIONS
    ):
        if fraction < 0.01:
            fraction_text = (
                f"{fraction:.1%}"
            )
        else:
            fraction_text = (
                f"{fraction:.0%}"
            )

        print()
        print(
            f"Top {fraction_text}:"
        )

        for n_estimators in BENCHMARK_TREES:
            result = all_metrics[
                n_estimators
            ]["top_fraction"][
                fraction_index
            ]

            print(
                f"  {n_estimators:>3} träd: "
                f"mean={result['mean_return']:+.4%}, "
                f"median={result['median_return']:+.4%}, "
                f"event={result['event_rate']:.4f}, "
                f"n={result['rows']:,}"
            )


def main() -> None:
    total_start = perf_counter()

    print(
        "================================"
    )
    print(
        "Blankdiss Random Forest benchmark"
    )
    print(
        "================================"
    )
    print()

    print(
        f"Experiment workers: "
        f"{MAX_PARALLEL_EXPERIMENTS}"
    )

    print(
        f"Walk-forward windows: "
        f"{len(WALK_FORWARD_WINDOWS)}"
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

    benchmark_results = {}

    for n_estimators in BENCHMARK_TREES:
        print()
        print(
            "================================"
        )
        print(
            f"TEST: Random Forest "
            f"{n_estimators} träd"
        )
        print(
            f"Experiment workers="
            f"{MAX_PARALLEL_EXPERIMENTS}"
        )
        print(
            "================================"
        )

        result = run_benchmark(
            feature_set_cache,
            experiments,
            n_estimators,
        )

        benchmark_results[
            n_estimators
        ] = result

        wall_seconds = result[
            "wall_seconds"
        ]

        performance = result[
            "performance"
        ]

        validation_scores = result[
            "validation_scores"
        ]

        average_validation_score = (
            sum(validation_scores)
            / len(validation_scores)
            if validation_scores
            else 0.0
        )

        print()
        print(
            f"Wall time: "
            f"{wall_seconds:.2f}s"
        )

        print(
            f"Window count: "
            f"{result['window_count']}"
        )

        print(
            f"Aggregate fit time: "
            f"{performance['fit_seconds']:.2f}s"
        )

        print(
            f"Aggregate validation prediction: "
            f"{performance['validation_prediction_seconds']:.2f}s"
        )

        print(
            f"Aggregate OOS prediction: "
            f"{performance['oos_prediction_seconds']:.2f}s"
        )

        print(
            f"Aggregate evaluation: "
            f"{performance['evaluation_seconds']:.2f}s"
        )

        print(
            f"Aggregate OOS row build: "
            f"{performance['oos_row_build_seconds']:.2f}s"
        )

        print(
            f"Average validation score: "
            f"{average_validation_score:.6f}"
        )

        print()
        print(
            "Selected models:"
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
                f"  {model_name}: "
                f"{count}"
            )

        print_economic_results(
            n_estimators,
            result[
                "economic_oos"
            ],
        )

    print_benchmark_summary(
        benchmark_results
    )

    print_economic_comparison(
        benchmark_results
    )

    for n_estimators in BENCHMARK_TREES:
        print_feature_set_economics(
            n_estimators,
            benchmark_results[
                n_estimators
            ][
                "experiment_results"
            ],
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
