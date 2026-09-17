"""Benchmark av Random Forest-storlek i hela Blankdiss ML-pipelinen.

Jämför:
- Random Forest 100 träd
- Random Forest 200 träd
- Random Forest 300 träd

Produktionslik belastning:
- 40 experiment
- 4 parallella experiment
- 2 walk-forward-fönster

Produktionsfiler ändras inte.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from time import perf_counter

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
        )

        cache[feature_set_name] = (
            feature_set_data,
            feature_columns,
        )

    return cache


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

    performance = {
        "fit_seconds": 0.0,
        "validation_prediction_seconds": 0.0,
        "oos_prediction_seconds": 0.0,
        "evaluation_seconds": 0.0,
        "oos_row_build_seconds": 0.0,
        "split_seconds": 0.0,
        "total_window_seconds": 0.0,
    }

    validation_scores = []
    selected_models = {}

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

        for key in performance:
            performance[key] += float(
                timing.get(key, 0.0)
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

        window_count += 1

    return (
        performance,
        window_count,
        validation_scores,
        selected_models,
    )


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

    aggregate = {
        "fit_seconds": 0.0,
        "validation_prediction_seconds": 0.0,
        "oos_prediction_seconds": 0.0,
        "evaluation_seconds": 0.0,
        "oos_row_build_seconds": 0.0,
        "split_seconds": 0.0,
        "total_window_seconds": 0.0,
    }

    validation_scores = []
    selected_models = {}

    window_count = 0

    for (
        performance,
        windows,
        scores,
        models,
    ) in results:
        for key in aggregate:
            aggregate[key] += (
                performance[key]
            )

        validation_scores.extend(
            scores
        )

        for model_name, count in models.items():
            selected_models[model_name] = (
                selected_models.get(
                    model_name,
                    0,
                )
                + count
            )

        window_count += windows

    return (
        wall_seconds,
        aggregate,
        window_count,
        validation_scores,
        selected_models,
    )


def main() -> None:
    total_start = perf_counter()

    print("================================")
    print("Blankdiss Random Forest benchmark")
    print("================================")
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

    print()
    print("Laddar features...")

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
    print("Bygger feature-set cache...")

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
        print("================================")
        print(
            f"TEST: Random Forest "
            f"{n_estimators} träd"
        )
        print(
            f"Experiment workers="
            f"{MAX_PARALLEL_EXPERIMENTS}"
        )
        print("================================")

        (
            wall_seconds,
            performance,
            window_count,
            validation_scores,
            selected_models,
        ) = run_benchmark(
            feature_set_cache,
            experiments,
            n_estimators,
        )

        benchmark_results[
            n_estimators
        ] = (
            wall_seconds,
            performance,
            window_count,
            validation_scores,
            selected_models,
        )

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
            f"{window_count}"
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

        print(
            "Selected models:"
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

    print()
    print("================================")
    print("JÄMFÖRELSE")
    print("================================")

    baseline_wall = benchmark_results[
        300
    ][0]

    baseline_score = (
        sum(
            benchmark_results[300][3]
        )
        / len(
            benchmark_results[300][3]
        )
        if benchmark_results[300][3]
        else 0.0
    )

    for n_estimators in BENCHMARK_TREES:
        (
            wall_seconds,
            performance,
            window_count,
            validation_scores,
            selected_models,
        ) = benchmark_results[
            n_estimators
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
    print("================================")
    print(
        f"Benchmark total: "
        f"{perf_counter() - total_start:.2f}s"
    )
    print("================================")


if __name__ == "__main__":
    main()
