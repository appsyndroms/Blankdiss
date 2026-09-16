"""Benchmark av hela Blankdiss ML-körningen.

Jämför produktionslik experimentparallellism med:
- Random Forest n_jobs=1
- Random Forest n_jobs=4

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

BENCHMARK_N_JOBS = (
    1,
    4,
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
    n_jobs: int = 1,
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
            model__n_jobs=n_jobs,
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

        window_count += 1

    return (
        performance,
        window_count,
    )


def run_benchmark(
    feature_set_cache,
    experiments,
    n_jobs,
):
    def build_models_for_benchmark(
        random_state: int,
        task: str = "classification",
    ):
        return build_benchmark_models(
            random_state,
            task=task,
            n_jobs=n_jobs,
        )

    # train_window() använder build_models som
    # importerats in i ml.walk_forward.
    # Vi byter bara den symbolen under benchmarken.
    walk_forward.build_models = (
        build_models_for_benchmark
    )

    start = perf_counter()

    with ThreadPoolExecutor(
        max_workers=MAX_PARALLEL_EXPERIMENTS,
        thread_name_prefix=(
            f"benchmark-rf-{n_jobs}"
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

    window_count = 0

    for (
        performance,
        windows,
    ) in results:
        for key in aggregate:
            aggregate[key] += (
                performance[key]
            )

        window_count += windows

    return (
        wall_seconds,
        aggregate,
        window_count,
    )


def main() -> None:
    total_start = perf_counter()

    print("================================")
    print("Blankdiss production ML benchmark")
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

    for n_jobs in BENCHMARK_N_JOBS:
        print()
        print("================================")
        print(
            f"TEST: Random Forest n_jobs={n_jobs}"
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
        ) = run_benchmark(
            feature_set_cache,
            experiments,
            n_jobs,
        )

        benchmark_results[n_jobs] = (
            wall_seconds,
            performance,
            window_count,
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

    print()
    print("================================")
    print("JÄMFÖRELSE")
    print("================================")

    baseline = benchmark_results[1][0]

    for n_jobs in BENCHMARK_N_JOBS:
        wall_seconds = (
            benchmark_results[n_jobs][0]
        )

        if n_jobs == 1:
            print(
                f"n_jobs=1: "
                f"{wall_seconds:.2f}s"
            )
            continue

        improvement = (
            1.0
            - wall_seconds / baseline
        ) * 100.0

        print(
            f"n_jobs={n_jobs}: "
            f"{wall_seconds:.2f}s "
            f"({improvement:+.1f}%)"
        )

    total_seconds = (
        perf_counter() - total_start
    )

    print()
    print("================================")
    print(
        f"Benchmark total: "
        f"{total_seconds:.2f}s"
    )
    print("================================")


if __name__ == "__main__":
    main()
