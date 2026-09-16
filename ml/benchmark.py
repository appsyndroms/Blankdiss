"""Benchmark av Blankdiss ML-pipeline med Random Forest n_jobs=4."""

from __future__ import annotations

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


BENCHMARK_TARGETS = {
    "up_5pct_5d",
    "down_5pct_5d",
}

RF_N_JOBS = 4


def build_benchmark_models(
    random_state: int,
    task: str = "classification",
):
    """
    Bygger exakt samma modeller som produktionen,
    men kör Random Forest med n_jobs=4.

    Produktionskoden ändras inte.
    """
    models = original_build_models(
        random_state,
        task=task,
    )

    random_forest = models.get("random_forest")

    if random_forest is not None:
        random_forest.set_params(
            model__n_jobs=RF_N_JOBS,
        )

    return models


# train_window har redan importerat build_models
# som en lokal symbol i ml.walk_forward.
# Vi ersätter endast den symbolen under benchmarkkörningen.
walk_forward.build_models = build_benchmark_models


def main() -> None:
    total_start = perf_counter()

    print("================================")
    print("Blankdiss ML benchmark")
    print("Random Forest n_jobs=4")
    print("================================")

    print()
    print("Laddar features...")

    start = perf_counter()

    features = load_features()

    print(
        f"Features: {len(features):,} "
        f"({perf_counter() - start:.2f}s)"
    )

    print()
    print("Förbereder fi_only...")

    start = perf_counter()

    feature_set_data, feature_columns = prepare_feature_set(
        features,
        include_price_features=False,
    )

    print(f"Rows: {len(feature_set_data):,}")
    print(f"Features: {len(feature_columns)}")
    print(
        f"Feature preparation: "
        f"{perf_counter() - start:.2f}s"
    )

    targets = [
        target
        for target in TARGETS
        if target.name in BENCHMARK_TARGETS
    ]

    total_window_seconds = 0.0

    for target in targets:
        print()
        print("--------------------------------")
        print(f"Target: {target.name}")
        print("--------------------------------")

        start = perf_counter()

        data, y, target_feature_columns = (
            prepare_ml_data_from_feature_set(
                feature_set_data,
                feature_columns,
                target,
            )
        )

        print(
            f"Dataset preparation: "
            f"{perf_counter() - start:.2f}s"
        )
        print(f"Rows: {len(data):,}")
        print(
            f"Features: "
            f"{len(target_feature_columns)}"
        )

        for window in WALK_FORWARD_WINDOWS:
            print()
            print(
                f"Window: "
                f"{window.train_end} -> "
                f"{window.validation_end} -> "
                f"{window.test_end}"
            )

            start = perf_counter()

            results, oos_predictions, timing = (
                walk_forward.train_window(
                    data=data,
                    y=y,
                    feature_columns=target_feature_columns,
                    window=window,
                    task=target.task,
                    direction=target.direction,
                )
            )

            wall_seconds = (
                perf_counter() - start
            )

            total_window_seconds += wall_seconds

            print(
                f"Total window: "
                f"{wall_seconds:.2f}s"
            )

            print(
                f"  split: "
                f"{timing['split_seconds']:.2f}s"
            )
            print(
                f"  fit: "
                f"{timing['fit_seconds']:.2f}s"
            )
            print(
                f"  validation prediction: "
                f"{timing['validation_prediction_seconds']:.2f}s"
            )
            print(
                f"  OOS prediction: "
                f"{timing['oos_prediction_seconds']:.2f}s"
            )
            print(
                f"  evaluation: "
                f"{timing['evaluation_seconds']:.2f}s"
            )
            print(
                f"  OOS row build: "
                f"{timing['oos_row_build_seconds']:.2f}s"
            )
            print(
                f"  models: "
                f"{len(results)}"
            )
            print(
                f"  OOS rows: "
                f"{len(oos_predictions):,}"
            )

    total_seconds = (
        perf_counter() - total_start
    )

    print()
    print("================================")
    print(
        f"WINDOW TOTAL: "
        f"{total_window_seconds:.2f}s"
    )
    print(
        f"TOTAL: "
        f"{total_seconds:.2f}s"
    )
    print("================================")


if __name__ == "__main__":
    main()
