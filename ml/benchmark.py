"""Benchmark av Blankdiss ML-pipeline med Random Forest n_jobs=4."""

from __future__ import annotations

from time import perf_counter

from ml.config import TARGETS
from ml.dataset import (
    load_features,
    prepare_feature_set,
    prepare_ml_data_from_feature_set,
)
from ml.models import build_models
from ml.walk_forward import train_window


BENCHMARK_TARGETS = {
    "up_5pct_5d",
    "down_5pct_5d",
}

RF_N_JOBS = 4


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

    total_ml_seconds = 0.0

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
        print(f"Features: {len(target_feature_columns)}")

        target_start = perf_counter()

        for window in __import__(
            "ml.config",
            fromlist=["WALK_FORWARD_WINDOWS"],
        ).WALK_FORWARD_WINDOWS:
            print()
            print(
                f"Window: "
                f"{window.train_end} -> "
                f"{window.validation_end} -> "
                f"{window.test_end}"
            )

            start = perf_counter()

            results, oos_predictions, timing = train_window(
                data=data,
                y=y,
                feature_columns=target_feature_columns,
                target=target,
                window=window,
                random_state=42,
            )

            window_seconds = perf_counter() - start

            print(
                f"Total window: "
                f"{window_seconds:.2f}s"
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
                f"  models: {len(results)}"
            )
            print(
                f"  OOS rows: {len(oos_predictions):,}"
            )

        target_seconds = perf_counter() - target_start
        total_ml_seconds += target_seconds

        print()
        print(
            f"Target total: "
            f"{target_seconds:.2f}s"
        )

    total_seconds = perf_counter() - total_start

    print()
    print("================================")
    print(
        f"ML TOTAL: {total_ml_seconds:.2f}s"
    )
    print(
        f"TOTAL: {total_seconds:.2f}s"
    )
    print("================================")


if __name__ == "__main__":
    main()
