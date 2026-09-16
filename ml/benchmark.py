"""Litet benchmark av Blankdiss ML-pipelinen."""

from __future__ import annotations

from time import perf_counter

from ml.config import TARGETS, WALK_FORWARD_WINDOWS
from ml.dataset import (
    load_features,
    prepare_feature_set,
    prepare_ml_data_from_feature_set,
)
from ml.walk_forward import train_window


BENCHMARK_TARGETS = {
    "up_5pct_5d",
    "down_5pct_5d",
}


def main() -> None:
    total_start = perf_counter()

    print("================================")
    print("Blankdiss ML benchmark")
    print("================================")

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

        for window in WALK_FORWARD_WINDOWS:
            print()
            print(
                f"Window: "
                f"{window.train_end} -> "
                f"{window.validation_end} -> "
                f"{window.test_end}"
            )

            start = perf_counter()

            results, oos_predictions, timing = train_window(
                data,
                y,
                target_feature_columns,
                window,
                task=target.task,
                direction=target.direction,
            )

            elapsed = perf_counter() - start

            print(f"Total window: {elapsed:.2f}s")
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
            print(f"  models: {len(results)}")
            print(
                f"  OOS rows: "
                f"{len(oos_predictions):,}"
            )

    total_elapsed = perf_counter() - total_start

    print()
    print("================================")
    print(f"TOTAL: {total_elapsed:.2f}s")
    print("================================")


if __name__ == "__main__":
    main()
