"""Benchmark av Random Forest med olika CPU-parallellism."""

from __future__ import annotations

from time import perf_counter

import numpy as np

from ml.config import RANDOM_STATE, TARGETS, WALK_FORWARD_WINDOWS
from ml.dataset import (
    load_features,
    prepare_feature_set,
    prepare_ml_data_from_feature_set,
)
from ml.models import build_models


BENCHMARK_TARGETS = {
    "up_5pct_5d",
    "down_5pct_5d",
}

BENCHMARK_N_JOBS = (
    1,
    2,
    4,
    -1,
)

# Bara första walk-forward-fönstret.
BENCHMARK_WINDOW = WALK_FORWARD_WINDOWS[0]


def main() -> None:
    total_start = perf_counter()

    print("================================")
    print("Blankdiss Random Forest benchmark")
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

    for target in targets:
        print()
        print("--------------------------------")
        print(f"Target: {target.name}")
        print("--------------------------------")

        data, y, target_feature_columns = (
            prepare_ml_data_from_feature_set(
                feature_set_data,
                feature_columns,
                target,
            )
        )

        train_end = np.datetime64(
            BENCHMARK_WINDOW.train_end
        )
        validation_end = np.datetime64(
            BENCHMARK_WINDOW.validation_end
        )
        test_end = np.datetime64(
            BENCHMARK_WINDOW.test_end
        )

        train_mask = (
            data["snapshot_date"].values
            <= train_end
        )

        validation_mask = (
            (data["snapshot_date"].values > train_end)
            & (
                data["snapshot_date"].values
                <= validation_end
            )
        )

        test_mask = (
            (data["snapshot_date"].values > validation_end)
            & (
                data["snapshot_date"].values
                <= test_end
            )
        )

        train = data.loc[
            train_mask,
            target_feature_columns,
        ]

        validation = data.loc[
            validation_mask,
            target_feature_columns,
        ]

        y_train = y.loc[train_mask]
        y_validation = y.loc[validation_mask]

        print(
            f"Training rows: {len(train):,}"
        )
        print(
            f"Validation rows: {len(validation):,}"
        )
        print(
            f"Features: {len(target_feature_columns)}"
        )

        print()
        print(
            f"Window: "
            f"{BENCHMARK_WINDOW.train_end} -> "
            f"{BENCHMARK_WINDOW.validation_end} -> "
            f"{BENCHMARK_WINDOW.test_end}"
        )

        print()
        print("Random Forest CPU scaling")
        print("=========================")

        for n_jobs in BENCHMARK_N_JOBS:
            models = build_models(
                RANDOM_STATE,
                task=target.task,
            )

            model = models["random_forest"]

            model.set_params(
                model__n_jobs=n_jobs,
            )

            print()
            print(
                f"n_jobs={n_jobs}"
            )

            start = perf_counter()

            model.fit(
                train,
                y_train,
            )

            fit_seconds = (
                perf_counter()
                - start
            )

            print(
                f"  fit: "
                f"{fit_seconds:.3f}s"
            )

            start = perf_counter()

            predictions = model.predict_proba(
                validation
            )[:, 1]

            prediction_seconds = (
                perf_counter()
                - start
            )

            print(
                f"  validation prediction: "
                f"{prediction_seconds:.3f}s"
            )

            print(
                f"  total: "
                f"{fit_seconds + prediction_seconds:.3f}s"
            )

    total_seconds = (
        perf_counter()
        - total_start
    )

    print()
    print("================================")
    print(
        f"TOTAL: {total_seconds:.3f}s"
    )
    print("================================")


if __name__ == "__main__":
    main()
