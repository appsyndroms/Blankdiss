"""Huvudprogram för iterativ Blankdiss ML-träning."""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from ml.config import (
    LATEST_RESULT_PATH,
    ML_OUTPUT_DIR,
    OOS_PREDICTIONS_PATH,
    PRICE_FEATURE_COLUMNS,
    RESULTS_PATH,
    RUNS_DIR,
    TARGETS,
    WALK_FORWARD_WINDOWS,
)
from ml.dataset import (
    dataset_summary,
    load_features,
    prepare_feature_set,
    prepare_ml_data_from_feature_set,
)
from ml.walk_forward import train_window


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
        set(PRICE_FEATURE_COLUMNS),
    ),
)


ABLATION_TARGETS = {
    "up_5pct_5d",
    "down_5pct_5d",
}


# GitHub-hosted ubuntu-latest har normalt 4 vCPU.
# Vi använder fyra parallella experiment och begränsar
# modellernas interna parallellism till en tråd.
MAX_PARALLEL_EXPERIMENTS = 4


def append_jsonl(
    path,
    records,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "a",
        encoding="utf-8",
    ) as handle:
        for record in records:
            handle.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )


def write_jsonl(
    path,
    records,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        for record in records:
            handle.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )


def prepare_feature_set_data(
    features,
    price_features,
):
    include_price_features = (
        price_features is not None
    )

    return prepare_feature_set(
        features,
        include_price_features,
    )


def prepare_feature_set(
    frame,
    target,
    price_features,
):
    """
    Bakåtkompatibel target-preparering från en redan cachead
    feature-set.

    Denna funktion används inte av den nya huvudloopen direkt,
    men behåller samma logik som tidigare kod för externa anrop.
    """
    include_price_features = (
        price_features is not None
    )

    (
        feature_set_data,
        feature_columns,
    ) = prepare_feature_set_data(
        frame,
        price_features,
    )

    (
        data,
        y,
        feature_columns,
    ) = prepare_ml_data_from_feature_set(
        feature_set_data,
        feature_columns,
        target,
    )

    return (
        data,
        y,
        feature_columns,
    )


def run_windows(
    data,
    y,
    feature_columns,
    task,
    direction,
):
    """
    Kör walk-forward-fönstren sekventiellt inom ett experiment.

    Parallellismen ligger på experimentnivå.
    """
    window_results = []

    for window in WALK_FORWARD_WINDOWS:
        window_results.append(
            train_window(
                data,
                y,
                feature_columns,
                window,
                task=task,
                direction=direction,
            )
        )

    return window_results


def _prepare_experiment(
    feature_set_data,
    feature_columns,
    feature_set_name,
    target,
):
    (
        data,
        y,
        feature_columns,
    ) = prepare_ml_data_from_feature_set(
        feature_set_data,
        feature_columns,
        target,
    )

    summary = dataset_summary(
        data,
        y,
        feature_columns,
        task=target.task,
    )

    window_results = run_windows(
        data,
        y,
        feature_columns,
        target.task,
        target.direction,
    )

    return {
        "feature_set_name": feature_set_name,
        "target": target,
        "data": data,
        "y": y,
        "feature_columns": feature_columns,
        "summary": summary,
        "window_results": window_results,
    }


def _run_experiment(
    feature_set_cache,
    feature_set_name,
    target,
):
    print(
        f"[START] {feature_set_name} / "
        f"{target.name}"
    )

    (
        feature_set_data,
        feature_columns,
    ) = feature_set_cache[
        feature_set_name
    ]

    result = _prepare_experiment(
        feature_set_data,
        feature_columns,
        feature_set_name,
        target,
    )

    print(
        f"[DONE]  {feature_set_name} / "
        f"{target.name}"
    )

    return result


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
    """
    Preparera varje feature-set exakt en gång.

    Tidigare gjordes motsvarande feature-preparering per
    feature-set/target-experiment.
    """
    cache = {}

    print()
    print(
        "================================"
    )
    print(
        "Förbereder feature-set cache"
    )
    print(
        "================================"
    )

    for (
        feature_set_name,
        price_features,
    ) in FEATURE_SETS:
        print(
            f"[CACHE] {feature_set_name}"
        )

        (
            feature_set_data,
            feature_columns,
        ) = prepare_feature_set(
            features,
            price_features is not None,
        )

        cache[feature_set_name] = (
            feature_set_data,
            feature_columns,
        )

        print(
            f"[CACHE] {feature_set_name}: "
            f"{len(feature_set_data):,} rader, "
            f"{len(feature_columns)} features"
        )

    return cache


def run_experiments_parallel(
    feature_set_cache,
    experiments,
):
    if len(experiments) <= 1:
        return [
            _run_experiment(
                feature_set_cache,
                feature_set_name,
                target,
            )
            for (
                feature_set_name,
                target,
            ) in experiments
        ]

    worker_count = min(
        MAX_PARALLEL_EXPERIMENTS,
        len(experiments),
    )

    print()
    print(
        "================================"
    )
    print(
        "Parallell ML-träning: "
        f"{worker_count} experiment samtidigt"
    )
    print(
        "Intern modellparallellism: "
        "1 tråd"
    )
    print(
        "================================"
    )

    with ThreadPoolExecutor(
        max_workers=worker_count,
        thread_name_prefix="blankdiss-experiment",
    ) as executor:
        futures = [
            executor.submit(
                _run_experiment,
                feature_set_cache,
                feature_set_name,
                target,
            )
            for (
                feature_set_name,
                target,
            ) in experiments
        ]

        return [
            future.result()
            for future in futures
        ]


def main() -> None:
    print("Blankdiss ML: startar.")
    print("================================")

    ML_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    RUNS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    features = load_features()

    print(
        "Features: "
        f"{len(features):,}"
    )

    # Preparera varje feature-set en gång.
    # Experimenten delar sedan dessa read-only DataFrames.
    feature_set_cache = (
        build_feature_set_cache(
            features
        )
    )

    experiments = build_experiment_list()

    print(
        "Experiment: "
        f"{len(experiments)}"
    )

    experiment_results = (
        run_experiments_parallel(
            feature_set_cache,
            experiments,
        )
    )

    all_results = []
    all_oos_predictions = []

    for experiment in experiment_results:
        feature_set_name = (
            experiment["feature_set_name"]
        )
        target = experiment["target"]
        summary = experiment["summary"]
        window_results = (
            experiment["window_results"]
        )

        if target.task == "classification":
            summary_text = (
                f"positiv rate "
                f"{summary['positive_rate']:.3f}"
            )
        else:
            summary_text = (
                f"target mean "
                f"{summary['target_mean']:.4f}"
            )

        print()
        print(
            "================================"
        )
        print(
            "Feature set: "
            f"{feature_set_name}"
        )
        print(
            "Target: "
            f"{target.name}"
        )
        print(
            "Dataset: "
            f"{summary['rows']:,} rader, "
            f"{summary['features']} features, "
            f"{summary_text}"
        )
        print(
            "================================"
        )

        for window, (
            results,
            oos_predictions,
        ) in zip(
            WALK_FORWARD_WINDOWS,
            window_results,
        ):
            print(
                "Window: "
                f"{window.train_end} -> "
                f"{window.validation_end} -> "
                f"{window.test_end}"
            )

            for result in results:
                record = {
                    "feature_set": (
                        feature_set_name
                    ),
                    "target": target.name,
                    "return_column": (
                        target.return_column
                    ),
                    "target_threshold": (
                        target.threshold
                    ),
                    "target_task": target.task,
                    "target_column": (
                        target.target_column
                    ),
                    "direction": (
                        target.direction
                    ),
                    "dataset_summary": summary,
                    "experiment": (
                        "price_feature_ablation"
                    ),
                    **result,
                }

                all_results.append(
                    record
                )

                if result[
                    "selected_for_oos"
                ]:
                    print(
                        "  VALDE MODELL: "
                        f"{result['model']} "
                        f"(score="
                        f"{result['validation_score']:.4f})"
                    )

            for prediction in oos_predictions:
                prediction[
                    "feature_set"
                ] = feature_set_name

                prediction[
                    "target"
                ] = target.name

                prediction[
                    "target_column"
                ] = target.target_column

                all_oos_predictions.append(
                    prediction
                )

    now = datetime.now(
        timezone.utc
    )

    run_id = now.strftime(
        "%Y%m%dT%H%M%SZ"
    )

    run_path = (
        RUNS_DIR
        / f"run_{run_id}.json"
    )

    run_document = {
        "run_id": run_id,
        "created_at": now.isoformat(),
        "experiment": (
            "price_feature_ablation"
        ),
        "feature_sets": [
            name
            for name, _ in FEATURE_SETS
        ],
        "ablation_targets": sorted(
            ABLATION_TARGETS
        ),
        "parallel_experiments": (
            MAX_PARALLEL_EXPERIMENTS
        ),
        "parallel_windows": 1,
        "feature_set_cache": True,
        "results": all_results,
        "oos_prediction_rows": (
            len(all_oos_predictions)
        ),
    }

    with run_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            run_document,
            handle,
            ensure_ascii=False,
            indent=2,
        )

    append_jsonl(
        RESULTS_PATH,
        all_results,
    )

    write_jsonl(
        OOS_PREDICTIONS_PATH,
        all_oos_predictions,
    )

    with LATEST_RESULT_PATH.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            run_document,
            handle,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print(
        "================================"
    )
    print(
        "Blankdiss ML: klart."
    )
    print(
        "Resultat: "
        f"{run_path}"
    )
    print(
        "OOS-prediktioner: "
        f"{len(all_oos_predictions):,}"
    )
    print(
        "Parallella experiment: "
        f"{MAX_PARALLEL_EXPERIMENTS}"
    )
    print(
        "Feature-set cache: aktiv"
    )
    print(
        "================================"
    )


if __name__ == "__main__":
    main()
