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
    prepare_ml_data,
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


def prepare_feature_set(
    frame,
    target,
    price_features,
):
    include_price_features = (
        price_features is not None
    )

    (
        data,
        y,
        base_feature_columns,
    ) = prepare_ml_data(
        frame,
        target,
        include_price_features,
    )

    if price_features is None:
        return (
            data,
            y,
            base_feature_columns,
        )

    fi_feature_columns = [
        column
        for column in base_feature_columns
        if column
        not in PRICE_FEATURE_COLUMNS
    ]

    selected_price_columns = [
        column
        for column in base_feature_columns
        if column in price_features
    ]

    selected_feature_columns = (
        fi_feature_columns
        + selected_price_columns
    )

    return (
        data,
        y,
        selected_feature_columns,
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

    Parallellismen ligger på experimentnivå. Det gör att flera
    oberoende feature-set/target-kombinationer kan använda CPU:n
    samtidigt utan att varje modell försöker använda alla kärnor.
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
    features,
    feature_set_name,
    price_features,
    target,
):
    (
        data,
        y,
        feature_columns,
    ) = prepare_feature_set(
        features,
        target,
        price_features,
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
    features,
    feature_set_name,
    price_features,
    target,
):
    print(
        f"[START] {feature_set_name} / "
        f"{target.name}"
    )

    result = _prepare_experiment(
        features,
        feature_set_name,
        price_features,
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
                    price_features,
                    target,
                )
            )

    return experiments


def run_experiments_parallel(
    features,
    experiments,
):
    if len(experiments) <= 1:
        return [
            _run_experiment(
                features,
                feature_set_name,
                price_features,
                target,
            )
            for (
                feature_set_name,
                price_features,
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
                features,
                feature_set_name,
                price_features,
                target,
            )
            for (
                feature_set_name,
                price_features,
                target,
            ) in experiments
        ]

        # Samma ordning som experiment-listan.
        # Resultaten blir därmed deterministiska.
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

    experiments = build_experiment_list()

    print(
        "Experiment: "
        f"{len(experiments)}"
    )

    experiment_results = (
        run_experiments_parallel(
            features,
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
        "================================"
    )


if __name__ == "__main__":
    main()
