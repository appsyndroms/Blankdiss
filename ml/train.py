"""Huvudprogram för iterativ Blankdiss ML-träning."""
from __future__ import annotations

import json
from datetime import datetime

import pandas as pd

from ml.config import (
    LATEST_RESULT_PATH,
    ML_OUTPUT_DIR,
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

from ml.walk_forward import (
    train_window,
)


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


def main() -> None:
    print(
        "Blankdiss ML: startar."
    )

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

    all_results = []

    for target in TARGETS:
        print(
            "Target: "
            f"{target.name}"
        )

        (
            data,
            y,
            feature_columns,
        ) = prepare_ml_data(
            features,
            target,
        )

        data = data.copy()

        data["target_return"] = pd.to_numeric(
            features.loc[
                data.index,
                target.return_column,
            ],
            errors="coerce",
        )

        summary = dataset_summary(
            data,
            y,
            feature_columns,
        )

        print(
            "Dataset: "
            f"{summary['rows']:,} rader, "
            f"{summary['features']} features, "
            f"positiv rate "
            f"{summary['positive_rate']:.3f}"
        )

        for window in WALK_FORWARD_WINDOWS:
            print(
                "Window: "
                f"{window.train_end} -> "
                f"{window.validation_end} -> "
                f"{window.test_end}"
            )

            results = train_window(
                data,
                y,
                feature_columns,
                window,
            )

            for result in results:
                record = {
                    "target": target.name,
                    "return_column": (
                        target.return_column
                    ),
                    "dataset_summary": summary,
                    **result,
                }

                all_results.append(
                    record
                )

                print(
                    "  "
                    f"{result['model']}: "
                    f"validation AUC="
                    f"{result['validation_roc_auc']:.4f}, "
                    f"test AUC="
                    f"{result['test']['roc_auc']:.4f}"
                )

    run_id = datetime.utcnow().strftime(
        "%Y%m%dT%H%M%SZ"
    )

    run_path = (
        RUNS_DIR
        / f"run_{run_id}.json"
    )

    run_document = {
        "run_id": run_id,
        "created_at": (
            datetime.utcnow()
            .isoformat()
            + "Z"
        ),
        "results": all_results,
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

    print(
        "Blankdiss ML: klart."
    )

    print(
        "Resultat: "
        f"{run_path}"
    )


if __name__ == "__main__":
    main()
