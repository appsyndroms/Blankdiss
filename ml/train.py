"""Huvudprogram för iterativ Blankdiss ML-träning."""
from __future__ import annotations
import json
from datetime import datetime, timezone
from ml.config import (
    LATEST_RESULT_PATH,
    ML_OUTPUT_DIR,
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
from ml.walk_forward import (
    train_window,
)
# ---------------------------------------------------------
# Feature-set för ablationsexperimentet.
#
# FI-only:
#   Baseline utan prisinformation.
#
# Enskilda prisfeatures:
#   Används för att se vilken prisfeature som faktiskt
#   står för signalen.
#
# Alla prisfeatures:
#   Full FI + pris-baseline.
# ---------------------------------------------------------
FEATURE_SETS = (
    (
        "fi_only",
        None,
    ),
    (
        "fi_plus_price_return_5d",
        {
            "price_return_5d",
        },
    ),
    (
        "fi_plus_price_return_20d",
        {
            "price_return_20d",
        },
    ),
    (
        "fi_plus_price_return_60d",
        {
            "price_return_60d",
        },
    ),
    (
        "fi_plus_price_volatility_20d",
        {
            "price_volatility_20d",
        },
    ),
    (
        "fi_plus_price_distance_from_20d_high",
        {
            "price_distance_from_20d_high",
        },
    ),
    (
        "fi_plus_price_distance_from_60d_high",
        {
            "price_distance_from_60d_high",
        },
    ),
    (
        "fi_plus_all_price",
        set(
            PRICE_FEATURE_COLUMNS
        ),
    ),
)
# ---------------------------------------------------------
# De två targets där prisfeatures redan har visat tydlig
# och stabil signal:
#
#   up_5pct_5d
#   down_5pct_5d
#
# För dessa kör vi varje prisfeature separat.
#
# För övriga targets kör vi bara:
#
#   FI-only
#   FI + alla prisfeatures
#
# Det ger ett fokuserat ablationsexperiment utan att
# multiplicera hela ML-körningen i onödan.
# ---------------------------------------------------------
ABLATION_TARGETS = {
    "up_5pct_5d",
    "down_5pct_5d",
}
def append_jsonl(
    path,
    records,
) -> None:
    """
    Appendar resultat till en JSONL-fil.
    """
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
def prepare_feature_set(
    frame,
    target,
    price_features,
):
    """
    Bygger ett ML-dataset med exakt vald uppsättning
    prisfeatures.
    prepare_ml_data() ansvarar fortfarande för:
    - target
    - leakage-skydd
    - numeriska features
    - NaN/inf-hantering
    - FI-only kontra FI+pris
    Här begränsar vi därefter prisdelen till exakt de
    features som experimentet vill testa.
    """
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
    # FI-only behöver ingen ytterligare filtrering.
    if price_features is None:
        return (
            data,
            y,
            base_feature_columns,
        )
    # Identifiera vilka kolumner som är FI-features.
    fi_feature_columns = [
        column
        for column in base_feature_columns
        if column
        not in PRICE_FEATURE_COLUMNS
    ]
    # Behåll endast de prisfeatures som hör till
    # just detta experiment.
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
def main() -> None:
    print(
        "Blankdiss ML: startar."
    )
    print(
        "Experiment:"
    )
    print(
        "  FI-only"
    )
    print(
        "  FI + en prisfeature i taget"
    )
    print(
        "  FI + alla prisfeatures"
    )
    print(
        "================================"
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
    for (
        feature_set_name,
        price_features,
    ) in FEATURE_SETS:
        print(
            "\n"
            "================================"
        )
        print(
            "Feature set: "
            f"{feature_set_name}"
        )
        print(
            "================================"
        )
        for target in TARGETS:
            # -------------------------------------------------
            # Enskilda prisfeatures testas endast på de två
            # targets där vi redan har sett tydlig pris-signal.
            #
            # FI-only och FI + alla prisfeatures körs däremot
            # för samtliga targets så att vi behåller den
            # breda baseline-serien.
            # -------------------------------------------------
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
            print(
                "\n"
                "Target: "
                f"{target.name}"
            )
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
                        "feature_set": (
                            feature_set_name
                        ),
                        "target": (
                            target.name
                        ),
                        "return_column": (
                            target.return_column
                        ),
                        "target_threshold": (
                            target.threshold
                        ),
                        "dataset_summary": (
                            summary
                        ),
                        "experiment": (
                            "price_feature_ablation"
                        ),
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
    # ---------------------------------------------------------
    # Spara körningen som ett separat run-dokument.
    # ---------------------------------------------------------
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
        "run_id": (
            run_id
        ),
        "created_at": (
            now.isoformat()
        ),
        "experiment": (
            "price_feature_ablation"
        ),
        "feature_sets": [
            name
            for name, _
            in FEATURE_SETS
        ],
        "ablation_targets": sorted(
            ABLATION_TARGETS
        ),
        "results": (
            all_results
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
    # ---------------------------------------------------------
    # Behåll befintlig historik i ml_results.jsonl.
    # ---------------------------------------------------------
    append_jsonl(
        RESULTS_PATH,
        all_results,
    )
    # ---------------------------------------------------------
    # latest_run.json pekar alltid på senaste körningen.
    # ---------------------------------------------------------
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
        "\n"
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
        "================================"
    )
if __name__ == "__main__":
    main()
