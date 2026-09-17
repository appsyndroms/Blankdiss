"""Jämförelse av FI-only, volatility-only och FI + volatility.

Syfte:
- avgöra hur mycket av signalen som kommer från volatility_20d
- avgöra om FI tillför information ovanpå volatility
- jämföra validation och faktisk OOS-ekonomi
- hålla beräkningskostnaden låg

Testade feature sets:
- FI-only
- volatility-only
- FI + volatility_20d

Ingen produktionsfil ändras.
"""

from __future__ import annotations

from time import perf_counter

import numpy as np

import ml.walk_forward as walk_forward
from ml.config import TARGETS, WALK_FORWARD_WINDOWS
from ml.dataset import (
    load_features,
    prepare_feature_set,
    prepare_ml_data_from_feature_set,
)
from ml.models import build_models as original_build_models


BENCHMARK_TREES = 100
ECONOMIC_TARGET = "down_5pct_5d"

ECONOMIC_TOP_FRACTIONS = (
    0.001,
    0.005,
    0.01,
    0.02,
    0.05,
)


FEATURE_SET_CONFIG = (
    ("fi_only", "fi"),
    ("volatility_only", "volatility"),
    ("fi_plus_volatility_20d", "fi_plus_volatility"),
)


def build_benchmark_models(
    random_state: int,
    task: str = "classification",
):
    models = original_build_models(
        random_state,
        task=task,
    )

    random_forest = models.get("random_forest")

    if random_forest is not None:
        random_forest.set_params(
            model__n_estimators=BENCHMARK_TREES,
            model__n_jobs=1,
        )

    random_forest_regression = models.get(
        "random_forest_regression"
    )

    if random_forest_regression is not None:
        random_forest_regression.set_params(
            model__n_estimators=BENCHMARK_TREES,
            model__n_jobs=1,
        )

    return models


def build_feature_sets(features):
    feature_sets = {}

    fi_data, fi_columns = prepare_feature_set(
        features,
        include_price_features=False,
    )

    price_data, price_columns = prepare_feature_set(
        features,
        include_price_features=True,
        price_features={"price_volatility_20d"},
    )

    if "price_volatility_20d" not in price_columns:
        raise ValueError(
            "price_volatility_20d saknas efter feature preparation."
        )

    volatility_data = price_data[
        [
            "snapshot_date",
            "security_key",
            "forward_return_1d",
            "forward_return_5d",
            "forward_return_20d",
            "forward_return_60d",
            "min_return_5d",
            "max_return_5d",
            "price_volatility_20d",
        ]
    ].copy()

    volatility_columns = [
        "price_volatility_20d"
    ]

    fi_plus_volatility_data = price_data.copy()
    fi_plus_volatility_columns = list(price_columns)

    feature_sets["fi_only"] = (
        fi_data,
        fi_columns,
    )

    feature_sets["volatility_only"] = (
        volatility_data,
        volatility_columns,
    )

    feature_sets["fi_plus_volatility_20d"] = (
        fi_plus_volatility_data,
        fi_plus_volatility_columns,
    )

    return feature_sets


def print_feature_set_qc(feature_sets):
    print()
    print("================================")
    print("FEATURE SET QC")
    print("================================")

    for feature_set_name, _ in FEATURE_SET_CONFIG:
        _, feature_columns = feature_sets[
            feature_set_name
        ]

        print(
            f"{feature_set_name}: "
            f"{len(feature_columns)} features"
        )

        print(
            "  Features: "
            + ", ".join(feature_columns)
        )


def calculate_economic_metrics(
    oos_predictions,
):
    if not oos_predictions:
        return {
            "rows": 0,
            "baseline_event_rate": None,
            "baseline_mean_return": None,
            "top_fraction": [],
        }

    scores = np.asarray(
        [
            row["score"]
            for row in oos_predictions
        ],
        dtype=float,
    )

    returns = np.asarray(
        [
            row["target_return"]
            for row in oos_predictions
        ],
        dtype=float,
    )

    events = (
        returns <= -0.05
    ).astype(float)

    valid = (
        np.isfinite(scores)
        & np.isfinite(returns)
    )

    scores = scores[valid]
    returns = returns[valid]
    events = events[valid]

    rows = len(scores)

    if rows == 0:
        return {
            "rows": 0,
            "baseline_event_rate": None,
            "baseline_mean_return": None,
            "top_fraction": [],
        }

    order = np.argsort(
        -scores,
        kind="mergesort",
    )

    baseline_event_rate = float(
        events.mean()
    )

    baseline_mean_return = float(
        returns.mean()
    )

    top_fraction = []

    for fraction in ECONOMIC_TOP_FRACTIONS:
        count = max(
            1,
            int(
                np.ceil(
                    rows * fraction
                )
            ),
        )

        selected = order[:count]

        selected_returns = returns[selected]
        selected_events = events[selected]

        event_rate = float(
            selected_events.mean()
        )

        mean_return = float(
            selected_returns.mean()
        )

        median_return = float(
            np.median(
                selected_returns
            )
        )

        lift = (
            event_rate / baseline_event_rate
            if baseline_event_rate > 0
            else None
        )

        top_fraction.append(
            {
                "fraction": fraction,
                "rows": count,
                "event_rate": event_rate,
                "lift": lift,
                "mean_return": mean_return,
                "median_return": median_return,
            }
        )

    return {
        "rows": rows,
        "baseline_event_rate": baseline_event_rate,
        "baseline_mean_return": baseline_mean_return,
        "top_fraction": top_fraction,
    }


def print_economic_results(
    oos_predictions,
    indent="  ",
):
    metrics = calculate_economic_metrics(
        oos_predictions
    )

    print(
        f"{indent}Rows: "
        f"{metrics['rows']:,}"
    )

    if metrics["baseline_event_rate"] is not None:
        print(
            f"{indent}Baseline event rate: "
            f"{metrics['baseline_event_rate']:.4f}"
        )

    if metrics["baseline_mean_return"] is not None:
        print(
            f"{indent}Baseline mean return: "
            f"{metrics['baseline_mean_return']:+.4%}"
        )

    for result in metrics["top_fraction"]:
        fraction = result["fraction"]

        if fraction < 0.01:
            fraction_text = f"{fraction:.1%}"
        else:
            fraction_text = f"{fraction:.0%}"

        lift = result["lift"]

        lift_text = (
            f"{lift:.2f}x"
            if lift is not None
            else "n/a"
        )

        print(
            f"{indent}Top {fraction_text:>5}: "
            f"n={result['rows']:,} "
            f"event={result['event_rate']:.4f} "
            f"lift={lift_text:>6} "
            f"mean={result['mean_return']:+.4%} "
            f"median={result['median_return']:+.4%}"
        )


def selected_validation_score(results):
    selected = [
        result
        for result in results
        if result.get("selected_for_oos")
    ]

    if not selected:
        return None, None

    selected.sort(
        key=lambda result: result["validation_score"],
        reverse=True,
    )

    result = selected[0]

    return (
        float(result["validation_score"]),
        result["model"],
    )


def run_one_experiment(
    feature_set_name,
    feature_set_data,
    feature_columns,
    target,
):
    experiment_start = perf_counter()

    data, y, feature_columns = (
        prepare_ml_data_from_feature_set(
            feature_set_data,
            feature_columns,
            target,
        )
    )

    validation_scores = []
    economic_oos = []
    window_results = []

    for window_index, window in enumerate(
        WALK_FORWARD_WINDOWS,
        start=1,
    ):
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

        validation_score, selected_model = (
            selected_validation_score(results)
        )

        if validation_score is not None:
            validation_scores.append(
                validation_score
            )

        economic_oos.extend(
            oos_predictions
        )

        window_results.append(
            {
                "window_index": window_index,
                "train_end": window.train_end,
                "validation_end": window.validation_end,
                "test_end": window.test_end,
                "validation_score": validation_score,
                "selected_model": selected_model,
                "economic_oos": list(
                    oos_predictions
                ),
                "timing": dict(timing),
            }
        )

    total_seconds = (
        perf_counter()
        - experiment_start
    )

    return {
        "feature_set": feature_set_name,
        "feature_count": len(feature_columns),
        "row_count": len(data),
        "validation_score": (
            sum(validation_scores)
            / len(validation_scores)
            if validation_scores
            else None
        ),
        "total_seconds": total_seconds,
        "fit_seconds": sum(
            result["timing"].get(
                "fit_seconds",
                0.0,
            )
            for result in window_results
        ),
        "economic_oos": economic_oos,
        "window_results": window_results,
    }


def print_window_results(results):
    print()
    print("================================")
    print("RESULTAT PER WALK-FORWARD-FÖNSTER")
    print("================================")

    for result in results:
        print()
        print(result["feature_set"])

        for window in result["window_results"]:
            print()
            print(
                f"  Window "
                f"{window['window_index']}"
            )

            print(
                f"    Train <= "
                f"{window['train_end']}"
            )

            print(
                f"    Validation <= "
                f"{window['validation_end']}"
            )

            print(
                f"    Test <= "
                f"{window['test_end']}"
            )

            if window["validation_score"] is not None:
                print(
                    f"    Selected model: "
                    f"{window['selected_model']}"
                )

                print(
                    f"    Selected validation AUC: "
                    f"{window['validation_score']:.6f}"
                )
            else:
                print(
                    "    Selected model: n/a"
                )

            print(
                f"    Time: "
                f"{window['timing'].get('total_window_seconds', 0.0):.2f}s"
            )

            print("    Economic OOS:")

            print_economic_results(
                window["economic_oos"],
                indent="      ",
            )


def print_combined_results(results):
    print()
    print("================================")
    print("RESULTAT PER FEATURE SET")
    print("================================")

    for result in results:
        print()
        print(result["feature_set"])

        print(
            f"  Features: "
            f"{result['feature_count']}"
        )

        print(
            f"  Rows: "
            f"{result['row_count']:,}"
        )

        print(
            f"  Walk-forward windows: "
            f"{len(result['window_results'])}"
        )

        print(
            f"  Total: "
            f"{result['total_seconds']:.2f}s"
        )

        print(
            f"  Fit: "
            f"{result['fit_seconds']:.2f}s"
        )

        if result["validation_score"] is not None:
            print(
                f"  Average selected-model "
                f"validation AUC: "
                f"{result['validation_score']:.6f}"
            )
        else:
            print(
                "  Average selected-model "
                "validation AUC: n/a"
            )

        print("  Selected models:")

        for window in result["window_results"]:
            print(
                f"    Window "
                f"{window['window_index']}: "
                f"{window['selected_model']}"
            )

        print("  Combined economic OOS:")

        print_economic_results(
            result["economic_oos"],
            indent="    ",
        )


def print_comparison(results):
    print()
    print("================================")
    print("FI vs VOLATILITY COMPARISON")
    print("================================")

    print(
        "Feature set | val AUC | top1 event | "
        "lift | top1 mean | total | fit"
    )

    print("-" * 105)

    for result in results:
        metrics = calculate_economic_metrics(
            result["economic_oos"]
        )

        top1 = next(
            (
                item
                for item in metrics["top_fraction"]
                if item["fraction"] == 0.01
            ),
            None,
        )

        if top1 is None:
            continue

        validation_score = (
            result["validation_score"]
            if result["validation_score"] is not None
            else float("nan")
        )

        lift = top1["lift"]

        print(
            f"{result['feature_set']:<28}"
            f"{validation_score:.6f}     "
            f"{top1['event_rate']:.4f}  "
            f"{lift:.2f}x   "
            f"{top1['mean_return']:+.4%}    "
            f"{result['total_seconds']:.2f}s    "
            f"{result['fit_seconds']:.2f}s"
        )


def print_key_deltas(results):
    lookup = {
        result["feature_set"]: result
        for result in results
    }

    fi = lookup["fi_only"]
    volatility = lookup["volatility_only"]
    combined = lookup["fi_plus_volatility_20d"]

    print()
    print("================================")
    print("KEY DELTAS")
    print("================================")

    if (
        fi["validation_score"] is not None
        and volatility["validation_score"] is not None
    ):
        print(
            "VOL vs FI validation AUC: "
            f"{volatility['validation_score'] "
            f"- fi['validation_score']:+.6f}"
        )

    if (
        volatility["validation_score"] is not None
        and combined["validation_score"] is not None
    ):
        print(
            "FI+VOL vs VOL validation AUC: "
            f"{combined['validation_score'] "
            f"- volatility['validation_score']:+.6f}"
        )

    fi_metrics = calculate_economic_metrics(
        fi["economic_oos"]
    )

    volatility_metrics = calculate_economic_metrics(
        volatility["economic_oos"]
    )

    combined_metrics = calculate_economic_metrics(
        combined["economic_oos"]
    )

    def top1(metrics):
        return next(
            (
                item
                for item in metrics["top_fraction"]
                if item["fraction"] == 0.01
            ),
            None,
        )

    fi_top1 = top1(fi_metrics)
    volatility_top1 = top1(volatility_metrics)
    combined_top1 = top1(combined_metrics)

    if (
        fi_top1 is not None
        and volatility_top1 is not None
    ):
        print(
            "VOL vs FI top1 event rate: "
            f"{volatility_top1['event_rate'] "
            f"- fi_top1['event_rate']:+.4f}"
        )

        print(
            "VOL vs FI top1 mean return: "
            f"{volatility_top1['mean_return'] "
            f"- fi_top1['mean_return']:+.4%}"
        )

    if (
        volatility_top1 is not None
        and combined_top1 is not None
    ):
        print(
            "FI+VOL vs VOL top1 event rate: "
            f"{combined_top1['event_rate'] "
            f"- volatility_top1['event_rate']:+.4f}"
        )

        print(
            "FI+VOL vs VOL top1 mean return: "
            f"{combined_top1['mean_return'] "
            f"- volatility_top1['mean_return']:+.4%}"
        )


def main():
    print("=" * 100)
    print("BLANKDISS VOLATILITY VS FI SCREENING")
    print("=" * 100)

    print(
        f"RF trees: {BENCHMARK_TREES}"
    )

    print("RF n_jobs: 1")

    print(
        f"Walk-forward windows: "
        f"{len(WALK_FORWARD_WINDOWS)}"
    )

    print(
        f"Economic target: "
        f"{ECONOMIC_TARGET}"
    )

    print()
    print("Loading feature data...")

    features = load_features()

    print(
        f"Feature rows: "
        f"{len(features):,}"
    )

    target = next(
        target
        for target in TARGETS
        if target.name == ECONOMIC_TARGET
    )

    print()
    print("Building feature sets...")

    feature_sets = build_feature_sets(
        features
    )

    print_feature_set_qc(
        feature_sets
    )

    # Patch walk_forward only for this diagnostic.
    walk_forward.build_models = (
        build_benchmark_models
    )

    results = []

    for feature_set_name, _ in FEATURE_SET_CONFIG:
        (
            feature_set_data,
            feature_columns,
        ) = feature_sets[
            feature_set_name
        ]

        print()
        print("=" * 96)
        print(
            f"RUN: {feature_set_name}"
        )
        print("=" * 96)

        result = run_one_experiment(
            feature_set_name,
            feature_set_data,
            feature_columns,
            target,
        )

        results.append(result)

    print_window_results(
        results
    )

    print_combined_results(
        results
    )

    print_comparison(
        results
    )

    print_key_deltas(
        results
    )

    print()
    print("=" * 100)
    print("SCREENING COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()
