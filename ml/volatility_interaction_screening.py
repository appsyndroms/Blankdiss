"""Screening av interaktioner mellan FI-shortning och prisvolatilitet.

Syfte:
- testa om volatility_20d interagerar med FI-signaler
- jämföra varje interaktion isolerat
- testa kombinationen av alla tre interaktioner
- mäta validation och faktisk OOS-ekonomi
- hålla beräkningskostnaden låg

Testade interaktioner:
- short_interest_pct * price_volatility_20d
- short_interest_delta_pp * price_volatility_20d
- short_interest_acceleration_pp * price_volatility_20d

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

BASE_PRICE_FEATURES = {
    "price_volatility_20d",
}

INTERACTION_FEATURES = {
    "short_interest_x_volatility": (
        "short_interest_pct",
        "price_volatility_20d",
    ),
    "short_interest_delta_x_volatility": (
        "short_interest_delta_pp",
        "price_volatility_20d",
    ),
    "short_interest_acceleration_x_volatility": (
        "short_interest_acceleration_pp",
        "price_volatility_20d",
    ),
}

FEATURE_SETS = (
    ("fi_plus_volatility_20d", ()),
    (
        "fi_plus_volatility_short_interest_x_volatility",
        ("short_interest_x_volatility",),
    ),
    (
        "fi_plus_volatility_delta_x_volatility",
        ("short_interest_delta_x_volatility",),
    ),
    (
        "fi_plus_volatility_acceleration_x_volatility",
        ("short_interest_acceleration_x_volatility",),
    ),
    (
        "fi_plus_volatility_all_interactions",
        (
            "short_interest_x_volatility",
            "short_interest_delta_x_volatility",
            "short_interest_acceleration_x_volatility",
        ),
    ),
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


def add_interactions(
    feature_set_data,
    interaction_names,
):
    data = feature_set_data.copy()

    for interaction_name in interaction_names:
        left_column, right_column = (
            INTERACTION_FEATURES[interaction_name]
        )

        if left_column not in data.columns:
            raise ValueError(
                f"Missing interaction source column: "
                f"{left_column}"
            )

        if right_column not in data.columns:
            raise ValueError(
                f"Missing interaction source column: "
                f"{right_column}"
            )

        data[interaction_name] = (
            data[left_column].astype(float)
            * data[right_column].astype(float)
        )

    return data


def build_feature_sets(features):
    feature_sets = {}

    (
        base_data,
        base_columns,
    ) = prepare_feature_set(
        features,
        include_price_features=True,
        price_features=BASE_PRICE_FEATURES,
    )

    for feature_set_name, interaction_names in FEATURE_SETS:
        data = add_interactions(
            base_data,
            interaction_names,
        )

        columns = list(base_columns)

        for interaction_name in interaction_names:
            if interaction_name not in columns:
                columns.append(interaction_name)

        feature_sets[feature_set_name] = (
            data,
            columns,
        )

    return feature_sets


def print_feature_set_qc(feature_sets):
    print()
    print("================================")
    print("FEATURE SET QC")
    print("================================")

    for feature_set_name, interaction_names in FEATURE_SETS:
        _, feature_columns = feature_sets[
            feature_set_name
        ]

        print(
            f"{feature_set_name}: "
            f"{len(feature_columns)} features"
        )

        if interaction_names:
            print(
                "  Interactions: "
                + ", ".join(interaction_names)
            )
        else:
            print("  Interactions: none")


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

        window_validation_scores = [
            float(result["validation_score"])
            for result in results
            if result.get("validation_score")
            is not None
        ]

        validation_scores.extend(
            window_validation_scores
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
                "validation_score": (
                    sum(window_validation_scores)
                    / len(window_validation_scores)
                    if window_validation_scores
                    else 0.0
                ),
                "economic_oos": list(
                    oos_predictions
                ),
                "timing": dict(timing),
            }
        )

    total_seconds = (
        perf_counter() - experiment_start
    )

    return {
        "feature_set": feature_set_name,
        "feature_count": len(feature_columns),
        "row_count": len(data),
        "validation_score": (
            sum(validation_scores)
            / len(validation_scores)
            if validation_scores
            else 0.0
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

            print(
                f"    Validation score: "
                f"{window['validation_score']:.6f}"
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

        print(
            f"  Average validation score: "
            f"{result['validation_score']:.6f}"
        )

        print("  Combined economic OOS:")

        print_economic_results(
            result["economic_oos"],
            indent="    ",
        )


def print_comparison(results):
    print()
    print("================================")
    print("INTERACTION COMPARISON")
    print("================================")

    print(
        "Feature set | val | top1 event | "
        "lift | top1 mean | total | fit"
    )

    print("-" * 100)

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

        lift = top1["lift"]

        print(
            f"{result['feature_set']:<48}"
            f"{result['validation_score']:.6f}     "
            f"{top1['event_rate']:.4f}  "
            f"{lift:.2f}x   "
            f"{top1['mean_return']:+.4%}    "
            f"{result['total_seconds']:.2f}s    "
            f"{result['fit_seconds']:.2f}s"
        )


def main():
    print("=" * 100)
    print("BLANKDISS VOLATILITY INTERACTION SCREENING")
    print("=" * 100)

    print(f"RF trees: {BENCHMARK_TREES}")
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

    def build_models_for_benchmark(
        random_state: int,
        task: str = "classification",
    ):
        return build_benchmark_models(
            random_state,
            task=task,
        )

    walk_forward.build_models = (
        build_models_for_benchmark
    )

    results = []

    for (
        feature_set_name,
        _interaction_names,
    ) in FEATURE_SETS:
        print()
        print("=" * 100)
        print(
            f"RUN: {feature_set_name}"
        )
        print("=" * 100)

        (
            feature_set_data,
            feature_columns,
        ) = feature_sets[
            feature_set_name
        ]

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

    print()
    print("=" * 100)
    print("SCREENING COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()
