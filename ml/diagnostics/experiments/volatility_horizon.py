from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from analysis.feature_config import PRICE_DIR
from analysis.feature_prices import (
    find_price_files,
    load_prices,
)
from ml.config import RANDOM_STATE, TARGETS
from ml.dataset import (
    get_feature_columns,
)
from ml.models import build_models
from ml.walk_forward import train_window

from .base import ExperimentResult


BENCHMARK_TREES = 100

ECONOMIC_FRACTIONS = (
    0.001,
    0.005,
    0.01,
    0.02,
    0.05,
)

FEATURE_SET_CONFIG = (
    (
        "volatility_20d",
        (
            "price_volatility_20d",
        ),
    ),
    (
        "volatility_relative_20d_60d",
        (
            "volatility_relative_20d_60d",
        ),
    ),
    (
        "volatility_change_20d_60d",
        (
            "volatility_change_20d_60d",
        ),
    ),
    (
        "volatility_20d_plus_relative",
        (
            "price_volatility_20d",
            "volatility_relative_20d_60d",
        ),
    ),
    (
        "volatility_20d_plus_change",
        (
            "price_volatility_20d",
            "volatility_change_20d_60d",
        ),
    ),
    (
        "volatility_20d_relative_plus_change",
        (
            "price_volatility_20d",
            "volatility_relative_20d_60d",
            "volatility_change_20d_60d",
        ),
    ),
)


def _find_target(target_name: str):
    for target in TARGETS:
        if target.name == target_name:
            return target

    raise ValueError(
        f"Okänd target: {target_name}"
    )


def _build_models() -> dict[str, Any]:
    models = build_models(
        RANDOM_STATE,
        task="classification",
    )

    random_forest = models.get(
        "random_forest"
    )

    if random_forest is not None:
        random_forest.set_params(
            model__n_estimators=BENCHMARK_TREES,
            model__n_jobs=1,
        )

    return models


@lru_cache(maxsize=1)
def _load_volatility_60d_lookup() -> pd.DataFrame:
    print()
    print(
        "Loading raw price data for 60d volatility..."
    )

    price_files = find_price_files(
        PRICE_DIR
    )

    print(
        f"Prisfiler: {len(price_files)}"
    )

    prices = load_prices(
        price_files
    )

    print(
        f"Prisrader: {len(prices):,}"
    )

    required = {
        "yahoo_symbol",
        "date",
        "close",
    }

    missing = sorted(
        required
        - set(prices.columns)
    )

    if missing:
        raise ValueError(
            "Prisdata saknar kolumner: "
            + ", ".join(missing)
        )

    prices = prices[
        [
            "yahoo_symbol",
            "date",
            "close",
        ]
    ].copy()

    prices["date"] = pd.to_datetime(
        prices["date"],
        errors="coerce",
    )

    prices["close"] = pd.to_numeric(
        prices["close"],
        errors="coerce",
    )

    prices = prices.loc[
        prices["date"].notna()
        & prices["close"].notna()
        & np.isfinite(prices["close"])
        & (prices["close"] > 0)
    ].copy()

    prices["yahoo_symbol"] = (
        prices["yahoo_symbol"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    prices = prices.sort_values(
        [
            "yahoo_symbol",
            "date",
        ],
        kind="mergesort",
    )

    prices["daily_return"] = (
        prices.groupby(
            "yahoo_symbol",
            sort=False,
        )["close"]
        .pct_change()
    )

    prices["volatility_60d"] = (
        prices.groupby(
            "yahoo_symbol",
            sort=False,
        )["daily_return"]
        .transform(
            lambda values: (
                values
                .rolling(
                    window=60,
                    min_periods=60,
                )
                .std()
            )
        )
    )

    lookup = prices[
        [
            "yahoo_symbol",
            "date",
            "volatility_60d",
        ]
    ].rename(
        columns={
            "date": "price_date",
        }
    )

    return lookup


def _add_horizon_features(
    features: pd.DataFrame,
) -> pd.DataFrame:
    required = {
        "yahoo_symbol",
        "price_date",
        "price_volatility_20d",
    }

    missing = sorted(
        required
        - set(features.columns)
    )

    if missing:
        raise ValueError(
            "Feature-data saknar kolumner för "
            "volatilitetshorisont: "
            + ", ".join(missing)
        )

    result = features.copy()

    result["price_date"] = pd.to_datetime(
        result["price_date"],
        errors="coerce",
    )

    result["yahoo_symbol"] = (
        result["yahoo_symbol"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    lookup = _load_volatility_60d_lookup().copy()

    lookup["price_date"] = pd.to_datetime(
        lookup["price_date"],
        errors="coerce",
    )

    lookup["yahoo_symbol"] = (
        lookup["yahoo_symbol"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    result = result.merge(
        lookup,
        on=[
            "yahoo_symbol",
            "price_date",
        ],
        how="left",
        sort=False,
        validate="many_to_one",
    )

    result["volatility_relative_20d_60d"] = (
        result["price_volatility_20d"]
        / result["volatility_60d"]
    )

    result["volatility_change_20d_60d"] = (
        result["price_volatility_20d"]
        - result["volatility_60d"]
    )

    for column in (
        "volatility_60d",
        "volatility_relative_20d_60d",
        "volatility_change_20d_60d",
    ):
        result[column] = result[column].replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )

    matched = result[
        "volatility_60d"
    ].notna()

    print()
    print(
        "VOLATILITY HORIZON QC"
    )
    print(
        f"Feature rows: {len(result):,}"
    )
    print(
        f"Rows with 60d volatility: "
        f"{matched.sum():,}"
    )
    print(
        f"Rows without 60d volatility: "
        f"{(~matched).sum():,}"
    )

    return result


def _economic_metrics(
    predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    if not predictions:
        return {
            "rows": 0,
            "baseline_event_rate": None,
            "baseline_mean_return": None,
            "top_fraction": [],
        }

    scores = np.asarray(
        [
            row["score"]
            for row in predictions
        ],
        dtype=float,
    )

    returns = np.asarray(
        [
            row["target_return"]
            for row in predictions
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

    if len(scores) == 0:
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

    top_fraction = []

    for fraction in ECONOMIC_FRACTIONS:
        count = max(
            1,
            int(
                np.ceil(
                    len(scores)
                    * fraction
                )
            ),
        )

        selected = order[:count]

        selected_events = (
            events[selected]
        )

        selected_returns = (
            returns[selected]
        )

        event_rate = float(
            selected_events.mean()
        )

        lift = (
            event_rate
            / baseline_event_rate
            if baseline_event_rate > 0
            else np.nan
        )

        top_fraction.append(
            {
                "fraction": fraction,
                "n": count,
                "event_rate": event_rate,
                "lift": lift,
                "mean_return": float(
                    selected_returns.mean()
                ),
                "median_return": float(
                    np.median(
                        selected_returns
                    )
                ),
            }
        )

    return {
        "rows": int(len(scores)),
        "baseline_event_rate": (
            baseline_event_rate
        ),
        "baseline_mean_return": float(
            returns.mean()
        ),
        "top_fraction": top_fraction,
    }


def _run_feature_set(
    feature_set_name: str,
    feature_data: pd.DataFrame,
    feature_columns: list[str],
    target,
    context,
) -> dict[str, Any]:
    from ml.dataset import (
        prepare_ml_data_from_feature_set,
    )

    data, y, feature_columns = (
        prepare_ml_data_from_feature_set(
            feature_data,
            feature_columns,
            target,
        )
    )

    (
        results,
        oos_predictions,
        timing,
    ) = train_window(
        data,
        y,
        feature_columns,
        context,
        task=target.task,
        direction=target.direction,
    )

    selected = [
        result
        for result in results
        if result.get(
            "selected_for_oos"
        )
    ]

    if selected:
        validation_auc = float(
            selected[0][
                "validation_score"
            ]
        )

        selected_model = selected[0][
            "model"
        ]
    else:
        validation_auc = np.nan
        selected_model = None

    economic = _economic_metrics(
        oos_predictions
    )

    return {
        "feature_set": feature_set_name,
        "feature_count": len(
            feature_columns
        ),
        "row_count": len(data),
        "validation_auc": validation_auc,
        "selected_model": selected_model,
        "economic": economic,
        "timing": timing,
    }


def run_volatility_horizon(
    context,
    *,
    economic_target: str = "down_5pct_5d",
) -> ExperimentResult:
    target = _find_target(
        economic_target
    )

    features = _add_horizon_features(
        context.data
    )

    feature_sets: dict[
        str,
        tuple[pd.DataFrame, list[str]],
    ] = {}

    required_columns = {
        feature
        for _, columns
        in FEATURE_SET_CONFIG
        for feature in columns
    }

    missing = sorted(
        required_columns
        - set(features.columns)
    )

    if missing:
        raise ValueError(
            "Följande volatilitetfeatures saknas: "
            + ", ".join(missing)
        )

    for (
        name,
        columns,
    ) in FEATURE_SET_CONFIG:
        feature_sets[name] = (
            features.copy(),
            list(columns),
        )

    rows = []

    for (
        feature_set_name,
        (
            feature_data,
            feature_columns,
        ),
    ) in feature_sets.items():
        print()
        print(
            "================================"
        )
        print(
            f"RUNNING: {feature_set_name}"
        )
        print(
            "================================"
        )

        result = _run_feature_set(
            feature_set_name,
            feature_data,
            feature_columns,
            target,
            context,
        )

        economic = result[
            "economic"
        ]

        top1 = next(
            (
                item
                for item in economic[
                    "top_fraction"
                ]
                if item["fraction"] == 0.01
            ),
            None,
        )

        rows.append(
            {
                "feature_set": (
                    feature_set_name
                ),
                "feature_count": (
                    result[
                        "feature_count"
                    ]
                ),
                "row_count": (
                    result[
                        "row_count"
                    ]
                ),
                "validation_auc": (
                    result[
                        "validation_auc"
                    ]
                ),
                "selected_model": (
                    result[
                        "selected_model"
                    ]
                ),
                "oos_rows": (
                    economic["rows"]
                ),
                "oos_event_rate": (
                    economic[
                        "baseline_event_rate"
                    ]
                ),
                "oos_mean_return": (
                    economic[
                        "baseline_mean_return"
                    ]
                ),
                "top_1pct_event_rate": (
                    top1["event_rate"]
                    if top1
                    else np.nan
                ),
                "top_1pct_lift": (
                    top1["lift"]
                    if top1
                    else np.nan
                ),
                "top_1pct_mean_return": (
                    top1["mean_return"]
                    if top1
                    else np.nan
                ),
            }
        )

    table = pd.DataFrame(rows)

    result = ExperimentResult(
        name="volatility_horizon_screening",
        description=(
            "Jämför 20d-volatilitet med "
            "relativ och förändrad volatilitet "
            "mot 60d."
        ),
    )

    result.add_table(
        "feature_sets",
        table,
    )

    result.add_metadata(
        "economic_target",
        economic_target,
    )

    result.add_metadata(
        "benchmark_trees",
        BENCHMARK_TREES,
    )

    result.add_metadata(
        "feature_sets",
        [
            name
            for name, _ in FEATURE_SET_CONFIG
        ],
    )

    return result
