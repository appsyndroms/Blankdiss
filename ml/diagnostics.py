"""Diagnostik av Blankdiss ML-signaler."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

from ml.config import (
    ML_OUTPUT_DIR,
    PRICE_FEATURE_COLUMNS,
    RANDOM_STATE,
    TARGETS,
    WALK_FORWARD_WINDOWS,
)
from ml.dataset import load_features, prepare_ml_data
from ml.models import build_models


OUTPUT_PATH = ML_OUTPUT_DIR / "diagnostics.json"

DIAGNOSTIC_TARGETS = (
    "up_5pct_5d",
    "down_5pct_5d",
)

DIAGNOSTIC_FEATURE_SETS = (
    (
        "fi_plus_price_volatility_20d",
        {"price_volatility_20d"},
    ),
    (
        "fi_plus_all_price",
        set(PRICE_FEATURE_COLUMNS),
    ),
)

TOP_FRACTIONS = (
    0.01,
    0.05,
    0.10,
    0.20,
)


def safe_auc(
    y_true: pd.Series,
    probabilities: np.ndarray,
) -> float | None:
    values = np.asarray(y_true)

    if len(np.unique(values)) < 2:
        return None

    return float(
        roc_auc_score(
            values,
            probabilities,
        )
    )


def window_masks(
    data: pd.DataFrame,
    window,
) -> dict[str, pd.Series]:
    train_end = pd.Timestamp(window.train_end)
    validation_end = pd.Timestamp(window.validation_end)
    test_end = pd.Timestamp(window.test_end)

    return {
        "train": (
            data["snapshot_date"] <= train_end
        ),
        "validation": (
            (data["snapshot_date"] > train_end)
            & (
                data["snapshot_date"]
                <= validation_end
            )
        ),
        "test": (
            (data["snapshot_date"] > validation_end)
            & (
                data["snapshot_date"]
                <= test_end
            )
        ),
    }


def feature_importance(
    model,
    feature_columns: list[str],
) -> list[dict[str, Any]]:
    if hasattr(model, "named_steps"):
        estimator = model.named_steps.get(
            "model",
            model,
        )
    else:
        estimator = model

    if not hasattr(
        estimator,
        "feature_importances_",
    ):
        return []

    importances = np.asarray(
        estimator.feature_importances_,
        dtype=float,
    )

    rows = [
        {
            "feature": feature,
            "importance": float(value),
        }
        for feature, value in zip(
            feature_columns,
            importances,
        )
    ]

    rows.sort(
        key=lambda row: row["importance"],
        reverse=True,
    )

    return rows


def prediction_buckets(
    model,
    data: pd.DataFrame,
    y: pd.Series,
    feature_columns: list[str],
    mask: pd.Series,
) -> dict[str, Any]:
    subset = data.loc[mask].copy()

    if subset.empty:
        return {
            "rows": 0,
            "baseline_event_rate": None,
            "baseline_mean_return": None,
            "buckets": [],
        }

    subset["target"] = (
        y.loc[mask].to_numpy()
    )

    subset["probability"] = (
        model.predict_proba(
            subset[feature_columns]
        )[:, 1]
    )

    subset["target_return"] = pd.to_numeric(
        subset["target_return"],
        errors="coerce",
    )

    subset = subset.sort_values(
        "probability",
        ascending=False,
    ).reset_index(drop=True)

    baseline_event_rate = float(
        subset["target"].mean()
    )

    baseline_mean_return = float(
        subset["target_return"].mean()
    )

    buckets = []

    for fraction in TOP_FRACTIONS:
        rows = max(
            1,
            int(
                np.ceil(
                    len(subset) * fraction
                )
            ),
        )

        top = subset.iloc[:rows]

        event_rate = float(
            top["target"].mean()
        )

        mean_return = float(
            top["target_return"].mean()
        )

        median_return = float(
            top["target_return"].median()
        )

        mean_probability = float(
            top["probability"].mean()
        )

        buckets.append(
            {
                "top_fraction": fraction,
                "rows": int(rows),
                "mean_probability": (
                    mean_probability
                ),
                "event_rate": event_rate,
                "baseline_event_rate": (
                    baseline_event_rate
                ),
                "event_rate_lift": (
                    event_rate
                    - baseline_event_rate
                ),
                "event_rate_lift_ratio": (
                    event_rate
                    / baseline_event_rate
                    if baseline_event_rate > 0
                    else None
                ),
                "mean_return": mean_return,
                "median_return": median_return,
                "baseline_mean_return": (
                    baseline_mean_return
                ),
                "mean_return_lift": (
                    mean_return
                    - baseline_mean_return
                ),
                "min_probability": float(
                    top["probability"].min()
                ),
                "max_probability": float(
                    top["probability"].max()
                ),
            }
        )

    return {
        "rows": int(len(subset)),
        "baseline_event_rate": (
            baseline_event_rate
        ),
        "baseline_mean_return": (
            baseline_mean_return
        ),
        "buckets": buckets,
    }


def yearly_evaluation(
    model,
    data: pd.DataFrame,
    y: pd.Series,
    feature_columns: list[str],
    start_year: int,
    end_year: int,
) -> list[dict[str, Any]]:
    rows = []

    for year in range(
        start_year,
        end_year + 1,
    ):
        mask = (
            data["snapshot_date"]
            .dt.year
            == year
        )

        if not mask.any():
            continue

        subset = data.loc[mask]

        y_year = y.loc[mask]

        probabilities = (
            model.predict_proba(
                subset[feature_columns]
            )[:, 1]
        )

        predictions = (
            probabilities >= 0.5
        ).astype(int)

        returns = pd.to_numeric(
            subset["target_return"],
            errors="coerce",
        )

        rows.append(
            {
                "year": year,
                "rows": int(len(subset)),
                "positive_rate": float(
                    y_year.mean()
                ),
                "roc_auc": safe_auc(
                    y_year,
                    probabilities,
                ),
                "accuracy": float(
                    accuracy_score(
                        y_year,
                        predictions,
                    )
                ),
                "mean_return": float(
                    returns.mean()
                ),
                "median_return": float(
                    returns.median()
                ),
            }
        )

    return rows


def company_diagnostics(
    model,
    data: pd.DataFrame,
    y: pd.Series,
    feature_columns: list[str],
    mask: pd.Series,
) -> dict[str, Any]:
    subset = data.loc[mask].copy()

    subset["target"] = (
        y.loc[mask].to_numpy()
    )

    subset["probability"] = (
        model.predict_proba(
            subset[feature_columns]
        )[:, 1]
    )

    records = []

    for (
        security_key,
        group,
    ) in subset.groupby(
        "security_key",
        sort=False,
    ):
        group_y = group["target"]

        returns = pd.to_numeric(
            group["target_return"],
            errors="coerce",
        )

        records.append(
            {
                "security_key": security_key,
                "rows": int(len(group)),
                "positive_rate": float(
                    group_y.mean()
                ),
                "roc_auc": safe_auc(
                    group_y,
                    group["probability"].to_numpy(),
                ),
                "accuracy": float(
                    accuracy_score(
                        group_y,
                        (
                            group["probability"]
                            >= 0.5
                        ).astype(int),
                    )
                ),
                "mean_return": float(
                    returns.mean()
                ),
                "median_return": float(
                    returns.median()
                ),
                "mean_probability": float(
                    group["probability"].mean()
                ),
            }
        )

    records.sort(
        key=lambda row: row["rows"],
        reverse=True,
    )

    auc_rows = [
        row
        for row in records
        if row["roc_auc"] is not None
    ]

    auc_values = [
        row["roc_auc"]
        for row in auc_rows
    ]

    return {
        "companies": int(len(records)),
        "companies_with_auc": int(
            len(auc_rows)
        ),
        "median_company_auc": (
            float(np.median(auc_values))
            if auc_values
            else None
        ),
        "mean_company_auc": (
            float(np.mean(auc_values))
            if auc_values
            else None
        ),
        "top_by_rows": records[:25],
    }


def run_window(
    data: pd.DataFrame,
    y: pd.Series,
    feature_columns: list[str],
    window,
) -> dict[str, Any]:
    masks = window_masks(
        data,
        window,
    )

    train = data.loc[
        masks["train"],
        feature_columns,
    ]

    validation = data.loc[
        masks["validation"],
        feature_columns,
    ]

    test = data.loc[
        masks["test"],
        feature_columns,
    ]

    y_train = y.loc[
        masks["train"]
    ]

    y_validation = y.loc[
        masks["validation"]
    ]

    y_test = y.loc[
        masks["test"]
    ]

    models = build_models(
        RANDOM_STATE
    )

    candidates = []

    for name, model in models.items():
        model.fit(
            train,
            y_train,
        )

        probabilities = (
            model.predict_proba(
                validation
            )[:, 1]
        )

        score = safe_auc(
            y_validation,
            probabilities,
        )

        if score is not None:
            candidates.append(
                (
                    score,
                    name,
                    model,
                )
            )

    candidates.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    if not candidates:
        raise RuntimeError(
            "Ingen modell kunde tränas "
            "för diagnostiken."
        )

    (
        validation_score,
        selected_name,
        selected_model,
    ) = candidates[0]

    test_probabilities = (
        selected_model.predict_proba(
            test
        )[:, 1]
    )

    test_predictions = (
        test_probabilities >= 0.5
    ).astype(int)

    test_returns = pd.to_numeric(
        data.loc[
            masks["test"],
            "target_return",
        ],
        errors="coerce",
    )

    test_start_year = (
        int(
            pd.Timestamp(
                window.validation_end
            ).year
        )
        + 1
    )

    test_end_year = int(
        pd.Timestamp(
            window.test_end
        ).year
    )

    yearly = {}

    for year in range(
        test_start_year,
        test_end_year + 1,
    ):
        year_mask = (
            masks["test"]
            & (
                data["snapshot_date"]
                .dt.year
                == year
            )
        )

        if not year_mask.any():
            continue

        yearly[str(year)] = (
            prediction_buckets(
                selected_model,
                data,
                y,
                feature_columns,
                year_mask,
            )
        )

    return {
        "window": {
            "train_end": (
                window.train_end
            ),
            "validation_end": (
                window.validation_end
            ),
            "test_end": (
                window.test_end
            ),
        },
        "selected_model": selected_name,
        "validation_roc_auc": float(
            validation_score
        ),
        "test_roc_auc": safe_auc(
            y_test,
            test_probabilities,
        ),
        "test_accuracy": float(
            accuracy_score(
                y_test,
                test_predictions,
            )
        ),
        "test_rows": int(
            len(test)
        ),
        "test_positive_rate": float(
            y_test.mean()
        ),
        "test_mean_return": float(
            test_returns.mean()
        ),
        "test_median_return": float(
            test_returns.median()
        ),
        "feature_importance": (
            feature_importance(
                selected_model,
                feature_columns,
            )[:25]
        ),
        "yearly_evaluation": (
            yearly_evaluation(
                selected_model,
                data,
                y,
                feature_columns,
                test_start_year,
                test_end_year,
            )
        ),
        "prediction_buckets": yearly,
        "company_diagnostics": (
            company_diagnostics(
                selected_model,
                data,
                y,
                feature_columns,
                masks["test"],
            )
        ),
    }


def prepare_diagnostic_data(
    features: pd.DataFrame,
    target_name: str,
    selected_price_features: set[str],
) -> tuple[
    pd.DataFrame,
    pd.Series,
    list[str],
]:
    target = next(
        target
        for target in TARGETS
        if target.name == target_name
    )

    (
        fi_data,
        y,
        fi_feature_columns,
    ) = prepare_ml_data(
        features,
        target,
        include_price_features=False,
    )

    (
        price_data,
        price_y,
        all_feature_columns,
    ) = prepare_ml_data(
        features,
        target,
        include_price_features=True,
    )

    if not y.equals(price_y):
        raise RuntimeError(
            "FI-only och FI+pris gav olika "
            "target-rader i diagnostiken."
        )

    missing_price_features = (
        selected_price_features
        - set(all_feature_columns)
    )

    if missing_price_features:
        raise RuntimeError(
            "Saknar diagnostik-features: "
            + ", ".join(
                sorted(missing_price_features)
            )
        )

    feature_columns = (
        fi_feature_columns
        + sorted(selected_price_features)
    )

    missing_columns = [
        column
        for column in feature_columns
        if column not in price_data.columns
    ]

    if missing_columns:
        raise RuntimeError(
            "Saknar kolumner i diagnostikdata: "
            + ", ".join(missing_columns)
        )

    data = price_data[
        [
            "snapshot_date",
            "security_key",
            "target_return",
        ]
        + feature_columns
    ].copy()

    return (
        data,
        y,
        feature_columns,
    )


def run_target_feature_set(
    features: pd.DataFrame,
    target_name: str,
    feature_set_name: str,
    selected_price_features: set[str],
) -> dict[str, Any]:
    (
        data,
        y,
        feature_columns,
    ) = prepare_diagnostic_data(
        features,
        target_name,
        selected_price_features,
    )

    return {
        "target": target_name,
        "feature_set": feature_set_name,
        "feature_count": len(
            feature_columns
        ),
        "feature_columns": feature_columns,
        "dataset_summary": {
            "rows": int(len(data)),
            "features": len(
                feature_columns
            ),
            "positive": int(y.sum()),
            "negative": int(
                len(y) - y.sum()
            ),
            "positive_rate": float(
                y.mean()
            ),
            "date_start": (
                data["snapshot_date"]
                .min()
                .date()
                .isoformat()
            ),
            "date_end": (
                data["snapshot_date"]
                .max()
                .date()
                .isoformat()
            ),
        },
        "windows": [
            run_window(
                data,
                y,
                feature_columns,
                window,
            )
            for window
            in WALK_FORWARD_WINDOWS
        ],
    }


def main() -> None:
    features = load_features()

    results = []

    for (
        feature_set_name,
        selected_price_features,
    ) in DIAGNOSTIC_FEATURE_SETS:
        for target_name in DIAGNOSTIC_TARGETS:
            print(
                "Kör diagnostik: "
                f"{feature_set_name} / "
                f"{target_name}"
            )

            results.append(
                run_target_feature_set(
                    features,
                    target_name,
                    feature_set_name,
                    selected_price_features,
                )
            )

    diagnostics = {
        "experiment": (
            "price_signal_diagnostics"
        ),
        "feature_sets": [
            name
            for name, _
            in DIAGNOSTIC_FEATURE_SETS
        ],
        "targets": list(
            DIAGNOSTIC_TARGETS
        ),
        "top_fractions": list(
            TOP_FRACTIONS
        ),
        "results": results,
    }

    ML_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            diagnostics,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            diagnostics,
            ensure_ascii=False,
            indent=2,
        )
    )

    print(
        f"Diagnostik sparad: "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
