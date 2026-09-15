"""Diagnostik av Blankdiss ML-signal."""
from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
)

from ml.config import (
    ML_OUTPUT_DIR,
    RANDOM_STATE,
    TARGETS,
    WALK_FORWARD_WINDOWS,
)
from ml.dataset import (
    load_features,
    prepare_ml_data,
)
from ml.models import build_models


OUTPUT_PATH = (
    ML_OUTPUT_DIR
    / "diagnostics.json"
)

TARGET_NAME = "positive_5d"


def safe_auc(
    y_true: pd.Series,
    probabilities: np.ndarray,
) -> float | None:
    values = np.asarray(
        y_true
    )

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
    train_end = pd.Timestamp(
        window.train_end
    )

    validation_end = pd.Timestamp(
        window.validation_end
    )

    test_end = pd.Timestamp(
        window.test_end
    )

    return {
        "train": (
            data["snapshot_date"]
            <= train_end
        ),
        "validation": (
            (data["snapshot_date"] > train_end)
            & (
                data["snapshot_date"]
                <= validation_end
            )
        ),
        "test": (
            (
                data["snapshot_date"]
                > validation_end
            )
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
    if hasattr(
        model,
        "named_steps",
    ):
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
            "importance": float(
                value
            ),
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

        subset = data.loc[
            mask
        ]

        y_year = y.loc[
            mask
        ]

        probabilities = (
            model.predict_proba(
                subset[
                    feature_columns
                ]
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
                "rows": int(
                    len(subset)
                ),
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
    subset = data.loc[
        mask
    ].copy()

    subset["target"] = (
        y.loc[mask]
        .to_numpy()
    )

    subset["probability"] = (
        model.predict_proba(
            subset[
                feature_columns
            ]
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
        group_y = group[
            "target"
        ]

        returns = pd.to_numeric(
            group["target_return"],
            errors="coerce",
        )

        records.append(
            {
                "security_key": security_key,
                "rows": int(
                    len(group)
                ),
                "positive_rate": float(
                    group_y.mean()
                ),
                "roc_auc": safe_auc(
                    group_y,
                    group[
                        "probability"
                    ].to_numpy(),
                ),
                "accuracy": float(
                    accuracy_score(
                        group_y,
                        (
                            group[
                                "probability"
                            ]
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
                    group[
                        "probability"
                    ].mean()
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
        "companies": int(
            len(records)
        ),
        "companies_with_auc": int(
            len(auc_rows)
        ),
        "median_company_auc": (
            float(
                np.median(
                    auc_values
                )
            )
            if auc_values
            else None
        ),
        "mean_company_auc": (
            float(
                np.mean(
                    auc_values
                )
            )
            if auc_values
            else None
        ),
        "top_by_rows": records[
            :25
        ],
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
            "Ingen modell kunde "
            "tränas för diagnostiken."
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
        "selected_model": (
            selected_name
        ),
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
                (
                    int(
                        pd.Timestamp(
                            window.train_end
                        ).year
                    )
                    + 1
                ),
                int(
                    pd.Timestamp(
                        window.test_end
                    ).year
                ),
            )
        ),
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


def main() -> None:
    features = load_features()

    target = next(
        target
        for target in TARGETS
        if target.name
        == TARGET_NAME
    )

    (
        data,
        y,
        feature_columns,
    ) = prepare_ml_data(
        features,
        target,
        include_price_features=False,
    )

    diagnostics = {
        "target": TARGET_NAME,
        "feature_set": "fi_only",
        "feature_count": len(
            feature_columns
        ),
        "feature_columns": (
            feature_columns
        ),
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
