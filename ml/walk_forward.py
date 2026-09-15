"""Walk-forward-träning för Blankdiss."""
from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from ml.config import (
    RANDOM_STATE,
    TEST_MIN_ROWS,
    VALIDATION_MIN_ROWS,
    WalkForwardWindow,
)

from ml.evaluate import (
    evaluate_predictions,
    return_by_probability_bucket,
)

from ml.models import (
    build_models,
)


def _split(
    data: pd.DataFrame,
    y: pd.Series,
    window: WalkForwardWindow,
):
    train_end = pd.Timestamp(
        window.train_end
    )

    validation_end = pd.Timestamp(
        window.validation_end
    )

    test_end = pd.Timestamp(
        window.test_end
    )

    train_mask = (
        data["snapshot_date"]
        <= train_end
    )

    validation_mask = (
        (data["snapshot_date"] > train_end)
        & (
            data["snapshot_date"]
            <= validation_end
        )
    )

    test_mask = (
        (data["snapshot_date"] > validation_end)
        & (
            data["snapshot_date"]
            <= test_end
        )
    )

    return (
        train_mask,
        validation_mask,
        test_mask,
    )


def _features_available_in_training(
    train: pd.DataFrame,
    feature_columns: list[str],
) -> list[str]:
    """
    Returnerar endast features som har minst
    ett observerat värde i training-datasetet.

    Viktigt:
    - endast training används för beslutet
    - validation/test får inte påverka featurevalet
    - förhindrar all-NaN-kolumner i sklearn-imputern
    """

    available: list[str] = []
    removed: list[str] = []

    for column in feature_columns:
        if train[column].notna().any():
            available.append(column)
        else:
            removed.append(column)

    if removed:
        print(
            "Tar bort features som saknar "
            "observerade värden i training:"
        )

        for column in removed:
            print(
                f"  {column}"
            )

    return available


def roc_auc_safe(
    y_true,
    probabilities,
) -> float:
    from sklearn.metrics import roc_auc_score

    if len(
        np.unique(y_true)
    ) < 2:
        return float("-inf")

    return float(
        roc_auc_score(
            y_true,
            probabilities,
        )
    )


def _validation_score(
    model,
    X_validation,
    y_validation,
) -> float:
    probabilities = model.predict_proba(
        X_validation
    )[:, 1]

    return roc_auc_safe(
        y_validation,
        probabilities,
    )


def train_window(
    data: pd.DataFrame,
    y: pd.Series,
    feature_columns: list[str],
    window: WalkForwardWindow,
) -> list[dict[str, Any]]:
    (
        train_mask,
        validation_mask,
        test_mask,
    ) = _split(
        data,
        y,
        window,
    )

    train = data.loc[
        train_mask,
        feature_columns,
    ].copy()

    validation = data.loc[
        validation_mask,
        feature_columns,
    ].copy()

    test = data.loc[
        test_mask,
        feature_columns,
    ].copy()

    y_train = y.loc[
        train_mask
    ]

    y_validation = y.loc[
        validation_mask
    ]

    y_test = y.loc[
        test_mask
    ]

    if len(train) == 0:
        return []

    if len(validation) < VALIDATION_MIN_ROWS:
        return []

    if len(test) < TEST_MIN_ROWS:
        return []

    if len(
        np.unique(y_train)
    ) < 2:
        return []

    if len(
        np.unique(y_validation)
    ) < 2:
        return []

    if len(
        np.unique(y_test)
    ) < 2:
        return []

    # ---------------------------------------------------------
    # Viktigt:
    #
    # En feature får bara användas om den har minst ett
    # observerat värde i TRAINING-perioden.
    #
    # Vi tittar inte på validation/test när vi fattar detta
    # beslut. Det förhindrar både sklearn-varningar och
    # framtidsläckage.
    # ---------------------------------------------------------

    available_features = (
        _features_available_in_training(
            train,
            feature_columns,
        )
    )

    if not available_features:
        return []

    train = train[
        available_features
    ]

    validation = validation[
        available_features
    ]

    test = test[
        available_features
    ]

    models = build_models(
        RANDOM_STATE
    )

    trained = []

    for name, model in models.items():
        model.fit(
            train,
            y_train,
        )

        validation_score = (
            _validation_score(
                model,
                validation,
                y_validation,
            )
        )

        trained.append(
            (
                validation_score,
                name,
                model,
            )
        )

    trained.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    results = []

    for (
        validation_score,
        name,
        model,
    ) in trained:
        test_probabilities = (
            model.predict_proba(
                test
            )[:, 1]
        )

        metrics = evaluate_predictions(
            y_test,
            test_probabilities,
        )

        returns = data.loc[
            test_mask,
            "target_return",
        ]

        bucket_results = (
            return_by_probability_bucket(
                y_test,
                test_probabilities,
                returns,
            )
        )

        results.append(
            {
                "model": name,
                "validation_roc_auc": (
                    validation_score
                ),
                "test": metrics,
                "return_buckets": bucket_results,
                "features": available_features,
                "feature_count": int(
                    len(available_features)
                ),
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
                "train_rows": int(
                    len(train)
                ),
                "validation_rows": int(
                    len(validation)
                ),
                "test_rows": int(
                    len(test)
                ),
                "created_at": (
                    datetime.utcnow()
                    .isoformat()
                    + "Z"
                ),
            }
        )

    return results
