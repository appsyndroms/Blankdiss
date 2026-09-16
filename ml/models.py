"""ML-modeller för Blankdiss."""
from __future__ import annotations

from typing import Any

from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
    LogisticRegression,
    Ridge,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_models(
    random_state: int,
    task: str = "classification",
) -> dict[str, Any]:
    if task == "classification":
        return {
            "logistic_regression": Pipeline(
                [
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="median"
                        ),
                    ),
                    (
                        "scaler",
                        StandardScaler(),
                    ),
                    (
                        "model",
                        LogisticRegression(
                            max_iter=2000,
                            random_state=random_state,
                        ),
                    ),
                ]
            ),
            "random_forest": Pipeline(
                [
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="median"
                        ),
                    ),
                    (
                        "model",
                        RandomForestClassifier(
                            n_estimators=300,
                            max_depth=12,
                            min_samples_leaf=20,
                            n_jobs=-1,
                            random_state=random_state,
                        ),
                    ),
                ]
            ),
            "gradient_boosting": Pipeline(
                [
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="median"
                        ),
                    ),
                    (
                        "model",
                        HistGradientBoostingClassifier(
                            max_iter=300,
                            learning_rate=0.05,
                            max_leaf_nodes=15,
                            l2_regularization=1.0,
                            random_state=random_state,
                        ),
                    ),
                ]
            ),
        }

    if task == "regression":
        return {
            "ridge_regression": Pipeline(
                [
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="median"
                        ),
                    ),
                    (
                        "scaler",
                        StandardScaler(),
                    ),
                    (
                        "model",
                        Ridge(
                            alpha=1.0,
                        ),
                    ),
                ]
            ),
            "random_forest_regression": Pipeline(
                [
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="median"
                        ),
                    ),
                    (
                        "model",
                        RandomForestRegressor(
                            n_estimators=300,
                            max_depth=12,
                            min_samples_leaf=20,
                            n_jobs=-1,
                            random_state=random_state,
                        ),
                    ),
                ]
            ),
            "gradient_boosting_regression": Pipeline(
                [
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="median"
                        ),
                    ),
                    (
                        "model",
                        HistGradientBoostingRegressor(
                            max_iter=300,
                            learning_rate=0.05,
                            max_leaf_nodes=15,
                            l2_regularization=1.0,
                            random_state=random_state,
                        ),
                    ),
                ]
            ),
        }

    raise ValueError(
        f"Okänd ML-task: {task}"
    )
