"""Kör Blankdiss signal-backtest."""
from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from ml.config import (
    RANDOM_STATE,
    TARGETS,
    TEST_MIN_ROWS,
    VALIDATION_MIN_ROWS,
    WALK_FORWARD_WINDOWS,
)
from ml.dataset import dataset_summary
from ml.models import build_models
from ml.train import prepare_feature_set
from analysis.signal_backtest.config import (
    BACKTEST_TARGETS,
    TOP_FRACTIONS,
)
from analysis.signal_backtest.metrics import (
    summarize_year,
)
def split_window(
    data: pd.DataFrame,
    window,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Dela data enligt Blankdiss walk-forward-konfiguration."""
    train_end = pd.Timestamp(window.train_end)
    validation_end = pd.Timestamp(window.validation_end)
    test_end = pd.Timestamp(window.test_end)
    train_mask = (
        data["snapshot_date"] <= train_end
    )
    validation_mask = (
        (data["snapshot_date"] > train_end)
        & (data["snapshot_date"] <= validation_end)
    )
    test_mask = (
        (data["snapshot_date"] > validation_end)
        & (data["snapshot_date"] <= test_end)
    )
    return (
        train_mask,
        validation_mask,
        test_mask,
    )
def features_available_in_training(
    train: pd.DataFrame,
    feature_columns: list[str],
) -> list[str]:
    """Behåll features som har data i training-perioden."""
    return [
        column
        for column in feature_columns
        if train[column].notna().any()
    ]
def roc_auc_safe(
    y_true: pd.Series,
    probabilities: np.ndarray,
) -> float:
    """Beräkna AUC eller returnera -inf om AUC inte är möjlig."""
    if len(np.unique(y_true)) < 2:
        return float("-inf")
    return float(
        roc_auc_score(
            y_true,
            probabilities,
        )
    )
def select_model_and_predict(
    data: pd.DataFrame,
    y: pd.Series,
    feature_columns: list[str],
    window,
) -> dict[str, Any] | None:
    """
    Välj modell på validation och skapa OOS-prediktioner på test.
    Testperioden används aldrig vid modellval.
    """
    (
        train_mask,
        validation_mask,
        test_mask,
    ) = split_window(
        data,
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
    y_train = y.loc[train_mask]
    y_validation = y.loc[validation_mask]
    y_test = y.loc[test_mask]
    if len(train) == 0:
        return None
    if len(validation) < VALIDATION_MIN_ROWS:
        return None
    if len(test) < TEST_MIN_ROWS:
        return None
    if len(np.unique(y_train)) < 2:
        return None
    if len(np.unique(y_validation)) < 2:
        return None
    if len(np.unique(y_test)) < 2:
        return None
    available_features = (
        features_available_in_training(
            train,
            feature_columns,
        )
    )
    if not available_features:
        return None
    train = train[available_features]
    validation = validation[available_features]
    test = test[available_features]
    models = build_models(RANDOM_STATE)
    candidates = []
    for name, model in models.items():
        model.fit(
            train,
            y_train,
        )
        validation_probabilities = (
            model.predict_proba(
                validation
            )[:, 1]
        )
        validation_auc = roc_auc_safe(
            y_validation,
            validation_probabilities,
        )
        candidates.append(
            (
                validation_auc,
                name,
                model,
            )
        )
    candidates.sort(
        key=lambda item: item[0],
        reverse=True,
    )
    (
        validation_auc,
        model_name,
        model,
    ) = candidates[0]
    test_probabilities = (
        model.predict_proba(
            test
        )[:, 1]
    )
    test_auc = roc_auc_safe(
        y_test,
        test_probabilities,
    )
    predictions = data.loc[
        test_mask,
        [
            "snapshot_date",
            "security_key",
            "target_return",
        ],
    ].copy()
    predictions["actual"] = y_test.to_numpy()
    predictions["probability"] = test_probabilities
    predictions = predictions.sort_values(
        [
            "probability",
            "security_key",
        ],
        ascending=[
            False,
            True,
        ],
        kind="mergesort",
    ).reset_index(drop=True)
    return {
        "model": model_name,
        "validation_auc": float(validation_auc),
        "test_auc": float(test_auc),
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "test_rows": int(len(test)),
        "feature_count": int(len(available_features)),
        "features": available_features,
        "window": {
            "train_end": window.train_end,
            "validation_end": window.validation_end,
            "test_end": window.test_end,
        },
        "predictions": predictions,
    }
def combine_window_predictions(
    window_results: list[dict[str, Any]],
) -> pd.DataFrame:
    """Slå ihop OOS-prediktioner från walk-forward-vinduer."""
    frames = []
    for result in window_results:
        predictions = result["predictions"].copy()
        predictions["model"] = result["model"]
        predictions["validation_auc"] = result["validation_auc"]
        predictions["test_auc"] = result["test_auc"]
        predictions["train_end"] = (
            result["window"]["train_end"]
        )
        predictions["validation_end"] = (
            result["window"]["validation_end"]
        )
        predictions["test_end"] = (
            result["window"]["test_end"]
        )
        frames.append(predictions)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(
        frames,
        ignore_index=True,
    )
    combined["snapshot_date"] = pd.to_datetime(
        combined["snapshot_date"]
    )
    return combined.sort_values(
        [
            "snapshot_date",
            "security_key",
        ],
        kind="mergesort",
    ).reset_index(drop=True)
def run_experiment(
    features: pd.DataFrame,
    feature_set_name: str,
    price_features: set[str],
    target_name: str,
) -> dict[str, Any]:
    """Kör ett komplett signal-backtest."""
    target = next(
        item
        for item in TARGETS
        if item.name == target_name
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
    window_results = []
    for window in WALK_FORWARD_WINDOWS:
        result = select_model_and_predict(
            data,
            y,
            feature_columns,
            window,
        )
        if result is not None:
            window_results.append(result)
    predictions = combine_window_predictions(
        window_results
    )
    if predictions.empty:
        raise RuntimeError(
            "Signal-backtest fick inga OOS-prediktioner för "
            f"{feature_set_name} / {target_name}."
        )
    years = sorted(
        predictions["snapshot_date"]
        .dt.year
        .unique()
        .tolist()
    )
    yearly = [
        summarize_year(
            predictions,
            int(year),
            TOP_FRACTIONS,
        )
        for year in years
    ]
    return {
        "feature_set": feature_set_name,
        "target": target_name,
        "return_column": target.return_column,
        "target_threshold": target.threshold,
        "dataset_summary": summary,
        "windows": [
            {
                "window": result["window"],
                "model": result["model"],
                "validation_auc": result["validation_auc"],
                "test_auc": result["test_auc"],
                "feature_count": result["feature_count"],
                "train_rows": result["train_rows"],
                "validation_rows": result["validation_rows"],
                "test_rows": result["test_rows"],
            }
            for result in window_results
        ],
        "yearly": yearly,
    }
def run_all(
    features: pd.DataFrame,
    feature_sets,
) -> list[dict[str, Any]]:
    """Kör alla definierade signal-backtest."""
    results = []
    for feature_set_name, price_features in feature_sets:
        for target_name in BACKTEST_TARGETS:
            results.append(
                run_experiment(
                    features,
                    feature_set_name,
                    price_features,
                    target_name,
                )
            )
    return results
