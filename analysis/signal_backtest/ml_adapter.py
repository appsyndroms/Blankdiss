"""Adapter mellan Blankdiss ML-OOS och ekonomiskt backtest."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import pandas as pd
from ml.config import OOS_PREDICTIONS_PATH
def load_oos_predictions(
    path: Path = OOS_PREDICTIONS_PATH,
) -> pd.DataFrame:
    """Läs aktuella OOS-prediktioner från ML-pipelinen."""
    if not path.exists():
        raise FileNotFoundError(
            f"OOS-prediktioner saknas: {path}"
        )
    rows: list[dict[str, Any]] = []
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(
                json.loads(line)
            )
    if not rows:
        raise ValueError(
            "OOS-prediktionerna är tomma."
        )
    return pd.DataFrame(rows)
def prepare_economic_predictions(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalisera ML-prediktioner för ekonomiskt backtest.
    Classification:
        score = probability
    Regression DOWN:
        score = -prediction
    Regression UP:
        score = prediction
    Högre score betyder alltid:
        "mer attraktiv för strategin".
    """
    required = {
        "snapshot_date",
        "security_key",
        "target_return",
        "prediction",
        "task",
        "direction",
        "feature_set",
        "target",
    }
    missing = required - set(
        predictions.columns
    )
    if missing:
        raise ValueError(
            "ML-OOS saknar kolumner: "
            + ", ".join(sorted(missing))
        )
    clean = predictions.copy()
    clean["snapshot_date"] = pd.to_datetime(
        clean["snapshot_date"],
        errors="coerce",
    )
    clean["target_return"] = pd.to_numeric(
        clean["target_return"],
        errors="coerce",
    )
    clean["prediction"] = pd.to_numeric(
        clean["prediction"],
        errors="coerce",
    )
    if "probability" in clean.columns:
        clean["probability"] = pd.to_numeric(
            clean["probability"],
            errors="coerce",
        )
    def build_score(row: pd.Series) -> float:
        task = str(row["task"])
        direction = str(row["direction"])
        if task == "classification":
            probability = row.get(
                "probability"
            )
            if pd.isna(probability):
                raise ValueError(
                    "Klassificeringsrad saknar probability."
                )
            return float(probability)
        prediction = row["prediction"]
        if pd.isna(prediction):
            raise ValueError(
                "Regressionsrad saknar prediction."
            )
        if direction == "below":
            return float(-prediction)
        if direction == "above":
            return float(prediction)
        raise ValueError(
            f"Okänd target direction: {direction}"
        )
    clean["score"] = clean.apply(
        build_score,
        axis=1,
    )
    clean["economic_direction"] = clean[
        "direction"
    ].map(
        {
            "above": "long",
            "below": "short",
        }
    )
    if clean["economic_direction"].isna().any():
        raise ValueError(
            "ML-OOS innehåller okänd economic direction."
        )
    clean = clean.dropna(
        subset=[
            "snapshot_date",
            "security_key",
            "target_return",
            "score",
        ]
    )
    clean["security_key"] = (
        clean["security_key"]
        .astype(str)
    )
    clean["feature_set"] = (
        clean["feature_set"]
        .astype(str)
    )
    clean["target"] = (
        clean["target"]
        .astype(str)
    )
    return clean.sort_values(
        [
            "feature_set",
            "target",
            "snapshot_date",
            "score",
            "security_key",
        ],
        ascending=[
            True,
            True,
            True,
            False,
            True,
        ],
        kind="mergesort",
    ).reset_index(drop=True)
def load_and_prepare_economic_predictions(
    path: Path = OOS_PREDICTIONS_PATH,
) -> pd.DataFrame:
    """Läs och normalisera ML-OOS i ett steg."""
    predictions = load_oos_predictions(path)
    return prepare_economic_predictions(
        predictions
    )
