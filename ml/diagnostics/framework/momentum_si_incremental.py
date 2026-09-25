from __future__ import annotations

import numpy as np
import pandas as pd

from .base import ExperimentResult
from .incremental_models import (
    IncrementalModelSpec,
    compare_incremental_models,
)
from ml.research.signals import build_signal


MOMENTUM_COLUMN = "price_momentum_20d"
SI_CHANGE_COLUMN = "short_interest_change"

TARGETS = (
    (
        "down_5pct_5d",
        "forward_return_5d",
        -0.05,
    ),
    (
        "down_7pct_5d",
        "forward_return_5d",
        -0.07,
    ),
    (
        "down_10pct_5d",
        "forward_return_5d",
        -0.10,
    ),
)


def _numeric(
    frame: pd.DataFrame,
    column: str,
) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(
            np.nan,
            index=frame.index,
            dtype=float,
        )

    return pd.to_numeric(
        frame[column],
        errors="coerce",
    ).replace(
        [np.inf, -np.inf],
        np.nan,
    )


def _prepare_frame(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    result = frame.copy()

    if "snapshot_date" not in result.columns:
        raise KeyError(
            "Momentum/SI incremental diagnostic "
            "kräver 'snapshot_date'."
        )

    result["snapshot_date"] = pd.to_datetime(
        result["snapshot_date"],
        errors="coerce",
    )

    result = result.loc[
        result["snapshot_date"].notna()
    ].copy()

    result["momentum_20d"] = build_signal(
        result,
        MOMENTUM_COLUMN,
    )

    result["si_change"] = build_signal(
        result,
        SI_CHANGE_COLUMN,
    )

    result["momentum_si_change"] = (
        _numeric(
            result,
            "momentum_20d",
        )
        * _numeric(
            result,
            "si_change",
        )
    )

    return result


def _target_model_specs(
    *,
    target_name: str,
) -> tuple[IncrementalModelSpec, ...]:
    return (
        IncrementalModelSpec(
            name=f"{target_name}_m0",
            features=(
                "momentum_20d",
            ),
        ),
        IncrementalModelSpec(
            name=f"{target_name}_m1",
            features=(
                "momentum_20d",
                "si_change",
            ),
        ),
        IncrementalModelSpec(
            name=f"{target_name}_m2",
            features=(
                "momentum_20d",
                "si_change",
                "momentum_si_change",
            ),
        ),
    )


def run_momentum_si_incremental(
    context,
) -> ExperimentResult:
    train = _prepare_frame(
        context.train
    )

    validation = _prepare_frame(
        context.validation
    )

    pretest = _prepare_frame(
        context.pretest
    )

    test = _prepare_frame(
        context.test
    )

    if test.empty:
        raise ValueError(
            "Momentum/SI incremental diagnostic "
            "fick ett tomt testdataset."
        )

    result = ExperimentResult(
        name="momentum_si_incremental",
        description=(
            "Testar inkrementell prediktiv information "
            "från SI-förändring och momentum × SI-förändring "
            "utöver 20-dagars momentum."
        ),
    )

    all_rows: list[pd.DataFrame] = []

    for (
        target_name,
        return_column,
        threshold,
    ) in TARGETS:
        model_specs = _target_model_specs(
            target_name=target_name,
        )

        table = compare_incremental_models(
            train=train,
            validation=validation,
            pretest=pretest,
            test=test,
            model_specs=model_specs,
            target_column=return_column,
            target_threshold=threshold,
        )

        table.insert(
            0,
            "target",
            target_name,
        )

        all_rows.append(
            table
        )

    comparison = pd.concat(
        all_rows,
        ignore_index=True,
    )

    result.add_table(
        "incremental_model_comparison",
        comparison,
    )

    result.add_metric(
        "targets",
        [
            target_name
            for (
                target_name,
                _,
                _,
            ) in TARGETS
        ],
    )

    result.add_metric(
        "model_sequence",
        [
            "M0 = momentum_20d",
            "M1 = momentum_20d + si_change",
            "M2 = momentum_20d + si_change + momentum_20d × si_change",
        ],
    )

    result.add_metric(
        "momentum_feature",
        "price_momentum_20d",
    )

    result.add_metric(
        "si_change_feature",
        "short_interest_change",
    )

    result.add_metric(
        "interaction_feature",
        "price_momentum_20d × short_interest_change",
    )

    result.add_metric(
        "walk_forward",
        True,
    )

    result.add_metric(
        "test_used_for_model_selection",
        False,
    )

    result.add_metric(
        "test_used_for_feature_selection",
        False,
    )

    result.add_metric(
        "test_used_for_threshold_selection",
        False,
    )

    return result
