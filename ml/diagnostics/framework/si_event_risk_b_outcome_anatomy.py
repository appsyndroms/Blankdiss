from __future__ import annotations

import numpy as np
import pandas as pd

from .base import ExperimentResult
from .event_risk_grid import (
    LOCKED_DOWNSIDE_TARGET,
    LOCKED_RISK_CUTOFF,
    LOCKED_SI_CHANGE_CUTOFF,
    _numeric,
)
from .si_event_risk_positive_momentum_context import (
    PRIMARY_GROUPS,
    _prepare_groups,
)
from .si_event_risk_reversal_path import (
    DISTANCE_COLUMNS,
    PRICE_RETURN_COLUMNS,
    _prepare_trajectory,
)
from .si_event_risk_signal_anatomy import (
    LOCKED_EVENT_THRESHOLD,
    LOCKED_HORIZON,
    _prepare_signal,
)


B_GROUP = PRIMARY_GROUPS[0]
D_GROUP = PRIMARY_GROUPS[1]

FORWARD_HORIZONS = (
    "5d",
    "20d",
    "60d",
)

FORWARD_COLUMNS = {
    "5d": "forward_return_5d",
    "20d": "forward_return_20d",
    "60d": "forward_return_60d",
}

OUTCOME_BUCKETS = (
    "down_le_minus_5pct",
    "neutral_minus_5_to_plus_5pct",
    "up_ge_plus_5pct",
)

PRE_SIGNAL_COLUMNS = (
    "price_return_5d",
    "price_return_20d",
    "price_return_60d",
    "return_acceleration_5d_vs_20d",
    "return_acceleration_20d_vs_60d",
    "price_distance_from_5d_high",
    "price_distance_from_10d_high",
    "price_distance_from_20d_high",
    "price_distance_from_60d_high",
    "short_interest_pct",
    "short_interest_pct_change",
    "event_score",
)


def _safe_values(
    frame: pd.DataFrame,
    column: str,
) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(
            dtype=float,
            index=frame.index,
        )

    return _numeric(
        frame,
        column,
    )


def _safe_mean(
    frame: pd.DataFrame,
    column: str,
) -> float:
    values = _safe_values(
        frame,
        column,
    ).dropna()

    if values.empty:
        return float("nan")

    return float(values.mean())


def _safe_median(
    frame: pd.DataFrame,
    column: str,
) -> float:
    values = _safe_values(
        frame,
        column,
    ).dropna()

    if values.empty:
        return float("nan")

    return float(values.median())


def _safe_quantile(
    frame: pd.DataFrame,
    column: str,
    quantile: float,
) -> float:
    values = _safe_values(
        frame,
        column,
    ).dropna()

    if values.empty:
        return float("nan")

    return float(
        values.quantile(quantile)
    )


def _outcome_bucket(
    values: pd.Series,
) -> pd.Series:
    numeric = pd.to_numeric(
        values,
        errors="coerce",
    )

    result = pd.Series(
        "insufficient_forward_data",
        index=values.index,
        dtype="object",
    )

    valid = numeric.notna()

    result.loc[
        valid & (numeric <= -0.05)
    ] = "down_le_minus_5pct"

    result.loc[
        valid
        & (numeric > -0.05)
        & (numeric < 0.05)
    ] = "neutral_minus_5_to_plus_5pct"

    result.loc[
        valid & (numeric >= 0.05)
    ] = "up_ge_plus_5pct"

    return result


def _prepare_b_outcomes(
    test: pd.DataFrame,
) -> pd.DataFrame:
    local = test.copy()

    for horizon in FORWARD_HORIZONS:
        column = FORWARD_COLUMNS[horizon]

        local[
            f"outcome_bucket_{horizon}"
        ] = _outcome_bucket(
            _safe_values(
                local,
                column,
            )
        )

    return local


def _b_outcome_distribution(
    test: pd.DataFrame,
) -> pd.DataFrame:
    b = test[
        test["context_group"]
        == B_GROUP
    ].copy()

    rows: list[dict] = []

    for horizon in FORWARD_HORIZONS:
        bucket_column = (
            f"outcome_bucket_{horizon}"
        )

        counts = (
            b[bucket_column]
            .value_counts(
                dropna=False
            )
        )

        total = len(b)

        for bucket in (
            *OUTCOME_BUCKETS,
            "insufficient_forward_data",
        ):
            count = int(
                counts.get(
                    bucket,
                    0,
                )
            )

            rows.append(
                {
                    "horizon": horizon,
                    "outcome_bucket": bucket,
                    "n": count,
                    "share": (
                        float(count / total)
                        if total > 0
                        else float("nan")
                    ),
                }
            )

    return pd.DataFrame(rows)


def _b_outcome_anatomy(
    test: pd.DataFrame,
) -> pd.DataFrame:
    b = test[
        test["context_group"]
        == B_GROUP
    ].copy()

    rows: list[dict] = []

    for horizon in FORWARD_HORIZONS:
        bucket_column = (
            f"outcome_bucket_{horizon}"
        )

        for bucket in OUTCOME_BUCKETS:
            frame = b[
                b[bucket_column]
                == bucket
            ].copy()

            row = {
                "horizon": horizon,
                "outcome_bucket": bucket,
                "n": int(len(frame)),
            }

            for column in PRE_SIGNAL_COLUMNS:
                if column not in frame.columns:
                    continue

                row[
                    f"mean_{column}"
                ] = _safe_mean(
                    frame,
                    column,
                )

                row[
                    f"median_{column}"
                ] = _safe_median(
                    frame,
                    column,
                )

            rows.append(row)

    return pd.DataFrame(rows)


def _b_outcome_vs_d(
    test: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict] = []

    for group_name, group in (
        ("B", B_GROUP),
        ("D", D_GROUP),
    ):
        frame = test[
            test["context_group"]
            == group
        ].copy()

        for horizon in FORWARD_HORIZONS:
            bucket_column = (
                f"outcome_bucket_{horizon}"
            )

            total = len(frame)

            counts = (
                frame[bucket_column]
                .value_counts(
                    dropna=False
                )
            )

            for bucket in OUTCOME_BUCKETS:
                count = int(
                    counts.get(
                        bucket,
                        0,
                    )
                )

                rows.append(
                    {
                        "group": group_name,
                        "horizon": horizon,
                        "outcome_bucket": bucket,
                        "n": count,
                        "share": (
                            float(count / total)
                            if total > 0
                            else float("nan")
                        ),
                    }
                )

    result = pd.DataFrame(rows)

    if result.empty:
        return result

    b = result[
        result["group"] == "B"
    ].copy()

    d = result[
        result["group"] == "D"
    ].copy()

    merged = b.merge(
        d,
        on=[
            "horizon",
            "outcome_bucket",
        ],
        suffixes=(
            "_B",
            "_D",
        ),
        how="outer",
    )

    return merged[
        [
            "horizon",
            "outcome_bucket",
            "n_B",
            "n_D",
            "share_B",
            "share_D",
        ]
    ]


def _b_outcome_yearly(
    test: pd.DataFrame,
) -> pd.DataFrame:
    if "snapshot_date" not in test.columns:
        return pd.DataFrame()

    b = test[
        test["context_group"]
        == B_GROUP
    ].copy()

    b["_signal_year"] = pd.to_datetime(
        b["snapshot_date"],
        errors="coerce",
    ).dt.year

    b = b[
        b["_signal_year"].notna()
    ].copy()

    if b.empty:
        return pd.DataFrame()

    b["_signal_year"] = (
        b["_signal_year"]
        .astype(int)
    )

    rows: list[dict] = []

    for year in sorted(
        b["_signal_year"].unique()
    ):
        year_frame = b[
            b["_signal_year"]
            == year
        ]

        for horizon in FORWARD_HORIZONS:
            bucket_column = (
                f"outcome_bucket_{horizon}"
            )

            total = len(year_frame)

            counts = (
                year_frame[bucket_column]
                .value_counts(
                    dropna=False
                )
            )

            for bucket in OUTCOME_BUCKETS:
                count = int(
                    counts.get(
                        bucket,
                        0,
                    )
                )

                rows.append(
                    {
                        "year": int(year),
                        "horizon": horizon,
                        "outcome_bucket": bucket,
                        "n": count,
                        "share": (
                            float(count / total)
                            if total > 0
                            else float("nan")
                        ),
                    }
                )

    return pd.DataFrame(rows)


def _b_outcome_quantiles(
    test: pd.DataFrame,
) -> pd.DataFrame:
    b = test[
        test["context_group"]
        == B_GROUP
    ].copy()

    rows: list[dict] = []

    for horizon in FORWARD_HORIZONS:
        column = FORWARD_COLUMNS[horizon]

        values = _safe_values(
            b,
            column,
        ).dropna()

        if values.empty:
            continue

        rows.append(
            {
                "horizon": horizon,
                "n": int(len(values)),
                "p10": float(
                    values.quantile(0.10)
                ),
                "p25": float(
                    values.quantile(0.25)
                ),
                "median": float(
                    values.quantile(0.50)
                ),
                "p75": float(
                    values.quantile(0.75)
                ),
                "p90": float(
                    values.quantile(0.90)
                ),
                "mean": float(
                    values.mean()
                ),
            }
        )

    return pd.DataFrame(rows)


def run_b_outcome_anatomy(
    context,
) -> ExperimentResult:
    (
        test,
        pretest,
        info,
    ) = _prepare_signal(
        context
    )

    test = _prepare_groups(
        test
    )

    test = _prepare_trajectory(
        test
    )

    test = _prepare_b_outcomes(
        test
    )

    b = test[
        test["context_group"]
        == B_GROUP
    ].copy()

    if b.empty:
        raise ValueError(
            "No B observations were "
            "available for outcome-anatomy analysis."
        )

    result = ExperimentResult(
        name="si_event_risk_b_outcome_anatomy",
        description=(
            "Deskriptiv anatomianalys av B-cellen "
            "från den låsta SI × event-risk-signalen. "
            "Analysen delar B efter fasta forward-return-"
            "intervall och jämför endast information "
            "som fanns vid signalögonblicket."
        ),
    )

    result.add_metric(
        "analysis_type",
        "locked_B_outcome_anatomy",
    )

    result.add_metric(
        "parameter_selection",
        "inherited_locked_configuration",
    )

    result.add_metric(
        "is_discovery_grid",
        False,
    )

    result.add_metric(
        "uses_test_for_parameter_selection",
        False,
    )

    result.add_metric(
        "outcome_bucket_thresholds",
        {
            "down": "<= -5%",
            "neutral": (
                "> -5% and < +5%"
            ),
            "up": ">= +5%",
        },
    )

    result.add_metric(
        "forward_horizons",
        list(
            FORWARD_HORIZONS
        ),
    )

    result.add_metric(
        "horizon_days",
        LOCKED_HORIZON,
    )

    result.add_metric(
        "event_threshold",
        LOCKED_EVENT_THRESHOLD,
    )

    result.add_metric(
        "downside_target",
        LOCKED_DOWNSIDE_TARGET,
    )

    result.add_metric(
        "risk_cutoff",
        LOCKED_RISK_CUTOFF,
    )

    result.add_metric(
        "si_change_cutoff",
        LOCKED_SI_CHANGE_CUTOFF,
    )

    result.add_metric(
        "event_model",
        info["event_model"],
    )

    result.add_metric(
        "event_features",
        info["event_features"],
    )

    result.add_metric(
        "event_validation_auc",
        info["event_validation_auc"],
    )

    result.add_metric(
        "risk_threshold",
        info["risk_threshold"],
    )

    result.add_metric(
        "si_change_threshold",
        info["si_change_threshold"],
    )

    result.add_metric(
        "B_definition",
        B_GROUP,
    )

    result.add_metric(
        "D_definition",
        D_GROUP,
    )

    result.add_metric(
        "pre_signal_columns_checked",
        list(
            PRE_SIGNAL_COLUMNS
        ),
    )

    result.add_metric(
        "test_rows",
        int(len(test)),
    )

    result.add_metric(
        "B_rows",
        int(len(b)),
    )

    result.add_metric(
        "research_question",
        (
            "När B-signalen uppstår, vad skiljer "
            "de B-observationer som senare får "
            "stor positiv, neutral eller negativ "
            "avkastning?"
        ),
    )

    result.add_metric(
        "interpretation_guardrail",
        (
            "Outcome buckets are descriptive "
            "post-signal groupings and are not "
            "used to select parameters, thresholds "
            "or trading rules."
        ),
    )

    result.add_table(
        "B_outcome_distribution",
        _b_outcome_distribution(
            test
        ),
    )

    result.add_table(
        "B_outcome_anatomy",
        _b_outcome_anatomy(
            test
        ),
    )

    result.add_table(
        "B_outcome_vs_D",
        _b_outcome_vs_d(
            test
        ),
    )

    result.add_table(
        "B_outcome_yearly",
        _b_outcome_yearly(
            test
        ),
    )

    result.add_table(
        "B_outcome_quantiles",
        _b_outcome_quantiles(
            test
        ),
    )

    result.add_metric(
        "pretest_rows_used_for_thresholds",
        int(len(pretest)),
    )

    return result
