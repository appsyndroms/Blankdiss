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
from .si_event_risk_signal_anatomy import (
    LOCKED_EVENT_THRESHOLD,
    LOCKED_HORIZON,
    _prepare_signal,
)


B_GROUP = PRIMARY_GROUPS[0]

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

ANATOMY_COLUMNS = (
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

QUARTILES = (
    "Q1",
    "Q2",
    "Q3",
    "Q4",
)


def _safe_values(
    frame: pd.DataFrame,
    column: str,
) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(dtype=float, index=frame.index)
    return _numeric(frame, column)


def _safe_mean(
    frame: pd.DataFrame,
    column: str,
) -> float:
    values = _safe_values(frame, column).dropna()
    if values.empty:
        return float("nan")
    return float(values.mean())


def _safe_median(
    frame: pd.DataFrame,
    column: str,
) -> float:
    values = _safe_values(frame, column).dropna()
    if values.empty:
        return float("nan")
    return float(values.median())


def _spearman(
    frame: pd.DataFrame,
    x_column: str,
    y_column: str,
) -> tuple[float, int]:
    if (
        x_column not in frame.columns
        or y_column not in frame.columns
    ):
        return float("nan"), 0

    local = pd.DataFrame({
        "x": _numeric(frame, x_column),
        "y": _numeric(frame, y_column),
    }).dropna()

    if len(local) < 3:
        return float("nan"), int(len(local))

    if local["x"].nunique() < 2 or local["y"].nunique() < 2:
        return float("nan"), int(len(local))

    value = local["x"].corr(
        local["y"],
        method="spearman",
    )
    return float(value), int(len(local))


def _quartile_labels(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(
        values,
        errors="coerce",
    )

    result = pd.Series(
        "insufficient_data",
        index=values.index,
        dtype="object",
    )

    valid = numeric.notna()
    if int(valid.sum()) < 4:
        return result

    ranks = numeric.loc[valid].rank(
        method="first",
        pct=True,
    )

    result.loc[valid] = pd.cut(
        ranks,
        bins=[0.0, 0.25, 0.50, 0.75, 1.0],
        labels=QUARTILES,
        include_lowest=True,
    ).astype(str)

    return result


def _prepare_quartiles(
    test: pd.DataFrame,
) -> pd.DataFrame:
    local = test.copy()

    for column in ANATOMY_COLUMNS:
        if column not in local.columns:
            continue

        safe_name = column.replace(
            "return_",
            "ret_",
        )
        local[f"quartile_{safe_name}"] = _quartile_labels(
            local[column]
        )

    return local


def _continuous_correlations(
    test: pd.DataFrame,
) -> pd.DataFrame:
    b = test[
        test["context_group"] == B_GROUP
    ].copy()

    rows: list[dict] = []

    for horizon in FORWARD_HORIZONS:
        forward_column = FORWARD_COLUMNS[horizon]

        for column in ANATOMY_COLUMNS:
            correlation, n = _spearman(
                b,
                column,
                forward_column,
            )

            rows.append({
                "horizon": horizon,
                "feature": column,
                "spearman_rho": correlation,
                "n": n,
                "absolute_rho": (
                    abs(correlation)
                    if np.isfinite(correlation)
                    else float("nan")
                ),
            })

    return pd.DataFrame(rows)


def _quartile_outcomes(
    test: pd.DataFrame,
) -> pd.DataFrame:
    b = test[
        test["context_group"] == B_GROUP
    ].copy()

    rows: list[dict] = []

    for horizon in FORWARD_HORIZONS:
        forward_column = FORWARD_COLUMNS[horizon]

        for feature in ANATOMY_COLUMNS:
            safe_name = feature.replace(
                "return_",
                "ret_",
            )
            quartile_column = (
                f"quartile_{safe_name}"
            )

            if quartile_column not in b.columns:
                continue

            for quartile in QUARTILES:
                frame = b[
                    b[quartile_column] == quartile
                ].copy()

                values = _safe_values(
                    frame,
                    forward_column,
                ).dropna()

                rows.append({
                    "horizon": horizon,
                    "feature": feature,
                    "quartile": quartile,
                    "n": int(len(values)),
                    "mean_forward_return": (
                        float(values.mean())
                        if not values.empty
                        else float("nan")
                    ),
                    "median_forward_return": (
                        float(values.median())
                        if not values.empty
                        else float("nan")
                    ),
                    "downside_rate_le_minus_5pct": (
                        float((values <= -0.05).mean())
                        if not values.empty
                        else float("nan")
                    ),
                    "upside_rate_ge_plus_5pct": (
                        float((values >= 0.05).mean())
                        if not values.empty
                        else float("nan")
                    ),
                })

    return pd.DataFrame(rows)


def _quartile_monotonicity(
    quartile_outcomes: pd.DataFrame,
) -> pd.DataFrame:
    if quartile_outcomes.empty:
        return pd.DataFrame()

    rows: list[dict] = []

    for (horizon, feature), group in (
        quartile_outcomes
        .groupby(["horizon", "feature"])
    ):
        local = group[
            group["quartile"].isin(QUARTILES)
        ].copy()

        local["quartile_number"] = (
            local["quartile"]
            .map({
                "Q1": 1,
                "Q2": 2,
                "Q3": 3,
                "Q4": 4,
            })
        )

        local = local.dropna(
            subset=[
                "quartile_number",
                "mean_forward_return",
            ]
        )

        if len(local) < 3:
            rows.append({
                "horizon": horizon,
                "feature": feature,
                "quartile_return_rho": float("nan"),
                "n_quartiles": int(len(local)),
            })
            continue

        rho = local["quartile_number"].corr(
            local["mean_forward_return"],
            method="spearman",
        )

        rows.append({
            "horizon": horizon,
            "feature": feature,
            "quartile_return_rho": float(rho),
            "n_quartiles": int(len(local)),
        })

    return pd.DataFrame(rows)


def _yearly_correlations(
    test: pd.DataFrame,
) -> pd.DataFrame:
    if "snapshot_date" not in test.columns:
        return pd.DataFrame()

    b = test[
        test["context_group"] == B_GROUP
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
        b["_signal_year"].astype(int)
    )

    rows: list[dict] = []

    for year in sorted(
        b["_signal_year"].unique()
    ):
        year_frame = b[
            b["_signal_year"] == year
        ]

        for horizon in FORWARD_HORIZONS:
            forward_column = FORWARD_COLUMNS[
                horizon
            ]

            for feature in ANATOMY_COLUMNS:
                correlation, n = _spearman(
                    year_frame,
                    feature,
                    forward_column,
                )

                rows.append({
                    "year": int(year),
                    "horizon": horizon,
                    "feature": feature,
                    "spearman_rho": correlation,
                    "n": n,
                })

    return pd.DataFrame(rows)


def _yearly_quartile_outcomes(
    test: pd.DataFrame,
) -> pd.DataFrame:
    if "snapshot_date" not in test.columns:
        return pd.DataFrame()

    b = test[
        test["context_group"] == B_GROUP
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
        b["_signal_year"].astype(int)
    )

    rows: list[dict] = []

    for year in sorted(
        b["_signal_year"].unique()
    ):
        year_frame = b[
            b["_signal_year"] == year
        ].copy()

        for horizon in FORWARD_HORIZONS:
            forward_column = FORWARD_COLUMNS[
                horizon
            ]

            for feature in ANATOMY_COLUMNS:
                safe_name = feature.replace(
                    "return_",
                    "ret_",
                )
                quartile_column = (
                    f"quartile_{safe_name}"
                )

                if quartile_column not in (
                    year_frame.columns
                ):
                    continue

                for quartile in QUARTILES:
                    frame = year_frame[
                        year_frame[quartile_column]
                        == quartile
                    ]

                    values = _safe_values(
                        frame,
                        forward_column,
                    ).dropna()

                    rows.append({
                        "year": int(year),
                        "horizon": horizon,
                        "feature": feature,
                        "quartile": quartile,
                        "n": int(len(values)),
                        "mean_forward_return": (
                            float(values.mean())
                            if not values.empty
                            else float("nan")
                        ),
                        "median_forward_return": (
                            float(values.median())
                            if not values.empty
                            else float("nan")
                        ),
                    })

    return pd.DataFrame(rows)


def _B_summary(
    test: pd.DataFrame,
) -> pd.DataFrame:
    b = test[
        test["context_group"] == B_GROUP
    ].copy()

    rows: list[dict] = []

    for feature in ANATOMY_COLUMNS:
        rows.append({
            "feature": feature,
            "n": int(
                _safe_values(
                    b,
                    feature,
                ).notna().sum()
            ),
            "mean": _safe_mean(
                b,
                feature,
            ),
            "median": _safe_median(
                b,
                feature,
            ),
            "q25": (
                float(
                    _safe_values(
                        b,
                        feature,
                    ).dropna().quantile(0.25)
                )
                if not _safe_values(
                    b,
                    feature,
                ).dropna().empty
                else float("nan")
            ),
            "q75": (
                float(
                    _safe_values(
                        b,
                        feature,
                    ).dropna().quantile(0.75)
                )
                if not _safe_values(
                    b,
                    feature,
                ).dropna().empty
                else float("nan")
            ),
        })

    return pd.DataFrame(rows)


def run_b_continuous_anatomy(
    context,
) -> ExperimentResult:
    test, pretest, info = _prepare_signal(
        context
    )

    test = _prepare_groups(test)
    test = _prepare_quartiles(test)

    b = test[
        test["context_group"] == B_GROUP
    ].copy()

    if b.empty:
        raise ValueError(
            "No B observations were available "
            "for continuous anatomy analysis."
        )

    correlations = _continuous_correlations(
        test
    )

    quartile_outcomes = _quartile_outcomes(
        test
    )

    monotonicity = _quartile_monotonicity(
        quartile_outcomes
    )

    yearly_correlations = _yearly_correlations(
        test
    )

    yearly_quartiles = (
        _yearly_quartile_outcomes(test)
    )

    result = ExperimentResult(
        name=(
            "si_event_risk_b_continuous_anatomy"
        ),
        description=(
            "Kontinuerlig deskriptiv analys av "
            "B-cellen från den låsta SI × "
            "event-risk-signalen. Analysen "
            "undersöker Spearman-samband och "
            "fasta kvartiler mellan information "
            "som fanns vid signalögonblicket "
            "och framtida avkastning."
        ),
    )

    result.add_metric(
        "analysis_type",
        "locked_B_continuous_anatomy",
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
        "quartile_method",
        "rank_based_fixed_quartiles",
    )
    result.add_metric(
        "forward_horizons",
        list(FORWARD_HORIZONS),
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
        "pre_signal_columns_checked",
        list(ANATOMY_COLUMNS),
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
            "Finns det en kontinuerlig struktur "
            "inne i B, där pris-, SI- eller "
            "event-risk-kontext vid signalen "
            "samvarierar med framtida avkastning?"
        ),
    )
    result.add_metric(
        "interpretation_guardrail",
        (
            "All correlations and quartiles are "
            "descriptive. No outcome information "
            "is used to select thresholds, "
            "parameters or trading rules."
        ),
    )
    result.add_metric(
        "yearly_breakdown",
        True,
    )
    result.add_metric(
        "pretest_rows_used_for_thresholds",
        int(len(pretest)),
    )

    result.add_table(
        "B_summary",
        _B_summary(test),
    )
    result.add_table(
        "B_continuous_correlations",
        correlations,
    )
    result.add_table(
        "B_quartile_outcomes",
        quartile_outcomes,
    )
    result.add_table(
        "B_quartile_monotonicity",
        monotonicity,
    )
    result.add_table(
        "B_yearly_correlations",
        yearly_correlations,
    )
    result.add_table(
        "B_yearly_quartile_outcomes",
        yearly_quartiles,
    )

    return result
