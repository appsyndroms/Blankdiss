from __future__ import annotations

import numpy as np
import pandas as pd

from .base import ExperimentResult


def _numeric(
    frame: pd.DataFrame,
    column: str,
) -> pd.Series:
    return pd.to_numeric(
        frame[column],
        errors="coerce",
    )


def _quantile(
    frame: pd.DataFrame,
    column: str,
    q: float,
) -> float:
    if column not in frame.columns:
        return float("nan")

    values = _numeric(
        frame,
        column,
    ).dropna()

    if values.empty:
        return float("nan")

    return float(values.quantile(q))


def _summary(
    frame: pd.DataFrame,
    column: str,
) -> dict[str, float | int]:
    if column not in frame.columns:
        return {
            "n": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "positive_rate": float("nan"),
        }

    values = _numeric(
        frame,
        column,
    ).dropna()

    if values.empty:
        return {
            "n": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "positive_rate": float("nan"),
        }

    return {
        "n": int(len(values)),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "positive_rate": float(
            (values > 0).mean()
        ),
    }


def _bootstrap_mean_difference(
    first: pd.Series,
    second: pd.Series,
    *,
    iterations: int = 2000,
    seed: int = 42,
) -> tuple[float, float, float]:
    first = pd.to_numeric(
        first,
        errors="coerce",
    ).dropna()

    second = pd.to_numeric(
        second,
        errors="coerce",
    ).dropna()

    if first.empty or second.empty:
        return (
            float("nan"),
            float("nan"),
            float("nan"),
        )

    observed = float(
        first.mean() - second.mean()
    )

    rng = np.random.default_rng(seed)

    first_values = first.to_numpy()
    second_values = second.to_numpy()

    bootstrap = np.empty(
        iterations,
        dtype=float,
    )

    for index in range(iterations):
        first_sample = rng.choice(
            first_values,
            size=len(first_values),
            replace=True,
        )

        second_sample = rng.choice(
            second_values,
            size=len(second_values),
            replace=True,
        )

        bootstrap[index] = (
            first_sample.mean()
            - second_sample.mean()
        )

    low, high = np.quantile(
        bootstrap,
        [0.025, 0.975],
    )

    return (
        observed,
        float(low),
        float(high),
    )


def run_si_level_change_joint(
    context,
    *,
    horizons: tuple[int, ...],
    level_quantiles: tuple[float, ...],
    change_cutoffs: tuple[float, ...],
) -> ExperimentResult:
    pretest = context.pretest
    test = context.test

    required = {
        "short_interest_pct",
        "short_interest_pct_change",
    }

    missing = [
        column
        for column in required
        if column not in test.columns
    ]

    if missing:
        raise KeyError(
            "SI level/change saknar kolumner: "
            + ", ".join(missing)
        )

    rows: list[dict] = []

    for level_quantile in level_quantiles:
        level_threshold = _quantile(
            pretest,
            "short_interest_pct",
            level_quantile,
        )

        if not np.isfinite(level_threshold):
            continue

        for change_cutoff in change_cutoffs:
            positive_changes = _numeric(
                pretest,
                "short_interest_pct_change",
            )

            positive_changes = positive_changes[
                positive_changes > 0
            ].dropna()

            if positive_changes.empty:
                continue

            change_threshold = float(
                positive_changes.quantile(
                    1.0 - change_cutoff
                )
            )

            frame = test.copy()

            level = _numeric(
                frame,
                "short_interest_pct",
            )

            change = _numeric(
                frame,
                "short_interest_pct_change",
            )

            frame["high_si_level"] = (
                level >= level_threshold
            )

            frame["high_si_change"] = (
                change >= change_threshold
            )

            frame = frame[
                change > 0
            ].copy()

            groups = {
                "low_level_low_change": frame[
                    (~frame["high_si_level"])
                    & (~frame["high_si_change"])
                ],
                "high_level_low_change": frame[
                    frame["high_si_level"]
                    & (~frame["high_si_change"])
                ],
                "low_level_high_change": frame[
                    (~frame["high_si_level"])
                    & frame["high_si_change"]
                ],
                "high_level_high_change": frame[
                    frame["high_si_level"]
                    & frame["high_si_change"]
                ],
            }

            for horizon in horizons:
                return_column = (
                    f"forward_return_{horizon}d"
                )

                if return_column not in frame.columns:
                    continue

                summaries = {
                    name: _summary(
                        group,
                        return_column,
                    )
                    for name, group in groups.items()
                }

                rows.append(
                    {
                        "level_quantile": level_quantile,
                        "level_threshold": level_threshold,
                        "change_cutoff": change_cutoff,
                        "change_threshold": change_threshold,
                        "horizon_days": horizon,
                        "low_level_low_change_n": summaries[
                            "low_level_low_change"
                        ]["n"],
                        "high_level_low_change_n": summaries[
                            "high_level_low_change"
                        ]["n"],
                        "low_level_high_change_n": summaries[
                            "low_level_high_change"
                        ]["n"],
                        "high_level_high_change_n": summaries[
                            "high_level_high_change"
                        ]["n"],
                        "low_level_low_change_mean": summaries[
                            "low_level_low_change"
                        ]["mean"],
                        "high_level_low_change_mean": summaries[
                            "high_level_low_change"
                        ]["mean"],
                        "low_level_high_change_mean": summaries[
                            "low_level_high_change"
                        ]["mean"],
                        "high_level_high_change_mean": summaries[
                            "high_level_high_change"
                        ]["mean"],
                    }
                )

    result = ExperimentResult(
        name="si_level_change_joint",
        description=(
            "Separera effekten av hög blankningsnivå "
            "från effekten av en stor positiv förändring "
            "i short interest."
        ),
    )

    result.add_table(
        "level_change_matrix",
        pd.DataFrame(rows),
    )

    return result


def _prepare_future_si(
    context,
) -> pd.DataFrame:
    data = context.data.copy()

    symbol_column = (
        "yahoo_symbol"
        if "yahoo_symbol" in data.columns
        else "security_key"
    )

    required = {
        symbol_column,
        "snapshot_date",
        "short_interest_pct",
    }

    missing = [
        column
        for column in required
        if column not in data.columns
    ]

    if missing:
        raise KeyError(
            "SI persistence saknar kolumner: "
            + ", ".join(missing)
        )

    data["snapshot_date"] = pd.to_datetime(
        data["snapshot_date"],
        errors="coerce",
    )

    data["short_interest_pct"] = _numeric(
        data,
        "short_interest_pct",
    )

    data = data.sort_values(
        [symbol_column, "snapshot_date"]
    ).copy()

    grouped = data.groupby(
        symbol_column,
        sort=False,
    )

    data["future_short_interest_pct"] = (
        grouped["short_interest_pct"]
        .shift(-1)
    )

    data["future_snapshot_date"] = (
        grouped["snapshot_date"]
        .shift(-1)
    )

    data["future_si_change"] = (
        data["future_short_interest_pct"]
        - data["short_interest_pct"]
    )

    test_end = (
        pd.Timestamp(context.test_end)
        if context.test_end is not None
        else None
    )

    validation_end = pd.Timestamp(
        context.validation_end
    )

    mask = (
        data["snapshot_date"]
        > validation_end
    )

    if test_end is not None:
        mask &= (
            data["snapshot_date"]
            <= test_end
        )

    data = data.loc[mask].copy()

    if test_end is not None:
        data = data[
            data["future_snapshot_date"]
            <= test_end
        ].copy()

    return data


def run_si_change_persistence(
    context,
    *,
    change_cutoffs: tuple[float, ...],
    horizons: tuple[int, ...],
) -> ExperimentResult:
    pretest = context.pretest

    test = _prepare_future_si(
        context
    )

    if "short_interest_pct_change" not in test.columns:
        # Reconstruct current SI change on the full
        # chronological dataset so that the first
        # test observation for a security can use
        # its immediately preceding observation.
        symbol_column = (
            "yahoo_symbol"
            if "yahoo_symbol" in context.data.columns
            else "security_key"
        )

        all_data = context.data.copy()

        all_data["snapshot_date"] = pd.to_datetime(
            all_data["snapshot_date"],
            errors="coerce",
        )

        all_data["short_interest_pct"] = _numeric(
            all_data,
            "short_interest_pct",
        )

        all_data = all_data.sort_values(
            [symbol_column, "snapshot_date"]
        )

        all_data["short_interest_pct_change"] = (
            all_data.groupby(
                symbol_column,
                sort=False,
            )["short_interest_pct"]
            .diff()
        )

        test_keys = test[
            [
                symbol_column,
                "snapshot_date",
            ]
        ].drop_duplicates()

        test = test.drop(
            columns=[
                "short_interest_pct_change"
            ],
            errors="ignore",
        )

        test = test.merge(
            all_data[
                [
                    symbol_column,
                    "snapshot_date",
                    "short_interest_pct_change",
                ]
            ],
            on=[
                symbol_column,
                "snapshot_date",
            ],
            how="left",
        )

    rows: list[dict] = []

    for cutoff in change_cutoffs:
        positive_pretest = _numeric(
            pretest,
            "short_interest_pct_change",
        )

        positive_pretest = positive_pretest[
            positive_pretest > 0
        ].dropna()

        if positive_pretest.empty:
            continue

        threshold = float(
            positive_pretest.quantile(
                1.0 - cutoff
            )
        )

        frame = test.copy()

        current_change = _numeric(
            frame,
            "short_interest_pct_change",
        )

        future_change = _numeric(
            frame,
            "future_si_change",
        )

        frame["high_si_change"] = (
            current_change >= threshold
        )

        frame = frame[
            current_change > 0
        ].copy()

        high = frame[
            frame["high_si_change"]
        ]

        other = frame[
            ~frame["high_si_change"]
        ]

        high_future = future_change.loc[
            high.index
        ].dropna()

        other_future = future_change.loc[
            other.index
        ].dropna()

        observed, ci_low, ci_high = (
            _bootstrap_mean_difference(
                high_future,
                other_future,
            )
        )

        for horizon in horizons:
            return_column = (
                f"forward_return_{horizon}d"
            )

            if return_column not in frame.columns:
                continue

            high_return = _summary(
                high,
                return_column,
            )

            other_return = _summary(
                other,
                return_column,
            )

            rows.append(
                {
                    "change_cutoff": cutoff,
                    "change_threshold": threshold,
                    "horizon_days": horizon,
                    "high_si_n": high_return["n"],
                    "other_positive_n": other_return["n"],
                    "high_si_mean_return": high_return[
                        "mean"
                    ],
                    "other_positive_mean_return": other_return[
                        "mean"
                    ],
                    "price_return_delta": (
                        high_return["mean"]
                        - other_return["mean"]
                    ),
                    "high_si_future_change_n": len(
                        high_future
                    ),
                    "other_positive_future_change_n": len(
                        other_future
                    ),
                    "high_si_future_change_mean": (
                        float(high_future.mean())
                        if not high_future.empty
                        else float("nan")
                    ),
                    "other_positive_future_change_mean": (
                        float(other_future.mean())
                        if not other_future.empty
                        else float("nan")
                    ),
                    "future_si_change_delta": observed,
                    "future_si_change_ci_low": ci_low,
                    "future_si_change_ci_high": ci_high,
                }
            )

    result = ExperimentResult(
        name="si_change_persistence",
        description=(
            "Testar om stora ökningar i short interest "
            "följs av fortsatt förändring i blankningen "
            "och hur detta hänger ihop med efterföljande "
            "kursutveckling."
        ),
    )

    result.add_table(
        "si_change_persistence",
        pd.DataFrame(rows),
    )

    return result


def run_si_concentration(
    context,
    *,
    concentration_columns: tuple[str, ...],
    concentration_quantile: float,
    horizons: tuple[int, ...],
    change_cutoff: float,
) -> ExperimentResult:
    pretest = context.pretest
    test = context.test.copy()

    if "short_interest_pct_change" not in test.columns:
        raise KeyError(
            "SI concentration saknar "
            "'short_interest_pct_change'."
        )

    positive_pretest = _numeric(
        pretest,
        "short_interest_pct_change",
    )

    positive_pretest = positive_pretest[
        positive_pretest > 0
    ].dropna()

    if positive_pretest.empty:
        raise ValueError(
            "Ingen positiv SI-förändring i pretest."
        )

    change_threshold = float(
        positive_pretest.quantile(
            1.0 - change_cutoff
        )
    )

    test["high_si_change"] = (
        _numeric(
            test,
            "short_interest_pct_change",
        )
        >= change_threshold
    )

    test = test[
        _numeric(
            test,
            "short_interest_pct_change",
        ) > 0
    ].copy()

    rows: list[dict] = []

    for concentration_column in concentration_columns:
        if concentration_column not in test.columns:
            continue

        concentration_threshold = _quantile(
            pretest,
            concentration_column,
            concentration_quantile,
        )

        if not np.isfinite(
            concentration_threshold
        ):
            continue

        concentration = _numeric(
            test,
            concentration_column,
        )

        test["high_concentration"] = (
            concentration
            >= concentration_threshold
        )

        high_concentration_high_change = test[
            test["high_concentration"]
            & test["high_si_change"]
        ]

        high_concentration_other_change = test[
            test["high_concentration"]
            & ~test["high_si_change"]
        ]

        low_concentration_high_change = test[
            ~test["high_concentration"]
            & test["high_si_change"]
        ]

        low_concentration_other_change = test[
            ~test["high_concentration"]
            & ~test["high_si_change"]
        ]

        groups = {
            "high_concentration_high_change":
                high_concentration_high_change,
            "high_concentration_other_change":
                high_concentration_other_change,
            "low_concentration_high_change":
                low_concentration_high_change,
            "low_concentration_other_change":
                low_concentration_other_change,
        }

        for horizon in horizons:
            return_column = (
                f"forward_return_{horizon}d"
            )

            if return_column not in test.columns:
                continue

            summaries = {
                name: _summary(
                    group,
                    return_column,
                )
                for name, group in groups.items()
            }

            rows.append(
                {
                    "concentration_column":
                        concentration_column,
                    "concentration_quantile":
                        concentration_quantile,
                    "concentration_threshold":
                        concentration_threshold,
                    "change_cutoff":
                        change_cutoff,
                    "change_threshold":
                        change_threshold,
                    "horizon_days":
                        horizon,
                    "high_concentration_high_change_n":
                        summaries[
                            "high_concentration_high_change"
                        ]["n"],
                    "high_concentration_other_change_n":
                        summaries[
                            "high_concentration_other_change"
                        ]["n"],
                    "low_concentration_high_change_n":
                        summaries[
                            "low_concentration_high_change"
                        ]["n"],
                    "low_concentration_other_change_n":
                        summaries[
                            "low_concentration_other_change"
                        ]["n"],
                    "high_concentration_high_change_mean":
                        summaries[
                            "high_concentration_high_change"
                        ]["mean"],
                    "high_concentration_other_change_mean":
                        summaries[
                            "high_concentration_other_change"
                        ]["mean"],
                    "low_concentration_high_change_mean":
                        summaries[
                            "low_concentration_high_change"
                        ]["mean"],
                    "low_concentration_other_change_mean":
                        summaries[
                            "low_concentration_other_change"
                        ]["mean"],
                }
            )

    result = ExperimentResult(
        name="si_concentration",
        description=(
            "Testar om stora förändringar i short interest "
            "ger olika signal beroende på hur koncentrerad "
            "blankningen är."
        ),
    )

    result.add_table(
        "si_concentration",
        pd.DataFrame(rows),
    )

    return result
