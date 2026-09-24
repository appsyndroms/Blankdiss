from __future__ import annotations

import numpy as np
import pandas as pd


HORIZONS = (
    1,
    3,
    5,
    10,
    20,
)

SI_CHANGE_CUTOFFS = (
    0.10,
    0.20,
    0.30,
)

PRIOR_RETURN_COLUMNS = (
    "price_return_5d",
)


def _numeric(
    frame: pd.DataFrame,
    column: str,
) -> pd.Series:
    return pd.to_numeric(
        frame[column],
        errors="coerce",
    )


def _summary(
    frame: pd.DataFrame,
    column: str,
) -> dict[str, float | int]:
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
    """
    Bootstrap 95% CI for:

        mean(first) - mean(second)

    This intentionally preserves the bootstrap semantics of the
    original diagnostic.
    """
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

    first_values = first.to_numpy(
        dtype=float
    )

    second_values = second.to_numpy(
        dtype=float
    )

    observed = float(
        first_values.mean()
        - second_values.mean()
    )

    rng = np.random.default_rng(
        seed
    )

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


def add_si_change(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add current short-interest change.

    The change is calculated within security and in chronological
    order, using the previous observed FI snapshot.
    """
    result = frame.copy()

    if "short_interest_pct_change" in result.columns:
        return result

    if "short_interest_pct" not in result.columns:
        raise KeyError(
            "SI-dynamik saknar "
            "'short_interest_pct'."
        )

    symbol_column = (
        "yahoo_symbol"
        if "yahoo_symbol" in result.columns
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
        if column not in result.columns
    ]

    if missing:
        raise KeyError(
            "SI-dynamik saknar kolumner: "
            + ", ".join(missing)
        )

    result["snapshot_date"] = pd.to_datetime(
        result["snapshot_date"],
        errors="coerce",
    )

    result["short_interest_pct"] = _numeric(
        result,
        "short_interest_pct",
    )

    result = result.sort_values(
        [symbol_column, "snapshot_date"]
    ).copy()

    result["short_interest_pct_change"] = (
        result.groupby(
            symbol_column,
            sort=False,
        )["short_interest_pct"]
        .diff()
    )

    return result


def _positive_si_threshold(
    pretest: pd.DataFrame,
    cutoff: float,
) -> float:
    values = _numeric(
        pretest,
        "short_interest_pct_change",
    )

    values = values[
        values > 0
    ].dropna()

    if values.empty:
        return float("nan")

    return float(
        values.quantile(
            1.0 - cutoff
        )
    )


def _build_prior_groups(
    frame: pd.DataFrame,
    *,
    prior_column: str,
    moderate_threshold: float,
    strong_threshold: float,
) -> pd.DataFrame:
    result = frame.copy()

    prior = _numeric(
        result,
        prior_column,
    )

    result["prior_return_group"] = np.select(
        [
            prior <= 0,
            (
                (prior > 0)
                & (prior <= moderate_threshold)
            ),
            (
                (prior > moderate_threshold)
                & (prior <= strong_threshold)
            ),
            prior > strong_threshold,
        ],
        [
            "negative_or_flat",
            "positive_moderate",
            "positive_strong",
            "positive_extreme",
        ],
        default="missing",
    )

    return result


def _build_results(
    pretest: pd.DataFrame,
    test: pd.DataFrame,
    *,
    prior_column: str,
    si_change_cutoffs: tuple[float, ...],
    horizons: tuple[int, ...],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    positive_prior = _numeric(
        pretest,
        prior_column,
    )

    positive_prior = positive_prior[
        positive_prior > 0
    ].dropna()

    if positive_prior.empty:
        return (
            pd.DataFrame(),
            pd.DataFrame(),
        )

    moderate_threshold = float(
        positive_prior.quantile(0.50)
    )

    strong_threshold = float(
        positive_prior.quantile(0.80)
    )

    thresholds = {
        cutoff: _positive_si_threshold(
            pretest,
            cutoff,
        )
        for cutoff in si_change_cutoffs
    }

    frame = _build_prior_groups(
        test,
        prior_column=prior_column,
        moderate_threshold=moderate_threshold,
        strong_threshold=strong_threshold,
    )

    frame = frame[
        frame["prior_return_group"]
        != "missing"
    ].copy()

    rows: list[dict] = []

    for cutoff in si_change_cutoffs:
        threshold = thresholds.get(
            cutoff,
            float("nan"),
        )

        if not np.isfinite(
            threshold
        ):
            continue

        change = _numeric(
            frame,
            "short_interest_pct_change",
        )

        grouped = frame.copy()

        grouped["positive_si_change"] = (
            change > 0
        )

        grouped["high_si_change"] = (
            change >= threshold
        )

        grouped = grouped[
            grouped["positive_si_change"]
        ].copy()

        for prior_group in (
            "negative_or_flat",
            "positive_moderate",
            "positive_strong",
            "positive_extreme",
        ):
            prior_frame = grouped[
                grouped["prior_return_group"]
                == prior_group
            ]

            high = prior_frame[
                prior_frame["high_si_change"]
            ]

            other = prior_frame[
                ~prior_frame["high_si_change"]
            ]

            for horizon in horizons:
                return_column = (
                    f"forward_return_{horizon}d"
                )

                if (
                    return_column
                    not in prior_frame.columns
                ):
                    continue

                high_summary = _summary(
                    high,
                    return_column,
                )

                other_summary = _summary(
                    other,
                    return_column,
                )

                (
                    delta,
                    ci_low,
                    ci_high,
                ) = _bootstrap_mean_difference(
                    high[return_column],
                    other[return_column],
                )

                rows.append(
                    {
                        "prior_return_column": (
                            prior_column
                        ),
                        "prior_group": (
                            prior_group
                        ),
                        "prior_moderate_threshold": (
                            moderate_threshold
                        ),
                        "prior_strong_threshold": (
                            strong_threshold
                        ),
                        "si_change_cutoff": (
                            cutoff
                        ),
                        "si_change_threshold": (
                            threshold
                        ),
                        "horizon_days": (
                            horizon
                        ),
                        "high_si_n": (
                            high_summary["n"]
                        ),
                        "other_positive_n": (
                            other_summary["n"]
                        ),
                        "high_si_mean_return": (
                            high_summary["mean"]
                        ),
                        "other_positive_mean_return": (
                            other_summary["mean"]
                        ),
                        "mean_return_delta": (
                            delta
                        ),
                        "mean_return_ci_low": (
                            ci_low
                        ),
                        "mean_return_ci_high": (
                            ci_high
                        ),
                        "high_si_median_return": (
                            high_summary["median"]
                        ),
                        "other_positive_median_return": (
                            other_summary["median"]
                        ),
                        "high_si_positive_rate": (
                            high_summary["positive_rate"]
                        ),
                        "other_positive_rate": (
                            other_summary["positive_rate"]
                        ),
                    }
                )

    result = pd.DataFrame(
        rows
    )

    thresholds_result = pd.DataFrame(
        [
            {
                "prior_return_column": (
                    prior_column
                ),
                "positive_prior_median": (
                    moderate_threshold
                ),
                "positive_prior_p80": (
                    strong_threshold
                ),
                "si_change_top10_threshold": (
                    thresholds.get(
                        0.10,
                        float("nan"),
                    )
                ),
                "si_change_top20_threshold": (
                    thresholds.get(
                        0.20,
                        float("nan"),
                    )
                ),
                "si_change_top30_threshold": (
                    thresholds.get(
                        0.30,
                        float("nan"),
                    )
                ),
            }
        ]
    )

    return (
        result,
        thresholds_result,
    )


def run(
    frame: pd.DataFrame,
    *,
    pretest_end: str | pd.Timestamp,
    test_end: str | pd.Timestamp | None = None,
    prior_column: str = "price_return_5d",
    si_change_cutoffs: tuple[float, ...] = (
        SI_CHANGE_CUTOFFS
    ),
    horizons: tuple[int, ...] = HORIZONS,
) -> pd.DataFrame:
    """
    Run the SI/prior-return interaction analysis.

    SI-change thresholds are calculated exclusively from pretest data.
    The actual interaction analysis is performed on the test period.

    Important:
    The old diagnostic calculates SI change separately inside the
    pretest and test slices. We preserve that behavior here so the
    first test observation for a security does not inherit an SI
    change from the final pretest observation.
    """
    data = frame.copy()

    data["snapshot_date"] = pd.to_datetime(
        data["snapshot_date"],
        errors="coerce",
    )

    pretest_end = pd.Timestamp(
        pretest_end
    )

    pretest = data[
        data["snapshot_date"]
        <= pretest_end
    ].copy()

    test = data[
        data["snapshot_date"]
        > pretest_end
    ].copy()

    if test_end is not None:
        test_end = pd.Timestamp(
            test_end
        )

        test = test[
            test["snapshot_date"]
            <= test_end
        ].copy()

    # Preserve the old diagnostic's exact window semantics:
    # SI change is calculated independently within pretest and test.
    pretest = add_si_change(
        pretest
    )

    test = add_si_change(
        test
    )

    results, _ = _build_results(
        pretest,
        test,
        prior_column=prior_column,
        si_change_cutoffs=si_change_cutoffs,
        horizons=horizons,
    )

    return results
