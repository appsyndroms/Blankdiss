from __future__ import annotations

import numpy as np
import pandas as pd

from ml.research.bootstrap import bootstrap_mean_difference


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
            "SI-dynamik saknar 'short_interest_pct'."
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
            + ", ".join(sorted(missing))
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


def _build_results(
    pretest: pd.DataFrame,
    test: pd.DataFrame,
    *,
    prior_column: str,
    si_change_cutoffs: tuple[float, ...],
    horizons: tuple[int, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
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
        frame["prior_return_group"] != "missing"
    ].copy()

    rows: list[dict] = []

    for cutoff in si_change_cutoffs:
        threshold = thresholds.get(
            cutoff,
            float("nan"),
        )

        if not np.isfinite(threshold):
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
            ].copy()

            for horizon in horizons:
                return_column = (
                    f"forward_return_{horizon}d"
                )

                if return_column not in prior_frame:
                    continue

                high = prior_frame[
                    prior_frame["high_si_change"]
                ]

                other = prior_frame[
                    ~prior_frame["high_si_change"]
                ]

                high_values = _numeric(
                    high,
                    return_column,
                ).dropna()

                other_values = _numeric(
                    other,
                    return_column,
                ).dropna()

                if high_values.empty:
                    continue

                row = {
                    "prior_return_group": prior_group,
                    "si_change_cutoff": cutoff,
                    "si_change_threshold": threshold,
                    "horizon": horizon,
                    "high_si_change_n": int(
                        len(high_values)
                    ),
                    "other_positive_si_change_n": int(
                        len(other_values)
                    ),
                    "high_si_change_mean_return": float(
                        high_values.mean()
                    ),
                    "other_positive_si_change_mean_return": (
                        float(other_values.mean())
                        if not other_values.empty
                        else float("nan")
                    ),
                }

                if not other_values.empty:
                    bootstrap = bootstrap_mean_difference(
                        high_values.to_numpy(),
                        other_values.to_numpy(),
                    )

                    row.update(
                        {
                            "mean_return_difference": (
                                float(
                                    high_values.mean()
                                    - other_values.mean()
                                )
                            ),
                            "bootstrap_ci_low": (
                                bootstrap[0]
                            ),
                            "bootstrap_ci_high": (
                                bootstrap[1]
                            ),
                        }
                    )
                else:
                    row.update(
                        {
                            "mean_return_difference": float(
                                "nan"
                            ),
                            "bootstrap_ci_low": float(
                                "nan"
                            ),
                            "bootstrap_ci_high": float(
                                "nan"
                            ),
                        }
                    )

                rows.append(row)

    return pd.DataFrame(rows), frame


def run(
    frame: pd.DataFrame,
    *,
    pretest_end: str | pd.Timestamp,
    test_end: str | pd.Timestamp | None = None,
    prior_column: str = "price_return_5d",
    si_change_cutoffs: tuple[float, ...] = SI_CHANGE_CUTOFFS,
    horizons: tuple[int, ...] = HORIZONS,
) -> pd.DataFrame:
    """
    Run the SI/prior-return interaction analysis.

    SI-change thresholds are calculated exclusively from pretest data.
    The actual interaction analysis is performed on the test period.
    """
    data = add_si_change(frame)

    data["snapshot_date"] = pd.to_datetime(
        data["snapshot_date"],
        errors="coerce",
    )

    pretest_end = pd.Timestamp(pretest_end)

    pretest = data[
        data["snapshot_date"] <= pretest_end
    ].copy()

    test = data[
        data["snapshot_date"] > pretest_end
    ].copy()

    if test_end is not None:
        test_end = pd.Timestamp(test_end)

        test = test[
            test["snapshot_date"] <= test_end
        ].copy()

    results, _ = _build_results(
        pretest,
        test,
        prior_column=prior_column,
        si_change_cutoffs=si_change_cutoffs,
        horizons=horizons,
    )

    return results
