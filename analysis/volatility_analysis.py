"""Beskrivande analys av 20-dagars prisvolatilitet.

Syfte:
    Förklara vad ML-ablationen faktiskt hittade.

Analysen använder befintliga features.jsonl och gör ingen extern
hämtning och ingen modellträning.

Den beräknar:
    1. Volatilitetens fem kvintiler per kalenderår.
    2. Sannolikheten för +5% respektive -5% inom 5 dagar per kvintil.
    3. Genomsnitt och median för 5-dagarsavkastningen per kvintil.
    4. En enkel 2x2-regim: låg/hög volatilitet x låg/hög short interest.

Kvintilerna beräknas separat per år. Det gör att Q5 betyder "högsta
20 procenten av volatiliteten det året" och gör 2025/2026 direkt
jämförbara utan att använda framtida års distributionsgränser.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


FEATURES_PATH = Path("data/processed/analysis/features.jsonl")
OUTPUT_PATH = Path("data/processed/analysis/volatility_analysis.json")

VOLATILITY_COLUMN = "price_volatility_20d"
RETURN_COLUMN = "forward_return_5d"
SHORT_INTEREST_COLUMN = "short_interest_pct"

YEARS = (2025, 2026)


def load_features() -> pd.DataFrame:
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Featurefil saknas: {FEATURES_PATH}"
        )

    frame = pd.read_json(FEATURES_PATH, lines=True)

    required = {
        "snapshot_date",
        VOLATILITY_COLUMN,
        RETURN_COLUMN,
        SHORT_INTEREST_COLUMN,
    }

    missing = required - set(frame.columns)

    if missing:
        raise RuntimeError(
            "Features saknar obligatoriska kolumner: "
            f"{sorted(missing)}"
        )

    frame["snapshot_date"] = pd.to_datetime(
        frame["snapshot_date"],
        errors="coerce",
    )

    frame[VOLATILITY_COLUMN] = pd.to_numeric(
        frame[VOLATILITY_COLUMN],
        errors="coerce",
    )

    frame[RETURN_COLUMN] = pd.to_numeric(
        frame[RETURN_COLUMN],
        errors="coerce",
    )

    frame[SHORT_INTEREST_COLUMN] = pd.to_numeric(
        frame[SHORT_INTEREST_COLUMN],
        errors="coerce",
    )

    frame = frame.dropna(
        subset=[
            "snapshot_date",
            VOLATILITY_COLUMN,
            RETURN_COLUMN,
            SHORT_INTEREST_COLUMN,
        ]
    ).copy()

    frame = frame[
        frame[VOLATILITY_COLUMN] >= 0
    ].copy()

    frame["year"] = frame["snapshot_date"].dt.year

    frame = frame[
        frame["year"].isin(YEARS)
    ].copy()

    if frame.empty:
        raise RuntimeError(
            "Ingen analysdata för 2025/2026."
        )

    return frame


def add_yearly_quantiles(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:

    result = frame.copy()

    thresholds: dict[str, dict[str, float]] = {}

    quantile_labels = [
        "Q1",
        "Q2",
        "Q3",
        "Q4",
        "Q5",
    ]

    def assign_quantile(
        group: pd.DataFrame,
    ) -> pd.Series:

        year = int(
            group["year"].iloc[0]
        )

        values = group[
            VOLATILITY_COLUMN
        ]

        quantiles = values.quantile(
            [0.2, 0.4, 0.6, 0.8]
        )

        thresholds[str(year)] = {
            "q20": float(
                quantiles.loc[0.2]
            ),
            "q40": float(
                quantiles.loc[0.4]
            ),
            "q60": float(
                quantiles.loc[0.6]
            ),
            "q80": float(
                quantiles.loc[0.8]
            ),
        }

        return pd.qcut(
            values.rank(
                method="first"
            ),
            5,
            labels=quantile_labels,
        )

    result["volatility_quantile"] = (
        result
        .groupby(
            "year",
            group_keys=False,
        )
        .apply(assign_quantile)
        .reset_index(
            level=0,
            drop=True,
        )
    )

    return result, thresholds


def _rate(
    mask: pd.Series,
) -> float:

    if not len(mask):
        return float("nan")

    return float(
        mask.mean()
    )


def quantile_analysis(
    frame: pd.DataFrame,
) -> list[dict]:

    rows: list[dict] = []

    for (
        year,
        quantile,
    ), group in frame.groupby(
        [
            "year",
            "volatility_quantile",
        ],
        observed=True,
    ):

        returns = group[
            RETURN_COLUMN
        ]

        rows.append(
            {
                "year": int(year),
                "quantile": str(
                    quantile
                ),
                "rows": int(
                    len(group)
                ),
                "volatility_mean": float(
                    group[
                        VOLATILITY_COLUMN
                    ].mean()
                ),
                "volatility_median": float(
                    group[
                        VOLATILITY_COLUMN
                    ].median()
                ),
                "forward_return_5d_mean": float(
                    returns.mean()
                ),
                "forward_return_5d_median": float(
                    returns.median()
                ),
                "positive_5pct_5d_rate": _rate(
                    returns >= 0.05
                ),
                "negative_5pct_5d_rate": _rate(
                    returns <= -0.05
                ),
                "positive_5pct_5d_count": int(
                    (returns >= 0.05).sum()
                ),
                "negative_5pct_5d_count": int(
                    (returns <= -0.05).sum()
                ),
            }
        )

    return rows


def regime_analysis(
    frame: pd.DataFrame,
) -> list[dict]:

    rows: list[dict] = []

    for year, group in frame.groupby(
        "year"
    ):

        volatility_median = float(
            group[
                VOLATILITY_COLUMN
            ].median()
        )

        short_median = float(
            group[
                SHORT_INTEREST_COLUMN
            ].median()
        )

        local = group.copy()

        local["volatility_regime"] = np.where(
            local[
                VOLATILITY_COLUMN
            ] >= volatility_median,
            "high_volatility",
            "low_volatility",
        )

        local["short_regime"] = np.where(
            local[
                SHORT_INTEREST_COLUMN
            ] >= short_median,
            "high_short_interest",
            "low_short_interest",
        )

        for (
            volatility_regime,
            short_regime,
        ), cell in local.groupby(
            [
                "volatility_regime",
                "short_regime",
            ],
            observed=True,
        ):

            returns = cell[
                RETURN_COLUMN
            ]

            rows.append(
                {
                    "year": int(year),
                    "volatility_regime": str(
                        volatility_regime
                    ),
                    "short_interest_regime": str(
                        short_regime
                    ),
                    "rows": int(
                        len(cell)
                    ),
                    "volatility_mean": float(
                        cell[
                            VOLATILITY_COLUMN
                        ].mean()
                    ),
                    "short_interest_mean": float(
                        cell[
                            SHORT_INTEREST_COLUMN
                        ].mean()
                    ),
                    "positive_5pct_5d_rate": _rate(
                        returns >= 0.05
                    ),
                    "negative_5pct_5d_rate": _rate(
                        returns <= -0.05
                    ),
                    "forward_return_5d_mean": float(
                        returns.mean()
                    ),
                }
            )

    return rows


def _clean(
    value,
):

    if (
        isinstance(value, float)
        and not np.isfinite(value)
    ):
        return None

    if isinstance(value, dict):
        return {
            key: _clean(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [
            _clean(item)
            for item in value
        ]

    return value


def main() -> None:

    print(
        "========================================"
    )

    print(
        "BLANKDISS VOLATILITY ANALYSIS"
    )

    print(
        "========================================"
    )

    print(
        f"Features: {FEATURES_PATH}"
    )

    frame = load_features()

    frame, thresholds = (
        add_yearly_quantiles(
            frame
        )
    )

    quantiles = (
        quantile_analysis(
            frame
        )
    )

    regimes = (
        regime_analysis(
            frame
        )
    )

    result = _clean(
        {
            "created_at": (
                pd.Timestamp.utcnow()
                .isoformat()
            ),
            "features_path": str(
                FEATURES_PATH
            ),
            "years": list(
                YEARS
            ),
            "rows": int(
                len(frame)
            ),
            "volatility_column": (
                VOLATILITY_COLUMN
            ),
            "return_column": (
                RETURN_COLUMN
            ),
            "short_interest_column": (
                SHORT_INTEREST_COLUMN
            ),
            "quantile_thresholds": (
                thresholds
            ),
            "quantiles": (
                quantiles
            ),
            "regimes": (
                regimes
            ),
        }
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("KVINTILER")
    print("----------------------------------------")

    for row in quantiles:

        print(
            f"{row['year']} "
            f"{row['quantile']}: "
            f"n={row['rows']:,}, "
            f"vol={row['volatility_mean']:.4f}, "
            f"+5%={row['positive_5pct_5d_rate']:.3%}, "
            f"-5%={row['negative_5pct_5d_rate']:.3%}, "
            f"mean5d={row['forward_return_5d_mean']:.3%}"
        )

    print()
    print("VOLATILITET x BLANKNING")
    print("----------------------------------------")

    for row in regimes:

        print(
            f"{row['year']} "
            f"{row['volatility_regime']} + "
            f"{row['short_interest_regime']}: "
            f"n={row['rows']:,}, "
            f"+5%={row['positive_5pct_5d_rate']:.3%}, "
            f"-5%={row['negative_5pct_5d_rate']:.3%}, "
            f"mean5d={row['forward_return_5d_mean']:.3%}"
        )

    print()
    print(
        f"Sparad: {OUTPUT_PATH}"
    )

    print(
        "========================================"
    )

    print(
        "VOLATILITY ANALYSIS KLAR"
    )

    print(
        "========================================"
    )


if __name__ == "__main__":
    main()
