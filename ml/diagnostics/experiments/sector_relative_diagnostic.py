from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml.diagnostics.framework import DiagnosticExperiment
from ml.diagnostics.framework.base import ExperimentResult
from ml.diagnostics.framework.si_additional import _add_si_change


SECTOR_MAP_PATH = Path(
    "data/analysis/sector_map.json"
)

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

BOOTSTRAP_ITERATIONS = 2_000
RANDOM_STATE = 42


def _numeric(
    frame: pd.DataFrame,
    column: str,
) -> pd.Series:
    return pd.to_numeric(
        frame[column],
        errors="coerce",
    )


def _load_sector_map() -> dict[str, str]:
    if not SECTOR_MAP_PATH.exists():
        raise RuntimeError(
            "Sector mapping saknas: "
            f"{SECTOR_MAP_PATH}"
        )

    try:
        payload = json.loads(
            SECTOR_MAP_PATH.read_text(
                encoding="utf-8"
            )
        )
    except Exception as exc:
        raise RuntimeError(
            "Kunde inte läsa sector mapping: "
            f"{SECTOR_MAP_PATH}: {exc}"
        ) from exc

    if not isinstance(payload, dict):
        raise RuntimeError(
            "Sector mapping måste vara ett JSON-objekt."
        )

    mapping = payload.get("mapping")

    if not isinstance(mapping, dict):
        raise RuntimeError(
            "Sector mapping saknar 'mapping'."
        )

    result: dict[str, str] = {}

    for symbol, sector in mapping.items():
        if not isinstance(symbol, str):
            continue

        if not isinstance(sector, str):
            continue

        symbol = symbol.strip()
        sector = sector.strip()

        if symbol and sector:
            result[symbol] = sector

    if not result:
        raise RuntimeError(
            "Sector mapping innehåller inga instrument."
        )

    return result


def _add_sector(
    frame: pd.DataFrame,
    sector_map: dict[str, str],
) -> pd.DataFrame:
    frame = frame.copy()

    if "yahoo_symbol" not in frame.columns:
        raise KeyError(
            "Sector-relative diagnostic requires "
            "'yahoo_symbol'."
        )

    frame["sector"] = (
        frame["yahoo_symbol"]
        .map(sector_map)
    )

    return frame


def _add_relative_returns(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    frame = frame.copy()

    if "snapshot_date" not in frame.columns:
        raise KeyError(
            "Sector-relative diagnostic requires "
            "'snapshot_date'."
        )

    frame["snapshot_date"] = pd.to_datetime(
        frame["snapshot_date"],
        errors="coerce",
    )

    for horizon in HORIZONS:
        raw_column = (
            f"forward_return_{horizon}d"
        )

        if raw_column not in frame.columns:
            raise KeyError(
                "Sector-relative diagnostic requires "
                f"'{raw_column}'."
            )

        frame[raw_column] = _numeric(
            frame,
            raw_column,
        )

        market_median = (
            frame
            .groupby(
                "snapshot_date",
                sort=False,
            )[raw_column]
            .transform("median")
        )

        sector_median = (
            frame
            .groupby(
                [
                    "snapshot_date",
                    "sector",
                ],
                sort=False,
            )[raw_column]
            .transform("median")
        )

        frame[
            f"market_relative_return_{horizon}d"
        ] = (
            frame[raw_column]
            - market_median
        )

        frame[
            f"sector_relative_return_{horizon}d"
        ] = (
            frame[raw_column]
            - sector_median
        )

    return frame


def _prepare(
    frame: pd.DataFrame,
    sector_map: dict[str, str],
) -> pd.DataFrame:
    frame = frame.copy()

    frame = _add_si_change(
        frame
    )

    frame = _add_sector(
        frame,
        sector_map,
    )

    frame = _add_relative_returns(
        frame
    )

    return frame


def _tail_threshold(
    values: pd.Series,
    fraction: float,
) -> float:
    values = pd.to_numeric(
        values,
        errors="coerce",
    ).dropna()

    if values.empty:
        return float("nan")

    return float(
        values.quantile(
            1.0 - fraction
        )
    )


def _bootstrap_difference(
    high: pd.Series,
    other: pd.Series,
) -> tuple[float, float, float]:
    high_values = (
        pd.to_numeric(
            high,
            errors="coerce",
        )
        .dropna()
        .to_numpy()
    )

    other_values = (
        pd.to_numeric(
            other,
            errors="coerce",
        )
        .dropna()
        .to_numpy()
    )

    if (
        len(high_values) == 0
        or len(other_values) == 0
    ):
        return (
            float("nan"),
            float("nan"),
            float("nan"),
        )

    observed = (
        high_values.mean()
        - other_values.mean()
    )

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    bootstrap = np.empty(
        BOOTSTRAP_ITERATIONS,
        dtype=float,
    )

    for index in range(
        BOOTSTRAP_ITERATIONS
    ):
        high_sample = rng.choice(
            high_values,
            size=len(high_values),
            replace=True,
        )

        other_sample = rng.choice(
            other_values,
            size=len(other_values),
            replace=True,
        )

        bootstrap[index] = (
            high_sample.mean()
            - other_sample.mean()
        )

    return (
        float(observed),
        float(
            np.quantile(
                bootstrap,
                0.025,
            )
        ),
        float(
            np.quantile(
                bootstrap,
                0.975,
            )
        ),
    )


def _mean_return(
    frame: pd.DataFrame,
    column: str,
) -> float:
    values = pd.to_numeric(
        frame[column],
        errors="coerce",
    ).dropna()

    if values.empty:
        return float("nan")

    return float(values.mean())


def _count(
    frame: pd.DataFrame,
    column: str,
) -> int:
    return int(
        pd.to_numeric(
            frame[column],
            errors="coerce",
        )
        .notna()
        .sum()
    )


class SectorRelativeReturnExperiment(
    DiagnosticExperiment
):
    name = "sector_relative_return"

    description = (
        "Testar om sambandet mellan stora "
        "short-interest-förändringar och "
        "framtida kursutveckling kvarstår "
        "efter kontroll för sektor- och "
        "marknadsrelativ avkastning."
    )

    horizons = HORIZONS
    si_change_cutoffs = SI_CHANGE_CUTOFFS

    def analyze_window(
        self,
        context,
    ) -> ExperimentResult:
        sector_map = _load_sector_map()

        pretest = _prepare(
            context.pretest,
            sector_map,
        )

        test = _prepare(
            context.test,
            sector_map,
        )

        rows: list[dict] = []

        for cutoff in self.si_change_cutoffs:
            threshold = _tail_threshold(
                pretest[
                    "si_change"
                ],
                cutoff,
            )

            if not np.isfinite(threshold):
                continue

            test = test.copy()

            test["high_si_change"] = (
                test["si_change"]
                >= threshold
            )

            for horizon in self.horizons:
                raw_column = (
                    f"forward_return_{horizon}d"
                )

                sector_column = (
                    "sector_relative_return_"
                    f"{horizon}d"
                )

                market_column = (
                    "market_relative_return_"
                    f"{horizon}d"
                )

                high = test[
                    test["high_si_change"]
                ]

                other = test[
                    ~test["high_si_change"]
                ]

                raw_delta = (
                    _mean_return(
                        high,
                        raw_column,
                    )
                    - _mean_return(
                        other,
                        raw_column,
                    )
                )

                sector_delta, sector_ci_low, sector_ci_high = (
                    _bootstrap_difference(
                        high[
                            sector_column
                        ],
                        other[
                            sector_column
                        ],
                    )
                )

                market_delta, market_ci_low, market_ci_high = (
                    _bootstrap_difference(
                        high[
                            market_column
                        ],
                        other[
                            market_column
                        ],
                    )
                )

                rows.append(
                    {
                        "si_change_cutoff": cutoff,
                        "si_change_threshold": threshold,
                        "horizon_days": horizon,
                        "high_si_change_n": len(high),
                        "other_si_change_n": len(other),
                        "raw_high_mean_return": (
                            _mean_return(
                                high,
                                raw_column,
                            )
                        ),
                        "raw_other_mean_return": (
                            _mean_return(
                                other,
                                raw_column,
                            )
                        ),
                        "raw_delta": raw_delta,
                        "sector_high_mean_return": (
                            _mean_return(
                                high,
                                sector_column,
                            )
                        ),
                        "sector_other_mean_return": (
                            _mean_return(
                                other,
                                sector_column,
                            )
                        ),
                        "sector_relative_delta": (
                            sector_delta
                        ),
                        "sector_relative_ci_low": (
                            sector_ci_low
                        ),
                        "sector_relative_ci_high": (
                            sector_ci_high
                        ),
                        "market_high_mean_return": (
                            _mean_return(
                                high,
                                market_column,
                            )
                        ),
                        "market_other_mean_return": (
                            _mean_return(
                                other,
                                market_column,
                            )
                        ),
                        "market_relative_delta": (
                            market_delta
                        ),
                        "market_relative_ci_low": (
                            market_ci_low
                        ),
                        "market_relative_ci_high": (
                            market_ci_high
                        ),
                        "mapped_sector_rows": (
                            int(
                                test["sector"]
                                .notna()
                                .sum()
                            )
                        ),
                        "total_rows": len(test),
                    }
                )

        result = ExperimentResult(
            name=self.name,
            description=self.description,
        )

        result.add_table(
            "sector_relative_analysis",
            pd.DataFrame(rows),
        )

        result.add_metadata(
            "sector_mapping_path",
            str(
                SECTOR_MAP_PATH
            ),
        )

        result.add_metadata(
            "sector_mapping_symbols",
            len(sector_map),
        )

        result.add_metadata(
            "relative_return_definition",
            (
                "stock forward return minus "
                "same-date sector median return"
            ),
        )

        result.add_metadata(
            "market_relative_return_definition",
            (
                "stock forward return minus "
                "same-date universe median return"
            ),
        )

        return result
