from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ml.diagnostics.framework import (
    DiagnosticExperiment,
    ExperimentResult,
)
from ml.research.signals import build_signal


# ---------------------------------------------------------------------------
# LOCKED HYPOTHESIS
# ---------------------------------------------------------------------------
# This file deliberately contains no data-driven selection logic.
#
# The regime and SI-change definition are fixed before the 2026 test:
#
#   momentum decile 9
#   short-interest level decile 10
#   high SI change = top 20% of positive SI changes
#
# The four groups are:
#
#   A = outside 9x10 regime + low SI change
#   B = outside 9x10 regime + high SI change
#   C = 9x10 regime + low SI change
#   D = 9x10 regime + high SI change
#
# Primary effect:
#
#   D - C
#
# The experiment is deliberately descriptive. It does not fit a model,
# search thresholds, select cells, or optimize against the 2026 data.


MOMENTUM_DECILE = 9
SI_LEVEL_DECILE = 10
SI_CHANGE_TOP_FRACTION = 0.20

DISCOVERY_END = pd.Timestamp("2024-12-31")
LOCK_START = pd.Timestamp("2025-01-01")
LOCK_END = pd.Timestamp("2025-12-31")
OOS_START = pd.Timestamp("2026-01-01")

MIN_GROUP_N = 20

TARGETS = (
    (
        "down_5pct_5d",
        "forward_return_5d",
        "below",
        -0.05,
    ),
    (
        "down_7pct_5d",
        "forward_return_5d",
        "below",
        -0.07,
    ),
    (
        "down_10pct_10d",
        "forward_return_10d",
        "below",
        -0.10,
    ),
)

RETURN_HORIZONS = (
    5,
    10,
)

SECTOR_MAP_PATH = Path(
    "data/analysis/sector_map.json"
)


def _numeric(
    frame: pd.DataFrame,
    column: str,
) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(
            index=frame.index,
            dtype=float,
        )

    return pd.to_numeric(
        frame[column],
        errors="coerce",
    )


def _cross_sectional_deciles(
    frame: pd.DataFrame,
    values: pd.Series,
) -> pd.Series:
    """
    Assign 1..10 deciles independently for every snapshot date.

    Decile 1 = lowest.
    Decile 10 = highest.
    """

    numeric = pd.to_numeric(
        values,
        errors="coerce",
    )

    ranks = numeric.groupby(
        frame["snapshot_date"]
    ).rank(
        method="first",
        pct=True,
    )

    deciles = (
        np.ceil(
            ranks * 10
        )
        .astype("Int64")
    )

    return deciles.clip(
        lower=1,
        upper=10,
    )


def _high_si_change_mask(
    frame: pd.DataFrame,
) -> pd.Series:
    """
    Locked SI-change definition.

    High SI change means the top 20% of positive SI changes
    within each snapshot date.

    This is a cross-sectional rule and therefore does not
    estimate a threshold from 2025 or 2026.
    """

    change = _numeric(
        frame,
        "short_interest_change",
    )

    positive = change.where(
        change > 0
    )

    rank = positive.groupby(
        frame["snapshot_date"]
    ).rank(
        method="first",
        pct=True,
    )

    return (
        positive.notna()
        & (
            rank
            > (
                1.0
                - SI_CHANGE_TOP_FRACTION
            )
        )
    )


def _prepare(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    result = frame.copy()

    result["snapshot_date"] = pd.to_datetime(
        result["snapshot_date"],
        errors="coerce",
    )

    result = result.loc[
        result["snapshot_date"].notna()
    ].copy()

    result["momentum"] = build_signal(
        result,
        "price_momentum_5d",
    )

    result["si_level"] = build_signal(
        result,
        "short_interest_level",
    )

    result["si_change"] = build_signal(
        result,
        "short_interest_change",
    )

    result["momentum_decile"] = (
        _cross_sectional_deciles(
            result,
            result["momentum"],
        )
    )

    result["si_level_decile"] = (
        _cross_sectional_deciles(
            result,
            result["si_level"],
        )
    )

    result["high_si_change"] = (
        _high_si_change_mask(
            result
        )
    )

    result["high_regime"] = (
        (
            result["momentum_decile"]
            == MOMENTUM_DECILE
        )
        & (
            result["si_level_decile"]
            == SI_LEVEL_DECILE
        )
    )

    # Four locked groups:
    #
    # A = outside high regime + low SI change
    # B = outside high regime + high SI change
    # C = high regime + low SI change
    # D = high regime + high SI change

    result["group"] = np.select(
        [
            (
                result["high_regime"]
                & result["high_si_change"]
            ),
            (
                result["high_regime"]
                & ~result["high_si_change"]
            ),
            (
                ~result["high_regime"]
                & result["high_si_change"]
            ),
        ],
        [
            "D",
            "C",
            "B",
        ],
        default="A",
    )

    return result


def _event_series(
    frame: pd.DataFrame,
    return_column: str,
    direction: str,
    threshold: float,
) -> pd.Series:
    values = _numeric(
        frame,
        return_column,
    )

    if direction == "below":
        return values <= threshold

    return values >= threshold


def _proportion_ci(
    n: int,
    events: int,
    z: float = 1.96,
) -> tuple[float, float]:
    if n <= 0:
        return (
            float("nan"),
            float("nan"),
        )

    p = events / n

    se = np.sqrt(
        max(
            p * (1.0 - p),
            0.0,
        )
        / n
    )

    return (
        p - z * se,
        p + z * se,
    )


def _difference_ci(
    n_a: int,
    p_a: float,
    n_b: int,
    p_b: float,
    z: float = 1.96,
) -> tuple[float, float]:
    if (
        n_a <= 0
        or n_b <= 0
        or not np.isfinite(p_a)
        or not np.isfinite(p_b)
    ):
        return (
            float("nan"),
            float("nan"),
        )

    se = np.sqrt(
        max(
            p_a * (1.0 - p_a),
            0.0,
        )
        / n_a
        +
        max(
            p_b * (1.0 - p_b),
            0.0,
        )
        / n_b
    )

    delta = p_a - p_b

    return (
        delta - z * se,
        delta + z * se,
    )


def _group_rows(
    frame: pd.DataFrame,
    return_column: str,
    direction: str,
    threshold: float,
) -> list[dict[str, Any]]:
    event = _event_series(
        frame,
        return_column,
        direction,
        threshold,
    )

    rows: list[dict[str, Any]] = []

    for group in (
        "A",
        "B",
        "C",
        "D",
    ):
        mask = (
            frame["group"]
            == group
        )

        valid = (
            mask
            & event.notna()
        )

        values = _numeric(
            frame.loc[valid],
            return_column,
        )

        events = int(
            event.loc[valid].sum()
        )

        n = int(
            valid.sum()
        )

        rate = (
            events / n
            if n
            else float("nan")
        )

        ci_low, ci_high = (
            _proportion_ci(
                n,
                events,
            )
        )

        rows.append(
            {
                "group": group,
                "n": n,
                "events": events,
                "event_rate": rate,
                "event_rate_pct": (
                    rate * 100
                    if np.isfinite(rate)
                    else np.nan
                ),
                "event_rate_ci95_low": ci_low,
                "event_rate_ci95_high": ci_high,
                "mean_return": (
                    float(values.mean())
                    if not values.empty
                    else np.nan
                ),
                "median_return": (
                    float(values.median())
                    if not values.empty
                    else np.nan
                ),
            }
        )

    return rows


def _dc_effect(
    rows: list[dict[str, Any]],
    target_name: str,
) -> dict[str, Any]:
    by_group = {
        row["group"]: row
        for row in rows
    }

    c = by_group.get(
        "C",
        {},
    )

    d = by_group.get(
        "D",
        {},
    )

    n_c = int(
        c.get(
            "n",
            0,
        )
    )

    n_d = int(
        d.get(
            "n",
            0,
        )
    )

    p_c = float(
        c.get(
            "event_rate",
            np.nan,
        )
    )

    p_d = float(
        d.get(
            "event_rate",
            np.nan,
        )
    )

    delta = (
        p_d - p_c
        if n_c and n_d
        else np.nan
    )

    ci_low, ci_high = (
        _difference_ci(
            n_d,
            p_d,
            n_c,
            p_c,
        )
    )

    return {
        "target": target_name,
        "c_n": n_c,
        "d_n": n_d,
        "c_event_rate": p_c,
        "d_event_rate": p_d,
        "d_minus_c": delta,
        "d_minus_c_pp": (
            delta * 100
            if np.isfinite(delta)
            else np.nan
        ),
        "d_minus_c_ci95_low": ci_low,
        "d_minus_c_ci95_high": ci_high,
        "d_minus_c_ci95_low_pp": (
            ci_low * 100
            if np.isfinite(ci_low)
            else np.nan
        ),
        "d_minus_c_ci95_high_pp": (
            ci_high * 100
            if np.isfinite(ci_high)
            else np.nan
        ),
        "meets_min_group_n": (
            n_c >= MIN_GROUP_N
            and n_d >= MIN_GROUP_N
        ),
    }


def _target_analysis(
    frame: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    group_rows: list[
        dict[str, Any]
    ] = []

    effect_rows: list[
        dict[str, Any]
    ] = []

    for (
        target_name,
        return_column,
        direction,
        threshold,
    ) in TARGETS:

        if (
            return_column
            not in frame.columns
        ):
            effect_rows.append(
                {
                    "target": target_name,
                    "status": "unavailable",
                    "return_column": return_column,
                    "reason": (
                        "Feature data saknar "
                        f"{return_column}"
                    ),
                }
            )

            continue

        rows = _group_rows(
            frame,
            return_column,
            direction,
            threshold,
        )

        for row in rows:
            row.update(
                {
                    "target": target_name,
                    "return_column": return_column,
                    "threshold": threshold,
                }
            )

        group_rows.extend(
            rows
        )

        effect = _dc_effect(
            rows,
            target_name,
        )

        effect["status"] = (
            "available"
        )

        effect["return_column"] = (
            return_column
        )

        effect["threshold"] = (
            threshold
        )

        effect_rows.append(
            effect
        )

    return (
        pd.DataFrame(
            group_rows
        ),
        pd.DataFrame(
            effect_rows
        ),
    )


def _return_comparison(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[
        dict[str, Any]
    ] = []

    for horizon in RETURN_HORIZONS:
        column = (
            f"forward_return_{horizon}d"
        )

        if column not in frame.columns:
            rows.append(
                {
                    "horizon_days": horizon,
                    "status": "unavailable",
                    "return_column": column,
                }
            )

            continue

        for group in (
            "A",
            "B",
            "C",
            "D",
        ):
            values = _numeric(
                frame.loc[
                    frame["group"]
                    == group
                ],
                column,
            ).dropna()

            rows.append(
                {
                    "horizon_days": horizon,
                    "group": group,
                    "status": "available",
                    "n": int(
                        len(values)
                    ),
                    "mean_return": (
                        float(values.mean())
                        if not values.empty
                        else np.nan
                    ),
                    "median_return": (
                        float(values.median())
                        if not values.empty
                        else np.nan
                    ),
                }
            )

        d = _numeric(
            frame.loc[
                frame["group"]
                == "D"
            ],
            column,
        ).dropna()

        c = _numeric(
            frame.loc[
                frame["group"]
                == "C"
            ],
            column,
        ).dropna()

        rows.append(
            {
                "horizon_days": horizon,
                "group": "D_minus_C",
                "status": "available",
                "n": min(
                    len(d),
                    len(c),
                ),
                "mean_return": (
                    float(
                        d.mean()
                        - c.mean()
                    )
                    if (
                        not d.empty
                        and not c.empty
                    )
                    else np.nan
                ),
                "median_return": (
                    float(
                        d.median()
                        - c.median()
                    )
                    if (
                        not d.empty
                        and not c.empty
                    )
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def _load_sector_map() -> dict[str, str]:
    if not SECTOR_MAP_PATH.exists():
        return {}

    payload = json.loads(
        SECTOR_MAP_PATH.read_text(
            encoding="utf-8"
        )
    )

    instruments = (
        payload.get(
            "instruments",
            {},
        )
        if isinstance(
            payload,
            dict,
        )
        else {}
    )

    mapping: dict[
        str,
        str,
    ] = {}

    for symbol, item in instruments.items():
        if (
            isinstance(
                item,
                dict,
            )
            and item.get("sector")
        ):
            mapping[
                str(symbol)
            ] = str(
                item["sector"]
            ).strip()

    return mapping


def _sector_relative(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """
    Sector-relative return for the same four groups.

    Benchmark:
        equal-weight sector mean excluding the stock itself,
        separately for every snapshot date.

    If the frozen sector map is unavailable, report that explicitly.
    """

    sector_map = _load_sector_map()

    symbol_column = (
        "yahoo_symbol"
        if "yahoo_symbol"
        in frame.columns
        else "security_key"
    )

    if not sector_map:
        return pd.DataFrame(
            [
                {
                    "status": "unavailable",
                    "reason": (
                        "sector_map.json saknas "
                        "eller är tom"
                    ),
                }
            ]
        )

    work = frame.copy()

    work["sector"] = (
        work[
            symbol_column
        ]
        .astype(str)
        .map(sector_map)
    )

    rows: list[
        dict[str, Any]
    ] = []

    for horizon in RETURN_HORIZONS:
        column = (
            f"forward_return_{horizon}d"
        )

        if column not in work.columns:
            rows.append(
                {
                    "horizon_days": horizon,
                    "status": "unavailable",
                    "return_column": column,
                }
            )

            continue

        work["_ret"] = _numeric(
            work,
            column,
        )

        stock = work.dropna(
            subset=[
                "snapshot_date",
                "sector",
                "_ret",
            ]
        ).copy()

        if stock.empty:
            continue

        stock_sum = (
            stock.groupby(
                [
                    "snapshot_date",
                    "sector",
                ]
            )["_ret"]
            .transform("sum")
        )

        stock_count = (
            stock.groupby(
                [
                    "snapshot_date",
                    "sector",
                ]
            )["_ret"]
            .transform("count")
        )

        stock[
            "sector_mean_ex_self"
        ] = np.where(
            stock_count > 1,
            (
                stock_sum
                - stock["_ret"]
            )
            / (
                stock_count - 1
            ),
            np.nan,
        )

        stock[
            "sector_relative"
        ] = (
            stock["_ret"]
            - stock[
                "sector_mean_ex_self"
            ]
        )

        for group in (
            "A",
            "B",
            "C",
            "D",
        ):
            values = stock.loc[
                stock["group"]
                == group,
                "sector_relative",
            ].dropna()

            rows.append(
                {
                    "horizon_days": horizon,
                    "group": group,
                    "status": "available",
                    "n": int(
                        len(values)
                    ),
                    "mean_sector_relative": (
                        float(
                            values.mean()
                        )
                        if not values.empty
                        else np.nan
                    ),
                    "median_sector_relative": (
                        float(
                            values.median()
                        )
                        if not values.empty
                        else np.nan
                    ),
                }
            )

        d = stock.loc[
            stock["group"]
            == "D",
            "sector_relative",
        ].dropna()

        c = stock.loc[
            stock["group"]
            == "C",
            "sector_relative",
        ].dropna()

        rows.append(
            {
                "horizon_days": horizon,
                "group": "D_minus_C",
                "status": "available",
                "n": min(
                    len(d),
                    len(c),
                ),
                "mean_sector_relative": (
                    float(
                        d.mean()
                        - c.mean()
                    )
                    if (
                        not d.empty
                        and not c.empty
                    )
                    else np.nan
                ),
                "median_sector_relative": (
                    float(
                        d.median()
                        - c.median()
                    )
                    if (
                        not d.empty
                        and not c.empty
                    )
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def _period_frame(
    context,
    start: pd.Timestamp | None,
    end: pd.Timestamp | None,
) -> pd.DataFrame:
    frame = context.data.copy()

    frame["snapshot_date"] = pd.to_datetime(
        frame["snapshot_date"],
        errors="coerce",
    )

    frame = frame.loc[
        frame["snapshot_date"].notna()
    ].copy()

    if start is not None:
        frame = frame.loc[
            frame["snapshot_date"]
            >= start
        ]

    if end is not None:
        frame = frame.loc[
            frame["snapshot_date"]
            <= end
        ]

    return _prepare(
        frame
    )


def _analyse_period(
    frame: pd.DataFrame,
    phase: str,
) -> dict[
    str,
    pd.DataFrame,
]:
    (
        group_table,
        effect_table,
    ) = _target_analysis(
        frame
    )

    returns = _return_comparison(
        frame
    )

    sector = _sector_relative(
        frame
    )

    for table in (
        group_table,
        effect_table,
        returns,
        sector,
    ):
        if not table.empty:
            table.insert(
                0,
                "phase",
                phase,
            )

    return {
        "groups": group_table,
        "effects": effect_table,
        "returns": returns,
        "sector_relative": sector,
    }


class MomentumSIIncrementalLockedOOSExperiment(
    DiagnosticExperiment
):
    name = (
        "momentum_si_incremental_locked_oos"
    )

    description = (
        "Låser hypotesen hög momentum + mycket hög SI "
        "+ ytterligare SI-ökning och testar den med "
        "fyra grupper i 2025 lock-period och helt "
        "orörd 2026 OOS."
    )

    def analyze_window(
        self,
        context,
    ) -> ExperimentResult:
        # Discovery is explicitly limited to 2022-2024.
        discovery = _period_frame(
            context,
            pd.Timestamp(
                "2022-01-01"
            ),
            DISCOVERY_END,
        )

        # 2025 is the lock/replication period.
        lock = _period_frame(
            context,
            LOCK_START,
            LOCK_END,
        )

        # 2026 is the final untouched OOS period.
        oos = _period_frame(
            context,
            OOS_START,
            (
                pd.Timestamp(
                    context.test_end
                )
                if context.test_end
                is not None
                else None
            ),
        )

        discovery_tables = (
            _analyse_period(
                discovery,
                "discovery_2022_2024",
            )
        )

        lock_tables = (
            _analyse_period(
                lock,
                "lock_2025",
            )
        )

        oos_tables = (
            _analyse_period(
                oos,
                "final_oos_2026",
            )
        )

        result = ExperimentResult(
            name=self.name,
            description=self.description,
        )

        tables = (
            (
                "discovery_groups",
                discovery_tables[
                    "groups"
                ],
            ),
            (
                "lock_2025_groups",
                lock_tables[
                    "groups"
                ],
            ),
            (
                "oos_2026_groups",
                oos_tables[
                    "groups"
                ],
            ),
            (
                "discovery_effects",
                discovery_tables[
                    "effects"
                ],
            ),
            (
                "lock_2025_effects",
                lock_tables[
                    "effects"
                ],
            ),
            (
                "oos_2026_effects",
                oos_tables[
                    "effects"
                ],
            ),
            (
                "discovery_returns",
                discovery_tables[
                    "returns"
                ],
            ),
            (
                "lock_2025_returns",
                lock_tables[
                    "returns"
                ],
            ),
            (
                "oos_2026_returns",
                oos_tables[
                    "returns"
                ],
            ),
            (
                "discovery_sector_relative",
                discovery_tables[
                    "sector_relative"
                ],
            ),
            (
                "lock_2025_sector_relative",
                lock_tables[
                    "sector_relative"
                ],
            ),
            (
                "oos_2026_sector_relative",
                oos_tables[
                    "sector_relative"
                ],
            ),
        )

        for (
            table_name,
            table,
        ) in tables:
            result.add_table(
                table_name,
                table,
            )

        result.add_metric(
            "hypothesis",
            (
                "När en aktie har hög momentum + mycket "
                "hög short interest, är en ytterligare "
                "ökning av short interest associerad med "
                "högre risk för större nedgång?"
            ),
        )

        result.add_metric(
            "locked_momentum_decile",
            MOMENTUM_DECILE,
        )

        result.add_metric(
            "locked_si_level_decile",
            SI_LEVEL_DECILE,
        )

        result.add_metric(
            "locked_high_si_change_definition",
            (
                "top 20% av positiva "
                "short-interest-förändringar "
                "per snapshot_date"
            ),
        )

        result.add_metric(
            "group_definition",
            {
                "A": (
                    "outside 9x10 regime "
                    "+ low SI change"
                ),
                "B": (
                    "outside 9x10 regime "
                    "+ high SI change"
                ),
                "C": (
                    "9x10 regime "
                    "+ low SI change"
                ),
                "D": (
                    "9x10 regime "
                    "+ high SI change"
                ),
            },
        )

        result.add_metric(
            "primary_effect",
            "D_minus_C",
        )

        result.add_metric(
            "discovery_period",
            "2022-01-01..2024-12-31",
        )

        result.add_metric(
            "lock_period",
            "2025-01-01..2025-12-31",
        )

        result.add_metric(
            "final_oos_period",
            "2026-01-01..test_end",
        )

        result.add_metric(
            "test_end",
            str(
                context.test_end
            ),
        )

        result.add_metric(
            "minimum_group_n",
            MIN_GROUP_N,
        )

        result.add_metric(
            "no_parameter_selection_in_test",
            True,
        )

        return result
