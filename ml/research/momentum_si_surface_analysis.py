from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ml.config import TARGETS
from ml.dataset import build_target, load_features
from ml.research.signals import build_signal


DISCOVERY_END = pd.Timestamp("2025-12-19")
TEST_START_EXCLUSIVE = pd.Timestamp("2024-12-31")

TARGET_NAMES = (
    "down_5pct_5d",
    "down_7pct_5d",
    "down_10pct_5d",
)

MOMENTUM_SIGNAL_NAME = "price_momentum_5d"
SI_SIGNAL_NAME = "short_interest_change"

N_BINS = 10
N_PERMUTATIONS = 1000
MIN_CELL_N = 50
RANDOM_STATE = 42

OUTPUT_DIR = Path(
    "data/processed/ml/research/momentum_si_surface"
)


def get_target(name: str):
    for target in TARGETS:
        if target.name == name:
            return target

    raise ValueError(
        f"Unknown target: {name}"
    )


def cross_sectional_deciles(
    frame: pd.DataFrame,
    values: pd.Series,
) -> np.ndarray:
    """
    Assign each observation to one of ten cross-sectional
    deciles independently for each snapshot date.

    Decile 1 = lowest values.
    Decile 10 = highest values.

    Only information available on the same snapshot date
    is used for the ranking.
    """
    result = np.full(
        len(frame),
        -1,
        dtype=np.int8,
    )

    working = pd.DataFrame(
        {
            "date": frame["snapshot_date"],
            "value": pd.to_numeric(
                values,
                errors="coerce",
            ),
        },
        index=frame.index,
    )

    for _, index in working.groupby(
        "date",
        sort=False,
    ).groups.items():

        local = working.loc[
            index,
            "value",
        ]

        valid = local.notna()

        if not valid.any():
            continue

        ranks = local.loc[
            valid
        ].rank(
            method="first",
            pct=True,
        ).to_numpy()

        bins = (
            np.ceil(
                ranks * N_BINS
            ).astype(int)
            - 1
        )

        bins = np.clip(
            bins,
            0,
            N_BINS - 1,
        )

        positions = (
            frame.index.get_indexer(
                local.loc[valid].index
            )
        )

        result[
            positions
        ] = bins

    return result


def cell_surface(
    momentum_bins: np.ndarray,
    si_bins: np.ndarray,
    target: np.ndarray,
) -> pd.DataFrame:
    """
    Build a 10 x 10 momentum/SI surface.

    The incremental effect is measured against the complete
    momentum-decile row, i.e. the event rate among all SI
    deciles within the same momentum decile.
    """
    rows: list[dict[str, Any]] = []

    valid = (
        (momentum_bins >= 0)
        & (si_bins >= 0)
        & np.isfinite(target)
    )

    overall_rate = (
        float(
            target[valid].mean()
        )
        if valid.any()
        else None
    )

    for momentum_decile in range(
        N_BINS
    ):
        row_mask = (
            valid
            & (
                momentum_bins
                == momentum_decile
            )
        )

        row_target = target[
            row_mask
        ]

        row_rate = (
            float(
                row_target.mean()
            )
            if len(row_target)
            else None
        )

        for si_decile in range(
            N_BINS
        ):
            cell_mask = (
                row_mask
                & (
                    si_bins
                    == si_decile
                )
            )

            cell_target = target[
                cell_mask
            ]

            cell_rate = (
                float(
                    cell_target.mean()
                )
                if len(cell_target)
                else None
            )

            incremental_difference = None

            if (
                cell_rate is not None
                and row_rate is not None
            ):
                incremental_difference = (
                    cell_rate
                    - row_rate
                )

            incremental_lift = None

            if (
                cell_rate is not None
                and row_rate is not None
                and row_rate > 0
            ):
                incremental_lift = (
                    cell_rate
                    / row_rate
                )

            rows.append(
                {
                    "momentum_decile": (
                        momentum_decile + 1
                    ),
                    "si_decile": (
                        si_decile + 1
                    ),
                    "n": int(
                        len(cell_target)
                    ),
                    "events": int(
                        cell_target.sum()
                    ),
                    "event_rate": (
                        cell_rate
                    ),
                    "momentum_row_n": int(
                        row_mask.sum()
                    ),
                    "momentum_row_event_rate": (
                        row_rate
                    ),
                    "incremental_vs_momentum_row": (
                        incremental_difference
                    ),
                    "incremental_lift_vs_momentum_row": (
                        incremental_lift
                    ),
                    "overall_event_rate": (
                        overall_rate
                    ),
                }
            )

    return pd.DataFrame(
        rows
    )


def surface_statistic(
    surface: pd.DataFrame,
) -> float | None:
    """
    Maximum absolute incremental event-rate difference
    among sufficiently populated cells.

    This is the statistic used for the surface-wide
    permutation test.
    """
    eligible = surface.loc[
        surface["n"] >= MIN_CELL_N,
        "incremental_vs_momentum_row",
    ].dropna()

    if eligible.empty:
        return None

    return float(
        np.max(
            np.abs(
                eligible.to_numpy()
            )
        )
    )


def high_high_value(
    surface: pd.DataFrame,
) -> float | None:
    """
    Incremental effect in:

        momentum decile 10
        SI decile 10

    relative to the complete momentum-decile-10 row.
    """
    row = surface[
        (
            surface[
                "momentum_decile"
            ]
            == N_BINS
        )
        & (
            surface[
                "si_decile"
            ]
            == N_BINS
        )
    ]

    if row.empty:
        return None

    if (
        row.iloc[0]["n"]
        < MIN_CELL_N
    ):
        return None

    value = row.iloc[0][
        "incremental_vs_momentum_row"
    ]

    if pd.isna(value):
        return None

    return float(value)


def permute_si_within_dates(
    si_bins: np.ndarray,
    date_values: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Break the association between SI and the target while
    preserving the complete cross-sectional SI-decile
    distribution separately for every snapshot date.

    Momentum bins and target values remain untouched.
    """
    permuted = si_bins.copy()

    for date in np.unique(
        date_values
    ):
        indices = np.flatnonzero(
            date_values == date
        )

        valid = indices[
            permuted[indices] >= 0
        ]

        if len(valid) > 1:
            permuted[
                valid
            ] = rng.permutation(
                permuted[valid]
            )

    return permuted


def permutation_statistics(
    momentum_bins: np.ndarray,
    si_bins: np.ndarray,
    target: np.ndarray,
    date_values: np.ndarray,
    rng: np.random.Generator,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    """
    Generate the null distributions for:

    1. maximum absolute surface effect
    2. high-momentum/high-SI cell effect
    """
    max_statistics: list[float] = []
    high_high_statistics: list[float] = []

    for permutation in range(
        N_PERMUTATIONS
    ):
        permuted_si = (
            permute_si_within_dates(
                si_bins,
                date_values,
                rng,
            )
        )

        surface = cell_surface(
            momentum_bins,
            permuted_si,
            target,
        )

        statistic = (
            surface_statistic(
                surface
            )
        )

        high_high = (
            high_high_value(
                surface
            )
        )

        if statistic is not None:
            max_statistics.append(
                statistic
            )

        if high_high is not None:
            high_high_statistics.append(
                high_high
            )

        if (
            (permutation + 1)
            % 100
            == 0
        ):
            print(
                f"  permutation "
                f"{permutation + 1:,}/"
                f"{N_PERMUTATIONS:,}",
                flush=True,
            )

    return (
        np.asarray(
            max_statistics,
            dtype=float,
        ),
        np.asarray(
            high_high_statistics,
            dtype=float,
        ),
    )


def run_target(
    frame: pd.DataFrame,
    momentum_bins: np.ndarray,
    si_bins: np.ndarray,
    target: np.ndarray,
    target_name: str,
    date_values: np.ndarray,
) -> tuple[
    pd.DataFrame,
    dict[str, Any],
]:
    observed = cell_surface(
        momentum_bins,
        si_bins,
        target,
    )

    observed_max = (
        surface_statistic(
            observed
        )
    )

    observed_high_high = (
        high_high_value(
            observed
        )
    )

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    (
        null_max,
        null_high_high,
    ) = permutation_statistics(
        momentum_bins,
        si_bins,
        target,
        date_values,
        rng,
    )

    max_statistic_p = None

    if (
        observed_max is not None
        and len(null_max)
    ):
        max_statistic_p = float(
            (
                1
                + np.sum(
                    null_max
                    >= observed_max
                )
            )
            / (
                len(null_max)
                + 1
            )
        )

    high_high_one_sided_p = None
    high_high_two_sided_p = None

    if (
        observed_high_high
        is not None
        and len(null_high_high)
    ):
        high_high_one_sided_p = float(
            (
                1
                + np.sum(
                    null_high_high
                    >= observed_high_high
                )
            )
            / (
                len(null_high_high)
                + 1
            )
        )

        high_high_two_sided_p = float(
            (
                1
                + np.sum(
                    np.abs(
                        null_high_high
                    )
                    >= abs(
                        observed_high_high
                    )
                )
            )
            / (
                len(null_high_high)
                + 1
            )
        )

    summary = {
        "target": target_name,
        "test_period": (
            f"({TEST_START_EXCLUSIVE.date()}, "
            f"{DISCOVERY_END.date()}]"
        ),
        "n": int(
            np.isfinite(
                target
            ).sum()
        ),
        "observed_max_abs_incremental_pp": (
            observed_max * 100
            if observed_max is not None
            else None
        ),
        "max_abs_incremental_permutation_p": (
            max_statistic_p
        ),
        "observed_high_high_incremental_pp": (
            observed_high_high * 100
            if observed_high_high is not None
            else None
        ),
        "high_high_one_sided_p": (
            high_high_one_sided_p
        ),
        "high_high_two_sided_p": (
            high_high_two_sided_p
        ),
        "permutations": int(
            len(null_max)
        ),
        "min_cell_n": MIN_CELL_N,
    }

    return (
        observed,
        summary,
    )


def main() -> None:
    print(
        "Loading features...",
        flush=True,
    )

    frame = load_features().copy()

    frame[
        "snapshot_date"
    ] = pd.to_datetime(
        frame[
            "snapshot_date"
        ],
        errors="coerce",
    )

    frame = frame[
        (
            frame[
                "snapshot_date"
            ]
            > TEST_START_EXCLUSIVE
        )
        & (
            frame[
                "snapshot_date"
            ]
            <= DISCOVERY_END
        )
    ].copy()

    frame.reset_index(
        drop=True,
        inplace=True,
    )

    print(
        f"Independent 2025 test rows "
        f"through {DISCOVERY_END.date()}: "
        f"{len(frame):,}",
        flush=True,
    )

    momentum = build_signal(
        frame,
        MOMENTUM_SIGNAL_NAME,
    )

    si = build_signal(
        frame,
        SI_SIGNAL_NAME,
    )

    momentum_bins = (
        cross_sectional_deciles(
            frame,
            momentum,
        )
    )

    si_bins = (
        cross_sectional_deciles(
            frame,
            si,
        )
    )

    date_values = (
        frame[
            "snapshot_date"
        ].to_numpy()
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_surfaces: list[
        pd.DataFrame
    ] = []

    all_summaries: list[
        dict[str, Any]
    ] = []

    for target_name in TARGET_NAMES:
        print(
            f"Surface + permutation: "
            f"{target_name}",
            flush=True,
        )

        target = (
            build_target(
                frame,
                get_target(
                    target_name
                ),
            )
            .to_numpy(
                dtype=float
            )
        )

        (
            surface,
            summary,
        ) = run_target(
            frame,
            momentum_bins,
            si_bins,
            target,
            target_name,
            date_values,
        )

        surface.insert(
            0,
            "target",
            target_name,
        )

        all_surfaces.append(
            surface
        )

        all_summaries.append(
            summary
        )

    surfaces = pd.concat(
        all_surfaces,
        ignore_index=True,
    )

    summary_frame = pd.DataFrame(
        all_summaries
    )

    surfaces.to_csv(
        OUTPUT_DIR
        / "surface_10x10.csv",
        index=False,
    )

    summary_frame.to_csv(
        OUTPUT_DIR
        / "permutation_summary.csv",
        index=False,
    )

    results = {
        "analysis": (
            "momentum_si_10x10_"
            "surface_permutation"
        ),
        "discovery_end": str(
            DISCOVERY_END.date()
        ),
        "test_start_exclusive": str(
            TEST_START_EXCLUSIVE.date()
        ),
        "targets": list(
            TARGET_NAMES
        ),
        "momentum_signal": (
            MOMENTUM_SIGNAL_NAME
        ),
        "si_signal": SI_SIGNAL_NAME,
        "bins": N_BINS,
        "permutations": N_PERMUTATIONS,
        "min_cell_n": MIN_CELL_N,
        "random_state": RANDOM_STATE,
        "summary": all_summaries,
        "method": {
            "surface": (
                "10x10 cross-sectional "
                "momentum decile x SI decile"
            ),
            "permutation": (
                "shuffle SI decile labels "
                "within each snapshot date"
            ),
            "max_statistic": (
                "maximum absolute cell "
                "event-rate difference "
                "versus its momentum-decile "
                "row baseline"
            ),
            "high_high_statistic": (
                "decile 10 momentum x "
                "decile 10 SI incremental "
                "event-rate difference "
                "versus momentum-decile-10 "
                "row baseline"
            ),
            "multiple_testing": (
                "maximum-cell permutation "
                "p-value accounts for searching "
                "across the complete 100-cell "
                "surface"
            ),
            "independent_oos_note": (
                "Only window_1/test calendar "
                "period in 2025 is used. "
                "window_2/validation covers "
                "the same calendar period and "
                "is not treated as independent."
            ),
        },
    }

    (
        OUTPUT_DIR
        / "results.json"
    ).write_text(
        json.dumps(
            results,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    lines = [
        "# Momentum × SI 10×10 Surface",
        "",
        f"- Independent test period: "
        f"`({TEST_START_EXCLUSIVE.date()}, "
        f"{DISCOVERY_END.date()}]`",
        f"- Rows: `{len(frame):,}`",
        f"- Permutations: "
        f"`{N_PERMUTATIONS:,}`",
        f"- Minimum cell N for "
        f"max-statistic: `{MIN_CELL_N}`",
        "",
        "## Summary",
        "",
        "| Target | Max abs incremental pp | Surface-wide permutation p | High-high incremental pp | High-high one-sided p | High-high two-sided p |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for row in all_summaries:
        def fmt(
            value: Any,
        ) -> str:
            if value is None:
                return ""

            return f"{value:.4f}"

        lines.append(
            f"| {row['target']} | "
            f"{fmt(row['observed_max_abs_incremental_pp'])} | "
            f"{fmt(row['max_abs_incremental_permutation_p'])} | "
            f"{fmt(row['observed_high_high_incremental_pp'])} | "
            f"{fmt(row['high_high_one_sided_p'])} | "
            f"{fmt(row['high_high_two_sided_p'])} |"
        )

    lines.extend(
        [
            "",
            "## What the test does",
            "",
            "The 10×10 surface describes the relationship "
            "between cross-sectional momentum and SI.",
            "",
            "For each momentum decile, the SI effect is "
            "measured relative to the complete momentum "
            "decile row. This removes the simple effect "
            "of momentum itself from the cell comparison.",
            "",
            "The permutation test shuffles SI decile labels "
            "within each snapshot date. Momentum and the "
            "target remain unchanged.",
            "",
            "The surface-wide p-value compares the observed "
            "largest absolute cell effect against the "
            "largest absolute cell effect from each "
            "permutation. This accounts for the fact that "
            "100 cells are being searched.",
            "",
            "The high-high p-values are a separate focused "
            "test of the extreme momentum × extreme SI "
            "corner.",
            "",
            "Only the 2025 independent test period is used. "
            "Window_2 validation is the same 2025 calendar "
            "period and is not counted as another "
            "independent test.",
            "",
            "## Important limitation",
            "",
            "This is an inference test of the pre-specified "
            "momentum × SI surface. It does not select a "
            "trading rule, optimize entry or exit thresholds, "
            "or establish economic profitability.",
        ]
    )

    (
        OUTPUT_DIR
        / "report.md"
    ).write_text(
        "\n".join(lines)
        + "\n",
        encoding="utf-8",
    )

    metadata = {
        "analysis": (
            "momentum_si_10x10_"
            "surface_permutation"
        ),
        "created_at_utc": (
            pd.Timestamp.now(
                tz="UTC"
            ).isoformat()
        ),
        "discovery_end": str(
            DISCOVERY_END.date()
        ),
        "test_start_exclusive": str(
            TEST_START_EXCLUSIVE.date()
        ),
        "targets": list(
            TARGET_NAMES
        ),
        "rows": int(
            len(frame)
        ),
        "permutations": N_PERMUTATIONS,
        "min_cell_n": MIN_CELL_N,
        "random_state": RANDOM_STATE,
    }

    (
        OUTPUT_DIR
        / "metadata.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        f"Written to {OUTPUT_DIR}",
        flush=True,
    )


if __name__ == "__main__":
    main()
