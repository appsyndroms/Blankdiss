from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ml.config import TARGETS
from ml.dataset import build_target, load_features
from ml.research.signals import build_signal


OUTPUT_DIR = Path(
    "data/processed/ml/research/momentum_si_surface_stability"
)

MIN_CELL_N = 50
TOP_K = 5
N_DECILES = 10

DATE_CUTOFF = pd.Timestamp("2025-12-19")
HALF_SPLIT = pd.Timestamp("2025-06-30")

TARGET_NAMES = (
    "down_5pct_5d",
    "down_7pct_5d",
    "down_10pct_5d",
)

MOMENTUM_SIGNAL_NAME = "price_momentum_5d"
SI_SIGNAL_NAME = "short_interest_change"


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
            "snapshot_date": frame["snapshot_date"],
            "value": pd.to_numeric(
                values,
                errors="coerce",
            ),
        },
        index=frame.index,
    )

    for _, index in working.groupby(
        "snapshot_date",
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
                ranks * N_DECILES
            ).astype(int)
            - 1
        )

        bins = np.clip(
            bins,
            0,
            N_DECILES - 1,
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
        N_DECILES
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
            N_DECILES
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

    return pd.DataFrame(rows)


def _select_top_cells(
    surface: pd.DataFrame,
) -> pd.DataFrame:
    eligible = surface.loc[
        surface["n"] >= MIN_CELL_N
    ].copy()

    if eligible.empty:
        return eligible

    eligible["abs_effect"] = (
        eligible[
            "incremental_vs_momentum_row"
        ].abs()
    )

    return (
        eligible.sort_values(
            "abs_effect",
            ascending=False,
        )
        .head(TOP_K)
        .copy()
    )


def _evaluate_cells(
    surface: pd.DataFrame,
    cells: pd.DataFrame,
) -> pd.DataFrame:
    if cells.empty:
        return pd.DataFrame()

    keys = cells[
        [
            "momentum_decile",
            "si_decile",
        ]
    ].drop_duplicates()

    return surface.merge(
        keys,
        on=[
            "momentum_decile",
            "si_decile",
        ],
        how="inner",
    )


def _prepare_data() -> pd.DataFrame:
    features = load_features().copy()

    if "snapshot_date" not in features.columns:
        raise ValueError(
            "Feature data must contain a "
            "'snapshot_date' column."
        )

    features["snapshot_date"] = pd.to_datetime(
        features["snapshot_date"],
        errors="coerce",
    )

    features = features.loc[
        features["snapshot_date"].notna()
    ].copy()

    features = features.loc[
        features["snapshot_date"] <= DATE_CUTOFF
    ].copy()

    momentum = build_signal(
        features,
        MOMENTUM_SIGNAL_NAME,
    )

    short_interest = build_signal(
        features,
        SI_SIGNAL_NAME,
    )

    features["momentum"] = momentum
    features["short_interest"] = short_interest

    target_frames: list[pd.Series] = []

    for target_name in TARGET_NAMES:
        target = build_target(
            features,
            get_target(target_name),
        )

        target_frames.append(
            target.rename(target_name)
        )

    data = features.copy()

    for target_series in target_frames:
        data[target_series.name] = target_series

    required = [
        "snapshot_date",
        "momentum",
        "short_interest",
        *TARGET_NAMES,
    ]

    missing = [
        column
        for column in required
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            "Missing required columns: "
            f"{missing}"
        )

    data = data.dropna(
        subset=[
            "snapshot_date",
            "momentum",
            "short_interest",
        ]
    ).copy()

    data = data.sort_values(
        [
            "snapshot_date",
            "security_key",
        ],
        kind="mergesort",
    ).reset_index(
        drop=True
    )

    return data


def _make_surface(
    df: pd.DataFrame,
    target_name: str,
) -> pd.DataFrame:
    momentum_bins = cross_sectional_deciles(
        df,
        df["momentum"],
    )

    si_bins = cross_sectional_deciles(
        df,
        df["short_interest"],
    )

    target = pd.to_numeric(
        df[target_name],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    return cell_surface(
        momentum_bins,
        si_bins,
        target,
    )


def _run_direction(
    data: pd.DataFrame,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    eval_start: pd.Timestamp,
    eval_end: pd.Timestamp,
    direction: str,
) -> tuple[
    list[dict[str, Any]],
    list[pd.DataFrame],
]:
    train = data.loc[
        (data["snapshot_date"] >= train_start)
        & (data["snapshot_date"] <= train_end)
    ].copy()

    evaluation = data.loc[
        (data["snapshot_date"] >= eval_start)
        & (data["snapshot_date"] <= eval_end)
    ].copy()

    replication_rows: list[dict[str, Any]] = []
    selected_surfaces: list[pd.DataFrame] = []

    selection_half = (
        "H1"
        if direction == "H1_to_H2"
        else "H2"
    )

    for target_name in TARGET_NAMES:
        train_surface = _make_surface(
            train,
            target_name,
        )

        eval_surface = _make_surface(
            evaluation,
            target_name,
        )

        selected = _select_top_cells(
            train_surface
        )

        if selected.empty:
            continue

        evaluated = _evaluate_cells(
            eval_surface,
            selected,
        )

        for _, selected_row in selected.iterrows():
            momentum_decile = int(
                selected_row[
                    "momentum_decile"
                ]
            )

            si_decile = int(
                selected_row[
                    "si_decile"
                ]
            )

            matching = evaluated.loc[
                (
                    evaluated[
                        "momentum_decile"
                    ]
                    == momentum_decile
                )
                & (
                    evaluated[
                        "si_decile"
                    ]
                    == si_decile
                )
            ]

            if matching.empty:
                eval_n = 0
                eval_event_rate = np.nan
                eval_momentum_baseline = np.nan
                eval_effect = np.nan
                eval_lift = np.nan

            else:
                eval_row = matching.iloc[0]

                eval_n = int(
                    eval_row["n"]
                )

                eval_event_rate = float(
                    eval_row["event_rate"]
                )

                eval_momentum_baseline = float(
                    eval_row[
                        "momentum_row_event_rate"
                    ]
                )

                eval_effect = float(
                    eval_row[
                        "incremental_vs_momentum_row"
                    ]
                )

                eval_lift = (
                    float(
                        eval_row[
                            "incremental_lift_vs_momentum_row"
                        ]
                    )
                    if pd.notna(
                        eval_row[
                            "incremental_lift_vs_momentum_row"
                        ]
                    )
                    else np.nan
                )

            replication_rows.append(
                {
                    "direction": direction,
                    "target": target_name,
                    "momentum_decile": momentum_decile,
                    "si_decile": si_decile,
                    "train_n": int(
                        selected_row["n"]
                    ),
                    "train_event_rate": float(
                        selected_row[
                            "event_rate"
                        ]
                    ),
                    "train_momentum_baseline": float(
                        selected_row[
                            "momentum_row_event_rate"
                        ]
                    ),
                    "train_incremental_effect": float(
                        selected_row[
                            "incremental_vs_momentum_row"
                        ]
                    ),
                    "train_lift": (
                        float(
                            selected_row[
                                "incremental_lift_vs_momentum_row"
                            ]
                        )
                        if pd.notna(
                            selected_row[
                                "incremental_lift_vs_momentum_row"
                            ]
                        )
                        else np.nan
                    ),
                    "eval_n": eval_n,
                    "eval_event_rate": eval_event_rate,
                    "eval_momentum_baseline": (
                        eval_momentum_baseline
                    ),
                    "eval_incremental_effect": (
                        eval_effect
                    ),
                    "eval_lift": eval_lift,
                }
            )

        selected = selected.copy()
        selected["direction"] = direction
        selected["selection_half"] = (
            selection_half
        )

        selected_surfaces.append(
            selected
        )

    return (
        replication_rows,
        selected_surfaces,
    )


def _write_report(
    replication: pd.DataFrame,
    metadata: dict[str, Any],
) -> None:
    lines = [
        "# Momentum × SI Surface Stability",
        "",
        "## Purpose",
        "",
        (
            "Tests whether the strongest Momentum × Short "
            "Interest surface effects found in one half of "
            "2025 replicate in the other half."
        ),
        "",
        "## Method",
        "",
        f"- Date range: through {DATE_CUTOFF.date()}",
        "- H1: 2025-01-01 through 2025-06-30",
        "- H2: 2025-07-01 through 2025-12-19",
        f"- Surface: {N_DECILES} × {N_DECILES}",
        f"- Minimum cell N: {MIN_CELL_N}",
        f"- Cells selected per target: {TOP_K}",
        (
            "- Momentum: price_momentum_5d "
            "(cross-sectional deciles)"
        ),
        (
            "- Short interest: short_interest_change "
            "(cross-sectional deciles)"
        ),
        (
            "- Cell effect = cell event rate minus the "
            "complete momentum-decile row event rate."
        ),
        (
            "- The same momentum/SI cell coordinates are "
            "evaluated in the opposite half."
        ),
        "",
        "## Cross-half replication",
        "",
    ]

    if replication.empty:
        lines.append(
            "No eligible replication results were produced."
        )

    else:
        for direction in sorted(
            replication["direction"].unique()
        ):
            lines.extend(
                [
                    f"### {direction}",
                    "",
                ]
            )

            subset = replication.loc[
                replication["direction"] == direction
            ]

            for target_name in TARGET_NAMES:
                target_rows = subset.loc[
                    subset["target"] == target_name
                ]

                if target_rows.empty:
                    continue

                lines.append(
                    f"#### `{target_name}`"
                )
                lines.append("")

                for _, row in target_rows.iterrows():
                    train_effect = row[
                        "train_incremental_effect"
                    ]

                    eval_effect = row[
                        "eval_incremental_effect"
                    ]

                    eval_effect_text = (
                        f"{eval_effect:.4f}"
                        if pd.notna(eval_effect)
                        else "n/a"
                    )

                    lines.append(
                        "- "
                        f"Momentum "
                        f"{int(row['momentum_decile'])}, "
                        f"SI "
                        f"{int(row['si_decile'])}: "
                        f"train effect "
                        f"{train_effect:.4f}, "
                        f"evaluation effect "
                        f"{eval_effect_text}, "
                        f"train N "
                        f"{int(row['train_n'])}, "
                        f"evaluation N "
                        f"{int(row['eval_n'])}."
                    )

                lines.append("")

    lines.extend(
        [
            "## Interpretation",
            "",
            (
                "This is a temporal stability test, not a "
                "trading strategy validation."
            ),
            "",
            (
                "A cell selected because of a large effect "
                "in one half can only be considered "
                "temporally stable if a similar effect is "
                "visible at the identical coordinates in "
                "the other half."
            ),
            "",
            (
                "The analysis does not establish causality "
                "and does not account for transaction costs, "
                "execution, position sizing, or portfolio "
                "construction."
            ),
            "",
        ]
    )

    (
        OUTPUT_DIR / "report.md"
    ).write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "=== Momentum × SI Surface Stability Analysis ===",
        flush=True,
    )

    data = _prepare_data()

    print(
        f"Rows after preparation: {len(data):,}",
        flush=True,
    )

    h1_start = pd.Timestamp(
        "2025-01-01"
    )

    h1_end = HALF_SPLIT

    h2_start = (
        HALF_SPLIT
        + pd.Timedelta(days=1)
    )

    h2_end = DATE_CUTOFF

    all_replication: list[
        dict[str, Any]
    ] = []

    all_surfaces: list[
        pd.DataFrame
    ] = []

    rows, surfaces = _run_direction(
        data,
        h1_start,
        h1_end,
        h2_start,
        h2_end,
        "H1_to_H2",
    )

    all_replication.extend(rows)
    all_surfaces.extend(surfaces)

    rows, surfaces = _run_direction(
        data,
        h2_start,
        h2_end,
        h1_start,
        h1_end,
        "H2_to_H1",
    )

    all_replication.extend(rows)
    all_surfaces.extend(surfaces)

    replication = pd.DataFrame(
        all_replication
    )

    if all_surfaces:
        selected_surfaces = pd.concat(
            all_surfaces,
            ignore_index=True,
        )
    else:
        selected_surfaces = pd.DataFrame()

    replication.to_csv(
        OUTPUT_DIR
        / "cross_half_replication.csv",
        index=False,
    )

    selected_surfaces.to_csv(
        OUTPUT_DIR
        / "surface_h1_h2.csv",
        index=False,
    )

    metadata = {
        "analysis": (
            "momentum_si_surface_stability"
        ),
        "date_cutoff": str(
            DATE_CUTOFF.date()
        ),
        "half_split": str(
            HALF_SPLIT.date()
        ),
        "n_rows": int(
            len(data)
        ),
        "n_deciles": N_DECILES,
        "min_cell_n": MIN_CELL_N,
        "top_k": TOP_K,
        "targets": list(
            TARGET_NAMES
        ),
        "momentum_signal": (
            MOMENTUM_SIGNAL_NAME
        ),
        "si_signal": (
            SI_SIGNAL_NAME
        ),
        "h1_start": str(
            h1_start.date()
        ),
        "h1_end": str(
            h1_end.date()
        ),
        "h2_start": str(
            h2_start.date()
        ),
        "h2_end": str(
            h2_end.date()
        ),
    }

    (
        OUTPUT_DIR / "metadata.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    results = {
        "metadata": metadata,
        "replication_rows": int(
            len(replication)
        ),
    }

    (
        OUTPUT_DIR / "results.json"
    ).write_text(
        json.dumps(
            results,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    _write_report(
        replication,
        metadata,
    )

    print(
        f"Replication rows: "
        f"{len(replication):,}",
        flush=True,
    )

    print(
        "Results written to:",
        flush=True,
    )

    print(
        OUTPUT_DIR,
        flush=True,
    )


if __name__ == "__main__":
    main()
