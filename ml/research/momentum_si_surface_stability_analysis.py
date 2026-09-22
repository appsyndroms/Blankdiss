from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml.config import TARGETS
from ml.dataset import build_target, load_features
from ml.research.signals import build_signal


OUTPUT_DIR = Path("data/processed/ml/research/momentum_si_surface_stability")

MIN_CELL_N = 50
TOP_K = 5
N_DECILES = 10

DATE_CUTOFF = pd.Timestamp("2025-12-19")
HALF_SPLIT = pd.Timestamp("2025-06-30")


def _event_rate(values: pd.Series) -> float:
    if len(values) == 0:
        return float("nan")
    return float(values.mean())


def _make_surface(
    df: pd.DataFrame,
    target: str,
) -> pd.DataFrame:
    work = df.copy()

    work["momentum_decile"] = (
        work.groupby("date")["momentum"]
        .transform(
            lambda x: pd.qcut(
                x.rank(method="first"),
                N_DECILES,
                labels=False,
                duplicates="drop",
            )
            + 1
        )
    )

    work["si_decile"] = (
        work.groupby("date")["short_interest"]
        .transform(
            lambda x: pd.qcut(
                x.rank(method="first"),
                N_DECILES,
                labels=False,
                duplicates="drop",
            )
            + 1
        )
    )

    work = work.dropna(
        subset=[
            "momentum_decile",
            "si_decile",
            target,
        ]
    )

    momentum_baseline = (
        work.groupby(
            ["momentum_decile"],
            observed=True,
        )[target]
        .mean()
        .rename("momentum_baseline")
    )

    surface = (
        work.groupby(
            ["momentum_decile", "si_decile"],
            observed=True,
        )[target]
        .agg(
            event_rate="mean",
            n="size",
        )
        .reset_index()
    )

    surface = surface.merge(
        momentum_baseline.reset_index(),
        on="momentum_decile",
        how="left",
    )

    surface["incremental_effect"] = (
        surface["event_rate"]
        - surface["momentum_baseline"]
    )

    surface["lift"] = (
        surface["event_rate"]
        / surface["momentum_baseline"].replace(0, np.nan)
    )

    surface["target"] = target

    return surface


def _select_top_cells(
    surface: pd.DataFrame,
) -> pd.DataFrame:
    eligible = surface.loc[
        surface["n"] >= MIN_CELL_N
    ].copy()

    if eligible.empty:
        return eligible

    eligible["abs_effect"] = eligible[
        "incremental_effect"
    ].abs()

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
    half_name: str,
) -> pd.DataFrame:
    if cells.empty:
        return pd.DataFrame()

    keys = cells[
        [
            "momentum_decile",
            "si_decile",
        ]
    ].drop_duplicates()

    result = surface.merge(
        keys,
        on=[
            "momentum_decile",
            "si_decile",
        ],
        how="inner",
    )

    result["evaluation_half"] = half_name

    return result


def _prepare_data() -> pd.DataFrame:
    features = load_features()

    if "date" not in features.columns:
        raise ValueError(
            "Feature data must contain a 'date' column."
        )

    features["date"] = pd.to_datetime(
        features["date"]
    )

    features = features.loc[
        features["date"] <= DATE_CUTOFF
    ].copy()

    target_frames = []

    for target in TARGETS:
        target_df = build_target(
            features,
            target,
        )

        if isinstance(target_df, pd.Series):
            target_df = target_df.rename(target).to_frame()

        if target not in target_df.columns:
            if len(target_df.columns) == 1:
                target_df = target_df.rename(
                    columns={
                        target_df.columns[0]: target
                    }
                )
            else:
                raise ValueError(
                    f"Could not identify target column '{target}'."
                )

        target_frames.append(
            target_df[[target]]
        )

    data = features.copy()

    for target_df in target_frames:
        data = data.join(
            target_df,
            how="inner",
        )

    signal = build_signal(data)

    if isinstance(signal, pd.Series):
        data["momentum"] = signal
    else:
        if "momentum" not in signal.columns:
            raise ValueError(
                "build_signal() did not return a 'momentum' column."
            )

        data["momentum"] = signal["momentum"]

    required = [
        "date",
        "momentum",
        "short_interest",
    ]

    missing = [
        column
        for column in required
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    data = data.dropna(
        subset=[
            "date",
            "momentum",
            "short_interest",
        ]
    )

    return data


def _run_direction(
    data: pd.DataFrame,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    eval_start: pd.Timestamp,
    eval_end: pd.Timestamp,
    direction: str,
) -> tuple[list[dict], list[pd.DataFrame]]:
    train = data.loc[
        (data["date"] >= train_start)
        & (data["date"] <= train_end)
    ].copy()

    evaluation = data.loc[
        (data["date"] >= eval_start)
        & (data["date"] <= eval_end)
    ].copy()

    replication_rows: list[dict] = []
    surface_frames: list[pd.DataFrame] = []

    for target in TARGETS:
        train_surface = _make_surface(
            train,
            target,
        )

        eval_surface = _make_surface(
            evaluation,
            target,
        )

        selected = _select_top_cells(
            train_surface
        )

        if selected.empty:
            continue

        evaluated = _evaluate_cells(
            eval_surface,
            selected,
            direction,
        )

        for _, selected_row in selected.iterrows():
            momentum_decile = int(
                selected_row["momentum_decile"]
            )
            si_decile = int(
                selected_row["si_decile"]
            )

            matching = evaluated.loc[
                (evaluated["momentum_decile"] == momentum_decile)
                & (
                    evaluated["si_decile"]
                    == si_decile
                )
            ]

            if matching.empty:
                eval_row = {
                    "direction": direction,
                    "target": target,
                    "momentum_decile": momentum_decile,
                    "si_decile": si_decile,
                    "train_n": int(selected_row["n"]),
                    "train_event_rate": float(
                        selected_row["event_rate"]
                    ),
                    "train_momentum_baseline": float(
                        selected_row[
                            "momentum_baseline"
                        ]
                    ),
                    "train_incremental_effect": float(
                        selected_row[
                            "incremental_effect"
                        ]
                    ),
                    "eval_n": 0,
                    "eval_event_rate": np.nan,
                    "eval_momentum_baseline": np.nan,
                    "eval_incremental_effect": np.nan,
                    "eval_lift": np.nan,
                }
            else:
                eval_row_raw = matching.iloc[0]

                eval_row = {
                    "direction": direction,
                    "target": target,
                    "momentum_decile": momentum_decile,
                    "si_decile": si_decile,
                    "train_n": int(selected_row["n"]),
                    "train_event_rate": float(
                        selected_row["event_rate"]
                    ),
                    "train_momentum_baseline": float(
                        selected_row[
                            "momentum_baseline"
                        ]
                    ),
                    "train_incremental_effect": float(
                        selected_row[
                            "incremental_effect"
                        ]
                    ),
                    "eval_n": int(
                        eval_row_raw["n"]
                    ),
                    "eval_event_rate": float(
                        eval_row_raw["event_rate"]
                    ),
                    "eval_momentum_baseline": float(
                        eval_row_raw[
                            "momentum_baseline"
                        ]
                    ),
                    "eval_incremental_effect": float(
                        eval_row_raw[
                            "incremental_effect"
                        ]
                    ),
                    "eval_lift": float(
                        eval_row_raw["lift"]
                    )
                    if pd.notna(
                        eval_row_raw["lift"]
                    )
                    else np.nan,
                }

            replication_rows.append(eval_row)

        selected["direction"] = direction
        selected["selection_half"] = (
            "H1"
            if direction == "H1_to_H2"
            else "H2"
        )
        surface_frames.append(selected)

    return replication_rows, surface_frames


def _write_report(
    replication: pd.DataFrame,
    metadata: dict,
) -> None:
    lines = [
        "# Momentum × SI Surface Stability",
        "",
        "## Purpose",
        "",
        (
            "Tests whether the strongest Momentum × Short Interest "
            "surface effects found in one half of 2025 replicate "
            "in the other half."
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
            "- Cell effect = cell event rate minus the complete "
            "momentum-decile row event rate."
        ),
        (
            "- The same momentum/SI cell coordinates are evaluated "
            "in the opposite half."
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

            for target in TARGETS:
                target_rows = subset.loc[
                    subset["target"] == target
                ]

                if target_rows.empty:
                    continue

                lines.append(
                    f"#### `{target}`"
                )
                lines.append("")

                for _, row in target_rows.iterrows():
                    train_effect = row[
                        "train_incremental_effect"
                    ]
                    eval_effect = row[
                        "eval_incremental_effect"
                    ]

                    lines.append(
                        "- "
                        f"Momentum {int(row['momentum_decile'])}, "
                        f"SI {int(row['si_decile'])}: "
                        f"train effect "
                        f"{train_effect:.4f}, "
                        f"evaluation effect "
                        f"{eval_effect:.4f}, "
                        f"train N {int(row['train_n'])}, "
                        f"evaluation N {int(row['eval_n'])}."
                    )

                lines.append("")

    lines.extend(
        [
            "## Interpretation",
            "",
            (
                "This is a temporal stability test, not a trading "
                "strategy validation. A cell selected because of a "
                "large effect in one half can only be considered "
                "temporally stable if a similar effect is visible "
                "at the identical coordinates in the other half."
            ),
            "",
            (
                "The analysis does not establish causality and does "
                "not account for transaction costs, execution, "
                "position sizing, or portfolio construction."
            ),
            "",
        ]
    )

    (OUTPUT_DIR / "report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "=== Momentum × SI Surface Stability Analysis ==="
    )

    data = _prepare_data()

    print(
        f"Rows after preparation: {len(data):,}"
    )

    h1_start = pd.Timestamp("2025-01-01")
    h1_end = HALF_SPLIT
    h2_start = HALF_SPLIT + pd.Timedelta(days=1)
    h2_end = DATE_CUTOFF

    all_replication = []
    all_surfaces = []

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
        OUTPUT_DIR / "cross_half_replication.csv",
        index=False,
    )

    selected_surfaces.to_csv(
        OUTPUT_DIR / "surface_h1_h2.csv",
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
        "n_rows": int(len(data)),
        "n_deciles": N_DECILES,
        "min_cell_n": MIN_CELL_N,
        "top_k": TOP_K,
        "targets": list(TARGETS),
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
            default=str,
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
            default=str,
        ),
        encoding="utf-8",
    )

    _write_report(
        replication,
        metadata,
    )

    print(
        f"Replication rows: {len(replication):,}"
    )

    print(
        "Results written to:"
    )
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
