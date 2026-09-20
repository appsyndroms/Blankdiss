from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.config import TARGETS, WALK_FORWARD_WINDOWS
from ml.dataset import build_target, load_features
from ml.research.signals import build_signal


DISCOVERY_END = pd.Timestamp("2025-12-19")
TARGET_NAME = "down_10pct_5d"
SI_SIGNAL_NAME = "short_interest_change"
VOL_SIGNAL_NAME = "price_volatility_20d"

VOL_GROUPS = 3
SI_DECILES = 10

MIN_MODEL_ROWS = 100
MIN_CLASS_COUNT = 5

OUTPUT_DIR = Path(
    "data/processed/ml/research/incremental_si"
)


@dataclass(frozen=True)
class SplitConfig:
    name: str
    train_end: pd.Timestamp
    validation_end: pd.Timestamp
    test_end: pd.Timestamp


def target_config():
    for target in TARGETS:
        if target.name == TARGET_NAME:
            return target

    raise ValueError(
        f"Unknown target: {TARGET_NAME}"
    )


def build_splits() -> list[SplitConfig]:
    splits = []

    for index, window in enumerate(
        WALK_FORWARD_WINDOWS,
        start=1,
    ):
        splits.append(
            SplitConfig(
                name=f"window_{index}",
                train_end=pd.Timestamp(
                    window.train_end
                ),
                validation_end=pd.Timestamp(
                    window.validation_end
                ),
                test_end=pd.Timestamp(
                    window.test_end
                ),
            )
        )

    return splits


def safe_auc(
    y_true: np.ndarray,
    scores: np.ndarray,
) -> float | None:
    if len(y_true) == 0:
        return None

    if np.unique(y_true).size < 2:
        return None

    return float(
        roc_auc_score(
            y_true,
            scores,
        )
    )


def safe_log_loss(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> float | None:
    if len(y_true) == 0:
        return None

    try:
        return float(
            log_loss(
                y_true,
                probabilities,
                labels=[0, 1],
            )
        )
    except ValueError:
        return None


def safe_brier(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> float | None:
    if len(y_true) == 0:
        return None

    return float(
        brier_score_loss(
            y_true,
            probabilities,
        )
    )


def model_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, Any]:
    return {
        "n": int(len(y_true)),
        "events": int(y_true.sum()),
        "event_rate": (
            float(y_true.mean())
            if len(y_true)
            else None
        ),
        "auc": safe_auc(
            y_true,
            probabilities,
        ),
        "log_loss": safe_log_loss(
            y_true,
            probabilities,
        ),
        "brier": safe_brier(
            y_true,
            probabilities,
        ),
    }


def rankdata_average(
    values: np.ndarray,
) -> np.ndarray:
    """
    Average ranks, implemented locally so the analysis does not
    require scipy.
    """
    values = np.asarray(
        values,
        dtype=float,
    )

    order = np.argsort(
        values,
        kind="mergesort",
    )

    sorted_values = values[order]

    ranks = np.empty(
        len(values),
        dtype=float,
    )

    start = 0

    while start < len(values):
        end = start + 1

        while (
            end < len(values)
            and sorted_values[end]
            == sorted_values[start]
        ):
            end += 1

        average_rank = (
            start
            + end
            - 1
        ) / 2.0 + 1.0

        ranks[
            order[start:end]
        ] = average_rank

        start = end

    return ranks


def spearman(
    x: np.ndarray,
    y: np.ndarray,
) -> float | None:
    valid = (
        np.isfinite(x)
        & np.isfinite(y)
    )

    if valid.sum() < 3:
        return None

    x_valid = x[valid]
    y_valid = y[valid]

    if (
        np.unique(x_valid).size < 2
        or np.unique(y_valid).size < 2
    ):
        return None

    x_rank = rankdata_average(
        x_valid
    )
    y_rank = rankdata_average(
        y_valid
    )

    x_centered = (
        x_rank
        - x_rank.mean()
    )
    y_centered = (
        y_rank
        - y_rank.mean()
    )

    denominator = (
        np.sqrt(
            np.sum(
                x_centered ** 2
            )
        )
        * np.sqrt(
            np.sum(
                y_centered ** 2
            )
        )
    )

    if denominator == 0:
        return None

    return float(
        np.sum(
            x_centered
            * y_centered
        )
        / denominator
    )


def cross_sectional_group(
    frame: pd.DataFrame,
    values: pd.Series,
    groups: int,
) -> pd.Series:
    """
    Assign cross-sectional quantile groups independently for every
    snapshot date.

    Group 0 = lowest values.
    Group groups-1 = highest values.
    """
    result = pd.Series(
        np.nan,
        index=frame.index,
        dtype=float,
    )

    working = pd.DataFrame(
        {
            "date": frame[
                "snapshot_date"
            ],
            "value": values,
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

        if valid.sum() < groups:
            continue

        ranks = local.loc[
            valid
        ].rank(
            method="first",
            pct=True,
        )

        group = np.floor(
            ranks * groups
        ).astype(int)

        group = group.clip(
            upper=groups - 1
        )

        result.loc[
            group.index
        ] = group.astype(float)

    return result


def conditional_si_decile(
    frame: pd.DataFrame,
    si: pd.Series,
    volatility: pd.Series,
) -> pd.DataFrame:
    """
    Assign:
      - volatility tercile cross-sectionally per date
      - SI decile within date + volatility tercile

    This directly answers whether SI has a monotonic relationship
    with the target after conditioning on current volatility.
    """
    vol_group = cross_sectional_group(
        frame,
        volatility,
        VOL_GROUPS,
    )

    si_group = pd.Series(
        np.nan,
        index=frame.index,
        dtype=float,
    )

    working = pd.DataFrame(
        {
            "date": frame[
                "snapshot_date"
            ],
            "vol_group": vol_group,
            "si": si,
        },
        index=frame.index,
    )

    for (
        _,
        index,
    ) in working.groupby(
        [
            "date",
            "vol_group",
        ],
        sort=False,
    ).groups.items():
        local = working.loc[
            index,
            "si",
        ]

        valid = local.notna()

        if valid.sum() < SI_DECILES:
            continue

        ranks = local.loc[
            valid
        ].rank(
            method="first",
            pct=True,
        )

        group = np.floor(
            ranks * SI_DECILES
        ).astype(int)

        group = group.clip(
            upper=SI_DECILES - 1
        )

        si_group.loc[
            group.index
        ] = group.astype(float)

    return pd.DataFrame(
        {
            "vol_group": vol_group,
            "si_decile": si_group,
        },
        index=frame.index,
    )


def decile_analysis(
    frame: pd.DataFrame,
    target: np.ndarray,
    returns: np.ndarray,
    si: pd.Series,
    volatility: pd.Series,
) -> list[dict[str, Any]]:
    groups = conditional_si_decile(
        frame,
        si,
        volatility,
    )

    rows = []

    for vol_group in range(
        VOL_GROUPS
    ):
        for si_decile in range(
            SI_DECILES
        ):
            mask = (
                groups[
                    "vol_group"
                ].to_numpy()
                == vol_group
            ) & (
                groups[
                    "si_decile"
                ].to_numpy()
                == si_decile
            )

            valid_target = (
                mask
                & np.isfinite(target)
            )

            y = target[
                valid_target
            ]

            return_mask = (
                mask
                & np.isfinite(returns)
            )

            selected_returns = returns[
                return_mask
            ]

            rows.append(
                {
                    "vol_group": int(
                        vol_group
                    ),
                    "vol_group_label": (
                        "low"
                        if vol_group == 0
                        else (
                            "middle"
                            if vol_group == 1
                            else "high"
                        )
                    ),
                    "si_decile": int(
                        si_decile + 1
                    ),
                    "n": int(
                        len(y)
                    ),
                    "events": int(
                        y.sum()
                    ),
                    "event_rate": (
                        float(y.mean())
                        if len(y)
                        else None
                    ),
                    "mean_return": (
                        float(
                            selected_returns.mean()
                        )
                        if len(
                            selected_returns
                        )
                        else None
                    ),
                }
            )

    return rows


def conditional_rank_analysis(
    frame: pd.DataFrame,
    target: np.ndarray,
    si: pd.Series,
    volatility: pd.Series,
) -> list[dict[str, Any]]:
    groups = conditional_si_decile(
        frame,
        si,
        volatility,
    )

    rows = []

    for vol_group in range(
        VOL_GROUPS
    ):
        mask = (
            groups[
                "vol_group"
            ].to_numpy()
            == vol_group
        )

        valid = (
            mask
            & np.isfinite(target)
            & np.isfinite(
                si.to_numpy(
                    dtype=float
                )
            )
        )

        si_values = si.to_numpy(
            dtype=float
        )[valid]

        y = target[valid]

        rows.append(
            {
                "vol_group": int(
                    vol_group
                ),
                "vol_group_label": (
                    "low"
                    if vol_group == 0
                    else (
                        "middle"
                        if vol_group == 1
                        else "high"
                    )
                ),
                "n": int(
                    len(y)
                ),
                "events": int(
                    y.sum()
                ),
                "event_rate": (
                    float(y.mean())
                    if len(y)
                    else None
                ),
                "spearman_si_target": spearman(
                    si_values,
                    y,
                ),
            }
        )

    return rows


def fit_model(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
) -> Pipeline | None:
    if len(x_train) < MIN_MODEL_ROWS:
        return None

    if (
        np.unique(y_train).size < 2
        or np.sum(y_train == 1)
        < MIN_CLASS_COUNT
        or np.sum(y_train == 0)
        < MIN_CLASS_COUNT
    ):
        return None

    model = Pipeline(
        [
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "logistic",
                LogisticRegression(
                    max_iter=2000,
                    C=1.0,
                    solver="lbfgs",
                    random_state=42,
                ),
            ),
        ]
    )

    model.fit(
        x_train,
        y_train,
    )

    return model


def prepare_model_frame(
    si: pd.Series,
    volatility: pd.Series,
) -> pd.DataFrame:
    si_values = si.to_numpy(
        dtype=float
    )

    vol_values = volatility.to_numpy(
        dtype=float
    )

    result = pd.DataFrame(
        {
            "volatility": vol_values,
            "short_interest_change": si_values,
            "si_x_volatility": (
                si_values
                * vol_values
            ),
        }
    )

    return result.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )


def run_model_family(
    model_frame: pd.DataFrame,
    target: np.ndarray,
    train_mask: np.ndarray,
    eval_mask: np.ndarray,
) -> list[dict[str, Any]]:
    specifications = {
        "volatility_only": [
            "volatility",
        ],
        "volatility_plus_si": [
            "volatility",
            "short_interest_change",
        ],
        "volatility_plus_si_interaction": [
            "volatility",
            "short_interest_change",
            "si_x_volatility",
        ],
    }

    rows = []

    for (
        model_name,
        columns,
    ) in specifications.items():
        train_valid = (
            train_mask
            & np.isfinite(target)
        )

        eval_valid = (
            eval_mask
            & np.isfinite(target)
        )

        for column in columns:
            train_valid &= (
                np.isfinite(
                    model_frame[
                        column
                    ].to_numpy()
                )
            )
            eval_valid &= (
                np.isfinite(
                    model_frame[
                        column
                    ].to_numpy()
                )
            )

        if not train_valid.any():
            rows.append(
                {
                    "model": model_name,
                    "status": "INSUFFICIENT_DATA",
                }
            )
            continue

        if not eval_valid.any():
            rows.append(
                {
                    "model": model_name,
                    "status": "NO_EVALUATION_DATA",
                }
            )
            continue

        x_train = model_frame.loc[
            train_valid,
            columns,
        ]

        y_train = target[
            train_valid
        ].astype(int)

        x_eval = model_frame.loc[
            eval_valid,
            columns,
        ]

        y_eval = target[
            eval_valid
        ].astype(int)

        model = fit_model(
            x_train,
            y_train,
        )

        if model is None:
            rows.append(
                {
                    "model": model_name,
                    "status": "INSUFFICIENT_TRAIN_DATA",
                }
            )
            continue

        probabilities = model.predict_proba(
            x_eval
        )[:, 1]

        metrics = model_metrics(
            y_eval,
            probabilities,
        )

        rows.append(
            {
                "model": model_name,
                "status": "ok",
                "features": columns,
                **metrics,
            }
        )

    by_name = {
        row["model"]: row
        for row in rows
        if row.get("status") == "ok"
    }

    baseline = by_name.get(
        "volatility_only"
    )

    if baseline is not None:
        for row in rows:
            if row.get("status") != "ok":
                continue

            for metric in (
                "auc",
                "log_loss",
                "brier",
            ):
                baseline_value = baseline.get(
                    metric
                )
                value = row.get(
                    metric
                )

                delta_key = (
                    f"delta_{metric}_vs_volatility_only"
                )

                if (
                    baseline_value is not None
                    and value is not None
                ):
                    row[delta_key] = (
                        value
                        - baseline_value
                    )
                else:
                    row[delta_key] = None

    return rows


def main() -> None:
    print(
        "Loading features...",
        flush=True,
    )

    frame = load_features().copy()

    frame["snapshot_date"] = pd.to_datetime(
        frame["snapshot_date"],
        errors="coerce",
    )

    frame = frame.loc[
        frame["snapshot_date"]
        <= DISCOVERY_END
    ].copy()

    frame.reset_index(
        drop=True,
        inplace=True,
    )

    print(
        f"Rows through "
        f"{DISCOVERY_END.date()}: "
        f"{len(frame):,}",
        flush=True,
    )

    target_cfg = target_config()

    target = build_target(
        frame,
        target_cfg,
    ).to_numpy(
        dtype=float
    )

    returns = pd.to_numeric(
        frame[
            target_cfg.return_column
        ],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    si = build_signal(
        frame,
        SI_SIGNAL_NAME,
    )

    volatility = build_signal(
        frame,
        VOL_SIGNAL_NAME,
    )

    model_frame = prepare_model_frame(
        si,
        volatility,
    )

    print(
        "Running conditional SI decile analysis...",
        flush=True,
    )

    deciles = decile_analysis(
        frame,
        target,
        returns,
        si,
        volatility,
    )

    conditional_ranks = (
        conditional_rank_analysis(
            frame,
            target,
            si,
            volatility,
        )
    )

    model_results = []

    for split in build_splits():
        train_mask = (
            frame["snapshot_date"]
            <= split.train_end
        ).to_numpy()

        validation_mask = (
            (
                frame["snapshot_date"]
                > split.train_end
            )
            & (
                frame["snapshot_date"]
                <= split.validation_end
            )
        ).to_numpy()

        test_mask = (
            (
                frame["snapshot_date"]
                > split.validation_end
            )
            & (
                frame["snapshot_date"]
                <= split.test_end
            )
        ).to_numpy()

        cutoff_mask = (
            frame["snapshot_date"]
            <= DISCOVERY_END
        ).to_numpy()

        train_mask &= cutoff_mask
        validation_mask &= cutoff_mask
        test_mask &= cutoff_mask

        for (
            evaluation_name,
            evaluation_mask,
        ) in (
            (
                "validation",
                validation_mask,
            ),
            (
                "test",
                test_mask,
            ),
        ):
            print(
                f"Model evaluation: "
                f"{split.name} / "
                f"{evaluation_name}",
                flush=True,
            )

            rows = run_model_family(
                model_frame=model_frame,
                target=target,
                train_mask=train_mask,
                eval_mask=evaluation_mask,
            )

            model_results.append(
                {
                    "window": split.name,
                    "evaluation": (
                        evaluation_name
                    ),
                    "train_end": str(
                        split.train_end.date()
                    ),
                    "validation_end": str(
                        split.validation_end.date()
                    ),
                    "test_end": str(
                        split.test_end.date()
                    ),
                    "train_rows": int(
                        train_mask.sum()
                    ),
                    "evaluation_rows": int(
                        evaluation_mask.sum()
                    ),
                    "results": rows,
                }
            )

    document = {
        "analysis": (
            "incremental_short_interest"
        ),
        "discovery_end": str(
            DISCOVERY_END.date()
        ),
        "target": TARGET_NAME,
        "signal": SI_SIGNAL_NAME,
        "conditioning_feature": (
            VOL_SIGNAL_NAME
        ),
        "method": {
            "volatility_groups": (
                VOL_GROUPS
            ),
            "si_deciles": SI_DECILES,
            "volatility_grouping": (
                "cross-sectional per snapshot_date"
            ),
            "si_grouping": (
                "within snapshot_date and "
                "volatility group"
            ),
            "model": (
                "standardized logistic regression"
            ),
            "regularization": "C=1.0",
            "interaction": (
                "short_interest_change "
                "* price_volatility_20d"
            ),
            "cutoff": str(
                DISCOVERY_END.date()
            ),
        },
        "rows": int(len(frame)),
        "decile_analysis": deciles,
        "conditional_rank_analysis": (
            conditional_ranks
        ),
        "walk_forward": model_results,
    }

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_path = (
        OUTPUT_DIR
        / "results.json"
    )

    results_path.write_text(
        json.dumps(
            document,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    pd.DataFrame(
        deciles
    ).to_csv(
        OUTPUT_DIR
        / "decile_analysis.csv",
        index=False,
    )

    pd.DataFrame(
        conditional_ranks
    ).to_csv(
        OUTPUT_DIR
        / "conditional_rank_analysis.csv",
        index=False,
    )

    flattened_models = []

    for run in model_results:
        for result in run[
            "results"
        ]:
            flattened = {
                "window": run[
                    "window"
                ],
                "evaluation": run[
                    "evaluation"
                ],
                "train_rows": run[
                    "train_rows"
                ],
                "evaluation_rows": run[
                    "evaluation_rows"
                ],
                **result,
            }

            flattened_models.append(
                flattened
            )

    pd.DataFrame(
        flattened_models
    ).to_csv(
        OUTPUT_DIR
        / "walk_forward_models.csv",
        index=False,
    )

    lines = [
        "# Incremental Short-Interest Analysis",
        "",
        f"- Target: `{TARGET_NAME}`",
        f"- SI signal: `{SI_SIGNAL_NAME}`",
        f"- Conditioning feature: `{VOL_SIGNAL_NAME}`",
        f"- Frozen cutoff: `{DISCOVERY_END.date()}`",
        "",
        "## Conditional SI analysis",
        "",
        "| Volatility | SI decile | N | Events | Event rate | Mean return |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for row in deciles:
        lines.append(
            "| "
            f"{row['vol_group_label']} | "
            f"{row['si_decile']} | "
            f"{row['n']} | "
            f"{row['events']} | "
            f"{_fmt(row['event_rate'])} | "
            f"{_fmt(row['mean_return'])} |"
        )

    lines.extend(
        [
            "",
            "## Conditional SI monotonicity",
            "",
            "| Volatility | N | Events | Event rate | Spearman SI vs target |",
            "|---|---:|---:|---:|---:|",
        ]
    )

    for row in conditional_ranks:
        lines.append(
            "| "
            f"{row['vol_group_label']} | "
            f"{row['n']} | "
            f"{row['events']} | "
            f"{_fmt(row['event_rate'])} | "
            f"{_fmt(row['spearman_si_target'])} |"
        )

    lines.extend(
        [
            "",
            "## Walk-forward model comparison",
            "",
            "| Window | Evaluation | Model | N | Events | AUC | Δ AUC | Log loss | Δ Log loss | Brier | Δ Brier |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )

    for row in flattened_models:
        lines.append(
            "| "
            f"{row['window']} | "
            f"{row['evaluation']} | "
            f"{row['model']} | "
            f"{row.get('n', '')} | "
            f"{row.get('events', '')} | "
            f"{_fmt(row.get('auc'))} | "
            f"{_fmt(row.get('delta_auc_vs_volatility_only'))} | "
            f"{_fmt(row.get('log_loss'))} | "
            f"{_fmt(row.get('delta_log_loss_vs_volatility_only'))} | "
            f"{_fmt(row.get('brier'))} | "
            f"{_fmt(row.get('delta_brier_vs_volatility_only'))} |"
        )

    report_path = (
        OUTPUT_DIR
        / "report.md"
    )

    report_path.write_text(
        "\n".join(lines)
        + "\n",
        encoding="utf-8",
    )

    metadata = {
        "analysis": (
            "incremental_short_interest"
        ),
        "created_at_utc": pd.Timestamp.now(
            tz="UTC"
        ).isoformat(),
        "discovery_end": str(
            DISCOVERY_END.date()
        ),
        "target": TARGET_NAME,
        "signal": SI_SIGNAL_NAME,
        "conditioning_feature": (
            VOL_SIGNAL_NAME
        ),
        "feature_rows": int(
            len(frame)
        ),
        "volatility_groups": (
            VOL_GROUPS
        ),
        "si_deciles": SI_DECILES,
        "model_count": 3,
        "walk_forward_windows": [
            asdict(split)
            for split in build_splits()
        ],
    }

    for window in metadata[
        "walk_forward_windows"
    ]:
        window[
            "train_end"
        ] = str(
            window[
                "train_end"
            ].date()
        )
        window[
            "validation_end"
        ] = str(
            window[
                "validation_end"
            ].date()
        )
        window[
            "test_end"
        ] = str(
            window[
                "test_end"
            ].date()
        )

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

    print()
    print(
        f"Written to {OUTPUT_DIR}"
    )


def _fmt(
    value: Any,
) -> str:
    if value is None:
        return ""

    if isinstance(
        value,
        float,
    ):
        return f"{value:.6f}"

    return str(value)


if __name__ == "__main__":
    main()
