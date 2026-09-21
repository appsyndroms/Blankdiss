from __future__ import annotations

import json
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

TARGET_NAMES = (
    "down_5pct_5d",
    "down_7pct_5d",
    "down_10pct_5d",
)

SI_SIGNAL_NAME = "short_interest_change"
MOMENTUM_SIGNAL_NAME = "price_momentum_5d"

MOMENTUM_TAILS = (
    0.20,
    0.10,
    0.05,
    0.025,
)

SI_TAILS = (
    0.20,
    0.10,
    0.05,
    0.025,
)

MIN_MODEL_ROWS = 100
MIN_CLASS_COUNT = 5

OUTPUT_DIR = Path(
    "data/processed/ml/research/momentum_incremental_si"
)


def get_target(
    name: str,
):
    for target in TARGETS:
        if target.name == name:
            return target

    raise ValueError(
        f"Unknown target: {name}"
    )


def cross_sectional_tail(
    frame: pd.DataFrame,
    values: pd.Series,
    fraction: float,
) -> np.ndarray:
    """
    Select the upper cross-sectional tail independently
    for each snapshot date.

    The grouping uses only values available on that
    snapshot date.
    """
    result = pd.Series(
        False,
        index=frame.index,
        dtype=bool,
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
            method="average",
            pct=True,
        )

        selected = (
            ranks
            >= (1.0 - fraction)
        )

        result.loc[
            selected.index
        ] = selected.astype(bool)

    return result.to_numpy()


def split_masks(
    frame: pd.DataFrame,
    train_end: pd.Timestamp,
    validation_end: pd.Timestamp,
    test_end: pd.Timestamp,
) -> dict[str, np.ndarray]:
    dates = frame[
        "snapshot_date"
    ]

    cutoff = (
        dates
        <= DISCOVERY_END
    )

    return {
        "train": (
            (dates <= train_end)
            & cutoff
        ).to_numpy(),

        "validation": (
            (dates > train_end)
            & (dates <= validation_end)
            & cutoff
        ).to_numpy(),

        "test": (
            (dates > validation_end)
            & (dates <= test_end)
            & cutoff
        ).to_numpy(),
    }


def safe_metrics(
    y: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, Any]:
    if len(y) == 0:
        return {
            "n": 0,
            "events": 0,
            "event_rate": None,
            "auc": None,
            "log_loss": None,
            "brier": None,
        }

    auc = None

    if np.unique(y).size >= 2:
        auc = float(
            roc_auc_score(
                y,
                probabilities,
            )
        )

    return {
        "n": int(len(y)),
        "events": int(y.sum()),
        "event_rate": float(
            y.mean()
        ),
        "auc": auc,
        "log_loss": float(
            log_loss(
                y,
                probabilities,
                labels=[0, 1],
            )
        ),
        "brier": float(
            brier_score_loss(
                y,
                probabilities,
            )
        ),
    }


def fit_predict(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_eval: pd.DataFrame,
) -> np.ndarray | None:
    if len(x_train) < MIN_MODEL_ROWS:
        return None

    if np.unique(
        y_train
    ).size < 2:
        return None

    if (
        (y_train == 1).sum()
        < MIN_CLASS_COUNT
    ):
        return None

    if (
        (y_train == 0).sum()
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

    return model.predict_proba(
        x_eval
    )[:, 1]


def tail_comparison(
    frame: pd.DataFrame,
    target: np.ndarray,
    momentum: pd.Series,
    si: pd.Series,
    evaluation_mask: np.ndarray,
    target_name: str,
) -> list[dict[str, Any]]:
    """
    Compare:

        high momentum

    against:

        high momentum + high SI

    This directly measures whether SI adds event-rate information
    inside an already-defined momentum regime.
    """
    rows: list[dict[str, Any]] = []

    for momentum_tail in MOMENTUM_TAILS:
        momentum_mask = (
            cross_sectional_tail(
                frame,
                momentum,
                momentum_tail,
            )
        )

        base = (
            evaluation_mask
            & np.isfinite(target)
            & momentum_mask
        )

        baseline_target = target[
            base
        ]

        baseline_rate = (
            float(
                baseline_target.mean()
            )
            if len(baseline_target)
            else None
        )

        for si_tail in SI_TAILS:
            si_mask = (
                cross_sectional_tail(
                    frame,
                    si,
                    si_tail,
                )
            )

            combined = (
                base
                & si_mask
            )

            combined_target = target[
                combined
            ]

            combined_rate = (
                float(
                    combined_target.mean()
                )
                if len(combined_target)
                else None
            )

            difference = None

            if (
                combined_rate is not None
                and baseline_rate is not None
            ):
                difference = (
                    combined_rate
                    - baseline_rate
                )

            lift = None

            if (
                combined_rate is not None
                and baseline_rate is not None
                and baseline_rate > 0
            ):
                lift = (
                    combined_rate
                    / baseline_rate
                )

            rows.append(
                {
                    "target": target_name,
                    "momentum_tail": momentum_tail,
                    "si_tail": si_tail,
                    "momentum_n": int(
                        base.sum()
                    ),
                    "momentum_events": int(
                        baseline_target.sum()
                    ),
                    "momentum_event_rate": (
                        baseline_rate
                    ),
                    "momentum_plus_si_n": int(
                        combined.sum()
                    ),
                    "momentum_plus_si_events": int(
                        combined_target.sum()
                    ),
                    "momentum_plus_si_event_rate": (
                        combined_rate
                    ),
                    "incremental_event_rate_difference": (
                        difference
                    ),
                    "incremental_event_rate_lift": (
                        lift
                    ),
                }
            )

    return rows


def model_comparison(
    frame: pd.DataFrame,
    target: np.ndarray,
    momentum: pd.Series,
    si: pd.Series,
    train_mask: np.ndarray,
    eval_mask: np.ndarray,
    target_name: str,
    window: str,
    evaluation: str,
) -> list[dict[str, Any]]:
    """
    Compare three nested models:

        1. momentum only
        2. momentum + SI
        3. momentum + SI + interaction

    The momentum-only model is the baseline.

    This is the main incremental-information test.
    """
    momentum_values = momentum.to_numpy(
        dtype=float
    )

    si_values = si.to_numpy(
        dtype=float
    )

    model_frame = pd.DataFrame(
        {
            "momentum": momentum_values,
            "short_interest_change": (
                si_values
            ),
            "momentum_x_si": (
                momentum_values
                * si_values
            ),
        }
    ).replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )

    specifications = {
        "momentum_only": [
            "momentum",
        ],
        "momentum_plus_si": [
            "momentum",
            "short_interest_change",
        ],
        "momentum_plus_si_interaction": [
            "momentum",
            "short_interest_change",
            "momentum_x_si",
        ],
    }

    rows: list[dict[str, Any]] = []

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
            values = model_frame[
                column
            ].to_numpy()

            train_valid &= (
                np.isfinite(values)
            )

            eval_valid &= (
                np.isfinite(values)
            )

        if (
            not train_valid.any()
            or not eval_valid.any()
        ):
            rows.append(
                {
                    "target": target_name,
                    "window": window,
                    "evaluation": evaluation,
                    "model": model_name,
                    "status": (
                        "INSUFFICIENT_DATA"
                    ),
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

        probabilities = fit_predict(
            x_train,
            y_train,
            x_eval,
        )

        if probabilities is None:
            rows.append(
                {
                    "target": target_name,
                    "window": window,
                    "evaluation": evaluation,
                    "model": model_name,
                    "status": (
                        "INSUFFICIENT_TRAIN_DATA"
                    ),
                }
            )

            continue

        rows.append(
            {
                "target": target_name,
                "window": window,
                "evaluation": evaluation,
                "model": model_name,
                "status": "ok",
                "train_n": int(
                    len(y_train)
                ),
                "train_events": int(
                    y_train.sum()
                ),
                **safe_metrics(
                    y_eval,
                    probabilities,
                ),
            }
        )

    baseline = next(
        (
            row
            for row in rows
            if (
                row.get("model")
                == "momentum_only"
                and row.get("status")
                == "ok"
            )
        ),
        None,
    )

    if baseline is not None:
        for row in rows:
            if row.get(
                "status"
            ) != "ok":
                continue

            for metric in (
                "auc",
                "log_loss",
                "brier",
            ):
                baseline_value = (
                    baseline.get(metric)
                )

                value = row.get(
                    metric
                )

                if (
                    baseline_value is None
                    or value is None
                ):
                    delta = None
                else:
                    delta = (
                        value
                        - baseline_value
                    )

                row[
                    f"delta_{metric}_vs_momentum_only"
                ] = delta

    return rows


def fmt(
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


def main() -> None:
    print(
        "Loading features...",
        flush=True,
    )

    frame = load_features().copy()

    frame["snapshot_date"] = (
        pd.to_datetime(
            frame["snapshot_date"],
            errors="coerce",
        )
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

    momentum = build_signal(
        frame,
        MOMENTUM_SIGNAL_NAME,
    )

    si = build_signal(
        frame,
        SI_SIGNAL_NAME,
    )

    tail_rows: list[
        dict[str, Any]
    ] = []

    model_rows: list[
        dict[str, Any]
    ] = []

    split_rows: list[
        dict[str, Any]
    ] = []

    for index, window in enumerate(
        WALK_FORWARD_WINDOWS,
        start=1,
    ):
        split_name = (
            f"window_{index}"
        )

        masks = split_masks(
            frame,
            pd.Timestamp(
                window.train_end
            ),
            pd.Timestamp(
                window.validation_end
            ),
            pd.Timestamp(
                window.test_end
            ),
        )

        split_rows.append(
            {
                "window": split_name,
                "train_rows": int(
                    masks["train"].sum()
                ),
                "validation_rows": int(
                    masks[
                        "validation"
                    ].sum()
                ),
                "test_rows": int(
                    masks["test"].sum()
                ),
                "test_available": bool(
                    masks["test"].any()
                ),
                "validation_period": (
                    f"({window.train_end}, "
                    f"{window.validation_end}]"
                ),
                "test_period": (
                    f"({window.validation_end}, "
                    f"{window.test_end}]"
                ),
            }
        )

        for target_name in TARGET_NAMES:
            target_cfg = get_target(
                target_name
            )

            target = build_target(
                frame,
                target_cfg,
            ).to_numpy(
                dtype=float
            )

            for evaluation in (
                "validation",
                "test",
            ):
                evaluation_mask = masks[
                    evaluation
                ]

                if not evaluation_mask.any():
                    print(
                        f"Skipping "
                        f"{split_name} / "
                        f"{evaluation}: "
                        f"no observations",
                        flush=True,
                    )

                    continue

                print(
                    f"Incremental SI: "
                    f"{split_name} / "
                    f"{evaluation} / "
                    f"{target_name}",
                    flush=True,
                )

                tail_rows.extend(
                    {
                        "window": split_name,
                        "evaluation": evaluation,
                        **row,
                    }
                    for row in tail_comparison(
                        frame,
                        target,
                        momentum,
                        si,
                        evaluation_mask,
                        target_name,
                    )
                )

                model_rows.extend(
                    model_comparison(
                        frame,
                        target,
                        momentum,
                        si,
                        masks["train"],
                        evaluation_mask,
                        target_name,
                        split_name,
                        evaluation,
                    )
                )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    document = {
        "analysis": (
            "momentum_incremental_short_interest"
        ),
        "discovery_end": str(
            DISCOVERY_END.date()
        ),
        "targets": list(
            TARGET_NAMES
        ),
        "momentum_signal": (
            MOMENTUM_SIGNAL_NAME
        ),
        "si_signal": SI_SIGNAL_NAME,
        "method": {
            "baseline": (
                "momentum_only"
            ),
            "incremental_model": (
                "momentum_plus_si"
            ),
            "interaction_model": (
                "momentum_plus_si_interaction"
            ),
            "cross_sectional_tails": True,
            "momentum_tails": list(
                MOMENTUM_TAILS
            ),
            "si_tails": list(
                SI_TAILS
            ),
            "walk_forward": True,
            "future_information_used_for_grouping": False,
            "independent_oos_note": (
                "window_1/test and "
                "window_2/validation "
                "cover the same 2025 "
                "calendar period"
            ),
        },
        "splits": split_rows,
        "tail_comparison": tail_rows,
        "model_comparison": model_rows,
    }

    (
        OUTPUT_DIR
        / "results.json"
    ).write_text(
        json.dumps(
            document,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    pd.DataFrame(
        tail_rows
    ).to_csv(
        OUTPUT_DIR
        / "tail_comparison.csv",
        index=False,
    )

    pd.DataFrame(
        model_rows
    ).to_csv(
        OUTPUT_DIR
        / "model_comparison.csv",
        index=False,
    )

    test_models = [
        row
        for row in model_rows
        if row["evaluation"]
        == "test"
    ]

    test_tails = [
        row
        for row in tail_rows
        if row["evaluation"]
        == "test"
    ]

    lines = [
        "# Momentum → Incremental Short Interest",
        "",
        f"- Frozen cutoff: "
        f"`{DISCOVERY_END.date()}`",
        f"- Momentum signal: "
        f"`{MOMENTUM_SIGNAL_NAME}`",
        f"- SI signal: "
        f"`{SI_SIGNAL_NAME}`",
        "",
        "## Question",
        "",
        "Does short-interest change add "
        "information beyond prior 5-day momentum?",
        "",
        "The main model comparison uses "
        "`momentum_only` as the baseline, "
        "then adds SI, followed by a "
        "momentum × SI interaction.",
        "",
        "## Walk-forward periods",
        "",
        "| Window | Train rows | Validation rows | Test rows | Test available | Validation period | Test period |",
        "|---|---:|---:|---:|---|---|---|",
    ]

    for row in split_rows:
        lines.append(
            f"| {row['window']} | "
            f"{row['train_rows']} | "
            f"{row['validation_rows']} | "
            f"{row['test_rows']} | "
            f"{row['test_available']} | "
            f"{row['validation_period']} | "
            f"{row['test_period']} |"
        )

    lines.extend(
        [
            "",
            "## OOS model comparison",
            "",
            "| Target | Window | Eval | Model | N | Events | AUC | ΔAUC vs momentum | Log loss | ΔLog loss | Brier | ΔBrier |",
            "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )

    for row in test_models:
        lines.append(
            f"| {row['target']} | "
            f"{row['window']} | "
            f"{row['evaluation']} | "
            f"{row['model']} | "
            f"{row.get('n', '')} | "
            f"{row.get('events', '')} | "
            f"{fmt(row.get('auc'))} | "
            f"{fmt(row.get('delta_auc_vs_momentum_only'))} | "
            f"{fmt(row.get('log_loss'))} | "
            f"{fmt(row.get('delta_log_loss_vs_momentum_only'))} | "
            f"{fmt(row.get('brier'))} | "
            f"{fmt(row.get('delta_brier_vs_momentum_only'))} |"
        )

    lines.extend(
        [
            "",
            "## OOS tail comparison",
            "",
            "| Target | Momentum tail | SI tail | Momentum N | Momentum event rate | Momentum + SI N | Momentum + SI event rate | Incremental pp | Incremental lift |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )

    for row in test_tails:
        lines.append(
            f"| {row['target']} | "
            f"{row['momentum_tail']:.3f} | "
            f"{row['si_tail']:.3f} | "
            f"{row['momentum_n']} | "
            f"{fmt(row['momentum_event_rate'])} | "
            f"{row['momentum_plus_si_n']} | "
            f"{fmt(row['momentum_plus_si_event_rate'])} | "
            f"{fmt(row['incremental_event_rate_difference'])} | "
            f"{fmt(row['incremental_event_rate_lift'])} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The primary test is whether "
            "`momentum_plus_si` improves out-of-sample "
            "metrics over `momentum_only`.",
            "",
            "The interaction model asks whether the "
            "incremental SI effect itself depends on "
            "momentum.",
            "",
            "Tail comparisons are descriptive. The "
            "walk-forward model comparison is the "
            "cleaner test of incremental predictive "
            "information.",
            "",
            "The 2025 test period appears again as "
            "window_2 validation. Those are the same "
            "calendar observations and must not be "
            "counted as independent OOS samples.",
            "",
            "No trading rule is selected automatically "
            "by this analysis.",
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
            "momentum_incremental_short_interest"
        ),
        "created_at_utc": (
            pd.Timestamp.now(
                tz="UTC"
            ).isoformat()
        ),
        "discovery_end": str(
            DISCOVERY_END.date()
        ),
        "targets": list(
            TARGET_NAMES
        ),
        "momentum_signal": (
            MOMENTUM_SIGNAL_NAME
        ),
        "si_signal": SI_SIGNAL_NAME,
        "tail_row_count": len(
            tail_rows
        ),
        "model_row_count": len(
            model_rows
        ),
        "walk_forward_windows": (
            split_rows
        ),
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
