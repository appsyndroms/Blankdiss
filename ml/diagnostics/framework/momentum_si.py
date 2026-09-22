from __future__ import annotations
import numpy as np
import pandas as pd
from .base import ExperimentResult
from ml.research.signals import build_signal
from ml.diagnostics.framework.event_risk import _prepare_event_risk_data
EVENT_RISK_BANDS = (
    ("top_5pct", 0.00, 0.05),
    ("5_20pct", 0.05, 0.20),
    ("20_50pct", 0.20, 0.50),
    ("bottom_50pct", 0.50, 1.00),
)
def _cross_sectional_deciles(
    frame: pd.DataFrame,
    column: str,
) -> pd.Series:
    """Assign 0-9 deciles independently for each snapshot date."""
    values = pd.to_numeric(frame[column], errors="coerce")
    ranks = values.groupby(frame["snapshot_date"]).rank(
        method="first",
        pct=True,
    )
    deciles = np.ceil(ranks * 10).astype("Int64") - 1
    deciles = deciles.clip(lower=0, upper=9)
    return deciles.fillna(-1).astype(int)
def _numeric_mean(frame: pd.DataFrame, column: str) -> float:
    if column not in frame.columns:
        return float("nan")
    values = pd.to_numeric(frame[column], errors="coerce")
    return float(values.mean()) if values.notna().any() else float("nan")
def _numeric_median(frame: pd.DataFrame, column: str) -> float:
    if column not in frame.columns:
        return float("nan")
    values = pd.to_numeric(frame[column], errors="coerce")
    return float(values.median()) if values.notna().any() else float("nan")
def _event_rate(frame: pd.DataFrame) -> float:
    """
    Five-day -5% event rate.
    The event is defined directly from the realized forward return so
    this comparison does not depend on the internals of the event model.
    """
    if "forward_return_5d" not in frame.columns:
        return float("nan")
    values = pd.to_numeric(
        frame["forward_return_5d"],
        errors="coerce",
    ).dropna()
    if values.empty:
        return float("nan")
    return float((values <= -0.05).mean())
def _event_n(frame: pd.DataFrame) -> int:
    if "forward_return_5d" not in frame.columns:
        return 0
    return int(
        pd.to_numeric(
            frame["forward_return_5d"],
            errors="coerce",
        ).notna().sum()
    )
def _delta_pp(
    numerator_rate: float,
    baseline_rate: float,
) -> float:
    if not np.isfinite(numerator_rate) or not np.isfinite(baseline_rate):
        return float("nan")
    return float(100.0 * (numerator_rate - baseline_rate))
def _prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if "snapshot_date" not in result.columns:
        raise ValueError("Momentum/SI analysis requires snapshot_date")
    result["snapshot_date"] = pd.to_datetime(
        result["snapshot_date"],
        errors="coerce",
    )
    result = result.loc[result["snapshot_date"].notna()].copy()
    result["price_momentum_5d"] = build_signal(
        result,
        "price_momentum_5d",
    )
    result["short_interest_change"] = build_signal(
        result,
        "short_interest_change",
    )
    result["momentum_decile"] = _cross_sectional_deciles(
        result,
        "price_momentum_5d",
    )
    result["si_decile"] = _cross_sectional_deciles(
        result,
        "short_interest_change",
    )
    return result
def _training_risk_bands(
    test_frame: pd.DataFrame,
    train_scores: pd.Series,
    test_scores: pd.Series,
) -> pd.DataFrame:
    """
    Assign event-risk regimes to the test set using only train-score
    quantiles.
    Higher event_score is treated as higher predicted event risk.
    """
    result = test_frame.copy()
    result["event_score"] = pd.to_numeric(
        test_scores,
        errors="coerce",
    )
    train_values = pd.to_numeric(
        train_scores,
        errors="coerce",
    ).dropna()
    if train_values.empty:
        result["event_risk_band"] = "missing"
        return result
    score = result["event_score"]
    cut_05 = float(train_values.quantile(0.95))
    cut_20 = float(train_values.quantile(0.80))
    cut_50 = float(train_values.quantile(0.50))
    result["event_risk_band"] = np.select(
        [
            score >= cut_05,
            (score >= cut_20) & (score < cut_05),
            (score >= cut_50) & (score < cut_20),
            score < cut_50,
        ],
        [
            "top_5pct",
            "5_20pct",
            "20_50pct",
            "bottom_50pct",
        ],
        default="missing",
    )
    result.attrs["event_risk_cutoffs"] = {
        "top_5pct": cut_05,
        "5_20pct": cut_20,
        "20_50pct": cut_50,
    }
    return result
def _cell_rows(
    frame: pd.DataFrame,
    *,
    focus_cells: tuple[tuple[int, int], ...],
    context_columns: tuple[str, ...],
    horizons: tuple[int, ...],
) -> list[dict]:
    rows: list[dict] = []
    for momentum_decile, si_decile in focus_cells:
        cell = frame.loc[
            (frame["momentum_decile"] == momentum_decile - 1)
            & (frame["si_decile"] == si_decile - 1)
        ].copy()
        row: dict = {
            "momentum_decile": momentum_decile,
            "si_decile": si_decile,
            "n": int(len(cell)),
            "event_n": _event_n(cell),
            "event_rate": _event_rate(cell),
            "momentum_mean": _numeric_mean(
                cell,
                "price_momentum_5d",
            ),
            "momentum_median": _numeric_median(
                cell,
                "price_momentum_5d",
            ),
            "si_change_mean": _numeric_mean(
                cell,
                "short_interest_change",
            ),
            "si_change_median": _numeric_median(
                cell,
                "short_interest_change",
            ),
        }
        for column in context_columns:
            row[f"{column}_mean"] = _numeric_mean(cell, column)
            row[f"{column}_median"] = _numeric_median(cell, column)
        for horizon in horizons:
            column = f"forward_return_{horizon}d"
            row[f"{column}_mean"] = _numeric_mean(cell, column)
            row[f"{column}_median"] = _numeric_median(cell, column)
        rows.append(row)
    return rows
def _momentum_control_rows(
    frame: pd.DataFrame,
    *,
    focus_cells: tuple[tuple[int, int], ...],
) -> list[dict]:
    """
    Compare each focal cell with the rest of the same momentum decile.
    The focal SI decile is excluded from the control population. This
    makes the comparison specifically about whether SI adds information
    within a fixed momentum decile.
    """
    rows: list[dict] = []
    for momentum_decile, si_decile in focus_cells:
        momentum_mask = frame["momentum_decile"] == momentum_decile - 1
        cell_mask = (
            momentum_mask
            & (frame["si_decile"] == si_decile - 1)
        )
        cell = frame.loc[cell_mask].copy()
        control = frame.loc[
            momentum_mask
            & (frame["si_decile"] != si_decile - 1)
        ].copy()
        cell_rate = _event_rate(cell)
        control_rate = _event_rate(control)
        rows.append(
            {
                "momentum_decile": momentum_decile,
                "si_decile": si_decile,
                "cell_n": _event_n(cell),
                "control_n": _event_n(control),
                "cell_event_rate": cell_rate,
                "same_momentum_other_si_event_rate": control_rate,
                "delta_vs_same_momentum_pp": _delta_pp(
                    cell_rate,
                    control_rate,
                ),
            }
        )
    return rows
def _risk_control_rows(
    frame: pd.DataFrame,
    *,
    focus_cells: tuple[tuple[int, int], ...],
) -> list[dict]:
    """
    Compare each focal cell against the same momentum decile AND
    the same event-risk regime, excluding the focal SI decile.
    This is the primary control for the question:
        Does SI add information after momentum and general event-risk
        have already been accounted for?
    """
    rows: list[dict] = []
    for momentum_decile, si_decile in focus_cells:
        momentum_mask = frame["momentum_decile"] == momentum_decile - 1
        cell_mask = (
            momentum_mask
            & (frame["si_decile"] == si_decile - 1)
        )
        cell_all = frame.loc[cell_mask].copy()
        for risk_band, _, _ in EVENT_RISK_BANDS:
            cell = cell_all.loc[
                cell_all["event_risk_band"] == risk_band
            ].copy()
            control = frame.loc[
                momentum_mask
                & (frame["si_decile"] != si_decile - 1)
                & (frame["event_risk_band"] == risk_band)
            ].copy()
            cell_rate = _event_rate(cell)
            control_rate = _event_rate(control)
            rows.append(
                {
                    "momentum_decile": momentum_decile,
                    "si_decile": si_decile,
                    "event_risk_band": risk_band,
                    "cell_n": _event_n(cell),
                    "control_n": _event_n(control),
                    "cell_event_rate": cell_rate,
                    "same_momentum_same_risk_other_si_event_rate": control_rate,
                    "risk_adjusted_delta_pp": _delta_pp(
                        cell_rate,
                        control_rate,
                    ),
                }
            )
    return rows
def _risk_summary_rows(
    frame: pd.DataFrame,
    *,
    focus_cells: tuple[tuple[int, int], ...],
) -> list[dict]:
    """
    Descriptive event-rate table for each focal cell across risk regimes.
    """
    rows: list[dict] = []
    for momentum_decile, si_decile in focus_cells:
        cell_all = frame.loc[
            (frame["momentum_decile"] == momentum_decile - 1)
            & (frame["si_decile"] == si_decile - 1)
        ].copy()
        for risk_band, _, _ in EVENT_RISK_BANDS:
            local = cell_all.loc[
                cell_all["event_risk_band"] == risk_band
            ].copy()
            rows.append(
                {
                    "momentum_decile": momentum_decile,
                    "si_decile": si_decile,
                    "event_risk_band": risk_band,
                    "n": _event_n(local),
                    "event_rate": _event_rate(local),
                }
            )
    return rows
def run_momentum_si_cell_context(
    context,
    *,
    focus_cells: tuple[tuple[int, int], ...],
    context_columns: tuple[str, ...],
    horizons: tuple[int, ...],
) -> ExperimentResult:
    """
    Test pre-defined momentum × SI cells against progressively tighter
    controls.
    Controls:
      1. overall test population
      2. same momentum decile, excluding focal SI decile
      3. same momentum decile + same event-risk regime,
         excluding focal SI decile
    Event:
      forward_return_5d <= -5%.
    """
    frame = _prepare_frame(context.test)
    required = {
        "forward_return_5d",
        "price_momentum_5d",
        "short_interest_change",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            "Momentum/SI analysis saknar kolumner: "
            + ", ".join(sorted(missing))
        )
    # Reuse the established event-risk model and walk-forward split.
    # The helper learns the model from pre-test data and returns
    # train/pre-test and test scores.
    pretest_risk, test_risk, model_name, event_features, validation_auc = (
        _prepare_event_risk_data(context)
    )
    train_scores = pd.to_numeric(
        pretest_risk["event_score"],
        errors="coerce",
    )
    test_scores = pd.to_numeric(
        test_risk["event_score"],
        errors="coerce",
    )
    # Align scores to the test rows using their original indices.
    test_frame = frame.copy()
    if len(test_scores) != len(test_frame):
        common_index = test_frame.index.intersection(test_scores.index)
        if len(common_index) == 0:
            raise ValueError(
                "Kunde inte aligna event-risk scores med momentum/SI-testdata."
            )
        test_frame = test_frame.loc[common_index].copy()
        test_scores = test_scores.loc[common_index]
    else:
        test_scores = pd.Series(
            test_scores.to_numpy(),
            index=test_frame.index,
        )
    test_frame = _training_risk_bands(
        test_frame,
        train_scores,
        test_scores,
    )
    cell_context = _cell_rows(
        test_frame,
        focus_cells=focus_cells,
        context_columns=context_columns,
        horizons=horizons,
    )
    momentum_control = _momentum_control_rows(
        test_frame,
        focus_cells=focus_cells,
    )
    risk_control = _risk_control_rows(
        test_frame,
        focus_cells=focus_cells,
    )
    risk_summary = _risk_summary_rows(
        test_frame,
        focus_cells=focus_cells,
    )
    overall_event_rate = _event_rate(test_frame)
    result = ExperimentResult(
        name="momentum_si_cell_context",
        description=(
            "Kontrollerar om de förut identifierade momentum × SI-cellerna "
            "har förhöjd 5d-risk för minst −5 %, först relativt totalpopulationen, "
            "sedan inom samma momentum-decile och slutligen inom samma "
            "momentum-decile och event-risk-regim."
        ),
    )
    result.add_metric(
        "event_definition",
        "forward_return_5d <= -0.05",
    )
    result.add_metric(
        "overall_test_event_rate",
        overall_event_rate,
    )
    result.add_metric(
        "event_model",
        model_name,
    )
    result.add_metric(
        "event_features",
        list(event_features),
    )
    result.add_metric(
        "validation_auc",
        validation_auc,
    )
    result.add_metric(
        "event_risk_bands",
        [name for name, _, _ in EVENT_RISK_BANDS],
    )
    result.add_metric(
        "risk_control_excludes_focal_si_decile",
        True,
    )
    result.add_metric(
        "test_rows",
        int(len(test_frame)),
    )
    result.add_metric(
        "focus_cell_count",
        int(len(focus_cells)),
    )
    result.add_table(
        "cell_context",
        pd.DataFrame(cell_context),
    )
    result.add_table(
        "momentum_control",
        pd.DataFrame(momentum_control),
    )
    result.add_table(
        "momentum_event_risk_control",
        pd.DataFrame(risk_control),
    )
    result.add_table(
        "cell_event_risk_profile",
        pd.DataFrame(risk_summary),
    )
    return result
