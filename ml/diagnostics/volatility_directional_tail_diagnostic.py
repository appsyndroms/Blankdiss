"""
Volatility directional/tail diagnostic.
The model is trained only on:
    down_5pct_5d
The resulting out-of-sample ranking is then evaluated against several
different future-return outcomes. This separates:
1. Directional downside signal
2. General large-move signal
3. Downside asymmetry
4. Volatility level vs volatility term structure
5. Dependence on recent price direction
6. Stability across calendar years
7. Concentration of the signal across volatility regimes
This is intentionally logistic-only so that the diagnostic tests the
information content of the features rather than model-family selection.
"""
from __future__ import annotations
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from ml.config import TARGETS
from ml.dataset import load_features
from ml.prices import find_price_files, load_prices
from ml.walk_forward import WalkForwardWindow, train_window
ROOT = Path(__file__).resolve().parents[2]
FEATURE_DIR = ROOT / "data" / "processed" / "analysis"
PRICE_DIR = ROOT / "data" / "processed" / "prices"
WINDOWS = (
    WalkForwardWindow(
        train_end="2023-12-31",
        validation_end="2024-12-31",
        test_end="2025-12-31",
    ),
    WalkForwardWindow(
        train_end="2024-12-31",
        validation_end="2025-12-31",
        test_end="2026-12-31",
    ),
)
TOP_FRACTIONS = (0.001, 0.005, 0.01, 0.02, 0.05)
FI_FEATURES = [
    "short_interest_pct",
    "active_holders",
    "max_individual_position_pct",
    "max_position_share_pct",
    "previous_short_interest_pct",
    "previous_active_holders",
    "previous_max_individual_position_pct",
    "previous_max_position_share_pct",
    "fi_observation_gap_days",
    "short_interest_delta_pp",
    "holder_delta",
    "max_position_delta_pp",
    "concentration_delta_pp",
    "short_interest_relative_change",
    "short_interest_acceleration_pp",
    "above_1_0pct",
    "entered_above_1_0pct",
    "exited_below_1_0pct",
    "above_2_0pct",
    "entered_above_2_0pct",
    "exited_below_2_0pct",
    "above_3_0pct",
    "entered_above_3_0pct",
    "exited_below_3_0pct",
    "above_5_0pct",
    "entered_above_5_0pct",
    "exited_below_5_0pct",
    "new_visible_observation",
]
def find_target_config(name: str):
    for target in TARGETS:
        if target.name == name:
            return target
    raise ValueError(f"Target not found: {name}")
def load_all_features() -> pd.DataFrame:
    files = sorted(FEATURE_DIR.glob("features_*.jsonl"))
    if not files:
        raise FileNotFoundError(
            f"No feature files found in {FEATURE_DIR}"
        )
    frames = []
    for path in files:
        frame = pd.read_json(path, lines=True)
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    if "date" in data.columns:
        data["date"] = pd.to_datetime(data["date"])
    return data
def normalise_price_data(prices: pd.DataFrame) -> pd.DataFrame:
    prices = prices.copy()
    date_candidates = ("date", "Date", "timestamp", "Timestamp")
    symbol_candidates = (
        "symbol",
        "ticker",
        "instrument",
        "yahoo_symbol",
    )
    close_candidates = (
        "close",
        "Close",
        "adj_close",
        "Adj Close",
    )
    date_col = next(
        (column for column in date_candidates if column in prices.columns),
        None,
    )
    symbol_col = next(
        (column for column in symbol_candidates if column in prices.columns),
        None,
    )
    close_col = next(
        (column for column in close_candidates if column in prices.columns),
        None,
    )
    if date_col is None:
        raise ValueError(
            f"Could not identify price date column. "
            f"Columns: {list(prices.columns)}"
        )
    if symbol_col is None:
        raise ValueError(
            f"Could not identify price symbol column. "
            f"Columns: {list(prices.columns)}"
        )
    if close_col is None:
        raise ValueError(
            f"Could not identify price close column. "
            f"Columns: {list(prices.columns)}"
        )
    prices = prices.rename(
        columns={
            date_col: "date",
            symbol_col: "symbol",
            close_col: "close",
        }
    )
    prices["date"] = pd.to_datetime(prices["date"])
    prices["close"] = pd.to_numeric(
        prices["close"],
        errors="coerce",
    )
    prices = prices.dropna(subset=["date", "symbol", "close"])
    prices = prices.sort_values(["symbol", "date"])
    return prices
def add_volatility_features(
    features: pd.DataFrame,
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add 20d/60d realised volatility and recent-return controls.
    VOL20:
        std of the previous 19 daily returns from 20 close observations.
    VOL60:
        std of the previous 59 daily returns from 60 close observations.
    The volatility values are therefore aligned to the feature date and
    contain no future information.
    """
    features = features.copy()
    prices = normalise_price_data(prices)
    price_features = []
    for symbol, group in prices.groupby("symbol", sort=False):
        group = group.sort_values("date").copy()
        returns = group["close"].pct_change()
        group["price_return_5d"] = group["close"].pct_change(5)
        group["price_return_20d"] = group["close"].pct_change(20)
        group["price_return_60d"] = group["close"].pct_change(60)
        group["price_volatility_20d"] = returns.rolling(
            19,
            min_periods=19,
        ).std()
        group["price_volatility_60d"] = returns.rolling(
            59,
            min_periods=59,
        ).std()
        group["volatility_20d_minus_60d"] = (
            group["price_volatility_20d"]
            - group["price_volatility_60d"]
        )
        group["volatility_20d_div_60d"] = (
            group["price_volatility_20d"]
            / group["price_volatility_60d"].replace(0, np.nan)
        )
        keep = group[
            [
                "symbol",
                "date",
                "price_return_5d",
                "price_return_20d",
                "price_return_60d",
                "price_volatility_20d",
                "price_volatility_60d",
                "volatility_20d_minus_60d",
                "volatility_20d_div_60d",
            ]
        ]
        price_features.append(keep)
    price_features = pd.concat(
        price_features,
        ignore_index=True,
    )
    features = features.merge(
        price_features,
        on=["symbol", "date"],
        how="left",
    )
    return features
def get_forward_return(
    prices: pd.DataFrame,
    horizon: int = 5,
) -> pd.DataFrame:
    """
    Create the actual forward return for every symbol/date.
    forward_return_5d is the close-to-close return from the current
    observation to the close five trading observations later.
    """
    prices = normalise_price_data(prices)
    frames = []
    for symbol, group in prices.groupby("symbol", sort=False):
        group = group.sort_values("date").copy()
        group["forward_return_5d"] = (
            group["close"].shift(-horizon)
            / group["close"]
            - 1.0
        )
        frames.append(
            group[
                [
                    "symbol",
                    "date",
                    "forward_return_5d",
                ]
            ]
        )
    return pd.concat(frames, ignore_index=True)
def add_analysis_targets(
    features: pd.DataFrame,
    prices: pd.DataFrame,
) -> pd.DataFrame:
    forward = get_forward_return(prices)
    data = features.merge(
        forward,
        on=["symbol", "date"],
        how="left",
    )
    r = data["forward_return_5d"]
    data["target_down_5pct_5d"] = r <= -0.05
    data["target_up_5pct_5d"] = r >= 0.05
    data["target_abs_5pct_5d"] = r.abs() >= 0.05
    data["target_abs_10pct_5d"] = r.abs() >= 0.10
    return data
def make_model() -> Pipeline:
    return Pipeline(
        [
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "logistic",
                LogisticRegression(
                    max_iter=2000,
                    random_state=42,
                ),
            ),
        ]
    )
def make_target_series(
    data: pd.DataFrame,
    target_name: str,
) -> pd.Series:
    target_config = find_target_config(target_name)
    if target_name == "down_5pct_5d":
        target = data["target_down_5pct_5d"].astype(int)
    else:
        raise ValueError(
            f"This diagnostic intentionally trains only on "
            f"{target_name}. Expected down_5pct_5d."
        )
    target = target.where(
        data[target_config.name].notna()
        if target_config.name in data.columns
        else True
    )
    return target
def get_forward_target_column(
    data: pd.DataFrame,
    target_name: str,
) -> pd.Series:
    mapping = {
        "down_5pct_5d": data["target_down_5pct_5d"],
        "up_5pct_5d": data["target_up_5pct_5d"],
        "abs_5pct_5d": data["target_abs_5pct_5d"],
        "abs_10pct_5d": data["target_abs_10pct_5d"],
    }
    return mapping[target_name].astype(float)
def get_feature_columns(
    data: pd.DataFrame,
    feature_set: str,
) -> list[str]:
    available = set(data.columns)
    if feature_set == "volatility_60d":
        columns = [
            "price_volatility_60d",
        ]
    elif feature_set == "volatility_20d_plus_60d":
        columns = [
            "price_volatility_20d",
            "price_volatility_60d",
        ]
    elif feature_set == "volatility_60d_plus_term_structure":
        columns = [
            "price_volatility_60d",
            "volatility_20d_minus_60d",
        ]
    elif feature_set == "volatility_20d_plus_60d_plus_term_structure":
        columns = [
            "price_volatility_20d",
            "price_volatility_60d",
            "volatility_20d_minus_60d",
        ]
    elif feature_set == "volatility_20d_plus_60d_plus_ratio":
        columns = [
            "price_volatility_20d",
            "price_volatility_60d",
            "volatility_20d_div_60d",
        ]
    elif feature_set == "volatility_plus_returns":
        columns = [
            "price_volatility_60d",
            "price_return_5d",
            "price_return_20d",
            "price_return_60d",
        ]
    elif feature_set == "volatility_20d_plus_60d_plus_returns":
        columns = [
            "price_volatility_20d",
            "price_volatility_60d",
            "price_return_5d",
            "price_return_20d",
            "price_return_60d",
        ]
    else:
        raise ValueError(f"Unknown feature set: {feature_set}")
    missing = [
        column
        for column in columns
        if column not in available
    ]
    if missing:
        raise ValueError(
            f"Feature set {feature_set} is missing columns: {missing}"
        )
    return columns
def build_xy(
    data: pd.DataFrame,
    feature_columns: list[str],
    target_name: str = "down_5pct_5d",
) -> tuple[pd.DataFrame, pd.Series]:
    x = data[feature_columns].copy()
    y = make_target_series(data, target_name)
    valid = (
        x.notna().all(axis=1)
        & y.notna()
        & data["forward_return_5d"].notna()
    )
    return (
        x.loc[valid].copy(),
        y.loc[valid].copy(),
    )
def fit_predict_walk_forward(
    data: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """
    Produce genuine OOS predictions using the existing walk-forward
    windows.
    Training target is always down_5pct_5d.
    """
    target_name = "down_5pct_5d"
    target_config = find_target_config(target_name)
    all_oos = []
    for window_number, window in enumerate(WINDOWS, start=1):
        train_mask = data["date"] <= pd.Timestamp(window.train_end)
        validation_mask = (
            (data["date"] > pd.Timestamp(window.train_end))
            & (data["date"] <= pd.Timestamp(window.validation_end))
        )
        test_mask = (
            (data["date"] > pd.Timestamp(window.validation_end))
            & (data["date"] <= pd.Timestamp(window.test_end))
        )
        window_data = data.loc[
            train_mask | validation_mask | test_mask
        ].copy()
        x, y = build_xy(
            window_data,
            feature_columns,
            target_name,
        )
        model = make_model()
        train_indices = x.index.intersection(
            data.index[train_mask]
        )
        validation_indices = x.index.intersection(
            data.index[validation_mask]
        )
        test_indices = x.index.intersection(
            data.index[test_mask]
        )
        if len(train_indices) == 0:
            continue
        model.fit(
            x.loc[train_indices],
            y.loc[train_indices],
        )
        prediction_indices = validation_indices.union(
            test_indices
        )
        if len(prediction_indices) == 0:
            continue
        scores = model.predict_proba(
            x.loc[prediction_indices]
        )[:, 1]
        result = data.loc[prediction_indices].copy()
        result["score"] = scores
        result["window"] = window_number
        result["trained_target"] = target_config.name
        result = result[
            [
                "symbol",
                "date",
                "score",
                "forward_return_5d",
                "target_down_5pct_5d",
                "target_up_5pct_5d",
                "target_abs_5pct_5d",
                "target_abs_10pct_5d",
                "price_volatility_20d",
                "price_volatility_60d",
                "volatility_20d_minus_60d",
                "volatility_20d_div_60d",
                "price_return_5d",
                "price_return_20d",
                "price_return_60d",
                "window",
            ]
        ]
        all_oos.append(result)
    if not all_oos:
        raise RuntimeError(
            "No OOS predictions were produced."
        )
    return pd.concat(
        all_oos,
        ignore_index=True,
    )
def safe_auc(
    score: pd.Series,
    target: pd.Series,
) -> float:
    valid = (
        score.notna()
        & target.notna()
    )
    score = score.loc[valid]
    target = target.loc[valid]
    if target.nunique() < 2:
        return float("nan")
    return float(
        roc_auc_score(
            target,
            score,
        )
    )
def describe_top_fraction(
    frame: pd.DataFrame,
    target_column: str,
    fraction: float,
) -> dict[str, float]:
    frame = frame.dropna(
        subset=["score", "forward_return_5d"]
    )
    if frame.empty:
        return {
            "event_rate": float("nan"),
            "lift": float("nan"),
            "mean_return": float("nan"),
            "median_return": float("nan"),
            "positive_return_fraction": float("nan"),
            "negative_return_fraction": float("nan"),
            "n": 0,
        }
    n = max(
        1,
        int(np.ceil(len(frame) * fraction)),
    )
    top = frame.nlargest(
        n,
        "score",
    )
    baseline_event = frame[target_column].mean()
    event_rate = top[target_column].mean()
    if baseline_event > 0:
        lift = event_rate / baseline_event
    else:
        lift = float("nan")
    returns = top["forward_return_5d"]
    return {
        "event_rate": float(event_rate),
        "lift": float(lift),
        "mean_return": float(returns.mean()),
        "median_return": float(returns.median()),
        "positive_return_fraction": float(
            (returns > 0).mean()
        ),
        "negative_return_fraction": float(
            (returns < 0).mean()
        ),
        "n": int(len(top)),
    }
def print_tail_analysis(
    oos: pd.DataFrame,
    label: str,
) -> dict[str, float]:
    print()
    print("=" * 100)
    print(label)
    print("=" * 100)
    target_columns = {
        "down_5pct_5d": "target_down_5pct_5d",
        "up_5pct_5d": "target_up_5pct_5d",
        "abs_5pct_5d": "target_abs_5pct_5d",
        "abs_10pct_5d": "target_abs_10pct_5d",
    }
    summary = {}
    for target_name, target_column in target_columns.items():
        auc = safe_auc(
            oos["score"],
            oos[target_column],
        )
        print(
            f"{target_name:20s} "
            f"AUC={auc:.6f} "
            f"baseline={oos[target_column].mean():.4f}"
        )
        summary[f"{target_name}_auc"] = auc
    print()
    print(
        "Top-tail analysis "
        "(ranking trained only on down_5pct_5d)"
    )
    for target_name, target_column in target_columns.items():
        print()
        print(f"Target: {target_name}")
        for fraction in TOP_FRACTIONS:
            result = describe_top_fraction(
                oos,
                target_column,
                fraction,
            )
            print(
                f"  top {fraction * 100:5.1f}% "
                f"event={result['event_rate']:.4f} "
                f"lift={result['lift']:.2f}x "
                f"mean={result['mean_return']:+.4%} "
                f"median={result['median_return']:+.4%} "
                f"pos={result['positive_return_fraction']:.4f} "
                f"neg={result['negative_return_fraction']:.4f} "
                f"n={result['n']}"
            )
    return summary
def print_direction_given_move(
    oos: pd.DataFrame,
) -> None:
    print()
    print("=" * 100)
    print("DIRECTION GIVEN LARGE MOVE")
    print("=" * 100)
    returns = oos["forward_return_5d"]
    for threshold in (0.05, 0.10):
        large = returns.abs() >= threshold
        if large.sum() == 0:
            continue
        large_returns = returns.loc[large]
        down_fraction = (
            large_returns <= -threshold
        ).mean()
        up_fraction = (
            large_returns >= threshold
        ).mean()
        print(
            f"|return| >= {threshold:.0%}: "
            f"n={len(large_returns)} "
            f"down={down_fraction:.4f} "
            f"up={up_fraction:.4f}"
        )
    print()
    print(
        "Same calculation restricted to the model's top 1%:"
    )
    top_n = max(
        1,
        int(np.ceil(len(oos) * 0.01)),
    )
    top = oos.nlargest(
        top_n,
        "score",
    )
    for threshold in (0.05, 0.10):
        large_returns = top.loc[
            top["forward_return_5d"].abs() >= threshold,
            "forward_return_5d",
        ]
        if large_returns.empty:
            continue
        print(
            f"top 1%, |return| >= {threshold:.0%}: "
            f"n={len(large_returns)} "
            f"down={(large_returns <= -threshold).mean():.4f} "
            f"up={(large_returns >= threshold).mean():.4f}"
        )
def print_return_buckets(
    oos: pd.DataFrame,
) -> None:
    print()
    print("=" * 100)
    print("ACTUAL 5-DAY RETURN DISTRIBUTION")
    print("=" * 100)
    buckets = [
        ("<= -10%", -np.inf, -0.10),
        ("-10% to -5%", -0.10, -0.05),
        ("-5% to 0%", -0.05, 0.0),
        ("0% to 5%", 0.0, 0.05),
        ("5% to 10%", 0.05, 0.10),
        (">= 10%", 0.10, np.inf),
    ]
    for fraction in TOP_FRACTIONS:
        n = max(
            1,
            int(np.ceil(len(oos) * fraction)),
        )
        top = oos.nlargest(
            n,
            "score",
        )
        print()
        print(
            f"Top {fraction * 100:.1f}% "
            f"(n={len(top)})"
        )
        for label, lower, upper in buckets:
            mask = (
                (top["forward_return_5d"] >= lower)
                & (top["forward_return_5d"] < upper)
            )
            print(
                f"  {label:12s} "
                f"{mask.mean():.4f}"
            )
def print_volatility_quintiles(
    oos: pd.DataFrame,
) -> None:
    print()
    print("=" * 100)
    print("VOLATILITY QUINTILES")
    print("=" * 100)
    for column in (
        "price_volatility_20d",
        "price_volatility_60d",
    ):
        frame = oos.dropna(
            subset=[
                column,
                "target_down_5pct_5d",
                "target_abs_5pct_5d",
            ]
        ).copy()
        if frame.empty:
            continue
        frame["quintile"] = pd.qcut(
            frame[column],
            5,
            labels=False,
            duplicates="drop",
        )
        print()
        print(column)
        grouped = frame.groupby(
            "quintile",
            observed=True,
        )
        for quintile, group in grouped:
            print(
                f"  Q{int(quintile) + 1}: "
                f"n={len(group):6d} "
                f"median_vol={group[column].median():.6f} "
                f"down={group['target_down_5pct_5d'].mean():.4f} "
                f"abs5={group['target_abs_5pct_5d'].mean():.4f}"
            )
def print_calendar_year_stability(
    oos: pd.DataFrame,
) -> None:
    print()
    print("=" * 100)
    print("CALENDAR-YEAR OOS STABILITY")
    print("=" * 100)
    frame = oos.copy()
    frame["year"] = frame["date"].dt.year
    for year, group in frame.groupby("year"):
        auc_down = safe_auc(
            group["score"],
            group["target_down_5pct_5d"],
        )
        auc_up = safe_auc(
            group["score"],
            group["target_up_5pct_5d"],
        )
        auc_abs5 = safe_auc(
            group["score"],
            group["target_abs_5pct_5d"],
        )
        print(
            f"{year}: "
            f"n={len(group):6d} "
            f"down_auc={auc_down:.6f} "
            f"up_auc={auc_up:.6f} "
            f"abs5_auc={auc_abs5:.6f} "
            f"down_rate={group['target_down_5pct_5d'].mean():.4f} "
            f"abs5_rate={group['target_abs_5pct_5d'].mean():.4f}"
        )
def print_score_deciles(
    oos: pd.DataFrame,
) -> None:
    print()
    print("=" * 100)
    print("SCORE DECILES")
    print("=" * 100)
    frame = oos.dropna(
        subset=[
            "score",
            "forward_return_5d",
        ]
    ).copy()
    frame["score_decile"] = pd.qcut(
        frame["score"],
        10,
        labels=False,
        duplicates="drop",
    )
    for decile, group in frame.groupby(
        "score_decile",
        observed=True,
    ):
        print(
            f"D{int(decile) + 1:02d}: "
            f"n={len(group):6d} "
            f"score_med={group['score'].median():.5f} "
            f"down={group['target_down_5pct_5d'].mean():.4f} "
            f"up={group['target_up_5pct_5d'].mean():.4f} "
            f"abs5={group['target_abs_5pct_5d'].mean():.4f} "
            f"abs10={group['target_abs_10pct_5d'].mean():.4f} "
            f"mean_ret={group['forward_return_5d'].mean():+.4%} "
            f"median_ret={group['forward_return_5d'].median():+.4%}"
        )
def print_feature_set_comparison(
    data: pd.DataFrame,
) -> None:
    print()
    print("=" * 100)
    print("FEATURE-SET COMPARISON")
    print("=" * 100)
    feature_sets = (
        "volatility_60d",
        "volatility_20d_plus_60d",
        "volatility_60d_plus_term_structure",
        "volatility_20d_plus_60d_plus_term_structure",
        "volatility_20d_plus_60d_plus_ratio",
        "volatility_plus_returns",
        "volatility_20d_plus_60d_plus_returns",
    )
    results = []
    for feature_set in feature_sets:
        started = time.perf_counter()
        columns = get_feature_columns(
            data,
            feature_set,
        )
        oos = fit_predict_walk_forward(
            data,
            columns,
        )
        auc_down = safe_auc(
            oos["score"],
            oos["target_down_5pct_5d"],
        )
        auc_up = safe_auc(
            oos["score"],
            oos["target_up_5pct_5d"],
        )
        auc_abs5 = safe_auc(
            oos["score"],
            oos["target_abs_5pct_5d"],
        )
        auc_abs10 = safe_auc(
            oos["score"],
            oos["target_abs_10pct_5d"],
        )
        top = describe_top_fraction(
            oos,
            "target_down_5pct_5d",
            0.01,
        )
        elapsed = time.perf_counter() - started
        results.append(
            {
                "feature_set": feature_set,
                "features": len(columns),
                "down_auc": auc_down,
                "up_auc": auc_up,
                "abs5_auc": auc_abs5,
                "abs10_auc": auc_abs10,
                "top1_down": top["event_rate"],
                "top1_lift": top["lift"],
                "top1_mean_return": top["mean_return"],
                "seconds": elapsed,
            }
        )
    comparison = pd.DataFrame(results)
    print(
        comparison.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )
def print_top1_by_window(
    oos: pd.DataFrame,
) -> None:
    print()
    print("=" * 100)
    print("TOP 1% BY WALK-FORWARD WINDOW")
    print("=" * 100)
    for window, group in oos.groupby("window"):
        top = describe_top_fraction(
            group,
            "target_down_5pct_5d",
            0.01,
        )
        auc_down = safe_auc(
            group["score"],
            group["target_down_5pct_5d"],
        )
        auc_up = safe_auc(
            group["score"],
            group["target_up_5pct_5d"],
        )
        auc_abs5 = safe_auc(
            group["score"],
            group["target_abs_5pct_5d"],
        )
        print(
            f"window {window}: "
            f"n={len(group):6d} "
            f"down_auc={auc_down:.6f} "
            f"up_auc={auc_up:.6f} "
            f"abs5_auc={auc_abs5:.6f} "
            f"top1_down={top['event_rate']:.4f} "
            f"lift={top['lift']:.2f}x "
            f"mean={top['mean_return']:+.4%} "
            f"median={top['median_return']:+.4%}"
        )
def print_top1_volatility_profile(
    oos: pd.DataFrame,
) -> None:
    print()
    print("=" * 100)
    print("TOP 1% VOLATILITY PROFILE")
    print("=" * 100)
    top = oos.nlargest(
        max(1, int(np.ceil(len(oos) * 0.01))),
        "score",
    )
    all_frame = oos.copy()
    for column in (
        "price_volatility_20d",
        "price_volatility_60d",
        "volatility_20d_minus_60d",
        "volatility_20d_div_60d",
        "price_return_5d",
        "price_return_20d",
        "price_return_60d",
    ):
        all_values = all_frame[column].dropna()
        top_values = top[column].dropna()
        if all_values.empty or top_values.empty:
            continue
        print(
            f"{column:32s} "
            f"all_median={all_values.median():+.6f} "
            f"top1_median={top_values.median():+.6f} "
            f"all_mean={all_values.mean():+.6f} "
            f"top1_mean={top_values.mean():+.6f}"
        )
def run_feature_set(
    data: pd.DataFrame,
    feature_set: str,
) -> pd.DataFrame:
    columns = get_feature_columns(
        data,
        feature_set,
    )
    print()
    print("#" * 100)
    print(
        f"RUNNING FEATURE SET: {feature_set}"
    )
    print(
        f"Features ({len(columns)}): {columns}"
    )
    print("#" * 100)
    started = time.perf_counter()
    oos = fit_predict_walk_forward(
        data,
        columns,
    )
    elapsed = time.perf_counter() - started
    print(
        f"OOS rows: {len(oos):,}"
    )
    print(
        f"Runtime: {elapsed:.2f}s"
    )
    print_tail_analysis(
        oos,
        feature_set,
    )
    print_direction_given_move(
        oos,
    )
    print_return_buckets(
        oos,
    )
    print_volatility_quintiles(
        oos,
    )
    print_calendar_year_stability(
        oos,
    )
    print_score_deciles(
        oos,
    )
    print_top1_by_window(
        oos,
    )
    print_top1_volatility_profile(
        oos,
    )
    return oos
def main() -> None:
    print("=" * 100)
    print("VOLATILITY DIRECTIONAL / TAIL DIAGNOSTIC")
    print("=" * 100)
    started = time.perf_counter()
    features = load_features(FEATURE_DIR)
    if not isinstance(features, pd.DataFrame):
        raise TypeError(
            "load_features() did not return a DataFrame."
        )
    print(
        f"Loaded feature rows: {len(features):,}"
    )
    prices = load_prices(
        find_price_files(PRICE_DIR)
    )
    print(
        f"Loaded price rows: {len(prices):,}"
    )
    data = add_volatility_features(
        features,
        prices,
    )
    data = add_analysis_targets(
        data,
        prices,
    )
    print(
        f"Rows with 60d volatility: "
        f"{data['price_volatility_60d'].notna().sum():,}"
    )
    print(
        f"Rows without 60d volatility: "
        f"{data['price_volatility_60d'].isna().sum():,}"
    )
    print(
        f"Target: down_5pct_5d"
    )
    print()
    print("Volatility statistics:")
    for column in (
        "price_volatility_20d",
        "price_volatility_60d",
        "volatility_20d_minus_60d",
        "volatility_20d_div_60d",
    ):
        series = data[column].dropna()
        if series.empty:
            continue
        print(
            f"  {column:32s} "
            f"median={series.median():.6f} "
            f"p20={series.quantile(.20):.6f} "
            f"p80={series.quantile(.80):.6f}"
        )
    feature_sets = (
        "volatility_60d",
        "volatility_20d_plus_60d",
    )
    outputs = {}
    for feature_set in feature_sets:
        outputs[feature_set] = run_feature_set(
            data,
            feature_set,
        )
    print_feature_set_comparison(
        data,
    )
    print()
    print("=" * 100)
    print("FINAL SUMMARY")
    print("=" * 100)
    for feature_set, oos in outputs.items():
        down_auc = safe_auc(
            oos["score"],
            oos["target_down_5pct_5d"],
        )
        up_auc = safe_auc(
            oos["score"],
            oos["target_up_5pct_5d"],
        )
        abs5_auc = safe_auc(
            oos["score"],
            oos["target_abs_5pct_5d"],
        )
        abs10_auc = safe_auc(
            oos["score"],
            oos["target_abs_10pct_5d"],
        )
        top = describe_top_fraction(
            oos,
            "target_down_5pct_5d",
            0.01,
        )
        print(
            f"{feature_set:42s} "
            f"down={down_auc:.6f} "
            f"up={up_auc:.6f} "
            f"abs5={abs5_auc:.6f} "
            f"abs10={abs10_auc:.6f} "
            f"top1_down={top['event_rate']:.4f} "
            f"lift={top['lift']:.2f}x "
            f"mean={top['mean_return']:+.4%}"
        )
    elapsed = time.perf_counter() - started
    print()
    print(
        f"Total runtime: {elapsed:.2f}s"
    )
if __name__ == "__main__":
    main()
