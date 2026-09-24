from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from ml.config import WALK_FORWARD_WINDOWS
from ml.dataset import load_features
from ml.research.cache import _build_signal_rank
from ml.research.signals import build_signal
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (
    ROOT
    / "data"
    / "processed"
    / "ml"
    / "research"
    / "diagnostics"
    / "momentum_si_mask.csv"
)
SI_SIGNAL = "short_interest_change"
MOMENTUM_SIGNAL = "price_momentum_20d"
def _build_tail_mask(
    rank: np.ndarray,
    fraction: float,
) -> np.ndarray:
    """
    Bygger exakt samma upper-tail-mask som research engine.
    Engine-semantik:
        rank >= (1.0 - fraction)
    NaN-ranks blir False.
    """
    return (
        np.isfinite(rank)
        & (rank >= (1.0 - fraction))
    )
def _build_test_window_mask(
    frame: pd.DataFrame,
    window_name: str,
) -> np.ndarray:
    """
    Bygger exakt samma test-window-mask som ResearchCache.
    window_1:
        snapshot_date > validation_end
        snapshot_date <= test_end
    window_2:
        samma princip för nästa walk-forward-fönster.
    """
    if not window_name.startswith("window_"):
        raise ValueError(
            f"Ogiltigt window-namn: {window_name}"
        )
    try:
        index = int(
            window_name.split("_", 1)[1]
        ) - 1
    except (ValueError, IndexError) as exc:
        raise ValueError(
            f"Ogiltigt window-namn: {window_name}"
        ) from exc
    if index < 0 or index >= len(
        WALK_FORWARD_WINDOWS
    ):
        raise ValueError(
            f"Window finns inte: {window_name}"
        )
    window = WALK_FORWARD_WINDOWS[index]
    dates = pd.to_datetime(
        frame["snapshot_date"],
        errors="coerce",
    )
    validation_end = pd.Timestamp(
        window.validation_end
    )
    test_end = pd.Timestamp(
        window.test_end
    )
    return (
        (dates > validation_end)
        & (dates <= test_end)
    ).to_numpy()
def _mask_counts(
    *,
    si10: np.ndarray,
    si20: np.ndarray,
    mom20: np.ndarray,
    window_mask: np.ndarray,
) -> dict[str, int]:
    """
    Räknar maskrelationerna inom exakt samma
    window/test-population som research engine.
    """
    selected_si10 = (
        window_mask
        & si10
    )
    selected_si20 = (
        window_mask
        & si20
    )
    selected_mom20 = (
        window_mask
        & mom20
    )
    return {
        "n": int(
            window_mask.sum()
        ),
        "si10": int(
            selected_si10.sum()
        ),
        "si20": int(
            selected_si20.sum()
        ),
        "mom20": int(
            selected_mom20.sum()
        ),
        "si20_and_mom20": int(
            (
                selected_si20
                & selected_mom20
            ).sum()
        ),
        "si10_and_mom20": int(
            (
                selected_si10
                & selected_mom20
            ).sum()
        ),
        "si20_not_si10": int(
            (
                selected_si20
                & ~selected_si10
            ).sum()
        ),
        "mom20_not_si10": int(
            (
                selected_mom20
                & ~selected_si10
            ).sum()
        ),
    }
def diagnose_si_momentum_masks(
    frame: pd.DataFrame,
    *,
    si10_fraction: float = 0.10,
    si20_fraction: float = 0.20,
    momentum20_fraction: float = 0.20,
    output_path: str | Path | None = DEFAULT_OUTPUT,
) -> pd.DataFrame:
    """
    Diagnostik av SI- och momentum-tailarnas faktiska medlemskap.
    Viktigt:
        Denna diagnostik använder samma:
        - signalbyggare
        - cross-sectional ranking
        - rank-metod
        - tail-mask-logik
        - walk-forward test-window
        som den nya research-engine.
    Kontrollerar separat för varje testfönster:
        SI20 ∩ MOM20
        SI10 ∩ MOM20
        SI20 \\ SI10
        MOM20 \\ SI10
    Kärnfrågan:
        Är MOM20 ⊆ SI10?
    Om:
        MOM20 \\ SI10 = 0
    inom ett testfönster betyder det att alla
    MOM20-observationer också ligger i SI10.
    Då blir:
        SI20 ∩ MOM20
        SI10 ∩ MOM20
    exakt samma mängd, oavsett att SI20 och SI10
    själva kan vara olika masker.
    """
    if "snapshot_date" not in frame.columns:
        raise ValueError(
            "Diagnostiken kräver kolumnen snapshot_date."
        )
    print(
        f"Building signal: {SI_SIGNAL}",
        flush=True,
    )
    si_series = build_signal(
        frame,
        SI_SIGNAL,
    )
    print(
        f"Ranking signal: {SI_SIGNAL}",
        flush=True,
    )
    si_rank = _build_signal_rank(
        frame,
        si_series,
    )
    print(
        f"Building signal: {MOMENTUM_SIGNAL}",
        flush=True,
    )
    momentum_series = build_signal(
        frame,
        MOMENTUM_SIGNAL,
    )
    print(
        f"Ranking signal: {MOMENTUM_SIGNAL}",
        flush=True,
    )
    momentum_rank = _build_signal_rank(
        frame,
        momentum_series,
    )
    si10 = _build_tail_mask(
        si_rank,
        si10_fraction,
    )
    si20 = _build_tail_mask(
        si_rank,
        si20_fraction,
    )
    mom20 = _build_tail_mask(
        momentum_rank,
        momentum20_fraction,
    )
    print()
    print(
        "=== SI / MOM MASK DIAGNOSTIC ==="
    )
    print()
    rows: list[dict[str, object]] = []
    for index, _window in enumerate(
        WALK_FORWARD_WINDOWS,
        start=1,
    ):
        window_name = (
            f"window_{index}"
        )
        window_mask = (
            _build_test_window_mask(
                frame,
                window_name,
            )
        )
        counts = _mask_counts(
            si10=si10,
            si20=si20,
            mom20=mom20,
            window_mask=window_mask,
        )
        rows.append(
            {
                "window": window_name,
                **counts,
            }
        )
        print(
            f"=== {window_name} / test ==="
        )
        print()
        print(
            f"Rows              : "
            f"{counts['n']:,}"
        )
        print(
            f"SI10              : "
            f"{counts['si10']:,}"
        )
        print(
            f"SI20              : "
            f"{counts['si20']:,}"
        )
        print(
            f"MOM20             : "
            f"{counts['mom20']:,}"
        )
        print(
            f"SI20 ∩ MOM20      : "
            f"{counts['si20_and_mom20']:,}"
        )
        print(
            f"SI10 ∩ MOM20      : "
            f"{counts['si10_and_mom20']:,}"
        )
        print(
            f"SI20 \\ SI10       : "
            f"{counts['si20_not_si10']:,}"
        )
        print(
            f"MOM20 \\ SI10      : "
            f"{counts['mom20_not_si10']:,}"
        )
        print()
        print("Kärnfråga:")
        if counts["mom20_not_si10"] == 0:
            print(
                "  MOM20 ⊆ SI10"
            )
            print(
                "  Alla MOM20-observationer "
                "ligger inom SI10."
            )
        else:
            print(
                "  MOM20 är INTE en delmängd "
                "av SI10."
            )
            print(
                f"  {counts['mom20_not_si10']:,} "
                "MOM20-observationer ligger "
                "utanför SI10."
            )
        print()
        if (
            counts["si20_and_mom20"]
            == counts["si10_and_mom20"]
        ):
            print(
                "  SI20×MOM20 och SI10×MOM20 "
                "har samma N."
            )
        else:
            print(
                "  SI20×MOM20 och SI10×MOM20 "
                "har olika N."
            )
        print()
    result = pd.DataFrame(rows)
    print(
        "=== JÄMFÖRELSE MOT OBSERVERADE N ==="
    )
    print()
    observed = {
        152: "SI20×MOM20 / SI10×MOM20",
        98: "SI20×MOM20 / SI10×MOM20",
        65: "SI20×MOM20 / SI10×MOM20",
    }
    for window_name, row in result.set_index(
        "window"
    ).iterrows():
        pairs = [
            (
                "SI20×MOM20",
                int(row["si20_and_mom20"]),
            ),
            (
                "SI10×MOM20",
                int(row["si10_and_mom20"]),
            ),
        ]
        print(
            f"{window_name}: "
            + ", ".join(
                f"{name}={value:,}"
                for name, value in pairs
            )
        )
        matched = [
            str(value)
            for _, value in pairs
            if value in observed
        ]
        if matched:
            print(
                "  Matchar observerat N: "
                + ", ".join(matched)
            )
        print()
    print(
        "=== TOTAL KÄRNKONTROLL ==="
    )
    print()
    if result.empty:
        print(
            "Inga testfönster hittades."
        )
    else:
        all_nested = bool(
            (
                result[
                    "mom20_not_si10"
                ]
                == 0
            ).all()
        )
        all_equal = bool(
            (
                result[
                    "si20_and_mom20"
                ]
                == result[
                    "si10_and_mom20"
                ]
            ).all()
        )
        if all_nested:
            print(
                "MOM20 ⊆ SI10 i samtliga "
                "testfönster."
            )
        else:
            print(
                "MOM20 ⊆ SI10 gäller inte "
                "i samtliga testfönster."
            )
        if all_equal:
            print(
                "SI20×MOM20 och SI10×MOM20 "
                "har samma N i samtliga "
                "testfönster."
            )
        else:
            print(
                "SI20×MOM20 och SI10×MOM20 "
                "skiljer sig i minst ett "
                "testfönster."
            )
    if output_path is not None:
        output_path = Path(
            output_path
        )
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        result.to_csv(
            output_path,
            index=False,
        )
        print()
        print(
            f"Diagnostic result: "
            f"{output_path}"
        )
    return result
def main() -> None:
    print(
        "Loading features...",
        flush=True,
    )
    frame = load_features()
    print(
        f"Loaded {len(frame):,} feature rows",
        flush=True,
    )
    diagnose_si_momentum_masks(
        frame
    )
if __name__ == "__main__":
    main()
