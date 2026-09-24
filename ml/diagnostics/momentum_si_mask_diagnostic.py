from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
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
    Bygger samma upper-tail-mask som research engine/cache.
    Engine-semantik:
        rank >= (1.0 - fraction)
    NaN-ranks blir False.
    """
    return (
        np.isfinite(rank)
        & (rank >= (1.0 - fraction))
    )
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
        Denna diagnostik använder exakt samma signalbyggare och
        cross-sectional ranking som den nya research-cachen.
    Kontrollerar:
        SI20 ∩ MOM20
        SI10 ∩ MOM20
        SI20 \\ SI10
        MOM20 \\ SI10
    Den viktigaste frågan är:
        Är MOM20 ⊆ SI10?
    Om ja för testdata förklarar det varför:
        SI20 × MOM20
        SI10 × MOM20
    kan ge exakt samma N även om SI20 och SI10 själva
    innehåller olika observationer.
    """
    if "snapshot_date" not in frame.columns:
        raise ValueError(
            "Diagnostiken kräver kolumnen snapshot_date."
        )
    # ---------------------------------------------------------
    # Build exakt samma signaler som research engine använder.
    # ---------------------------------------------------------
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
    # ---------------------------------------------------------
    # Bygg exakt samma tail-masker som cache.py.
    # ---------------------------------------------------------
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
    # ---------------------------------------------------------
    # Set relations.
    # ---------------------------------------------------------
    si20_and_mom20 = (
        si20 & mom20
    )
    si10_and_mom20 = (
        si10 & mom20
    )
    si20_not_si10 = (
        si20 & ~si10
    )
    mom20_not_si10 = (
        mom20 & ~si10
    )
    # ---------------------------------------------------------
    # Total.
    # ---------------------------------------------------------
    print()
    print("=== SI / MOM MASK DIAGNOSTIC ===")
    print()
    print("=== TOTALT ===")
    print(
        f"Rows              : {len(frame):,}"
    )
    print(
        f"SI10              : {int(si10.sum()):,}"
    )
    print(
        f"SI20              : {int(si20.sum()):,}"
    )
    print(
        f"MOM20             : {int(mom20.sum()):,}"
    )
    print(
        f"SI20 ∩ MOM20      : "
        f"{int(si20_and_mom20.sum()):,}"
    )
    print(
        f"SI10 ∩ MOM20      : "
        f"{int(si10_and_mom20.sum()):,}"
    )
    print(
        f"SI20 \\ SI10       : "
        f"{int(si20_not_si10.sum()):,}"
    )
    print(
        f"MOM20 \\ SI10      : "
        f"{int(mom20_not_si10.sum()):,}"
    )
    # ---------------------------------------------------------
    # Per snapshot_date.
    # ---------------------------------------------------------
    working = pd.DataFrame(
        {
            "snapshot_date": frame[
                "snapshot_date"
            ],
            "si10": si10,
            "si20": si20,
            "mom20": mom20,
        },
        index=frame.index,
    )
    rows: list[dict[str, object]] = []
    for date, group in working.groupby(
        "snapshot_date",
        sort=True,
    ):
        group_si10 = group["si10"]
        group_si20 = group["si20"]
        group_mom20 = group["mom20"]
        rows.append(
            {
                "snapshot_date": date,
                "n": len(group),
                "si10": int(
                    group_si10.sum()
                ),
                "si20": int(
                    group_si20.sum()
                ),
                "mom20": int(
                    group_mom20.sum()
                ),
                "si20_and_mom20": int(
                    (
                        group_si20
                        & group_mom20
                    ).sum()
                ),
                "si10_and_mom20": int(
                    (
                        group_si10
                        & group_mom20
                    ).sum()
                ),
                "si20_not_si10": int(
                    (
                        group_si20
                        & ~group_si10
                    ).sum()
                ),
                "mom20_not_si10": int(
                    (
                        group_mom20
                        & ~group_si10
                    ).sum()
                ),
            }
        )
    result = pd.DataFrame(rows)
    print()
    print("=== PER SNAPSHOT_DATE ===")
    print()
    if result.empty:
        print(
            "Ingen snapshots hittades."
        )
        return result
    print(
        result.to_string(
            index=False
        )
    )
    # ---------------------------------------------------------
    # Kärnfrågan.
    # ---------------------------------------------------------
    outside = int(
        mom20_not_si10.sum()
    )
    print()
    print("=== KÄRNFRÅGAN ===")
    print()
    if outside == 0:
        print(
            "MOM20 ⊆ SI10 för hela datasetet."
        )
        print(
            "Detta förklarar direkt varför "
            "SI20×MOM20 och SI10×MOM20 "
            "kan få samma N."
        )
    else:
        print(
            f"MOM20 har {outside:,} "
            "observationer utanför SI10."
        )
        print(
            "MOM20 är alltså inte en fullständig "
            "delmängd av SI10."
        )
        print(
            "Lika interaktions-N måste då "
            "förklaras på annat sätt."
        )
    # ---------------------------------------------------------
    # Extra kontroll:
    #
    # Om SI10 ⊆ SI20 ska SI20 \\ SI10 vara positivt.
    # Detta bekräftar att SI10 och SI20 faktiskt skiljer sig
    # trots att deras interaktion med MOM20 kan vara identisk.
    # ---------------------------------------------------------
    print()
    print("=== MASKRELATION ===")
    print()
    si20_outside_si10 = int(
        si20_not_si10.sum()
    )
    if si20_outside_si10 > 0:
        print(
            "SI20 och SI10 är inte identiska masker."
        )
        print(
            f"SI20 innehåller "
            f"{si20_outside_si10:,} "
            "observationer som inte finns i SI10."
        )
    else:
        print(
            "SI20 \\ SI10 = 0."
        )
        print(
            "SI10 och SI20 är därför identiska "
            "för de giltiga observationerna."
        )
    # ---------------------------------------------------------
    # Spara.
    # ---------------------------------------------------------
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
            f"Diagnostic result: {output_path}"
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
