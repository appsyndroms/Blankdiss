from __future__ import annotations
import pandas as pd
def diagnose_si_momentum_masks(
    frame: pd.DataFrame,
    si_fraction: float = 0.10,
    momentum_fraction: float = 0.20,
) -> None:
    """
    Diagnostik av medlemskap i SI- och momentum-tailarna.
    Besvarar:
      - SI20 ∩ MOM20
      - SI10 ∩ MOM20
      - SI20 \\ SI10
      - MOM20 \\ SI10
    Körs per snapshot_date så att vi ser om relationen gäller
    generellt eller bara i aggregeringen över hela testperioden.
    """
    required_columns = {
        "snapshot_date",
        "short_interest_change",
        "price_momentum_20d",
    }
    missing = required_columns - set(frame.columns)
    if missing:
        raise ValueError(
            "Saknade kolumner: "
            + ", ".join(sorted(missing))
        )
    df = frame[
        [
            "snapshot_date",
            "short_interest_change",
            "price_momentum_20d",
        ]
    ].copy()
    # ---------------------------------------------------------
    # Cross-sectional percentile rank per snapshot_date
    # ---------------------------------------------------------
    df["si_rank"] = (
        pd.to_numeric(
            df["short_interest_change"],
            errors="coerce",
        )
        .groupby(df["snapshot_date"])
        .rank(
            method="first",
            pct=True,
        )
    )
    df["momentum_rank"] = (
        pd.to_numeric(
            df["price_momentum_20d"],
            errors="coerce",
        )
        .groupby(df["snapshot_date"])
        .rank(
            method="first",
            pct=True,
        )
    )
    df = df.dropna(
        subset=[
            "si_rank",
            "momentum_rank",
        ]
    )
    # ---------------------------------------------------------
    # Tail masks
    # ---------------------------------------------------------
    df["si10"] = (
        df["si_rank"] >= 1.0 - si_fraction
    )
    df["si20"] = (
        df["si_rank"] >= 0.80
    )
    df["mom20"] = (
        df["momentum_rank"] >= 0.80
    )
    # ---------------------------------------------------------
    # Set relations
    # ---------------------------------------------------------
    si20_and_mom20 = (
        df["si20"] & df["mom20"]
    )
    si10_and_mom20 = (
        df["si10"] & df["mom20"]
    )
    si20_not_si10 = (
        df["si20"] & ~df["si10"]
    )
    mom20_not_si10 = (
        df["mom20"] & ~df["si10"]
    )
    # ---------------------------------------------------------
    # Total
    # ---------------------------------------------------------
    print("\n=== TOTALT ===")
    print(
        f"SI20 ∩ MOM20 : "
        f"{si20_and_mom20.sum()}"
    )
    print(
        f"SI10 ∩ MOM20 : "
        f"{si10_and_mom20.sum()}"
    )
    print(
        f"SI20 \\ SI10  : "
        f"{si20_not_si10.sum()}"
    )
    print(
        f"MOM20 \\ SI10 : "
        f"{mom20_not_si10.sum()}"
    )
    # ---------------------------------------------------------
    # Per snapshot_date
    # ---------------------------------------------------------
    print("\n=== PER SNAPSHOT_DATE ===")
    rows = []
    for date, group in df.groupby(
        "snapshot_date",
        sort=True,
    ):
        rows.append(
            {
                "snapshot_date": date,
                "n": len(group),
                "si20": int(
                    group["si20"].sum()
                ),
                "si10": int(
                    group["si10"].sum()
                ),
                "mom20": int(
                    group["mom20"].sum()
                ),
                "si20_and_mom20": int(
                    (
                        group["si20"]
                        & group["mom20"]
                    ).sum()
                ),
                "si10_and_mom20": int(
                    (
                        group["si10"]
                        & group["mom20"]
                    ).sum()
                ),
                "si20_not_si10": int(
                    (
                        group["si20"]
                        & ~group["si10"]
                    ).sum()
                ),
                "mom20_not_si10": int(
                    (
                        group["mom20"]
                        & ~group["si10"]
                    ).sum()
                ),
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        print("Ingen giltig data efter rankning.")
        return
    print(
        result.to_string(
            index=False
        )
    )
    # ---------------------------------------------------------
    # Kärnfrågan
    # ---------------------------------------------------------
    print("\n=== KÄRNFRÅGAN ===")
    outside = int(
        result["mom20_not_si10"].sum()
    )
    if outside == 0:
        print(
            "MOM20 ⊆ SI10 för hela perioden."
        )
        print(
            "Det förklarar direkt varför "
            "SI20×MOM20 och SI10×MOM20 "
            "får samma N."
        )
    else:
        print(
            f"MOM20 innehåller {outside} "
            "observationer utanför SI10."
        )
        print(
            "Då kan lika N för "
            "SI20×MOM20 och SI10×MOM20 "
            "inte förklaras av fullständig "
            "inbäddning."
        )
