"""Rekonstruerar synlig aggregerad blankning från FI:s individhistorik.

VIKTIGT:
FI:s officiella aggregat omfattar rapporterade positioner över 0,1 %.
FI:s historiska positionsfil omfattar publicerade individuella
positioner och innehåller även markeringar när positioner fallit
under 0,5 %.

Resultatet från denna modul är därför en REKONSTRUERAD serie över de
synliga individuella positionerna. Den får inte beskrivas som FI:s
officiella aggregat.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT
    / "data"
    / "raw"
    / "fi"
    / "positions"
    / "historical"
    / "fi_historical_positions.jsonl"
)

OUTPUT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "fi"
    / "aggregate"
)

OUTPUT = OUTPUT_DIR / "reconstructed.jsonl"

METADATA = (
    OUTPUT_DIR
    / "reconstructed_metadata.json"
)

START_DATE = pd.Timestamp("2022-05-25")


def parse_position(value: object) -> float:
    """Tolkar en normaliserad FI-position."""

    if value is None:
        return 0.0

    if pd.isna(value):
        return 0.0

    text = str(value).strip().replace(
        ",",
        ".",
    )

    if not text or text.lower() == "nan":
        return 0.0

    # En position som markerats som <0,5 i FI:s historik
    # normaliseras till None i historical_positions.py.
    # Här betyder None därför att den tidigare synliga
    # positionen ska avslutas.
    if text.startswith("<"):
        return 0.0

    try:
        return float(
            text.replace(
                "%",
                "",
            ).strip()
        )

    except ValueError:
        return 0.0


def load_history() -> pd.DataFrame:
    """Läser den normaliserade FI-historiken."""

    if not INPUT.exists():
        raise FileNotFoundError(
            f"Saknar FI-historik: {INPUT}"
        )

    frame = pd.read_json(
        INPUT,
        lines=True,
    )

    required = {
        "holder",
        "issuer",
        "position",
        "position_date",
        "isin",
    }

    missing = required.difference(
        frame.columns
    )

    if missing:
        raise ValueError(
            "Saknade kolumner: "
            + ", ".join(
                sorted(missing)
            )
            + ". Tillgängliga kolumner: "
            + ", ".join(
                map(
                    str,
                    frame.columns,
                )
            )
        )

    frame = frame.rename(
        columns={
            "position": "position_pct",
        }
    )

    frame["position_date"] = pd.to_datetime(
        frame["position_date"],
        errors="coerce",
    )

    frame["position_pct"] = frame[
        "position_pct"
    ].map(
        parse_position
    )

    for column in (
        "holder",
        "issuer",
        "isin",
    ):
        frame[column] = (
            frame[column]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    frame = frame.loc[
        frame["position_date"].notna()
        & (frame["holder"] != "")
        & (frame["issuer"] != "")
    ].copy()

    # ISIN är primär identitet.
    # Äldre poster utan ISIN använder emittentnamnet.
    frame["issuer_key"] = frame["isin"].where(
        frame["isin"] != "",
        frame["issuer"],
    )

    frame = frame.sort_values(
        [
            "position_date",
            "holder",
            "issuer_key",
        ],
        kind="mergesort",
    )

    # En state per innehavare/emittent/dag.
    frame = frame.drop_duplicates(
        subset=[
            "holder",
            "issuer_key",
            "position_date",
        ],
        keep="last",
    )

    return frame


def reconstruct(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Spelar igenom positionshistoriken kronologiskt."""

    state: dict[
        tuple[str, str],
        dict,
    ] = {}

    output: list[dict] = []

    for (
        position_date,
        day,
    ) in frame.groupby(
        "position_date",
        sort=True,
    ):
        # Uppdatera samtliga positioner för dagen.
        for row in day.itertuples(
            index=False
        ):
            key = (
                row.holder,
                row.issuer_key,
            )

            state[key] = {
                "holder": row.holder,
                "issuer": row.issuer,
                "isin": row.isin,
                "position_pct": float(
                    row.position_pct
                ),
            }

        # Spela igenom äldre historik för att bygga upp
        # korrekt state, men börja skriva observationer
        # från START_DATE.
        if position_date < START_DATE:
            continue

        active = [
            value
            for value in state.values()
            if value["position_pct"] > 0
        ]

        if not active:
            continue

        active_frame = pd.DataFrame(
            active
        )

        grouped = active_frame.groupby(
            [
                "issuer",
                "isin",
            ],
            dropna=False,
            sort=True,
        )

        for (
            issuer,
            isin,
        ), rows in grouped:
            positions = rows[
                "position_pct"
            ]

            total = float(
                positions.sum()
            )

            maximum = float(
                positions.max()
            )

            concentration = (
                maximum / total * 100
                if total > 0
                else 0
            )

            output.append(
                {
                    "snapshot_date": (
                        position_date
                        .date()
                        .isoformat()
                    ),
                    "issuer": issuer,
                    "isin": (
                        isin
                        if isin
                        else None
                    ),
                    "short_interest_pct": round(
                        total,
                        6,
                    ),
                    "active_holders": int(
                        (
                            positions > 0
                        ).sum()
                    ),
                    "max_individual_position_pct": (
                        round(
                            maximum,
                            6,
                        )
                    ),
                    "max_position_share_pct": (
                        round(
                            concentration,
                            6,
                        )
                    ),
                    "source": (
                        "reconstructed_from_"
                        "fi_historical_positions"
                    ),
                    "coverage": (
                        "visible_positions_"
                        "from_fi_history"
                    ),
                }
            )

    if not output:
        return pd.DataFrame()

    return pd.DataFrame(
        output
    ).sort_values(
        [
            "snapshot_date",
            "issuer",
            "isin",
        ],
        kind="mergesort",
    )


def write_result(
    result: pd.DataFrame,
) -> None:
    """Skriver JSONL och metadata."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT.open(
        "w",
        encoding="utf-8",
    ) as file:
        for record in result.to_dict(
            orient="records"
        ):
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

    metadata = {
        "dataset": (
            "reconstructed_fi_aggregate"
        ),
        "source_file": str(
            INPUT.relative_to(ROOT)
        ),
        "output_file": str(
            OUTPUT.relative_to(ROOT)
        ),
        "rows": int(
            len(result)
        ),
        "unique_issuers": int(
            result["issuer"].nunique()
        ),
        "first_snapshot_date": (
            result["snapshot_date"].min()
        ),
        "last_snapshot_date": (
            result["snapshot_date"].max()
        ),
        "source": (
            "Finansinspektionen GetHistFile"
        ),
        "official_aggregate_threshold": (
            ">0.1%"
        ),
        "method": (
            "Chronological replay of "
            "normalized individual FI "
            "position changes. A missing "
            "position value represents a "
            "previously visible position "
            "that has fallen below the "
            "publication threshold."
        ),
        "limitations": [
            (
                "Not FI's official aggregate "
                "series."
            ),
            (
                "Positions below FI's "
                "individual publication "
                "threshold are not available "
                "as exact values."
            ),
            (
                "Only dates with position "
                "changes are emitted."
            ),
            (
                "Pre-2022 history is replayed "
                "to establish state."
            ),
        ],
    }

    METADATA.write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    frame = load_history()

    print(
        "FI reconstruction: "
        f"{len(frame)} "
        "positionshändelser."
    )

    print(
        "Kolumner: "
        + ", ".join(
            frame.columns
        )
    )

    print(
        "Period: "
        f"{frame['position_date'].min().date()} "
        "till "
        f"{frame['position_date'].max().date()}"
    )

    result = reconstruct(
        frame
    )

    if result.empty:
        raise RuntimeError(
            "Rekonstruktionen gav inga "
            "observationer."
        )

    write_result(
        result
    )

    print(
        "FI reconstruction klar: "
        f"{len(result)} "
        "observationer."
    )

    print(
        f"Unika emittenter: "
        f"{result['issuer'].nunique()}"
    )

    print(
        f"Period: "
        f"{result['snapshot_date'].min()} "
        "till "
        f"{result['snapshot_date'].max()}"
    )

    print(
        f"Output: {OUTPUT}"
    )

    print(
        f"Metadata: {METADATA}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
