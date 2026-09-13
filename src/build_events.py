"""
Bygger event-data från blankningssnapshots och priser.

En event inträffar när blankningen förändras mellan två
observerade positioner för samma emittent.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT / "data" / "raw"
EVENT_DIR = ROOT / "data" / "events"


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []

    records = []

    with path.open(
        encoding="utf-8"
    ) as handle:
        for line in handle:
            line = line.strip()

            if line:
                records.append(
                    json.loads(line)
                )

    return records


def latest_fi_file() -> Path | None:
    files = sorted(
        RAW_DIR.glob("fi_aggregate_*.jsonl")
    )

    return files[-1] if files else None


def price_file() -> Path | None:
    files = sorted(
        RAW_DIR.glob("prices_*.jsonl")
    )

    return files[-1] if files else None


def calculate_forward_returns(
    prices: pd.DataFrame,
) -> pd.DataFrame:
    prices = prices.copy()

    prices["date"] = pd.to_datetime(
        prices["date"]
    )

    prices = prices.sort_values(
        ["ticker", "date"]
    )

    for days in (1, 5, 20, 60):
        prices[
            f"return_{days}d"
        ] = (
            prices.groupby("ticker")["close"]
            .shift(-days)
            / prices["close"]
            - 1
        )

    return prices


def build_events() -> list[dict]:
    fi_path = latest_fi_file()
    price_path = price_file()

    if fi_path is None:
        return []

    fi = pd.DataFrame(
        read_jsonl(fi_path)
    )

    if fi.empty:
        return []

    fi["position_date"] = pd.to_datetime(
        fi["position_date"]
    )

    fi = fi.sort_values(
        ["lei", "position_date"]
    )

    fi["previous_short_interest_pct"] = (
        fi.groupby("lei")[
            "short_interest_pct"
        ].shift(1)
    )

    fi["change_pp"] = (
        fi["short_interest_pct"]
        - fi["previous_short_interest_pct"]
    )

    events = fi[
        fi["previous_short_interest_pct"].notna()
        & (fi["change_pp"] != 0)
    ].copy()

    if price_path is not None:
        prices = pd.DataFrame(
            read_jsonl(price_path)
        )

        if not prices.empty:
            prices = calculate_forward_returns(
                prices
            )

            # V0.1: ticker/LEI mapping är ännu inte
            # komplett. Prisjoin görs därför endast
            # när ticker finns i event-data.
            if "ticker" in events.columns:
                events = events.merge(
                    prices,
                    left_on=[
                        "ticker",
                        "position_date",
                    ],
                    right_on=[
                        "ticker",
                        "date",
                    ],
                    how="left",
                )

    result = []

    for _, row in events.iterrows():
        record = {
            "event_date": row[
                "position_date"
            ].strftime("%Y-%m-%d"),

            "lei": row["lei"],

            "issuer": row["issuer"],

            "short_interest_pct": float(
                row["short_interest_pct"]
            ),

            "previous_short_interest_pct": float(
                row[
                    "previous_short_interest_pct"
                ]
            ),

            "change_pp": float(
                row["change_pp"]
            ),
        }

        for days in (1, 5, 20, 60):
            column = f"return_{days}d"

            if column in row and pd.notna(
                row[column]
            ):
                record[column] = float(
                    row[column]
                )
            else:
                record[column] = None

        result.append(record)

    return result


def write_events(
    events: list[dict],
) -> Path:

    EVENT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        EVENT_DIR
        / f"short_events_{date.today().isoformat()}.jsonl"
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        for event in events:
            handle.write(
                json.dumps(
                    event,
                    ensure_ascii=False,
                )
                + "\n"
            )

    return path


def main() -> None:
    events = build_events()
    path = write_events(events)

    print(
        f"Events: {len(events)} → {path}"
    )


if __name__ == "__main__":
    main()
