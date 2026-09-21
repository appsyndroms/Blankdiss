from __future__ import annotations
import json
import math
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
import pandas as pd
import yfinance as yf
OUTPUT_DIR = Path(
    "data/raw/prices"
)
def _valid_symbol(
    symbol: Any,
) -> bool:
    if symbol is None:
        return False
    value = str(symbol).strip()
    if not value:
        return False
    if value.upper() in {
        "NONE",
        "NULL",
        "NAN",
        "NAT",
    }:
        return False
    return True
def _normalise_date(
    value: str | date,
) -> date:
    """
    Convert a date or ISO date string to a date object.
    """
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        return date.fromisoformat(
            value
        )
    raise TypeError(
        "Date must be a date or ISO "
        f"date string, got {type(value).__name__}"
    )
def _next_weekday(
    value: date,
) -> date:
    """
    Flytta ett datum framåt till närmaste vardag.
    """
    while value.weekday() >= 5:
        value += timedelta(days=1)
    return value
def _last_weekday_before(
    value: date,
) -> date:
    """
    Senaste vardag före angivet datum.
    """
    value -= timedelta(days=1)
    while value.weekday() >= 5:
        value -= timedelta(days=1)
    return value
def _extract_close(
    data,
    symbol: str,
):
    """
    Extract Close from yfinance output.
    Handles both:
      - MultiIndex columns
      - ordinary DataFrame columns
    """
    if data is None or data.empty:
        return None
    if hasattr(
        data.columns,
        "levels",
    ):
        try:
            level0 = (
                data.columns
                .get_level_values(0)
            )
            if "Close" in level0:
                close = data["Close"]
                if hasattr(
                    close,
                    "columns",
                ):
                    if (
                        symbol
                        in close.columns
                    ):
                        return close[symbol]
                    if (
                        len(close.columns)
                        == 1
                    ):
                        return close.iloc[
                            :,
                            0,
                        ]
                return close
        except Exception:
            pass
    if "Close" in data.columns:
        return data["Close"]
    return None
def _existing_latest_dates(
    price_dir: Path,
) -> dict[str, date]:
    """
    Läs befintliga prisfiler och hitta senaste
    sparade datum per Yahoo-symbol.
    Detta används för inkrementell hämtning.
    """
    latest: dict[str, date] = {}
    files = sorted(
        price_dir.glob(
            "prices_*.jsonl"
        )
    )
    for path in files:
        try:
            frame = pd.read_json(
                path,
                lines=True,
            )
        except (
            ValueError,
            OSError,
        ):
            continue
        required = {
            "date",
            "yahoo_symbol",
        }
        if not required.issubset(
            frame.columns
        ):
            continue
        frame["date"] = pd.to_datetime(
            frame["date"],
            errors="coerce",
        )
        frame["yahoo_symbol"] = (
            frame["yahoo_symbol"]
            .fillna("")
            .astype(str)
            .str.strip()
        )
        frame = frame.loc[
            frame["date"].notna()
            & frame["yahoo_symbol"].ne("")
        ]
        for symbol, group in frame.groupby(
            "yahoo_symbol",
            sort=False,
        ):
            latest_timestamp = group[
                "date"
            ].max()
            latest_date = (
                latest_timestamp.date()
            )
            previous = latest.get(
                symbol
            )
            if (
                previous is None
                or latest_date > previous
            ):
                latest[symbol] = latest_date
    return latest
def _download_batch(
    instruments: list[dict[str, Any]],
    start_date: date,
    end_date: date,
) -> list[dict[str, Any]]:
    """
    Hämtar ett intervall från Yahoo för en grupp instrument.
    """
    if not instruments:
        return []
    symbols = sorted(
        {
            instrument[
                "yahoo_symbol"
            ]
            for instrument in instruments
            if _valid_symbol(
                instrument.get(
                    "yahoo_symbol"
                )
            )
        }
    )
    if not symbols:
        return []
    print(
        "Pris: Yahoo-hämtning - "
        f"{len(symbols)} instrument, "
        f"{start_date.isoformat()} -> "
        f"{end_date.isoformat()}."
    )
    data = yf.download(
        tickers=symbols,
        start=start_date.isoformat(),
        end=(
            end_date + timedelta(days=1)
        ).isoformat(),
        auto_adjust=False,
        actions=False,
        threads=True,
        timeout=20,
        progress=False,
        group_by="column",
    )
    if data is None or data.empty:
        print(
            "Pris: Yahoo returnerade "
            "ingen data för intervallet."
        )
        return []
    instrument_by_symbol = {
        instrument[
            "yahoo_symbol"
        ]: instrument
        for instrument in instruments
    }
    records: list[
        dict[str, Any]
    ] = []
    symbols_with_prices = 0
    symbols_without_prices = 0
    skipped_nonfinite = 0
    for symbol in symbols:
        instrument = (
            instrument_by_symbol[
                symbol
            ]
        )
        close_series = (
            _extract_close(
                data,
                symbol,
            )
        )
        if close_series is None:
            symbols_without_prices += 1
            continue
        count = 0
        for timestamp, value in (
            close_series.items()
        ):
            try:
                close = float(value)
            except (
                TypeError,
                ValueError,
            ):
                skipped_nonfinite += 1
                continue
            if not math.isfinite(
                close
            ):
                skipped_nonfinite += 1
                continue
            timestamp_date = (
                pd.Timestamp(
                    timestamp
                ).date()
            )
            if (
                timestamp_date < start_date
                or timestamp_date > end_date
            ):
                continue
            records.append(
                {
                    "date": (
                        timestamp_date.isoformat()
                    ),
                    "isin": instrument.get(
                        "isin"
                    ),
                    "lei": instrument.get(
                        "lei"
                    ),
                    "issuer": instrument.get(
                        "issuer"
                    ),
                    "ticker": instrument.get(
                        "ticker"
                    ),
                    "yahoo_symbol": symbol,
                    "mapping_source": (
                        instrument.get(
                            "mapping_source"
                        )
                    ),
                    "close": close,
                }
            )
            count += 1
        if count > 0:
            symbols_with_prices += 1
        else:
            symbols_without_prices += 1
    print(
        "Pris: Yahoo-resultat - "
        f"{symbols_with_prices} symboler med data, "
        f"{symbols_without_prices} utan användbara priser."
    )
    if skipped_nonfinite:
        print(
            "Pris: "
            f"{skipped_nonfinite} icke-finit prisvärden "
            "filtrerades bort."
        )
    return records
def fetch_prices(
    instruments: list[dict[str, Any]],
    start: str | date,
    end: str | date | None = None,
) -> list[dict[str, Any]]:
    """
    Hämta prisdata inkrementellt.
    Om det redan finns lokal historik för ett instrument
    hämtas endast data efter instrumentets senaste lokala datum.
    Detta innebär att:
        python -m prices --start 2022-01-01
    första gången hämtar historiken från 2022.
    Nästa gång hämtas endast det som saknas.
    Yahoo använder slutdatum exklusivt i sin API-hämtning.
    Därför begränsas den effektiva slutdagen till senaste
    vardag före angivet slutdatum. Helgintervall skickas
    aldrig till Yahoo.
    """
    start_date = _normalise_date(
        start
    )
    if end is None:
        end_date = date.today()
    else:
        end_date = _normalise_date(
            end
        )
    if end_date <= start_date:
        raise ValueError(
            "End date must be later "
            "than start date: "
            f"{start_date} -> {end_date}"
        )
    # Yahoo använder slutdatum exklusivt.
    # Begränsa därför hämtningen till senaste vardag
    # före end_date.
    effective_end_date = _last_weekday_before(
        end_date
    )
    valid_instruments: list[
        dict[str, Any]
    ] = []
    skipped = 0
    for instrument in instruments:
        symbol = instrument.get(
            "yahoo_symbol"
        )
        if not _valid_symbol(
            symbol
        ):
            skipped += 1
            continue
        valid_instruments.append(
            instrument
        )
    if skipped:
        print(
            f"Pris: {skipped} instrument "
            "hoppades över eftersom "
            "Yahoo-symbol saknas."
        )
    if not valid_instruments:
        print(
            "Pris: inga giltiga "
            "Yahoo-symboler att hämta."
        )
        return []
    latest_local = (
        _existing_latest_dates(
            OUTPUT_DIR
        )
    )
    instruments_to_fetch: dict[
        date,
        list[dict[str, Any]],
    ] = {}
    already_current = 0
    for instrument in valid_instruments:
        symbol = instrument[
            "yahoo_symbol"
        ]
        local_latest = latest_local.get(
            symbol
        )
        if local_latest is None:
            fetch_start = start_date
        else:
            fetch_start = max(
                start_date,
                local_latest
                + timedelta(days=1),
            )
        # Ett intervall får inte börja på en helg.
        fetch_start = _next_weekday(
            fetch_start
        )
        if fetch_start > effective_end_date:
            already_current += 1
            continue
        instruments_to_fetch.setdefault(
            fetch_start,
            [],
        ).append(
            instrument
        )
    if already_current:
        print(
            "Pris: lokal historik är redan aktuell "
            f"för {already_current} instrument."
        )
    if not instruments_to_fetch:
        print(
            "Pris: ingen Yahoo-hämtning behövs."
        )
        return []
    print(
        "Pris: inkrementell hämtning - "
        f"{sum(len(v) for v in instruments_to_fetch.values())} "
        "instrument."
    )
    records: list[
        dict[str, Any]
    ] = []
    for fetch_start, batch in sorted(
        instruments_to_fetch.items(),
        key=lambda item: item[0],
    ):
        batch_records = _download_batch(
            batch,
            fetch_start,
            effective_end_date,
        )
        records.extend(
            batch_records
        )
    print(
        "Pris: inkrementell hämtning klar - "
        f"{len(records):,} nya observationer."
    )
    return records
def write_jsonl(
    records: list[dict[str, Any]],
    *,
    start: str | date,
    end: str | date | None = None,
) -> Path:
    """
    Skriv prisdata som JSONL.
    När records kommer från en inkrementell hämtning
    används faktiskt första och sista observationsdatum
    i filnamnet.
    """
    if not records:
        raise ValueError(
            "Kan inte skriva prisfil: "
            "records är tom."
        )
    start_date = _normalise_date(
        start
    )
    record_dates: list[date] = []
    for record in records:
        value = record.get(
            "date"
        )
        if value is None:
            continue
        try:
            record_dates.append(
                _normalise_date(
                    str(value)
                )
            )
        except ValueError:
            continue
    if record_dates:
        actual_start = min(
            record_dates
        )
        actual_end = max(
            record_dates
        )
    else:
        actual_start = start_date
        if end is None:
            actual_end = actual_start
        else:
            actual_end = _normalise_date(
                end
            )
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    output = (
        OUTPUT_DIR
        / (
            "prices_"
            f"{actual_start.isoformat()}"
            "_"
            f"{actual_end.isoformat()}"
            ".jsonl"
        )
    )
    written = 0
    skipped = 0
    with output.open(
        "w",
        encoding="utf-8",
    ) as handle:
        for record in records:
            close = record.get(
                "close"
            )
            try:
                numeric_close = float(
                    close
                )
            except (
                TypeError,
                ValueError,
            ):
                skipped += 1
                continue
            if not math.isfinite(
                numeric_close
            ):
                skipped += 1
                continue
            record_to_write = dict(
                record
            )
            record_to_write[
                "close"
            ] = numeric_close
            handle.write(
                json.dumps(
                    record_to_write,
                    ensure_ascii=False,
                )
            )
            handle.write(
                "\n"
            )
            written += 1
    print(
        "Pris: JSONL skriven - "
        f"{written:,} rader -> "
        f"{output.name}"
    )
    if skipped:
        print(
            "Pris: "
            f"{skipped} ogiltiga rader "
            "filtrerades bort vid skrivning."
        )
    return output
def save_prices(
    records: list[dict[str, Any]],
    *,
    start: str | date,
    end: str | date,
) -> Path:
    """
    Convenience wrapper.
    """
    return write_jsonl(
        records,
        start=start,
        end=end,
    )
