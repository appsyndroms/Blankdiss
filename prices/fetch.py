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
def _instrument_key(
    instrument: dict[str, Any],
) -> str:
    """
    Returnerar den stabila nyckeln för ett instrument.
    Instrumentidentitet ska inte baseras på Yahoo-symbol,
    eftersom samma Yahoo-symbol kan förekomma för flera
    ISIN under instrumentets livscykel.
    Prioritet:
      1. ISIN
      2. LEI + issuer
      3. Yahoo-symbol
    Yahoo-symbol används endast som sista fallback för
    instrument som saknar annan identifierare.
    """
    isin = instrument.get(
        "isin"
    )
    if isin is not None:
        value = str(isin).strip()
        if value:
            return f"isin:{value}"
    lei = instrument.get(
        "lei"
    )
    issuer = instrument.get(
        "issuer"
    )
    lei_value = (
        str(lei).strip()
        if lei is not None
        else ""
    )
    issuer_value = (
        str(issuer).strip()
        if issuer is not None
        else ""
    )
    if lei_value and issuer_value:
        return (
            "lei_issuer:"
            f"{lei_value}:"
            f"{issuer_value}"
        )
    if lei_value:
        return f"lei:{lei_value}"
    yahoo_symbol = instrument.get(
        "yahoo_symbol"
    )
    if yahoo_symbol is not None:
        value = str(
            yahoo_symbol
        ).strip()
        if value:
            return f"yahoo:{value}"
    raise ValueError(
        "Instrument saknar identifierbar "
        "instrumentnyckel."
    )
def _record_instrument_key(
    record: dict[str, Any],
) -> str:
    """
    Samma instrumentnyckel som _instrument_key(),
    men för en prisrecord.
    """
    return _instrument_key(
        record
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
def _first_weekday_after(
    value: date,
) -> date:
    """
    Första vardag efter angivet datum.
    """
    value += timedelta(days=1)
    while value.weekday() >= 5:
        value += timedelta(days=1)
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
def _existing_price_bounds(
    price_dir: Path,
) -> dict[str, dict[str, date]]:
    """
    Läs befintliga prisfiler och hitta första och senaste
    sparade datum per instrument.
    VIKTIGT:
    Täckning räknas per instrument och INTE per
    yahoo_symbol.
    Exempel:
        isin:SE0007439112 -> 2022-01-03 -> 2026-10-01
        isin:SE0016101844 -> ingen historik
    Även om båda använder:
        SINCH.ST
    Dubbletter i befintliga prisfiler påverkar inte
    resultatet. Vi använder endast min/max-datum för
    varje instrument.
    """
    bounds: dict[
        str,
        dict[str, date],
    ] = {}
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
        }
        if not required.issubset(
            frame.columns
        ):
            continue
        frame["date"] = pd.to_datetime(
            frame["date"],
            errors="coerce",
        )
        frame = frame.loc[
            frame["date"].notna()
        ]
        if frame.empty:
            continue
        if "isin" in frame.columns:
            frame["isin"] = (
                frame["isin"]
                .fillna("")
                .astype(str)
                .str.strip()
            )
        if "lei" in frame.columns:
            frame["lei"] = (
                frame["lei"]
                .fillna("")
                .astype(str)
                .str.strip()
            )
        if "issuer" in frame.columns:
            frame["issuer"] = (
                frame["issuer"]
                .fillna("")
                .astype(str)
                .str.strip()
            )
        if "yahoo_symbol" in frame.columns:
            frame["yahoo_symbol"] = (
                frame["yahoo_symbol"]
                .fillna("")
                .astype(str)
                .str.strip()
            )
        for row in frame.to_dict(
            orient="records"
        ):
            try:
                instrument_key = (
                    _record_instrument_key(
                        row
                    )
                )
            except ValueError:
                continue
            timestamp = pd.Timestamp(
                row["date"]
            )
            row_date = timestamp.date()
            existing = bounds.get(
                instrument_key
            )
            if existing is None:
                bounds[
                    instrument_key
                ] = {
                    "first": row_date,
                    "last": row_date,
                }
                continue
            if (
                row_date
                < existing["first"]
            ):
                existing["first"] = (
                    row_date
                )
            if (
                row_date
                > existing["last"]
            ):
                existing["last"] = (
                    row_date
                )
    return bounds
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
    if start_date > end_date:
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
def _build_fetch_intervals(
    instrument: dict[str, Any],
    requested_start: date,
    effective_end: date,
    existing_bounds: dict[str, dict[str, date]],
) -> list[
    tuple[
        date,
        date,
        str,
    ]
]:
    """
    Bestämmer vilka datumintervall som faktiskt behöver
    hämtas för ett instrument.
    Täckningen beräknas på instrumentnyckeln, inte
    Yahoo-symbolen.
    Möjliga situationer:
    1. Ingen lokal historik
       -> requested_start -> effective_end
    2. Lokal historik finns men börjar efter requested_start
       -> requested_start -> dagen före first
    3. Lokal historik finns och behöver uppdateras framåt
       -> dagen efter last -> effective_end
    4. Lokal historik täcker hela det begärda intervallet
       -> inget intervall
    Returnerar även en beskrivning av varför intervallet
    hämtas: "initial", "backfill" eller "incremental".
    """
    instrument_key = _instrument_key(
        instrument
    )
    bounds = existing_bounds.get(
        instrument_key
    )
    if bounds is None:
        initial_start = _next_weekday(
            requested_start
        )
        if initial_start > effective_end:
            return []
        return [
            (
                initial_start,
                effective_end,
                "initial",
            )
        ]
    first_local = bounds[
        "first"
    ]
    last_local = bounds[
        "last"
    ]
    intervals: list[
        tuple[
            date,
            date,
            str,
        ]
    ] = []
    if first_local > requested_start:
        backfill_end = min(
            first_local - timedelta(days=1),
            effective_end,
        )
        backfill_start = _next_weekday(
            requested_start
        )
        if (
            backfill_start
            <= backfill_end
        ):
            intervals.append(
                (
                    backfill_start,
                    backfill_end,
                    "backfill",
                )
            )
    if last_local < effective_end:
        incremental_start = _next_weekday(
            last_local
            + timedelta(days=1)
        )
        if (
            incremental_start
            <= effective_end
        ):
            intervals.append(
                (
                    incremental_start,
                    effective_end,
                    "incremental",
                )
            )
    return intervals
def _deduplicate_records(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Deduplicera records innan skrivning.
    Primär nyckel:
        datum + instrumentnyckel
    Yahoo-symbol används alltså inte som identitet.
    Om samma datum + instrument förekommer flera gånger
    behålls den första recorden.
    Detta skyddar mot att en framtida batch eller flera
    överlappande hämtningar bygger nya dubbletter.
    """
    unique: dict[
        tuple[str, str],
        dict[str, Any],
    ] = {}
    duplicates = 0
    for record in records:
        value = record.get(
            "date"
        )
        if value is None:
            continue
        try:
            record_date = _normalise_date(
                str(value)
            ).isoformat()
        except ValueError:
            continue
        try:
            instrument_key = (
                _record_instrument_key(
                    record
                )
            )
        except ValueError:
            continue
        key = (
            record_date,
            instrument_key,
        )
        if key in unique:
            duplicates += 1
            continue
        record_to_store = dict(
            record
        )
        record_to_store[
            "date"
        ] = record_date
        unique[key] = record_to_store
    if duplicates:
        print(
            "Pris: "
            f"{duplicates:,} dubbletter "
            "filtrerades bort före skrivning."
        )
    return list(
        unique.values()
    )
def fetch_prices(
    instruments: list[dict[str, Any]],
    start: str | date,
    end: str | date | None = None,
) -> list[dict[str, Any]]:
    """
    Hämta prisdata inkrementellt och med backfill.
    Yahoo använder slutdatum exklusivt i sin API-hämtning.
    Lokal täckning beräknas per instrument, inte per
    Yahoo-symbol.
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
    effective_end_date = (
        _last_weekday_before(
            end_date
        )
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
        try:
            _instrument_key(
                instrument
            )
        except ValueError:
            skipped += 1
            continue
        valid_instruments.append(
            instrument
        )
    if skipped:
        print(
            f"Pris: {skipped} instrument "
            "hoppades över eftersom "
            "Yahoo-symbol eller instrumentidentitet "
            "saknas."
        )
    if not valid_instruments:
        print(
            "Pris: inga giltiga "
            "instrument att hämta."
        )
        return []
    existing_bounds = (
        _existing_price_bounds(
            OUTPUT_DIR
        )
    )
    fetch_jobs: list[
        tuple[
            date,
            date,
            str,
            dict[str, Any],
        ]
    ] = []
    already_current = 0
    for instrument in valid_instruments:
        intervals = (
            _build_fetch_intervals(
                instrument=instrument,
                requested_start=start_date,
                effective_end=effective_end_date,
                existing_bounds=existing_bounds,
            )
        )
        if not intervals:
            already_current += 1
            continue
        for (
            fetch_start,
            fetch_end,
            reason,
        ) in intervals:
            fetch_jobs.append(
                (
                    fetch_start,
                    fetch_end,
                    reason,
                    instrument,
                )
            )
    if already_current:
        print(
            "Pris: lokal historik är redan aktuell "
            f"för {already_current} instrument."
        )
    if not fetch_jobs:
        print(
            "Pris: ingen Yahoo-hämtning behövs."
        )
        return []
    backfill_jobs = sum(
        1
        for job in fetch_jobs
        if job[2] == "backfill"
    )
    incremental_jobs = sum(
        1
        for job in fetch_jobs
        if job[2] == "incremental"
    )
    initial_jobs = sum(
        1
        for job in fetch_jobs
        if job[2] == "initial"
    )
    print(
        "Pris: hämtningar planerade - "
        f"{len(fetch_jobs)} intervall "
        f"({backfill_jobs} backfill, "
        f"{incremental_jobs} incremental, "
        f"{initial_jobs} initial)."
    )
    records: list[
        dict[str, Any]
    ] = []
    for (
        fetch_start,
        fetch_end,
        reason,
        instrument,
    ) in sorted(
        fetch_jobs,
        key=lambda job: (
            job[0],
            job[1],
            job[2],
            job[3].get(
                "yahoo_symbol",
                "",
            ),
            job[3].get(
                "isin",
                "",
            ) or "",
        ),
    ):
        symbol = instrument[
            "yahoo_symbol"
        ]
        print(
            "Pris: "
            f"{reason} - "
            f"{symbol} - "
            f"{fetch_start.isoformat()} -> "
            f"{fetch_end.isoformat()}"
        )
        batch_records = _download_batch(
            [instrument],
            fetch_start,
            fetch_end,
        )
        records.extend(
            batch_records
        )
    records = _deduplicate_records(
        records
    )
    print(
        "Pris: hämtning klar - "
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
    Records dedupliceras på datum + instrument innan
    filen skrivs.
    Befintliga prisfiler skrivs inte över eller städas
    här. Den gamla filstrukturen hanteras separat.
    """
    if not records:
        raise ValueError(
            "Kan inte skriva prisfil: "
            "records är tom."
        )
    records = _deduplicate_records(
        records
    )
    if not records:
        raise ValueError(
            "Kan inte skriva prisfil: "
            "inga giltiga records efter deduplicering."
        )
    start_date = _normalise_date(
        start
    )
    record_dates: list[
        date
    ] = []
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
