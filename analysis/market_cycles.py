"""
Marknadsrelativ avkastning och cykliska blankningsmönster.
Detta analyssteg gör två saker:
1. Jämför aktiernas framtida avkastning med OMXSPI.
2. Identifierar återkommande lokala toppar och dalar i
   short interest för enskilda bolag.
Marknadsdata hämtas från FRED-serien NASDAQOMXSPI.
FRED-serien är Nasdaq-data och innehåller dagliga
OMX Stockholm Price Index-värden.
Marknadsdata persisteras lokalt som JSONL.
Arkitektur:
    FRED / NASDAQOMXSPI
        |
        v
    data/raw/market/omxspi.jsonl
        |
        v
    market_cycles.py
        |
        +--> market_adjusted_returns.jsonl
        |
        +--> short_cycles.jsonl
        |
        +--> short_cycles_summary.json
Marknadsdata är diagnostik/analysdata och används ännu inte
som ML-feature.
"""
from __future__ import annotations
import json
import math
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import requests
FEATURES_PATH = Path(
    "data/processed/analysis/features.jsonl"
)
MARKET_RAW_PATH = Path(
    "data/raw/market/omxspi.jsonl"
)
MARKET_ADJUSTED_PATH = Path(
    "data/processed/analysis/market_adjusted_returns.jsonl"
)
SHORT_CYCLES_PATH = Path(
    "data/processed/analysis/short_cycles.jsonl"
)
SHORT_CYCLES_SUMMARY_PATH = Path(
    "data/processed/analysis/short_cycles_summary.json"
)
MARKET_SYMBOL = "OMXSPI"
MARKET_SOURCE = "FRED"
MARKET_SOURCE_SERIES = "NASDAQOMXSPI"
FRED_URL = (
    "https://fred.stlouisfed.org/graph/fredgraph.csv"
)
FRED_HEADERS = {
    "User-Agent": (
        "Blankdiss/1.0 "
        "(market analysis; "
        "NASDAQOMXSPI)"
    ),
    "Accept": (
        "text/csv,text/plain,*/*"
    ),
}
RETURN_HORIZONS = (
    5,
    20,
    60,
)
MIN_CYCLE_PROMINENCE = 0.25
MIN_DAYS_BETWEEN_CYCLE_EVENTS = 30
CYCLE_LOOKBACK_DAYS = 180
REQUEST_TIMEOUT_SECONDS = 60
def _parse_number(
    value: Any,
) -> float:
    """
    Konvertera ett numeriskt värde till float.
    Hanterar exempelvis:
        1234.56
        1,234.56
        1 234,56
        1234,56
    FRED använder normalt punkt som decimaltecken.
    """
    if value is None:
        return math.nan
    if isinstance(
        value,
        (int, float),
    ):
        return float(value)
    text = str(value).strip()
    if not text:
        return math.nan
    if text == ".":
        return math.nan
    text = (
        text
        .replace("\xa0", "")
        .replace(" ", "")
        .replace("%", "")
    )
    if "," in text and "." in text:
        if text.rfind(",") > text.rfind("."):
            text = (
                text
                .replace(".", "")
                .replace(",", ".")
            )
        else:
            text = text.replace(",", "")
    elif "," in text:
        text = text.replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return math.nan
def _json_value(
    value: Any,
) -> Any:
    """
    Gör ett värde säkert för JSON.
    """
    if value is None:
        return None
    if isinstance(
        value,
        (
            np.integer,
            np.int64,
            np.int32,
        ),
    ):
        return int(value)
    if isinstance(
        value,
        (
            np.floating,
            np.float64,
            np.float32,
        ),
    ):
        value = float(value)
        if not np.isfinite(value):
            return None
        return value
    if isinstance(
        value,
        (
            pd.Timestamp,
            np.datetime64,
        ),
    ):
        return pd.Timestamp(
            value
        ).strftime(
            "%Y-%m-%d"
        )
    if pd.isna(value):
        return None
    return value
def load_features() -> pd.DataFrame:
    """
    Läs befintliga Blankdiss-features.
    """
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            "Featurefil saknas: "
            f"{FEATURES_PATH}"
        )
    frame = pd.read_json(
        FEATURES_PATH,
        lines=True,
    )
    if frame.empty:
        raise RuntimeError(
            "Featurefilen är tom."
        )
    required_columns = {
        "security_key",
        "snapshot_date",
        "price_date",
        "short_interest_pct",
    }
    missing = (
        required_columns
        - set(frame.columns)
    )
    if missing:
        raise RuntimeError(
            "Features saknar obligatoriska "
            f"kolumner: {sorted(missing)}"
        )
    frame["snapshot_date"] = pd.to_datetime(
        frame["snapshot_date"],
        errors="coerce",
    )
    frame["price_date"] = pd.to_datetime(
        frame["price_date"],
        errors="coerce",
    )
    frame["short_interest_pct"] = pd.to_numeric(
        frame["short_interest_pct"],
        errors="coerce",
    )
    for horizon in RETURN_HORIZONS:
        column = (
            f"forward_return_{horizon}d"
        )
        if column in frame.columns:
            frame[column] = pd.to_numeric(
                frame[column],
                errors="coerce",
            )
    frame = frame.dropna(
        subset=[
            "security_key",
            "snapshot_date",
        ]
    ).copy()
    return frame
def _parse_fred_csv(
    text: str,
) -> pd.DataFrame:
    """
    Tolka FRED:s CSV-svar.
    Förväntat format:
        DATE,VALUE
        2022-05-25,738.12
        2022-05-26,741.42
        ...
    FRED kan använda '.' för saknade värden.
    Dessa observationer tas bort.
    """
    if not text.strip():
        raise RuntimeError(
            "FRED returnerade ett tomt svar."
        )
    try:
        from io import StringIO
        frame = pd.read_csv(
            StringIO(text),
            dtype=str,
        )
    except Exception as error:
        raise RuntimeError(
            "FRED-svaret kunde inte läsas "
            "som CSV."
        ) from error
    if frame.empty:
        raise RuntimeError(
            "FRED CSV innehåller inga rader."
        )
    # FRED använder DATE och VALUE.
    # Vi accepterar även lowercase för robusthet.
    columns = {
        str(column).strip().upper(): column
        for column in frame.columns
    }
    date_column = columns.get(
        "DATE"
    )
    value_column = columns.get(
        "VALUE"
    )
    if (
        date_column is None
        or value_column is None
    ):
        raise RuntimeError(
            "FRED CSV saknar förväntade "
            "kolumner DATE/VALUE. "
            f"Kolumner: {list(frame.columns)}"
        )
    result = pd.DataFrame(
        {
            "market_date": pd.to_datetime(
                frame[date_column],
                errors="coerce",
            ),
            "market_close": [
                _parse_number(value)
                for value in frame[value_column]
            ],
        }
    )
    result = result.dropna(
        subset=[
            "market_date",
            "market_close",
        ]
    )
    result = result[
        np.isfinite(
            result["market_close"]
        )
    ]
    result = result[
        result["market_close"] > 0
    ]
    result = (
        result
        .drop_duplicates(
            subset=[
                "market_date",
            ]
        )
        .sort_values(
            "market_date"
        )
        .reset_index(
            drop=True
        )
    )
    if result.empty:
        raise RuntimeError(
            "FRED CSV innehåller inga "
            "giltiga OMXSPI-observationer."
        )
    return result
def download_market_data(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    """
    Hämta OMXSPI från FRED.
    FRED-serie:
        NASDAQOMXSPI
    Serien är daglig och kommer från Nasdaq.
    Vi hämtar ett litet överlapp runt intervallet
    så att inkrementella uppdateringar inte riskerar
    att lämna luckor vid exempelvis helger eller
    korrigerade observationer.
    """
    requested_start = (
        pd.Timestamp(start_date)
        - pd.Timedelta(days=10)
    )
    requested_end = (
        pd.Timestamp(end_date)
        + pd.Timedelta(days=10)
    )
    start_text = requested_start.strftime(
        "%Y-%m-%d"
    )
    end_text = requested_end.strftime(
        "%Y-%m-%d"
    )
    params = {
        "id": MARKET_SOURCE_SERIES,
        "cosd": start_text,
        "coed": end_text,
    }
    print(
        "Laddar marknadsdata: "
        f"{MARKET_SOURCE}/"
        f"{MARKET_SOURCE_SERIES}"
    )
    print(
        "FRED-intervall: "
        f"{start_text} -> {end_text}"
    )
    try:
        response = requests.get(
            FRED_URL,
            params=params,
            headers=FRED_HEADERS,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException as error:
        raise RuntimeError(
            "Kunde inte hämta "
            "OMXSPI från FRED: "
            f"{error}"
        ) from error
    print(
        "FRED HTTP-status: "
        f"{response.status_code}"
    )
    print(
        "FRED Content-Type: "
        f"{response.headers.get('Content-Type', '')}"
    )
    try:
        response.raise_for_status()
    except requests.HTTPError as error:
        preview = response.text[:1000]
        print(
            "FRED-svar, början:\n"
            f"{preview}"
        )
        raise RuntimeError(
            "FRED returnerade HTTP-fel: "
            f"{response.status_code}"
        ) from error
    preview = response.text[:500]
    print(
        "FRED-svar, början:\n"
        f"{preview}"
    )
    market = _parse_fred_csv(
        response.text
    )
    market = market[
        (
            market["market_date"]
            >= requested_start
        )
        & (
            market["market_date"]
            <= requested_end
        )
    ].reset_index(
        drop=True
    )
    if market.empty:
        raise RuntimeError(
            "FRED/OMXSPI innehåller inga "
            "observationer inom det begärda "
            "intervallet."
        )
    print(
        "OMXSPI-period: "
        f"{market['market_date'].min().date()} "
        "-> "
        f"{market['market_date'].max().date()}"
    )
    print(
        "OMXSPI-observationer: "
        f"{len(market):,}"
    )
    return market
def load_raw_market_data() -> pd.DataFrame:
    """
    Läs redan sparad OMXSPI-historik.
    """
    if not MARKET_RAW_PATH.exists():
        return pd.DataFrame(
            columns=[
                "market_date",
                "market_close",
            ]
        )
    rows = []
    with MARKET_RAW_PATH.open(
        "r",
        encoding="utf-8",
    ) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(
                    line
                )
            except json.JSONDecodeError:
                continue
            rows.append(
                record
            )
    if not rows:
        return pd.DataFrame(
            columns=[
                "market_date",
                "market_close",
            ]
        )
    frame = pd.DataFrame(
        rows
    )
    if (
        "market_date"
        not in frame.columns
        or
        "market_close"
        not in frame.columns
    ):
        return pd.DataFrame(
            columns=[
                "market_date",
                "market_close",
            ]
        )
    frame["market_date"] = pd.to_datetime(
        frame["market_date"],
        errors="coerce",
    )
    frame["market_close"] = pd.to_numeric(
        frame["market_close"],
        errors="coerce",
    )
    frame = frame.dropna(
        subset=[
            "market_date",
            "market_close",
        ]
    )
    frame = frame[
        np.isfinite(
            frame["market_close"]
        )
    ]
    frame = frame[
        frame["market_close"] > 0
    ]
    return (
        frame
        .drop_duplicates(
            subset=[
                "market_date",
            ]
        )
        .sort_values(
            "market_date"
        )
        .reset_index(
            drop=True
        )
    )
def merge_and_persist_market_data(
    existing: pd.DataFrame,
    downloaded: pd.DataFrame,
) -> pd.DataFrame:
    """
    Slå ihop lokal och ny marknadsdata.
    Den lokala filen blir därmed den
    långsiktiga historiska OMXSPI-serien.
    """
    combined = pd.concat(
        [
            existing,
            downloaded,
        ],
        ignore_index=True,
    )
    if combined.empty:
        raise RuntimeError(
            "Ingen marknadsdata att spara."
        )
    combined["market_date"] = pd.to_datetime(
        combined["market_date"],
        errors="coerce",
    )
    combined["market_close"] = pd.to_numeric(
        combined["market_close"],
        errors="coerce",
    )
    combined = combined.dropna(
        subset=[
            "market_date",
            "market_close",
        ]
    )
    combined = combined[
        np.isfinite(
            combined["market_close"]
        )
    ]
    combined = combined[
        combined["market_close"] > 0
    ]
    combined = (
        combined
        .drop_duplicates(
            subset=[
                "market_date",
            ],
            keep="last",
        )
        .sort_values(
            "market_date"
        )
        .reset_index(
            drop=True
        )
    )
    MARKET_RAW_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    with MARKET_RAW_PATH.open(
        "w",
        encoding="utf-8",
    ) as handle:
        for record in combined.to_dict(
            orient="records"
        ):
            cleaned = {
                key: _json_value(value)
                for key, value in record.items()
            }
            handle.write(
                json.dumps(
                    cleaned,
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(
        "Sparad OMXSPI-historik: "
        f"{MARKET_RAW_PATH}"
    )
    print(
        "Totalt OMXSPI-rader: "
        f"{len(combined):,}"
    )
    return combined
def add_market_forward_returns(
    market: pd.DataFrame,
) -> pd.DataFrame:
    """
    Beräkna framtida OMXSPI-avkastning
    per handelsdag.
    """
    result = market.copy()
    result = (
        result
        .sort_values(
            "market_date"
        )
        .reset_index(
            drop=True
        )
    )
    prices = (
        result["market_close"]
        .to_numpy(
            dtype=float
        )
    )
    for horizon in RETURN_HORIZONS:
        target_index = (
            np.arange(
                len(result)
            )
            + horizon
        )
        values = np.full(
            len(result),
            np.nan,
            dtype=float,
        )
        valid = (
            target_index
            < len(result)
        )
        if valid.any():
            values[valid] = (
                prices[
                    target_index[valid]
                ]
                / prices[valid]
                - 1.0
            )
        result[
            f"market_forward_return_{horizon}d"
        ] = values
    return result
def align_market_returns(
    features: pd.DataFrame,
    market: pd.DataFrame,
) -> pd.DataFrame:
    """
    Matcha varje aktierad mot första
    OMXSPI-handelsdagen på eller efter
    aktiens price_date.
    """
    frame = features.copy()
    market = (
        market
        .sort_values(
            "market_date"
        )
        .reset_index(
            drop=True
        )
    )
    market_dates = (
        market["market_date"]
        .to_numpy()
    )
    market_lookup = market.set_index(
        "market_date"
    )
    for horizon in RETURN_HORIZONS:
        column = (
            f"market_forward_return_{horizon}d"
        )
        values: list[float] = []
        for price_date in frame[
            "price_date"
        ]:
            if pd.isna(price_date):
                values.append(
                    np.nan
                )
                continue
            target_date = pd.Timestamp(
                price_date
            )
            index = np.searchsorted(
                market_dates,
                target_date,
                side="left",
            )
            if index >= len(
                market_dates
            ):
                values.append(
                    np.nan
                )
                continue
            matched_date = (
                market_dates[index]
            )
            value = market_lookup.loc[
                matched_date,
                column,
            ]
            values.append(
                float(value)
                if pd.notna(value)
                else np.nan
            )
        frame[column] = values
    return frame
def build_market_adjusted_returns(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """
    Skapa abnormal return:
        aktiens framtida avkastning
        -
        OMXSPI:s framtida avkastning.
    """
    result = frame[
        [
            "security_key",
            "snapshot_date",
            "price_date",
        ]
    ].copy()
    for horizon in RETURN_HORIZONS:
        stock_column = (
            f"forward_return_{horizon}d"
        )
        market_column = (
            f"market_forward_return_{horizon}d"
        )
        abnormal_column = (
            f"abnormal_return_{horizon}d"
        )
        if stock_column not in frame.columns:
            result[abnormal_column] = np.nan
            continue
        result[stock_column] = (
            frame[stock_column]
        )
        result[market_column] = (
            frame[market_column]
        )
        result[abnormal_column] = (
            frame[stock_column]
            - frame[market_column]
        )
    return result
def detect_local_extrema(
    series: pd.Series,
    dates: pd.Series,
) -> list[dict[str, Any]]:
    """
    Identifiera lokala toppar och dalar
    i short interest.
    Ett event måste:
      - ha minst 0.25 procentenheters
        prominence
      - ligga minst 30 dagar från
        föregående event
      - ha en lokal referensnivå inom
        180 dagar
    """
    values = pd.to_numeric(
        series,
        errors="coerce",
    ).to_numpy(
        dtype=float
    )
    date_values = pd.to_datetime(
        dates,
        errors="coerce",
    ).to_numpy()
    events: list[
        dict[str, Any]
    ] = []
    for index in range(
        1,
        len(values) - 1,
    ):
        current = values[index]
        if not np.isfinite(
            current
        ):
            continue
        previous = values[
            index - 1
        ]
        following = values[
            index + 1
        ]
        if not (
            np.isfinite(previous)
            and np.isfinite(following)
        ):
            continue
        is_peak = (
            current > previous
            and current >= following
        )
        is_trough = (
            current < previous
            and current <= following
        )
        if not (
            is_peak
            or is_trough
        ):
            continue
        current_date = pd.Timestamp(
            date_values[index]
        )
        if pd.isna(
            current_date
        ):
            continue
        lookback_start = (
            current_date
            - pd.Timedelta(
                days=CYCLE_LOOKBACK_DAYS
            )
        )
        historical_values = []
        for j in range(
            index + 1
        ):
            historical_date = pd.Timestamp(
                date_values[j]
            )
            if pd.isna(
                historical_date
            ):
                continue
            if (
                historical_date
                < lookback_start
            ):
                continue
            value = values[j]
            if np.isfinite(value):
                historical_values.append(
                    value
                )
        if not historical_values:
            continue
        local_minimum = min(
            historical_values
        )
        local_maximum = max(
            historical_values
        )
        if is_peak:
            prominence = (
                current
                - local_minimum
            )
            event_type = "peak"
        else:
            prominence = (
                local_maximum
                - current
            )
            event_type = "trough"
        if (
            prominence
            < MIN_CYCLE_PROMINENCE
        ):
            continue
        if events:
            previous_event_date = pd.Timestamp(
                events[-1][
                    "snapshot_date"
                ]
            )
            days_since_previous = (
                current_date
                - previous_event_date
            ).days
            if (
                days_since_previous
                < MIN_DAYS_BETWEEN_CYCLE_EVENTS
            ):
                continue
        events.append(
            {
                "snapshot_date": (
                    current_date.strftime(
                        "%Y-%m-%d"
                    )
                ),
                "event_type": event_type,
                "short_interest_pct": float(
                    current
                ),
                "prominence_pct": float(
                    prominence
                ),
            }
        )
    return events
def detect_short_cycles(
    features: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Identifiera blankningscykler per bolag.
    Returnerar:
      1. Alla identifierade toppar/dalar.
      2. Summering per bolag.
    """
    cycle_rows: list[
        dict[str, Any]
    ] = []
    summary_rows: list[
        dict[str, Any]
    ] = []
    grouped = (
        features
        .sort_values(
            [
                "security_key",
                "snapshot_date",
            ]
        )
        .groupby(
            "security_key",
            sort=False,
        )
    )
    for security_key, group in grouped:
        group = group.reset_index(
            drop=True
        )
        events = detect_local_extrema(
            group["short_interest_pct"],
            group["snapshot_date"],
        )
        if not events:
            continue
        for event in events:
            cycle_rows.append(
                {
                    "security_key": security_key,
                    **event,
                }
            )
        peaks = sum(
            event["event_type"]
            == "peak"
            for event in events
        )
        troughs = sum(
            event["event_type"]
            == "trough"
            for event in events
        )
        summary_rows.append(
            {
                "security_key": security_key,
                "cycle_events": len(
                    events
                ),
                "peaks": peaks,
                "troughs": troughs,
                "first_event": min(
                    event[
                        "snapshot_date"
                    ]
                    for event in events
                ),
                "last_event": max(
                    event[
                        "snapshot_date"
                    ]
                    for event in events
                ),
            }
        )
    cycles = pd.DataFrame(
        cycle_rows
    )
    summary = pd.DataFrame(
        summary_rows
    )
    if not cycles.empty:
        cycles = (
            cycles
            .sort_values(
                [
                    "security_key",
                    "snapshot_date",
                ]
            )
            .reset_index(
                drop=True
            )
        )
    if not summary.empty:
        summary = (
            summary
            .sort_values(
                [
                    "cycle_events",
                    "security_key",
                ],
                ascending=[
                    False,
                    True,
                ],
            )
            .reset_index(
                drop=True
            )
        )
    return cycles, summary
def write_jsonl(
    frame: pd.DataFrame,
    path: Path,
) -> None:
    """
    Skriv DataFrame som JSONL.
    """
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        for record in frame.to_dict(
            orient="records"
        ):
            cleaned = {
                key: _json_value(value)
                for key, value in record.items()
            }
            handle.write(
                json.dumps(
                    cleaned,
                    ensure_ascii=False,
                )
                + "\n"
            )
def write_summary(
    summary: pd.DataFrame,
    path: Path,
) -> None:
    """
    Skriv cykelsummering som JSON.
    """
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    records = []
    for record in summary.to_dict(
        orient="records"
    ):
        records.append(
            {
                key: _json_value(value)
                for key, value in record.items()
            }
        )
    payload = {
        "market_symbol": MARKET_SYMBOL,
        "market_source": MARKET_SOURCE,
        "market_source_series": (
            MARKET_SOURCE_SERIES
        ),
        "min_cycle_prominence_pct": (
            MIN_CYCLE_PROMINENCE
        ),
        "min_days_between_events": (
            MIN_DAYS_BETWEEN_CYCLE_EVENTS
        ),
        "cycle_lookback_days": (
            CYCLE_LOOKBACK_DAYS
        ),
        "companies_with_cycles": len(
            records
        ),
        "companies": records,
    }
    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            payload,
            handle,
            ensure_ascii=False,
            indent=2,
        )
def get_required_market_interval(
    features: pd.DataFrame,
) -> tuple[
    pd.Timestamp,
    pd.Timestamp,
]:
    """
    Bestäm vilket marknadsintervall som krävs
    av featurefilen.
    """
    price_dates = features[
        "price_date"
    ].dropna()
    if price_dates.empty:
        raise RuntimeError(
            "Features innehåller inga "
            "giltiga price_date."
        )
    return (
        pd.Timestamp(
            price_dates.min()
        ).normalize(),
        pd.Timestamp(
            price_dates.max()
        ).normalize(),
    )
def market_covers_required_interval(
    market: pd.DataFrame,
    required_start: pd.Timestamp,
    required_end: pd.Timestamp,
) -> bool:
    """
    Kontrollera om lokal marknadsdata täcker
    hela det intervall som krävs.
    Vi använder datumgränser snarare än antal
    kalenderdagar eftersom marknadsserien bara
    innehåller handelsdagar.
    """
    if market.empty:
        return False
    market_start = pd.Timestamp(
        market["market_date"].min()
    ).normalize()
    market_end = pd.Timestamp(
        market["market_date"].max()
    ).normalize()
    return (
        market_start <= required_start
        and market_end >= required_end
    )
def update_market_data(
    existing: pd.DataFrame,
    required_start: pd.Timestamp,
    required_end: pd.Timestamp,
) -> pd.DataFrame:
    """
    Uppdatera lokal OMXSPI-data.
    Första körningen:
        hämta hela det historiska intervallet.
    Senare körningar:
        hämta från strax före senaste lokala
        observation till required_end.
    Om extern källa misslyckas och lokal data
    redan finns, fortsätter vi med lokal data.
    """
    if existing.empty:
        print(
            "Ingen lokal OMXSPI-historik "
            "finns ännu."
        )
        try:
            downloaded = download_market_data(
                required_start,
                required_end,
            )
        except RuntimeError as error:
            raise RuntimeError(
                "Kunde inte bygga OMXSPI-historik "
                "och ingen lokal marknadsdata finns."
            ) from error
        return merge_and_persist_market_data(
            existing,
            downloaded,
        )
    print(
        "Lokal OMXSPI-historik: "
        f"{existing['market_date'].min().date()}"
        " -> "
        f"{existing['market_date'].max().date()}"
    )
    if market_covers_required_interval(
        existing,
        required_start,
        required_end,
    ):
        print(
            "Lokal OMXSPI-historik täcker "
            "hela det nödvändiga intervallet."
        )
        print(
            "Ingen extern hämtning behövs."
        )
        return existing
    existing_start = pd.Timestamp(
        existing["market_date"].min()
    ).normalize()
    existing_end = pd.Timestamp(
        existing["market_date"].max()
    ).normalize()
    download_start = min(
        required_start,
        existing_end - pd.Timedelta(
            days=10
        ),
    )
    download_end = max(
        required_end,
        existing_end,
    )
    print(
        "Lokal historik behöver kompletteras."
    )
    print(
        "Kompletterar: "
        f"{download_start.date()}"
        " -> "
        f"{download_end.date()}"
    )
    try:
        downloaded = download_market_data(
            download_start,
            download_end,
        )
    except RuntimeError as error:
        print(
            "VARNING: Kunde inte uppdatera "
            "OMXSPI från FRED."
        )
        print(
            f"Orsak: {error}"
        )
        print(
            "Fortsätter med lokal "
            "OMXSPI-historik."
        )
        if market_covers_required_interval(
            existing,
            required_start,
            required_end,
        ):
            return existing
        raise RuntimeError(
            "FRED kunde inte uppdatera "
            "marknadsdata och den lokala "
            "historiken täcker inte hela "
            "det nödvändiga intervallet."
        ) from error
    return merge_and_persist_market_data(
        existing,
        downloaded,
    )
def main() -> None:
    print(
        "=========================================="
    )
    print(
        "BLANKDISS MARKET & SHORT CYCLE ANALYSIS"
    )
    print(
        "=========================================="
    )
    features = load_features()
    print(
        "Feature-rader: "
        f"{len(features):,}"
    )
    print(
        "Datum: "
        f"{features['snapshot_date'].min().date()}"
        " -> "
        f"{features['snapshot_date'].max().date()}"
    )
    required_start, required_end = (
        get_required_market_interval(
            features
        )
    )
    print(
        "Krävd marknadsperiod: "
        f"{required_start.date()}"
        " -> "
        f"{required_end.date()}"
    )
    existing_market = (
        load_raw_market_data()
    )
    market = update_market_data(
        existing_market,
        required_start,
        required_end,
    )
    market = add_market_forward_returns(
        market
    )
    aligned = align_market_returns(
        features,
        market,
    )
    market_adjusted = (
        build_market_adjusted_returns(
            aligned
        )
    )
    cycles, cycle_summary = (
        detect_short_cycles(
            features
        )
    )
    write_jsonl(
        market_adjusted,
        MARKET_ADJUSTED_PATH,
    )
    write_jsonl(
        cycles,
        SHORT_CYCLES_PATH,
    )
    write_summary(
        cycle_summary,
        SHORT_CYCLES_SUMMARY_PATH,
    )
    companies_with_cycles = len(
        cycle_summary
    )
    print(
        "=========================================="
    )
    print(
        "MARKET ANALYS KLAR"
    )
    print(
        "=========================================="
    )
    print(
        "Market-adjusted rows: "
        f"{len(market_adjusted):,}"
    )
    print(
        "Cycle events: "
        f"{len(cycles):,}"
    )
    print(
        "Bolag med cykler: "
        f"{companies_with_cycles:,}"
    )
    print(
        "OMXSPI-period: "
        f"{market['market_date'].min().date()}"
        " -> "
        f"{market['market_date'].max().date()}"
    )
    print(
        "OMXSPI-rader: "
        f"{len(market):,}"
    )
    print(
        "Output:"
    )
    print(
        f"  {MARKET_RAW_PATH}"
    )
    print(
        f"  {MARKET_ADJUSTED_PATH}"
    )
    print(
        f"  {SHORT_CYCLES_PATH}"
    )
    print(
        f"  {SHORT_CYCLES_SUMMARY_PATH}"
    )
if __name__ == "__main__":
    main()
