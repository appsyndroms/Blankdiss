"""
Marknadsrelativ avkastning och cykliska blankningsmönster.

Detta analyssteg gör två saker:

1. Jämför aktiernas framtida avkastning med OMXSPI.
2. Identifierar återkommande lokala toppar och dalar i
   short interest för enskilda bolag.

Marknadsdata hämtas direkt från Nasdaq GIW:s
Equities Index Level History Service.

Nasdaq-endpoint:
    https://indexes.nasdaqomx.com/reports2/history.ashx

Marknadsdata persisteras lokalt som JSONL.

Arkitektur:

    Nasdaq GIW / OMXSPI
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

Nasdaq GIW History Service enligt specifikationen:
    IndexSymbol
    StartDate
    EndDate
    Type
    FileType

Vi använder:
    IndexSymbol = OMXSPI
    Type = CSV
    FileType = EOD

Parsern accepterar semikolon, komma och pipe eftersom Nasdaq
har använt olika CSV-separatorer i olika GIW-specifikationer.
"""

from __future__ import annotations

import csv
import io
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

MARKET_SOURCE = "NASDAQ"

MARKET_SOURCE_SERIES = "OMXSPI"

NASDAQ_HISTORY_URL = (
    "https://indexes.nasdaqomx.com/"
    "reports2/history.ashx"
)

NASDAQ_HEADERS = {
    "User-Agent": (
        "Blankdiss/1.0 "
        "(market analysis; "
        "OMXSPI)"
    ),
    "Accept": (
        "text/csv,text/plain,"
        "application/octet-stream,"
        "*/*"
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

    Nasdaq levererar normalt indexvärden med punkt som
    decimaltecken, men parsern är medvetet tolerant.
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


def _looks_like_html(
    text: str,
) -> bool:
    """
    Kontrollera om svaret uppenbart är HTML.

    Detta är viktigt eftersom Nasdaq tidigare har
    returnerat HTTP 200 tillsammans med en HTML-felsida.
    """

    preview = text[:5000].lower()

    html_markers = (
        "<html",
        "<!doctype",
        "<head",
        "<body",
        "<title",
        "<form",
        "<script",
    )

    return any(
        marker in preview
        for marker in html_markers
    )


def _detect_delimiter(
    text: str,
) -> str:
    """
    Detektera Nasdaq-svarets separator.

    Nasdaq GIW har dokumenterats med både komma,
    semikolon och pipe i olika versioner.

    Prioriteringsordning:
        ;
        ,
        |

    Om csv.Sniffer lyckas används dess resultat.
    """

    sample_lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    if not sample_lines:
        raise RuntimeError(
            "Nasdaq-svaret innehåller inga rader."
        )

    sample = "\n".join(
        sample_lines[:10]
    )

    try:
        dialect = csv.Sniffer().sniff(
            sample,
            delimiters=";,|",
        )

        if dialect.delimiter in (
            ";",
            ",",
            "|",
        ):
            return dialect.delimiter

    except csv.Error:
        pass

    counts = {
        ";": sample.count(";"),
        ",": sample.count(","),
        "|": sample.count("|"),
    }

    delimiter = max(
        counts,
        key=counts.get,
    )

    if counts[delimiter] == 0:
        raise RuntimeError(
            "Kunde inte identifiera separator "
            "i Nasdaq-svaret."
        )

    return delimiter


def _normalize_column_name(
    value: Any,
) -> str:
    """
    Normalisera kolumnnamn för tolerant parsing.
    """

    return (
        str(value)
        .strip()
        .lower()
        .replace(
            "\ufeff",
            "",
        )
        .replace(
            " ",
            "",
        )
        .replace(
            "_",
            "",
        )
    )


def _find_column(
    columns: list[Any],
    candidates: tuple[str, ...],
) -> Any | None:
    """
    Hitta en kolumn genom normaliserat namn.
    """

    normalized = {
        _normalize_column_name(column): column
        for column in columns
    }

    for candidate in candidates:
        found = normalized.get(
            _normalize_column_name(
                candidate
            )
        )

        if found is not None:
            return found

    return None


def _parse_nasdaq_date(
    value: Any,
) -> pd.Timestamp:
    """
    Tolka Nasdaq Trade Date.

    Nasdaq GIW anger YYYYMMDD i aktuell
    History Service-specifikation.

    Vi accepterar även:
        YYYY-MM-DD
        YYYY/MM/DD
    """

    if value is None:
        return pd.NaT

    text = str(value).strip()

    if not text:
        return pd.NaT

    text = (
        text
        .replace(
            "\ufeff",
            "",
        )
        .strip()
    )

    for fmt in (
        "%Y%m%d",
        "%Y-%m-%d",
        "%Y/%m/%d",
    ):
        try:
            return pd.Timestamp.strptime(
                text,
                fmt,
            )

        except (ValueError, TypeError):
            pass

    parsed = pd.to_datetime(
        text,
        errors="coerce",
    )

    if pd.isna(parsed):
        return pd.NaT

    return pd.Timestamp(parsed)


def _parse_nasdaq_csv(
    text: str,
) -> pd.DataFrame:
    """
    Tolka Nasdaq GIW History Service.

    Förväntad information:

        Trade Date
        Index Value
        Net Change
        High
        Low

    Vi använder endast:

        Trade Date
        Index Value

    Parsern accepterar både header och headerlösa svar.

    Den accepterar även separatorerna:

        ;
        ,
        |
    """

    if not text.strip():
        raise RuntimeError(
            "Nasdaq returnerade ett tomt svar."
        )

    if _looks_like_html(text):
        preview = text[:2000]

        raise RuntimeError(
            "Nasdaq returnerade HTML i stället "
            "för marknadsdata.\n"
            "Svar, början:\n"
            f"{preview}"
        )

    delimiter = _detect_delimiter(
        text
    )

    print(
        "Nasdaq CSV-separator: "
        f"{repr(delimiter)}"
    )

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    if not lines:
        raise RuntimeError(
            "Nasdaq-svaret innehåller inga "
            "icke-tomma rader."
        )

    reader = csv.reader(
        io.StringIO(
            "\n".join(lines)
        ),
        delimiter=delimiter,
    )

    rows = list(reader)

    if not rows:
        raise RuntimeError(
            "Nasdaq-svaret kunde inte parsas."
        )

    first_row = [
        str(value).strip()
        for value in rows[0]
    ]

    first_row_normalized = {
        _normalize_column_name(value)
        for value in first_row
    }

    has_header = (
        "tradedate"
        in first_row_normalized
        or
        "indexvalue"
        in first_row_normalized
    )

    if has_header:
        header = first_row

        data_rows = rows[1:]

        trade_date_index = None
        index_value_index = None

        for index, column in enumerate(
            header
        ):
            normalized = (
                _normalize_column_name(
                    column
                )
            )

            if normalized == "tradedate":
                trade_date_index = index

            elif normalized == "indexvalue":
                index_value_index = index

        if (
            trade_date_index is None
            or index_value_index is None
        ):
            raise RuntimeError(
                "Nasdaq-svaret innehåller en "
                "header men saknar Trade Date "
                "eller Index Value.\n"
                f"Kolumner: {header}"
            )

    else:
        """
        Headerlös fallback.

        Enligt GIW History Service är de första
        relevanta fälten:

            Trade Date
            Index Value

        Därför använder vi kolumn 0 och 1.
        """

        trade_date_index = 0
        index_value_index = 1

        data_rows = rows

    parsed_rows: list[
        dict[str, Any]
    ] = []

    invalid_rows = 0

    for row in data_rows:
        if len(row) <= max(
            trade_date_index,
            index_value_index,
        ):
            invalid_rows += 1
            continue

        raw_date = row[
            trade_date_index
        ]

        raw_value = row[
            index_value_index
        ]

        market_date = _parse_nasdaq_date(
            raw_date
        )

        market_close = _parse_number(
            raw_value
        )

        if pd.isna(
            market_date
        ):
            invalid_rows += 1
            continue

        if not np.isfinite(
            market_close
        ):
            invalid_rows += 1
            continue

        if market_close <= 0:
            invalid_rows += 1
            continue

        parsed_rows.append(
            {
                "market_date": market_date,
                "market_close": market_close,
            }
        )

    result = pd.DataFrame(
        parsed_rows
    )

    if result.empty:
        preview = "\n".join(
            lines[:20]
        )

        raise RuntimeError(
            "Nasdaq-svaret innehåller inga "
            "giltiga OMXSPI-observationer.\n"
            "Parserad separator: "
            f"{repr(delimiter)}\n"
            "Svar, början:\n"
            f"{preview}"
        )

    result = (
        result
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

    print(
        "Nasdaq-parsering:"
    )

    print(
        "  Giltiga observationer: "
        f"{len(result):,}"
    )

    print(
        "  Ogiltiga/skippade rader: "
        f"{invalid_rows:,}"
    )

    print(
        "  Datum: "
        f"{result['market_date'].min().date()}"
        " -> "
        f"{result['market_date'].max().date()}"
    )

    print(
        "  Första värde: "
        f"{result.iloc[0]['market_close']}"
    )

    print(
        "  Sista värde: "
        f"{result.iloc[-1]['market_close']}"
    )

    return result


def _validate_nasdaq_response(
    response: requests.Response,
) -> None:
    """
    Validera HTTP-svaret innan parsern körs.

    HTTP 200 är inte tillräckligt.
    Nasdaq får inte returnera HTML eller ett tomt svar.
    """

    print(
        "Nasdaq HTTP-status: "
        f"{response.status_code}"
    )

    print(
        "Nasdaq Content-Type: "
        f"{response.headers.get('Content-Type', '')}"
    )

    print(
        "Nasdaq Content-Length: "
        f"{response.headers.get('Content-Length', '')}"
    )

    try:
        response.raise_for_status()

    except requests.HTTPError as error:
        preview = response.text[:2000]

        print(
            "Nasdaq-svar, början:\n"
            f"{preview}"
        )

        raise RuntimeError(
            "Nasdaq returnerade HTTP-fel: "
            f"{response.status_code}"
        ) from error

    if not response.text.strip():
        raise RuntimeError(
            "Nasdaq returnerade HTTP 200 men "
            "svaret är tomt."
        )

    if _looks_like_html(
        response.text
    ):
        preview = response.text[:2000]

        print(
            "Nasdaq returnerade HTML trots "
            "HTTP 200."
        )

        print(
            "Svar, början:\n"
            f"{preview}"
        )

        raise RuntimeError(
            "Nasdaq returnerade HTML i stället "
            "för OMXSPI-data."
        )


def download_market_data(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    """
    Hämta OMXSPI direkt från Nasdaq GIW.

    Nasdaq GIW History Service:

        /reports2/history.ashx

    Parametrar:

        IndexSymbol = OMXSPI
        StartDate
        EndDate
        Type = CSV
        FileType = EOD

    Vi hämtar ett litet överlapp runt intervallet
    så att inkrementella uppdateringar inte riskerar
    att lämna luckor vid helger eller korrigerade
    observationer.
    """

    requested_start = (
        pd.Timestamp(start_date)
        - pd.Timedelta(days=10)
    )

    requested_end = (
        pd.Timestamp(end_date)
        + pd.Timedelta(days=10)
    )

    start_text = (
        requested_start.strftime(
            "%Y-%m-%d"
        )
    )

    end_text = (
        requested_end.strftime(
            "%Y-%m-%d"
        )
    )

    params = {
        "IndexSymbol": MARKET_SOURCE_SERIES,
        "StartDate": start_text,
        "EndDate": end_text,
        "Type": "CSV",
        "FileType": "EOD",
    }

    print(
        "Laddar marknadsdata direkt "
        "från Nasdaq GIW."
    )

    print(
        "Nasdaq URL: "
        f"{NASDAQ_HISTORY_URL}"
    )

    print(
        "Nasdaq IndexSymbol: "
        f"{MARKET_SOURCE_SERIES}"
    )

    print(
        "Nasdaq-intervall: "
        f"{start_text} -> {end_text}"
    )

    print(
        "Nasdaq Type: CSV"
    )

    print(
        "Nasdaq FileType: EOD"
    )

    try:
        response = requests.get(
            NASDAQ_HISTORY_URL,
            params=params,
            headers=NASDAQ_HEADERS,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )

    except requests.RequestException as error:
        raise RuntimeError(
            "Kunde inte hämta OMXSPI "
            "från Nasdaq GIW: "
            f"{error}"
        ) from error

    print(
        "Nasdaq faktisk URL:"
    )

    print(
        response.url
    )

    _validate_nasdaq_response(
        response
    )

    preview = response.text[:2000]

    print(
        "Nasdaq-svar, början:\n"
        f"{preview}"
    )

    market = _parse_nasdaq_csv(
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
            "Nasdaq/OMXSPI innehåller inga "
            "observationer inom det begärda "
            "intervallet."
        )

    print(
        "OMXSPI-period efter filtrering: "
        f"{market['market_date'].min().date()}"
        " -> "
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
        OMXSPI:s framtida avkastning
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
            result[
                abnormal_column
            ] = np.nan

            continue

        result[
            stock_column
        ] = frame[
            stock_column
        ]

        result[
            market_column
        ] = frame[
            market_column
        ]

        result[
            abnormal_column
        ] = (
            frame[
                stock_column
            ]
            - frame[
                market_column
            ]
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

        current = values[
            index
        ]

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
    redan täcker hela intervallet används den
    lokala datan.

    Om lokal data inte täcker intervallet
    stoppas analysen.
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
                "från Nasdaq och ingen lokal "
                "marknadsdata finns."
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

    existing_end = pd.Timestamp(
        existing["market_date"].max()
    ).normalize()

    download_start = min(
        required_start,
        existing_end
        - pd.Timedelta(
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
            "OMXSPI från Nasdaq."
        )

        print(
            f"Orsak: {error}"
        )

        if market_covers_required_interval(
            existing,
            required_start,
            required_end,
        ):

            print(
                "Lokal historik täcker ändå "
                "hela det nödvändiga intervallet."
            )

            return existing

        raise RuntimeError(
            "Nasdaq kunde inte uppdatera "
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

    print(
        "Marknadskälla: Nasdaq GIW"
    )

    print(
        "Index: OMXSPI"
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
