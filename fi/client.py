"""HTTP-klient för Finansinspektionens blankningsregister."""
from __future__ import annotations
from urllib.parse import urljoin
import requests
from .config import (
    FI_AGGREGATE_TIMEOUT,
    FI_AGGREGATE_URL,
    FI_URL,
    HEADERS,
)
from .errors import FIError
def fetch_html() -> str:
    """Hämtar FI:s blankningsregister."""
    try:
        response = requests.get(
            FI_URL,
            headers=HEADERS,
            timeout=30,
        )
    except requests.RequestException as exc:
        raise FIError(
            "HTTP-fel vid hämtning från FI: "
            f"{type(exc).__name__}"
        ) from exc
    if response.status_code != 200:
        raise FIError(
            "FI svarade med HTTP "
            f"{response.status_code}."
        )
    if not response.text.strip():
        raise FIError(
            "FI returnerade ett tomt HTML-svar."
        )
    return response.text
def download_file(
    url: str,
) -> tuple[bytes, str]:
    """
    Hämtar en fil från FI.
    Behålls som generell klientfunktion.
    """
    try:
        response = requests.get(
            url,
            headers=HEADERS,
            timeout=60,
        )
    except requests.RequestException as exc:
        raise FIError(
            "HTTP-fel vid hämtning av FI-fil: "
            f"{type(exc).__name__}"
        ) from exc
    if response.status_code != 200:
        raise FIError(
            "FI-fil svarade med HTTP "
            f"{response.status_code}: {url}"
        )
    data = response.content
    if not data:
        raise FIError(
            f"FI-filen var tom: {url}"
        )
    lowered = url.lower()
    if lowered.endswith(".xls"):
        extension = ".xls"
    elif lowered.endswith(".xlsx"):
        extension = ".xlsx"
    else:
        extension = ".ods"
    return data, extension
def fetch_aggregate() -> bytes:
    """
    Hämtar FI:s aggregerade blankningsfil.
    Detta är FI:s riktiga aggregatkälla och innehåller
    den aggregerade korta nettopositionen över 0,1 %.
    """
    try:
        response = requests.get(
            FI_AGGREGATE_URL,
            headers=HEADERS,
            timeout=FI_AGGREGATE_TIMEOUT,
        )
    except requests.RequestException as exc:
        raise FIError(
            "HTTP-fel vid hämtning av FI:s "
            "aggregerade blankningsfil: "
            f"{type(exc).__name__}"
        ) from exc
    if response.status_code != 200:
        raise FIError(
            "FI:s aggregat-endpoint svarade med HTTP "
            f"{response.status_code}."
        )
    data = response.content
    if not data:
        raise FIError(
            "FI:s aggregat-endpoint returnerade "
            "en tom fil."
        )
    return data
def absolute_url(href: str) -> str:
    """Gör en relativ FI-länk absolut."""
    return urljoin(
        FI_URL,
        href,
    )
