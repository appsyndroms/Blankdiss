"""HTTP-klient för Finansinspektionens blankningsregister."""
from __future__ import annotations
import time
from urllib.parse import urljoin
import requests
from .config import (
    FI_AGGREGATE_TIMEOUT,
    FI_AGGREGATE_URL,
    FI_URL,
    HEADERS,
)
from .errors import FIError
FI_RETRIES = 3
FI_RETRY_DELAY = 10
def _get_with_retry(
    url: str,
    *,
    timeout: int,
) -> requests.Response:
    """
    Hämtar en URL med retry vid tillfälliga fel.
    HTTP 5xx och tillfälliga nätverksfel retryas.
    HTTP 4xx returneras direkt eftersom ett nytt försök
    normalt inte löser ett klientfel.
    """
    for attempt in range(1, FI_RETRIES + 1):
        try:
            response = requests.get(
                url,
                headers=HEADERS,
                timeout=timeout,
            )
        except requests.RequestException as exc:
            if attempt == FI_RETRIES:
                raise FIError(
                    "HTTP-fel vid hämtning från FI: "
                    f"{type(exc).__name__}"
                ) from exc
            print(
                "WARN: HTTP-fel mot FI vid försök "
                f"{attempt}/{FI_RETRIES}: "
                f"{type(exc).__name__}. "
                f"Försöker igen om "
                f"{FI_RETRY_DELAY} sekunder."
            )
            time.sleep(FI_RETRY_DELAY)
            continue
        if 500 <= response.status_code < 600:
            if attempt == FI_RETRIES:
                return response
            print(
                "WARN: FI svarade med HTTP "
                f"{response.status_code} vid försök "
                f"{attempt}/{FI_RETRIES}. "
                f"Försöker igen om "
                f"{FI_RETRY_DELAY} sekunder."
            )
            time.sleep(FI_RETRY_DELAY)
            continue
        return response
    raise RuntimeError(
        "FI: oväntat slut på retry-loop."
    )
def fetch_html() -> str:
    """Hämtar FI:s blankningsregister."""
    response = _get_with_retry(
        FI_URL,
        timeout=30,
    )
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
    """Hämtar en fil från FI."""
    response = _get_with_retry(
        url,
        timeout=60,
    )
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
    """Hämtar FI:s aggregerade blankningsfil."""
    response = _get_with_retry(
        FI_AGGREGATE_URL,
        timeout=FI_AGGREGATE_TIMEOUT,
    )
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
