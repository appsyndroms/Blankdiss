"""HTTP-klient för Finansinspektionens blankningsregister."""

from __future__ import annotations

from urllib.parse import urljoin

import requests

from .config import FI_URL, HEADERS
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
    """Hämtar en fil från FI."""

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
    else:
        extension = ".xlsx"

    return data, extension


def absolute_url(href: str) -> str:
    """Gör en relativ FI-länk absolut."""

    return urljoin(FI_URL, href)
