from __future__ import annotations

import re
from html.parser import HTMLParser
from urllib.parse import parse_qs, urljoin, urlparse

import requests

from fi.client import fetch_html
from fi.config import FI_URL, HEADERS
from fi.errors import FIError


MAX_ISSUER_PAGES = 3
MAX_HOLDER_PAGES = 3
MAX_EXTERNAL_SCRIPTS = 5
REQUEST_TIMEOUT = 30


REGISTER_RE = re.compile(
    r"/(?:blankningsregistret|BlankningsRegister)/",
    re.IGNORECASE,
)

INTERESTING_PATH_RE = re.compile(
    r"(?:blankning|position|emittent|issuer|histor|aggregat|aggregate)",
    re.IGNORECASE,
)

INTERESTING_TERM_RE = re.compile(
    r"(?:"
    r"positionsinnehavare|"
    r"emittent|"
    r"issuer|"
    r"blankning|"
    r"histor|"
    r"aggregat|"
    r"aggregate|"
    r"summa procent|"
    r"positionsdatum|"
    r"positiondate|"
    r"isin|"
    r"lei"
    r")",
    re.IGNORECASE,
)

URL_RE = re.compile(
    r"""
    (?:
        https?://[^"' <>\s]+
        |
        /[^"' <>\s]+
        |
        (?:Positionsinnehavare|emittent)\?[^"' <>\s]+
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

DATE_RE = re.compile(r"\b20\d{2}-\d{2}-\d{2}\b")

PARAM_RE = re.compile(
    r"""
    (?:
        [?&]([A-Za-z][A-Za-z0-9_-]*)=
        |
        \b(name|id|lei|isin|date|datum|positionDate|positionsdatum)
        \s*=
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)


class ResourceParser(HTMLParser):
    """Samlar endast länkar, script, formulär och kontroller som behövs."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)

        self.links: list[str] = []
        self.scripts: list[str] = []
        self.inline_scripts: list[str] = []

        self.forms: list[dict[str, str]] = []
        self.controls: list[dict[str, str]] = []

        self._current_script: list[str] | None = None
        self._current_form: dict[str, str] | None = None

    def handle_starttag(self, tag: str, attrs) -> None:
        attrs_dict = {
            str(key): str(value)
            for key, value in attrs
            if value is not None
        }

        tag = tag.lower()

        if tag == "a":
            href = attrs_dict.get("href")

            if href:
                self.links.append(href)

        elif tag == "script":
            src = attrs_dict.get("src")

            if src:
                self.scripts.append(src)
            else:
                self._current_script = []

        elif tag == "form":
            self._current_form = {
                "action": attrs_dict.get("action", ""),
                "method": attrs_dict.get("method", "get"),
                "id": attrs_dict.get("id", ""),
                "name": attrs_dict.get("name", ""),
            }

        elif tag in {
            "input",
            "select",
            "option",
            "button",
            "textarea",
        }:
            control = {
                "tag": tag,
                "type": attrs_dict.get("type", ""),
                "name": attrs_dict.get("name", ""),
                "value": attrs_dict.get("value", ""),
                "id": attrs_dict.get("id", ""),
            }

            if self._current_form:
                control["form_action"] = self._current_form.get(
                    "action",
                    "",
                )

            values = " ".join(control.values())

            if INTERESTING_TERM_RE.search(values):
                self.controls.append(control)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()

        if tag == "script" and self._current_script is not None:
            self.inline_scripts.append(
                "".join(self._current_script)
            )
            self._current_script = None

        elif tag == "form" and self._current_form is not None:
            self.forms.append(self._current_form)
            self._current_form = None

    def handle_data(self, data: str) -> None:
        if self._current_script is not None:
            self._current_script.append(data)


def unique(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()

    for value in values:
        value = value.strip()

        if not value or value in seen:
            continue

        seen.add(value)
        result.append(value)

    return result


def normalize_url(value: str, base_url: str) -> str:
    value = value.strip()

    if not value:
        return ""

    if value.lower().startswith("javascript:"):
        return ""

    return urljoin(base_url, value)


def is_register_url(url: str) -> bool:
    return bool(REGISTER_RE.search(url))


def extract_urls(
    text: str,
    base_url: str,
) -> list[str]:
    urls: list[str] = []

    for match in URL_RE.findall(text):
        url = normalize_url(match, base_url)

        if
