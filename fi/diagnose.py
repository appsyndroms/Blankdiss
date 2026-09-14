from __future__ import annotations

import re
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse

import requests

from fi.client import fetch_html
from fi.config import FI_URL, HEADERS
from fi.errors import FIError


MAX_RELEVANT_PAGES = 12
REQUEST_TIMEOUT = 30


class ResourceParser(HTMLParser):
    """Samlar länkar, script och attribut som kan innehålla backend-anrop."""

    INTERESTING_ATTRIBUTES = {
        "data-url",
        "data-href",
        "data-endpoint",
        "data-api",
        "data-ajax-url",
        "onclick",
        "href",
        "src",
    }

    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []
        self.scripts: list[str] = []
        self.inline_scripts: list[str] = []
        self.attributes: list[tuple[str, str, str]] = []
        self._in_script = False
        self._script_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        attrs_dict = dict(attrs)

        if tag.lower() == "script":
            self._in_script = True
            self._script_parts = []

            src = attrs_dict.get("src")
            if src:
                self.scripts.append(src)

        for name, value in attrs:
            if not value:
                continue

            if name.lower() in self.INTERESTING_ATTRIBUTES:
                self.attributes.append((tag, name, value))

            if tag.lower() == "a" and name.lower() == "href":
                self.links.append(value)

    def handle_data(self, data: str) -> None:
        if self._in_script:
            self._script_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "script" and self._in_script:
            script = "".join(self._script_parts).strip()
            if script:
                self.inline_scripts.append(script)

            self._in_script = False
            self._script_parts = []


def absolute_url(base_url: str, value: str) -> str:
    """Gör en URL absolut."""
    return urljoin(base_url, value)


def same_host(url: str, base_url: str) -> bool:
    """Returnerar True om URL:en ligger på samma host."""
    return urlparse(url).netloc == urlparse(base_url).netloc


def extract_endpoints(text: str) -> set[str]:
    """
    Försöker hitta backend-endpoints i HTML/JavaScript.

    Vi letar inte bara efter FI:s kända BlankningsRegister-paths,
    utan även generiska AJAX/fetch-mönster.
    """

    endpoints: set[str] = set()

    patterns = [
        # Direkta FI-paths
        r"""['"](/BlankningsRegister/
