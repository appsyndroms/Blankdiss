"""Diagnostik för FI:s blankningsregister.
Hämtar samma HTML som den vanliga FI-klienten och skriver ut
länkar, script-resurser och andra referenser som kan avslöja
backend-/filresurser för historiska aggregate-data.
Kör med:
    python -m fi.diagnose
"""
from __future__ import annotations
import re
from html.parser import HTMLParser
from urllib.parse import urljoin
from .client import fetch_html
from .config import FI_URL
from .errors import FIError
INTERESTING_TERMS = (
    "aggregate",
    "aggreger",
    "blank",
    "position",
    "contentassets",
    "api",
    "json",
    "xlsx",
    "xls",
    "csv",
)
class ResourceParser(HTMLParser):
    """Samlar URL-referenser från HTML."""
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []
        self.scripts: list[str] = []
        self.raw_attributes: list[tuple[str, str]] = []
    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        for name, value in attrs:
            if not value:
                continue
            lowered_name = name.lower()
            if lowered_name in {
                "href",
                "src",
                "data-url",
                "data-href",
                "data-endpoint",
                "data-api",
            }:
                absolute = urljoin(FI_URL, value)
                self.raw_attributes.append((name, absolute))
                if lowered_name == "href":
                    self.links.append(absolute)
                if tag.lower() == "script" and lowered_name == "src":
                    self.scripts.append(absolute)
def is_interesting(value: str) -> bool:
    lowered = value.lower()
    return any(term in lowered for term in INTERESTING_TERMS)
def print_section(
    title: str,
    values: list[str],
) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)
    unique = sorted(set(values))
    if not unique:
        print("(inga träffar)")
        return
    for value in unique:
        print(value)
def main() -> int:
    try:
        html = fetch_html()
    except FIError as exc:
        print(f"FI: FEL: {exc}")
        return 1
    parser = ResourceParser()
    parser.feed(html)
    all_urls = [value for _, value in parser.raw_attributes]
    interesting_urls = [
        value
        for value in all_urls
        if is_interesting(value)
    ]
    print("FI-DIAGNOSTIK")
    print(f"URL: {FI_URL}")
    print(f"HTML-bytes: {len(html.encode('utf-8'))}")
    print(f"Alla URL-attribut: {len(all_urls)}")
    print(f"Script-resurser: {len(parser.scripts)}")
    print_section(
        "INTRESSANTA URL-REFERENSER",
        interesting_urls,
    )
    print_section(
        "SCRIPT-RESURSER",
        parser.scripts,
    )
    print_section(
        "ALLA HREF",
        parser.links,
    )
    print()
    print("=" * 72)
    print("INTRESSANTA TEXTREFERENSER")
    print("=" * 72)
    text_matches = sorted(
        {
            line.strip()
            for line in html.splitlines()
            if is_interesting(line)
        }
    )
    if text_matches:
        for line in text_matches:
            print(line[:1000])
    else:
        print("(inga träffar)")
    # Hjälper oss även hitta hårdkodade URL:er i inline-JavaScript.
    print()
    print("=" * 72)
    print("URL-LIKNANDE STRÄNGAR I HTML")
    print("=" * 72)
    urls_in_html = sorted(
        set(
            re.findall(
                r"https?://[^\"'<>\\s]+",
                html,
            )
        )
    )
    for value in urls_in_html:
        if is_interesting(value):
            print(value)
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
