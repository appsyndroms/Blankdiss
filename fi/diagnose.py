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

        if url and is_register_url(url):
            urls.append(url)

    return unique(urls)


def extract_interesting_endpoints(
    text: str,
    base_url: str,
) -> list[str]:
    endpoints: list[str] = []

    for match in URL_RE.findall(text):
        url = normalize_url(match, base_url)

        if not url:
            continue

        parsed = urlparse(url)

        if INTERESTING_PATH_RE.search(parsed.path):
            endpoints.append(url)

    return unique(endpoints)


def extract_parameters(text: str) -> list[str]:
    parameters: list[str] = []

    for match in PARAM_RE.finditer(text):
        for value in match.groups():
            if value:
                parameters.append(value)

    return unique(parameters)


def extract_dates(text: str) -> list[str]:
    return unique(DATE_RE.findall(text))


def relevant_links(
    links: list[str],
    base_url: str,
) -> list[str]:
    result: list[str] = []

    for link in links:
        url = normalize_url(link, base_url)

        if not url:
            continue

        parsed = urlparse(url)
        path = parsed.path.lower()

        if (
            "blankningsregistret" in path
            or "positionsinnehavare" in path
            or "/emittent" in path
        ):
            result.append(url)

    return unique(result)


def issuer_links(
    links: list[str],
    base_url: str,
) -> list[str]:
    result: list[str] = []

    for link in links:
        url = normalize_url(link, base_url)

        if not url:
            continue

        parsed = urlparse(url)

        if "/emittent" not in parsed.path.lower():
            continue

        query = parse_qs(parsed.query)

        if "id" in query:
            result.append(url)

    return unique(result)


def holder_links(
    links: list[str],
    base_url: str,
) -> list[str]:
    result: list[str] = []

    for link in links:
        url = normalize_url(link, base_url)

        if not url:
            continue

        parsed = urlparse(url)

        if "positionsinnehavare" not in parsed.path.lower():
            continue

        query = parse_qs(parsed.query)

        if "id" in query:
            result.append(url)

    return unique(result)


def print_endpoints(endpoints: list[str]) -> None:
    for endpoint in endpoints:
        print(f"  {endpoint}")


def inspect_page(
    url: str,
    label: str,
) -> tuple[list[str], list[str]]:
    print()
    print(f"--- {label} ---")
    print(url)

    try:
        response = requests.get(
            url,
            headers=HEADERS,
            timeout=REQUEST_TIMEOUT,
        )
    except requests.RequestException as exc:
        print(f"HTTP-FEL: {exc}")
        return [], []

    print(
        f"HTTP {response.status_code} | "
        f"{len(response.content)} bytes"
    )

    if response.status_code != 200:
        return [], []

    html = response.text

    parser = ResourceParser()

    try:
        parser.feed(html)
    except Exception as exc:
        print(f"PARSER-FEL: {exc}")
        return [], []

    endpoints = extract_interesting_endpoints(
        html,
        url,
    )

    parameters = extract_parameters(html)
    dates = extract_dates(html)

    if endpoints:
        print("Endpoints:")
        print_endpoints(endpoints[:30])

    if parameters:
        print(
            "Parametrar:",
            ", ".join(parameters[:30]),
        )

    if dates:
        print(
            "Datum:",
            ", ".join(dates[:20]),
        )

    forms = [
        form
        for form in parser.forms
        if INTERESTING_TERM_RE.search(
            " ".join(form.values())
        )
    ]

    if forms:
        print("Relevanta formulär:")

        for form in forms[:10]:
            print(
                f"  action={form['action']} "
                f"method={form['method']} "
                f"id={form['id']}"
            )

    if parser.controls:
        print("Relevanta kontroller:")

        for control in parser.controls[:20]:
            print(f"  {control}")

    holders = holder_links(
        parser.links,
        url,
    )

    issuers = issuer_links(
        parser.links,
        url,
    )

    if issuers:
        print(
            f"Emittentlänkar: {len(issuers)}"
        )

    if holders:
        print(
            f"Positionsinnehavare: {len(holders)}"
        )

        for holder in holders[:10]:
            print(f"  {holder}")

    return relevant_links(parser.links, url), holders


def inspect_inline_scripts(
    scripts: list[str],
    base_url: str,
) -> None:
    found: set[str] = set()

    for script in scripts:
        endpoints = extract_interesting_endpoints(
            script,
            base_url,
        )

        found.update(endpoints)

    if found:
        print()
        print("--- ENDPOINTS I INLINE-JS ---")
        print_endpoints(sorted(found))


def inspect_external_scripts(
    scripts: list[str],
    base_url: str,
) -> None:
    inspected = 0
    found_endpoints: set[str] = set()

    for script in scripts:
        if inspected >= MAX_EXTERNAL_SCRIPTS:
            break

        script_url = normalize_url(
            script,
            base_url,
        )

        if not script_url:
            continue

        try:
            response = requests.get(
                script_url,
                headers=HEADERS,
                timeout=REQUEST_TIMEOUT,
            )
        except requests.RequestException:
            continue

        if response.status_code != 200:
            continue

        text = response.text

        endpoints = extract_interesting_endpoints(
            text,
            script_url,
        )

        if not endpoints:
            continue

        inspected += 1

        print()
        print("--- EXTERNT SCRIPT MED RELEVANT TRÄFF ---")
        print(script_url)

        print_endpoints(endpoints[:30])

        found_endpoints.update(endpoints)

    if found_endpoints:
        print()
        print("--- UNIKA SCRIPT-ENDPOINTS ---")
        print_endpoints(sorted(found_endpoints))


def inspect_known_aggregate_endpoint() -> None:
    endpoint = urljoin(
        FI_URL,
        "/BlankningsRegister/GetBlankningsregisterAggregat",
    )

    print()
    print("--- KÄNT AGGREGAT-ENDPOINT ---")
    print(endpoint)

    try:
        response = requests.get(
            endpoint,
            headers=HEADERS,
            timeout=REQUEST_TIMEOUT,
        )
    except requests.RequestException as exc:
        print(f"HTTP-FEL: {exc}")
        return

    print(
        f"HTTP {response.status_code} | "
        f"{len(response.content)} bytes | "
        f"{response.headers.get('content-type', '')}"
    )


def inspect_date_parameters() -> None:
    endpoint = urljoin(
        FI_URL,
        "/BlankningsRegister/GetBlankningsregisterAggregat",
    )

    parameters = (
        ("date", "2026-08-01"),
        ("datum", "2026-08-01"),
        ("positionDate", "2026-08-01"),
        ("positionsdatum", "2026-08-01"),
    )

    print()
    print("--- DATUMPARAMETRAR PÅ AGGREGAT-ENDPOINT ---")

    try:
        baseline = requests.get(
            endpoint,
            headers=HEADERS,
            timeout=REQUEST_TIMEOUT,
        )
    except requests.RequestException as exc:
        print(f"HTTP-FEL: {exc}")
        return

    baseline_length = len(baseline.content)

    for name, value in parameters:
        try:
            response = requests.get(
                endpoint,
                params={name: value},
                headers=HEADERS,
                timeout=REQUEST_TIMEOUT,
            )
        except requests.RequestException as exc:
            print(
                f"  {name}: HTTP-FEL: {exc}"
            )
            continue

        changed = (
            len(response.content)
            != baseline_length
        )

        print(
            f"  {name}={value} -> "
            f"HTTP {response.status_code}, "
            f"{len(response.content)} bytes, "
            f"ändrad={changed}"
        )


def main() -> int:
    print("FI BACKEND-DIAGNOSTIK")
    print("=" * 80)
    print(f"FI_URL: {FI_URL}")

    try:
        html = fetch_html()
    except FIError as exc:
        print(f"FI-FEL: {exc}")
        return 1
    except Exception as exc:
        print(
            f"FEL VID HÄMTNING AV FI-SIDAN: {exc}"
        )
        return 1

    parser = ResourceParser()

    try:
        parser.feed(html)
    except Exception as exc:
        print(f"PARSER-FEL: {exc}")
        return 1

    print()
    print("--- HUVUDSIDAN ---")

    register_urls = extract_urls(
        html,
        FI_URL,
    )

    endpoints = extract_interesting_endpoints(
        html,
        FI_URL,
    )

    parameters = extract_parameters(html)

    if register_urls:
        print("Register-URL:er:")
        print_endpoints(register_urls[:30])

    if endpoints:
        print("Endpoints:")
        print_endpoints(endpoints[:30])

    if parameters:
        print(
            "Parametrar:",
            ", ".join(parameters[:30]),
        )

    inspect_inline_scripts(
        parser.inline_scripts,
        FI_URL,
    )

    issuer_pages = issuer_links(
        parser.links,
        FI_URL,
    )[:MAX_ISSUER_PAGES]

    if issuer_pages:
        print()
        print("--- EMITTENTER ATT UNDERSÖKA ---")

        for issuer in issuer_pages:
            print(f"  {issuer}")

    all_holder_pages: list[str] = []

    for issuer in issuer_pages:
        _, holders = inspect_page(
            issuer,
            "EMITTENT",
        )

        all_holder_pages.extend(holders)

        all_holder_pages = unique(
            all_holder_pages
        )[:MAX_HOLDER_PAGES]

        if len(all_holder_pages) >= MAX_HOLDER_PAGES:
            break

    if all_holder_pages:
        print()
        print("--- FÖLJER POSITIONSINNEHAVARE ---")

        for holder in all_holder_pages:
            print(f"  {holder}")

        for holder in all_holder_pages:
            inspect_page(
                holder,
                "POSITIONSINNEHAVARE",
            )

    inspect_external_scripts(
        parser.scripts,
        FI_URL,
    )

    inspect_known_aggregate_endpoint()
    inspect_date_parameters()

    print()
    print("=" * 80)
    print("DIAGNOSTIK KLAR")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
