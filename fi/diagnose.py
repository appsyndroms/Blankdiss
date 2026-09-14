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
        r"""['"](/BlankningsRegister/[^'"]+)['"]""",
        r"""[`"](/BlankningsRegister/[^`"]+)[`"]""",

        # url: "/..."
        r"""url\s*:\s*['"]([^'"]+)['"]""",

        # fetch("/...")
        r"""fetch\s*\(\s*['"]([^'"]+)['"]""",

        # $.get("/...")
        r"""\$\.(?:get|getJSON|post)\s*\(\s*['"]([^'"]+)['"]""",

        # $.ajax({ url: "/..." })
        r"""url\s*:\s*['"]([^'"]*BlankningsRegister[^'"]*)['"]""",

        # href/data-url etc
        r"""(?:href|data-url|data-href|data-endpoint|data-api|data-ajax-url)\s*=\s*['"]([^'"]+)['"]""",

        # ASP.NET/MVC-style paths
        r"""['"](/[^'"]*(?:Get|Post|Download|Report|Export|History|Historical|Aggregate|Aggregat)[^'"]*)['"]""",
    ]

    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            value = match.group(1).strip()

            if not value:
                continue

            # Undvik uppenbart irrelevanta värden.
            if value.startswith("javascript:"):
                continue

            if value.startswith("/") or value.startswith("http://") or value.startswith("https://"):
                endpoints.add(value)

    return endpoints


def extract_parameter_names(url_or_text: str) -> set[str]:
    """Hittar potentiella query-parametrar."""
    params: set[str] = set()

    # Riktiga query strings.
    parsed = urlparse(url_or_text)

    if parsed.query:
        for part in parsed.query.split("&"):
            if "=" in part:
                key = part.split("=", 1)[0].strip()
                if key:
                    params.add(key)

    # Även parametrar som förekommer i JavaScript.
    patterns = [
        r"""[?&]([A-Za-z][A-Za-z0-9_]*)=""",
        r"""(?:name|param|parameter)\s*[:=]\s*['"]([A-Za-z][A-Za-z0-9_]*)['"]""",
    ]

    for pattern in patterns:
        for match in re.finditer(pattern, url_or_text):
            params.add(match.group(1))

    return params


def print_contexts(text: str, terms: list[str], radius: int = 350) -> None:
    """Skriver relevanta kodstycken runt givna sökord."""

    lower = text.lower()

    for term in terms:
        start = 0

        while True:
            index = lower.find(term.lower(), start)

            if index == -1:
                break

            begin = max(0, index - radius)
            end = min(len(text), index + len(term) + radius)

            snippet = text[begin:end].replace("\r", " ")

            print()
            print(f"--- Kontext: {term} ---")
            print(snippet)

            start = index + len(term)


def inspect_page(
    session: requests.Session,
    url: str,
    label: str,
) -> tuple[set[str], list[str]]:
    """
    Hämtar och analyserar en FI-sida.

    Returnerar:
      - endpoints
      - externa scripts
    """

    print()
    print("=" * 80)
    print(f"SIDA: {label}")
    print(url)
    print("=" * 80)

    try:
        response = session.get(
            url,
            headers=HEADERS,
            timeout=REQUEST_TIMEOUT,
        )
    except requests.RequestException as exc:
        print(f"HTTP-fel: {exc}")
        return set(), []

    print(f"HTTP: {response.status_code}")
    print(f"Content-Type: {response.headers.get('content-type', '')}")
    print(f"Length: {len(response.content)}")

    if response.status_code != 200:
        return set(), []

    html = response.text

    parser = ResourceParser()

    try:
        parser.feed(html)
    except Exception as exc:
        print(f"HTML-parserfel: {exc}")

    endpoints = extract_endpoints(html)

    external_scripts: list[str] = []

    for script in parser.scripts:
        script_url = absolute_url(url, script)

        if same_host(script_url, FI_URL):
            external_scripts.append(script_url)

    print()
    print(f"Länkar: {len(parser.links)}")
    print(f"Externa script: {len(external_scripts)}")
    print(f"Inline script: {len(parser.inline_scripts)}")
    print(f"Intressanta attribut: {len(parser.attributes)}")
    print(f"Identifierade endpoints: {len(endpoints)}")

    if endpoints:
        print()
        print("--- Identifierade endpoints ---")

        for endpoint in sorted(endpoints):
            print(endpoint)

    relevant_attributes = [
        item
        for item in parser.attributes
        if any(
            term.lower() in item[2].lower()
            for term in (
                "blankning",
                "aggregat",
                "histor",
                "position",
                "report",
                "get",
                "download",
            )
        )
    ]

    if relevant_attributes:
        print()
        print("--- Relevanta attribut ---")

        for tag, name, value in relevant_attributes:
            print(f"<{tag}> {name}={value}")

    for script in parser.inline_scripts:
        script_endpoints = extract_endpoints(script)
        endpoints.update(script_endpoints)

        if any(
            term in script.lower()
            for term in (
                "getblankningsregisteraggregat",
                "gethistfile",
                "getaktuellfile",
                "runreport",
                "blankningsregister",
                "aggregat",
                "histor",
            )
        ):
            print_contexts(
                script,
                [
                    "GetBlankningsregisterAggregat",
                    "GetHistFile",
                    "GetAktuellFile",
                    "RunReport",
                    "BlankningsRegister",
                    "aggregat",
                    "histor",
                ],
            )

    return endpoints, external_scripts


def inspect_external_script(
    session: requests.Session,
    url: str,
) -> set[str]:
    """Hämtar och analyserar ett externt JavaScript på FI:s egen host."""

    print()
    print("-" * 80)
    print(f"EXTERNT SCRIPT: {url}")
    print("-" * 80)

    try:
        response = session.get(
            url,
            headers=HEADERS,
            timeout=REQUEST_TIMEOUT,
        )
    except requests.RequestException as exc:
        print(f"HTTP-fel: {exc}")
        return set()

    print(f"HTTP: {response.status_code}")
    print(f"Content-Type: {response.headers.get('content-type', '')}")
    print(f"Length: {len(response.content)}")

    if response.status_code != 200:
        return set()

    text = response.text
    endpoints = extract_endpoints(text)

    if endpoints:
        print()
        print("--- Endpoints i script ---")

        for endpoint in sorted(endpoints):
            print(endpoint)

    if any(
        term in text.lower()
        for term in (
            "blankningsregister",
            "aggregat",
            "histor",
            "position",
            "getblanknings",
        )
    ):
        print_contexts(
            text,
            [
                "GetBlankningsregisterAggregat",
                "GetHistFile",
                "GetAktuellFile",
                "BlankningsRegister",
                "aggregat",
                "histor",
            ],
        )

    return endpoints


def candidate_pages(
    links: list[str],
    base_url: str,
) -> list[str]:
    """
    Väljer relevanta FI-sidor för vidare undersökning.

    Vi prioriterar sidor som verkar kunna visa emittent-, innehavar-
    eller positionshistorik.
    """

    candidates: list[str] = []
    seen: set[str] = set()

    keywords = (
        "BlankningsRegister",
        "Positionsinnehavare",
        "emittent",
        "position",
        "blankning",
    )

    for link in links:
        absolute = absolute_url(base_url, link)

        if not same_host(absolute, FI_URL):
            continue

        path_lower = urlparse(absolute).path.lower()

        if not any(keyword.lower() in path_lower for keyword in keywords):
            continue

        if absolute in seen:
            continue

        seen.add(absolute)
        candidates.append(absolute)

    # Prioritera de mest intressanta sidorna.
    def score(url: str) -> tuple[int, int]:
        lower = url.lower()

        score_value = 0

        if "positionsinnehavare" in lower:
            score_value -= 50

        if "emittent" in lower:
            score_value -= 40

        if "blankningsregister" in lower:
            score_value -= 20

        if "position" in lower:
            score_value -= 10

        return score_value, len(url)

    candidates.sort(key=score)

    return candidates[:MAX_RELEVANT_PAGES]


def main() -> int:
    print("=" * 80)
    print("FI BACKEND-DIAGNOSTIK")
    print("=" * 80)
    print()
    print(f"FI_URL: {FI_URL}")

    session = requests.Session()
    session.headers.update(HEADERS)

    # ------------------------------------------------------------------
    # 1. Hämta huvudsidan
    # ------------------------------------------------------------------

    try:
        html = fetch_html(FI_URL)
    except FIError as exc:
        print(f"Kunde inte hämta FI-sidan: {exc}")
        return 1
    except Exception as exc:
        print(f"Oväntat fel vid hämtning av FI-sidan: {exc}")
        return 1

    print()
    print("=" * 80)
    print("FI HUVUDSIDA")
    print("=" * 80)
    print(f"Length: {len(html)}")

    parser = ResourceParser()

    try:
        parser.feed(html)
    except Exception as exc:
        print(f"HTML-parserfel: {exc}")

    print(f"Länkar: {len(parser.links)}")
    print(f"Externa script: {len(parser.scripts)}")
    print(f"Inline script: {len(parser.inline_scripts)}")
    print(f"Attribut: {len(parser.attributes)}")

    # ------------------------------------------------------------------
    # 2. Alla BlankningsRegister-länkar
    # ------------------------------------------------------------------

    blank_links: set[str] = set()

    for link in parser.links:
        absolute = absolute_url(FI_URL, link)

        if "/BlankningsRegister/" in absolute:
            blank_links.add(absolute)

    for tag, name, value in parser.attributes:
        absolute = absolute_url(FI_URL, value)

        if "/BlankningsRegister/" in absolute:
            blank_links.add(absolute)

    print()
    print("--- BlankningsRegister-länkar ---")

    for link in sorted(blank_links):
        print(link)

    # ------------------------------------------------------------------
    # 3. Hitta endpoints direkt i huvudsidan
    # ------------------------------------------------------------------

    endpoints = extract_endpoints(html)

    print()
    print("--- Endpoints identifierade från huvudsidan ---")

    if endpoints:
        for endpoint in sorted(endpoints):
            print(endpoint)
    else:
        print("(inga)")

    # ------------------------------------------------------------------
    # 4. Visa RunReport och relevanta inline-script
    # ------------------------------------------------------------------

    print()
    print("=" * 80)
    print("RELEVANT JAVASCRIPT PÅ HUVUDSIDAN")
    print("=" * 80)

    for script in parser.inline_scripts:
        lower = script.lower()

        if any(
            term in lower
            for term in (
                "runreport",
                "getblankningsregisteraggregat",
                "gethistfile",
                "getaktuellfile",
                "blankningsregister",
                "aggregat",
            )
        ):
            print_contexts(
                script,
                [
                    "RunReport",
                    "GetBlankningsregisterAggregat",
                    "GetHistFile",
                    "GetAktuellFile",
                    "BlankningsRegister",
                    "aggregat",
                ],
                radius=700,
            )

    # ------------------------------------------------------------------
    # 5. Analysera externa FI-script
    # ------------------------------------------------------------------

    external_scripts: list[str] = []

    for script in parser.scripts:
        script_url = absolute_url(FI_URL, script)

        if same_host(script_url, FI_URL):
            external_scripts.append(script_url)

    # Dubbletter bort.
    external_scripts = sorted(set(external_scripts))

    print()
    print("=" * 80)
    print("EXTERNA FI-SCRIPT")
    print("=" * 80)

    for script_url in external_scripts:
        inspect_external_script(session, script_url)

    # ------------------------------------------------------------------
    # 6. Hitta relevanta undersidor
    # ------------------------------------------------------------------

    pages = candidate_pages(parser.links, FI_URL)

    print()
    print("=" * 80)
    print("RELEVANTA UNDERSIDOR")
    print("=" * 80)

    for page in pages:
        print(page)

    # ------------------------------------------------------------------
    # 7. Inspektera undersidor
    # ------------------------------------------------------------------

    discovered_endpoints: set[str] = set(endpoints)

    for page in pages:
        page_endpoints, page_scripts = inspect_page(
            session,
            page,
            label="Blankningsregister-relaterad sida",
        )

        discovered_endpoints.update(page_endpoints)

        for script_url in page_scripts:
            script_endpoints = inspect_external_script(
                session,
                script_url,
            )

            discovered_endpoints.update(script_endpoints)

    # ------------------------------------------------------------------
    # 8. Samlad endpoint-lista
    # ------------------------------------------------------------------

    print()
    print("=" * 80)
    print("SAMLAD ENDPOINT-KARTA")
    print("=" * 80)

    for endpoint in sorted(discovered_endpoints):
        print(endpoint)

    # ------------------------------------------------------------------
    # 9. Parameternamn
    # ------------------------------------------------------------------

    parameter_names: set[str] = set()

    for endpoint in discovered_endpoints:
        parameter_names.update(
            extract_parameter_names(endpoint)
        )

    # Även HTML/JS kan innehålla parametrar som inte syntes i endpoints.
    parameter_names.update(
        extract_parameter_names(html)
    )

    print()
    print("=" * 80)
    print("IDENTIFIERADE PARAMETERNAMN")
    print("=" * 80)

    if parameter_names:
        for parameter in sorted(parameter_names):
            print(parameter)
    else:
        print("(inga)")

    # ------------------------------------------------------------------
    # 10. Kontrollera kända aggregate-endpointen separat
    # ------------------------------------------------------------------

    aggregate_url = absolute_url(
        FI_URL,
        "/BlankningsRegister/GetBlankningsregisterAggregat",
    )

    print()
    print("=" * 80)
    print("KÄND AGGREGATE-ENDPOINT")
    print("=" * 80)
    print(aggregate_url)

    try:
        response = session.get(
            aggregate_url,
            headers=HEADERS,
            timeout=REQUEST_TIMEOUT,
        )

        print(f"HTTP: {response.status_code}")
        print(
            "Content-Type:",
            response.headers.get("content-type", ""),
        )
        print("Length:", len(response.content))

    except requests.RequestException as exc:
        print(f"HTTP-fel: {exc}")

    # ------------------------------------------------------------------
    # 11. Testa om dokumenterade/kända endpoint-paths förekommer
    # ------------------------------------------------------------------

    known_paths = {
        "aggregate": "/BlankningsRegister/GetBlankningsregisterAggregat",
        "historical": "/BlankningsRegister/GetHistFile",
        "current": "/BlankningsRegister/GetAktuellFile",
    }

    print()
    print("=" * 80)
    print("KÄNDA FI-ENDPOINTS")
    print("=" * 80)

    for name, path in known_paths.items():
        found = any(
            path.lower() in endpoint.lower()
            for endpoint in discovered_endpoints
        )

        print(
            f"{name:12} {path:65} "
            f"{'HITTAD' if found else 'EJ HITTAD'}"
        )

    print()
    print("=" * 80)
    print("DIAGNOSTIK KLAR")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
