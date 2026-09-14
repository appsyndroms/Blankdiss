from __future__ import annotations

import re
from html import unescape
from html.parser import HTMLParser
from urllib.parse import parse_qs, urljoin, urlparse

import requests

from fi.client import fetch_html
from fi.config import FI_URL, HEADERS
from fi.errors import FIError


MAX_ISSUER_PAGES = 3
MAX_HOLDER_PAGES = 3
REQUEST_TIMEOUT = 30

REGISTER_RE = re.compile(r"/blankningsregistret/", re.IGNORECASE)

INTERESTING_TERMS = (
    "positionsinnehavare",
    "emittent",
    "issuer",
    "blankning",
    "position",
    "histor",
    "aggregate",
    "aggregat",
    "summa procent",
    "positionsdatum",
    "isin",
    "lei",
)

INTERESTING_ENDPOINT_RE = re.compile(
    r"""
    (?:
        /[^"' ]*(?:blankning|position|emittent|issuer|histor|aggregat)[^"' ]*
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

URL_RE = re.compile(
    r"""
    (?:
        https?://[^"' <>\s]+
        |
        /[^"' <>\s]+
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
        \b(name|id|lei|isin|date|datum|positionDate|positionsdatum)\s*=
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)


class ResourceParser(HTMLParser):
    """Minimal parser that collects only data useful for FI diagnostics."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)

        self.links: list[str] = []
        self.scripts: list[str] = []
        self.inline_scripts: list[str] = []

        self.forms: list[dict[str, str]] = []
        self.controls: list[dict[str, str]] = []

        self._current_form: dict[str, str] | None = None
        self._current_script: list[str] | None = None

    def handle_starttag(self, tag: str, attrs) -> None:
        attrs_dict = {str(k): str(v) for k, v in attrs if v is not None}

        tag_lower = tag.lower()

        if tag_lower == "a":
            href = attrs_dict.get("href")
            if href:
                self.links.append(href)

        elif tag_lower == "script":
            src = attrs_dict.get("src")

            if src:
                self.scripts.append(src)
            else:
                self._current_script = []

        elif tag_lower == "form":
            self._current_form = {
                "action": attrs_dict.get("action", ""),
                "method": attrs_dict.get("method", "get"),
                "id": attrs_dict.get("id", ""),
                "name": attrs_dict.get("name", ""),
            }

        elif tag_lower in {"input", "select", "option", "button", "textarea"}:
            control = {
                "tag": tag_lower,
                "type": attrs_dict.get("type", ""),
                "name": attrs_dict.get("name", ""),
                "value": attrs_dict.get("value", ""),
                "id": attrs_dict.get("id", ""),
            }

            if self._current_form:
                control["form_action"] = self._current_form.get("action", "")

            interesting = any(
                value and is_interesting_text(value)
                for value in control.values()
            )

            if interesting or tag_lower in {"button", "select"}:
                self.controls.append(control)

        if self._current_script is not None:
            self._current_script.append(self.get_starttag_text() or "")

    def handle_endtag(self, tag: str) -> None:
        tag_lower = tag.lower()

        if tag_lower == "script" and self._current_script is not None:
            self.inline_scripts.append("".join(self._current_script))
            self._current_script = None

        elif tag_lower == "form" and self._current_form is not None:
            self.forms.append(self._current_form)
            self._current_form = None

        if self._current_script is not None:
            self._current_script.append(f"</{tag}>")

    def handle_data(self, data: str) -> None:
        if self._current_script is not None:
            self._current_script.append(data)


def is_interesting_text(value: str) -> bool:
    value_lower = value.lower()

    return any(term in value_lower for term in INTERESTING_TERMS)


def normalize_url(url: str, base_url: str) -> str:
    url = unescape(url).strip()

    if url.startswith("javascript:"):
        return ""

    return urljoin(base_url, url)


def is_register_url(url: str) -> bool:
    return bool(REGISTER_RE.search(url))


def extract_register_urls(text: str, base_url: str) -> list[str]:
    results: list[str] = []

    for match in URL_RE.findall(text):
        url = normalize_url(match, base_url)

        if not url:
            continue

        if is_register_url(url):
            results.append(url)

    return unique(results)


def extract_interesting_endpoints(text: str, base_url: str) -> list[str]:
    results: list[str] = []

    for match in INTERESTING_ENDPOINT_RE.findall(text):
        url = normalize_url(match, base_url)

        if url:
            results.append(url)

    return unique(results)


def extract_parameter_names(text: str) -> list[str]:
    names: list[str] = []

    for match in PARAM_RE.finditer(text):
        for value in match.groups():
            if value:
                names.append(value)

    return unique(names)


def extract_dates(text: str) -> list[str]:
    return unique(DATE_RE.findall(text))


def unique(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()

    for value in values:
        if value in seen:
            continue

        seen.add(value)
        result.append(value)

    return result


def print_section(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def print_interesting_lines(text: str) -> None:
    """
    Print only short HTML/JS lines that contain something useful.
    """

    seen: set[str] = set()

    for raw_line in text.splitlines():
        line = " ".join(raw_line.split()).strip()

        if not line or len(line) > 500:
            continue

        if not is_interesting_text(line):
            continue

        if line in seen:
            continue

        seen.add(line)
        print(line)


def print_relevant_contexts(
    text: str,
    base_url: str,
    max_contexts: int = 12,
) -> None:
    """
    Print compact contexts around genuinely interesting terms.
    """

    clean = " ".join(text.split())

    patterns = (
        "Positionsinnehavare",
        "Summa procent",
        "Positionsdatum",
        "Efterföljande Position",
        "Föregående Position",
        "historiska",
        "aggregat",
        "aggregate",
        "Positionsinnehavare?id=",
    )

    printed: set[str] = set()
    count = 0

    for pattern in patterns:
        start = 0

        while count < max_contexts:
            index = clean.lower().find(pattern.lower(), start)

            if index == -1:
                break

            context_start = max(0, index - 180)
            context_end = min(len(clean), index + 500)

            context = clean[context_start:context_end]

            if context not in printed:
                printed.add(context)
                print(f"...{context}...")
                count += 1

            start = index + len(pattern)


def print_relevant_links(
    links: list[str],
    base_url: str,
) -> list[str]:
    results: list[str] = []

    for link in links:
        url = normalize_url(link, base_url)

        if not url:
            continue

        parsed = urlparse(url)
        path_lower = parsed.path.lower()

        if (
            "positionsinnehavare" in path_lower
            or "/emittent" in path_lower
            or "blankningsregistret" in path_lower
        ):
            results.append(url)

    results = unique(results)

    for url in results:
        print(url)

    return results


def print_forms(parser: ResourceParser) -> None:
    interesting_forms = []

    for form in parser.forms:
        values = " ".join(form.values()).lower()

        if is_interesting_text(values):
            interesting_forms.append(form)

    if not interesting_forms:
        return

    print()
    print("--- RELEVANTA FORMULÄR ---")

    for form in interesting_forms:
        print(
            f"action='{form['action']}' "
            f"method='{form['method']}' "
            f"id='{form['id']}' "
            f"name='{form['name']}'"
        )


def print_controls(parser: ResourceParser) -> None:
    interesting = []

    for control in parser.controls:
        values = " ".join(control.values()).lower()

        if is_interesting_text(values):
            interesting.append(control)

    if not interesting:
        return

    print()
    print("--- RELEVANTA KONTROLLER ---")

    for control in interesting:
        print(control)


def inspect_external_script(url: str) -> None:
    print_section(f"SCRIPT: {url}")

    try:
        response = requests.get(
            url,
            headers=HEADERS,
            timeout=REQUEST_TIMEOUT,
        )

        print(f"HTTP: {response.status_code}")

        if response.status_code != 200:
            return

        text = response.text

        endpoints = extract_interesting_endpoints(text, url)
        register_urls = extract_register_urls(text, url)
        params = extract_parameter_names(text)

        if endpoints:
            print()
            print("--- INTRESSANTA ENDPOINTS ---")

            for endpoint in endpoints[:30]:
                print(endpoint)

        if register_urls:
            print()
            print("--- REGISTER-URL:ER ---")

            for register_url in register_urls[:30]:
                print(register_url)

        if params:
            print()
            print("--- PARAMETRAR ---")
            print(", ".join(params[:50]))

        if not endpoints and not register_urls and not params:
            print("Inga relevanta blanknings-endpoints hittades.")

    except requests.RequestException as exc:
        print(f"HTTP-FEL: {exc}")


def fetch_page(url: str) -> tuple[str, str] | None:
    try:
        response = requests.get(
            url,
            headers=HEADERS,
            timeout=REQUEST_TIMEOUT,
        )

        print(f"HTTP: {response.status_code}")

        if response.status_code != 200:
            return None

        content_type = response.headers.get("content-type", "")
        print(f"Content-Type: {content_type}")
        print(f"Length: {len(response.content)}")

        return response.text, content_type

    except requests.RequestException as exc:
        print(f"HTTP-FEL: {exc}")
        return None


def inspect_page(
    url: str,
    label: str = "BLANKNINGSREGISTER-RELATERAD SIDA",
) -> tuple[list[str], list[str]]:
    print_section(f"{label}\n{url}")

    result = fetch_page(url)

    if result is None:
        return [], []

    html, _ = result

    parser = ResourceParser()

    try:
        parser.feed(html)
    except Exception as exc:
        print(f"HTML-PARSER-FEL: {exc}")

    print(
        f"Länkar: {len(parser.links)} | "
        f"Formulär: {len(parser.forms)} | "
        f"Kontroller: {len(parser.controls)} | "
        f"Inline script: {len(parser.inline_scripts)}"
    )

    register_urls = extract_register_urls(html, url)
    endpoints = extract_interesting_endpoints(html, url)
    parameters = extract_parameter_names(html)
    dates = extract_dates(html)

    relevant_links = print_relevant_links(
        parser.links,
        url,
    )

    if endpoints:
        print()
        print("--- INTRESSANTA ENDPOINTS ---")

        for endpoint in endpoints[:40]:
            print(endpoint)

    if parameters:
        print()
        print("--- PARAMETRAR ---")
        print(", ".join(parameters[:50]))

    if dates:
        print()
        print("--- DATUM ---")
        print(", ".join(dates[:30]))

    print_forms(parser)
    print_controls(parser)

    if relevant_links:
        print()
        print("--- POSITIONSINNEHAVARE / EMITTENT ---")

        for link in relevant_links[:30]:
            print(link)

    print()
    print("--- RELEVANTA HTML-KONTEXTER ---")
    print_relevant_contexts(html, url)

    if parser.inline_scripts:
        print()
        print("--- RELEVANT INLINE-JS ---")

        for script in parser.inline_scripts:
            if is_interesting_text(script):
                print_interesting_lines(script)

    return relevant_links, register_urls


def candidate_issuer_pages(
    links: list[str],
    base_url: str,
) -> list[str]:
    candidates: list[str] = []

    for link in links:
        url = normalize_url(link, base_url)

        if not url:
            continue

        parsed = urlparse(url)

        if "/emittent" not in parsed.path.lower():
            continue

        query = parse_qs(parsed.query)

        if "id" not in query:
            continue

        candidates.append(url)

    return unique(candidates)[:MAX_ISSUER_PAGES]


def candidate_holder_pages(
    links: list[str],
    base_url: str,
) -> list[str]:
    candidates: list[str] = []

    for link in links:
        url = normalize_url(link, base_url)

        if not url:
            continue

        parsed = urlparse(url)

        if "positionsinnehavare" not in parsed.path.lower():
            continue

        query = parse_qs(parsed.query)

        if "id" not in query:
            continue

        candidates.append(url)

    return unique(candidates)[:MAX_HOLDER_PAGES]


def inspect_holder_pages(urls: list[str]) -> None:
    for url in urls:
        inspect_page(
            url,
            "POSITIONSINNEHAVARE",
        )


def inspect_known_aggregate_endpoint() -> None:
    url = urljoin(
        FI_URL,
        "/BlankningsRegister/GetBlankningsregisterAggregat",
    )

    print_section("KÄNT AGGREGAT-ENDPOINT")

    try:
        response = requests.get(
            url,
            headers=HEADERS,
            timeout=REQUEST_TIMEOUT,
        )

        print(f"HTTP: {response.status_code}")
        print(
            f"Content-Type: "
            f"{response.headers.get('content-type', '')}"
        )
        print(f"Length: {len(response.content)}")

        if response.status_code == 200:
            print(
                "Endpointet fungerar, men vi vet ännu inte "
                "om historiska datum kan hämtas härifrån."
            )

    except requests.RequestException as exc:
        print(f"HTTP-FEL: {exc}")


def inspect_date_variants() -> None:
    """
    Testa endast ett fåtal uppenbara varianter.
    Detta är medvetet begränsat så diagnostiken inte spammar FI.
    """

    endpoint = urljoin(
        FI_URL,
        "/BlankningsRegister/GetBlankningsregisterAggregat",
    )

    test_parameters = (
        ("date", "2026-08-01"),
        ("datum", "2026-08-01"),
        ("positionDate", "2026-08-01"),
        ("positionsdatum", "2026-08-01"),
    )

    print_section("DATUMPARAMETRAR PÅ KÄNT AGGREGAT-ENDPOINT")

    baseline_length: int | None = None

    try:
        baseline = requests.get(
            endpoint,
            headers=HEADERS,
            timeout=REQUEST_TIMEOUT,
        )

        if baseline.status_code == 200:
            baseline_length = len(baseline.content)

    except requests.RequestException:
        pass

    for name, value in test_parameters:
        try:
            response = requests.get(
                endpoint,
                params={name: value},
                headers=HEADERS,
                timeout=REQUEST_TIMEOUT,
            )

            length = len(response.content)

            changed = (
                baseline_length is not None
                and length != baseline_length
            )

            print(
                f"{name}={value} -> "
                f"HTTP {response.status_code}, "
                f"length={length}, "
                f"changed={changed}"
            )

        except requests.RequestException as exc:
            print(f"{name}: HTTP-FEL: {exc}")


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
        print(f"FEL VID HÄMTNING AV FI-SIDAN: {exc}")
        return 1

    parser = ResourceParser()

    try:
        parser.feed(html)
    except Exception as exc:
        print(f"HTML-PARSER-FEL: {exc}")
        return 1

    print_section("HUVUDSIDAN – ENDAST RELEVANT INFORMATION")

    print(
        f"Länkar: {len(parser.links)} | "
        f"Script: {len(parser.scripts)} | "
        f"Inline script: {len(parser.inline_scripts)}"
    )

    relevant_links = print_relevant_links(
        parser.links,
        FI_URL,
    )

    endpoints = extract_interesting_endpoints(
        html,
        FI_URL,
    )

    if endpoints:
        print()
        print("--- INTRESSANTA ENDPOINTS ---")

        for endpoint in endpoints[:40]:
            print(endpoint)

    parameters = extract_parameter_names(html)

    if parameters:
        print()
        print("--- PARAMETRAR ---")
        print(", ".join(parameters[:50]))

    print()
    print("--- RELEVANTA INLINE-JS ---")

    for script in parser.inline_scripts:
        if is_interesting_text(script):
            print_interesting_lines(script)

    issuer_pages = candidate_issuer_pages(
        relevant_links,
        FI_URL,
    )

    print_section("EMITTENT-SIDOR")

    for issuer_url in issuer_pages:
        print(issuer_url)

    holder_pages: list[str] = []

    for issuer_url in issuer_pages:
        links, _ = inspect_page(
            issuer_url,
            "EMITTENT",
        )

        holder_pages.extend(
            candidate_holder_pages(
                links,
                issuer_url,
            )
        )

    holder_pages = unique(holder_pages)[:MAX_HOLDER_PAGES]

    if holder_pages:
        print_section("POSITIONSINNEHAVARE – URVAL")

        for holder_url in holder_pages:
            print(holder_url)

        inspect_holder_pages(holder_pages)

    print_section("EXTERNA SCRIPT – BARA BLANKNINGSRELATERADE TRÄFFAR")

    inspected_scripts = 0

    for script in parser.scripts:
        if inspected_scripts >= 5:
            break

        script_url = normalize_url(
            script,
            FI_URL,
        )

        if not script_url:
            continue

        try:
            response = requests.get(
                script_url,
                headers=HEADERS,
                timeout=REQUEST_TIMEOUT,
            )

            if response.status_code != 200:
                continue

            text = response.text

            interesting = (
                extract_interesting_endpoints(
                    text,
                    script_url,
                )
                or extract_register_urls(
                    text,
                    script_url,
                )
                or any(
                    term in text.lower()
                    for term in (
                        "positionsinnehavare",
                        "blankningsregistret",
                        "getblanknings",
                    )
                )
            )

            if not interesting:
                continue

            inspect_external_script(script_url)
            inspected_scripts += 1

        except requests.RequestException:
            continue

    inspect_known_aggregate_endpoint()
    inspect_date_variants()

    print_section("SLUTSATS")

    print(
        "Diagnostiken är nu begränsad till register-, "
        "emittent-, positionsinnehavare-, historik- och "
        "aggregatrelaterad information."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
