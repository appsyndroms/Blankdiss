from __future__ import annotations

import re
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse

import requests

from fi.client import fetch_html
from fi.config import FI_URL, HEADERS
from fi.errors import FIError


MAX_RELEVANT_PAGES = 12
MAX_POSITION_HOLDER_PAGES = 5
REQUEST_TIMEOUT = 30


class ResourceParser(HTMLParser):
    """Samlar länkar, formulär, script och attribut som kan innehålla backend-anrop."""

    INTERESTING_ATTRIBUTES = {
        "data-url",
        "data-href",
        "data-endpoint",
        "data-api",
        "data-ajax-url",
        "data-action",
        "data-target",
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

        self.forms: list[dict[str, str]] = []
        self.inputs: list[dict[str, str]] = []
        self.selects: list[dict[str, str]] = []
        self.options: list[dict[str, str]] = []
        self.buttons: list[dict[str, str]] = []

        self._in_script = False
        self._script_parts: list[str] = []

        self._current_form: dict[str, str] | None = None
        self._current_select: dict[str, str] | None = None

        self._in_button = False
        self._button_parts: list[str] = []
        self._current_button: dict[str, str] | None = None

    def handle_starttag(self, tag: str, attrs) -> None:
        tag_lower = tag.lower()
        attrs_dict = {
            str(name).lower(): str(value)
            for name, value in attrs
            if value is not None
        }

        if tag_lower == "script":
            self._in_script = True
            self._script_parts = []

            src = attrs_dict.get("src")

            if src:
                self.scripts.append(src)

        for name, value in attrs:
            if not value:
                continue

            name_lower = name.lower()

            if name_lower in self.INTERESTING_ATTRIBUTES:
                self.attributes.append(
                    (tag, name, value)
                )

            if tag_lower == "a" and name_lower == "href":
                self.links.append(value)

        if tag_lower == "form":
            self._current_form = {
                "action": attrs_dict.get("action", ""),
                "method": attrs_dict.get("method", "get"),
                "name": attrs_dict.get("name", ""),
                "id": attrs_dict.get("id", ""),
            }

            self.forms.append(self._current_form)

        elif tag_lower == "input":
            self.inputs.append(
                {
                    "type": attrs_dict.get("type", ""),
                    "name": attrs_dict.get("name", ""),
                    "value": attrs_dict.get("value", ""),
                    "id": attrs_dict.get("id", ""),
                    "form": attrs_dict.get("form", ""),
                    "data-url": attrs_dict.get("data-url", ""),
                    "data-href": attrs_dict.get("data-href", ""),
                    "data-endpoint": attrs_dict.get(
                        "data-endpoint",
                        "",
                    ),
                }
            )

        elif tag_lower == "select":
            self._current_select = {
                "name": attrs_dict.get("name", ""),
                "id": attrs_dict.get("id", ""),
                "form": attrs_dict.get("form", ""),
                "data-url": attrs_dict.get("data-url", ""),
                "data-endpoint": attrs_dict.get(
                    "data-endpoint",
                    "",
                ),
            }

            self.selects.append(self._current_select)

        elif tag_lower == "option":
            self.options.append(
                {
                    "name": (
                        self._current_select.get("name", "")
                        if self._current_select
                        else ""
                    ),
                    "value": attrs_dict.get("value", ""),
                    "id": attrs_dict.get("id", ""),
                    "text": "",
                }
            )

        elif tag_lower == "button":
            self._in_button = True
            self._button_parts = []

            self._current_button = {
                "type": attrs_dict.get("type", ""),
                "name": attrs_dict.get("name", ""),
                "value": attrs_dict.get("value", ""),
                "id": attrs_dict.get("id", ""),
                "onclick": attrs_dict.get("onclick", ""),
                "data-url": attrs_dict.get("data-url", ""),
                "data-href": attrs_dict.get("data-href", ""),
                "data-endpoint": attrs_dict.get(
                    "data-endpoint",
                    "",
                ),
            }

    def handle_data(self, data: str) -> None:
        if self._in_script:
            self._script_parts.append(data)

        if self._in_button:
            self._button_parts.append(data)

        if self.options:
            current = self.options[-1]

            if current["text"] == "":
                current["text"] = data.strip()

    def handle_endtag(self, tag: str) -> None:
        tag_lower = tag.lower()

        if tag_lower == "script" and self._in_script:
            script = "".join(self._script_parts).strip()

            if script:
                self.inline_scripts.append(script)

            self._in_script = False
            self._script_parts = []

        elif tag_lower == "select":
            self._current_select = None

        elif tag_lower == "button" and self._in_button:
            if self._current_button is not None:
                self._current_button["text"] = (
                    " ".join(self._button_parts).strip()
                )

                self.buttons.append(self._current_button)

            self._in_button = False
            self._button_parts = []
            self._current_button = None

        elif tag_lower == "form":
            self._current_form = None


def absolute_url(base_url: str, value: str) -> str:
    """Gör en URL absolut."""
    return urljoin(base_url, value)


def same_host(url: str, base_url: str) -> bool:
    """Returnerar True om URL:en ligger på samma host."""
    return urlparse(url).netloc == urlparse(base_url).netloc


def extract_endpoints(text: str) -> set[str]:
    """Försöker hitta backend-endpoints i HTML/JavaScript."""

    endpoints: set[str] = set()

    patterns = [
        r"""['"](/BlankningsRegister/[^'"]+)['"]""",
        r"""[`"](/BlankningsRegister/[^`"]+)[`"]""",
        r"""url\s*:\s*['"]([^'"]+)['"]""",
        r"""fetch\s*\(\s*['"]([^'"]+)['"]""",
        r"""\$\.(?:get|getJSON|post)\s*\(\s*['"]([^'"]+)['"]""",
        r"""url\s*:\s*['"]([^'"]*BlankningsRegister[^'"]*)['"]""",
        r"""(?:href|data-url|data-href|data-endpoint|data-api|data-ajax-url|data-action)\s*=\s*['"]([^'"]+)['"]""",
        r"""['"](/[^'"]*(?:Get|Post|Download|Report|Export|History|Historical|Aggregate|Aggregat)[^'"]*)['"]""",
        r"""(?:location\.href|window\.location)\s*=\s*['"]([^'"]+)['"]""",
    ]

    for pattern in patterns:
        for match in re.finditer(
            pattern,
            text,
            re.IGNORECASE,
        ):
            value = match.group(1).strip()

            if not value:
                continue

            if value.startswith("javascript:"):
                continue

            if value.startswith(
                ("/", "http://", "https://")
            ):
                endpoints.add(value)

    return endpoints


def extract_parameter_names(url_or_text: str) -> set[str]:
    """Hittar potentiella query-parametrar."""

    params: set[str] = set()

    parsed = urlparse(url_or_text)

    if parsed.query:
        for part in parsed.query.split("&"):
            if "=" in part:
                key = part.split("=", 1)[0].strip()

                if key:
                    params.add(key)

    patterns = [
        r"""[?&]([A-Za-z][A-Za-z0-9_]*)=""",
        r"""(?:name|param|parameter)\s*[:=]\s*['"]([A-Za-z][A-Za-z0-9_]*)['"]""",
        r"""(?:data-[A-Za-z0-9_-]*)(?:=)['"]([^'"]+)['"]""",
    ]

    for pattern in patterns:
        for match in re.finditer(
            pattern,
            url_or_text,
            re.IGNORECASE,
        ):
            params.add(match.group(1))

    return params


def print_contexts(
    text: str,
    terms: list[str],
    radius: int = 350,
    max_contexts_per_term: int = 5,
) -> None:
    """Skriver relevanta kodstycken runt givna sökord."""

    lower = text.lower()

    for term in terms:
        start = 0
        count = 0

        while count < max_contexts_per_term:
            index = lower.find(
                term.lower(),
                start,
            )

            if index == -1:
                break

            begin = max(0, index - radius)
            end = min(
                len(text),
                index + len(term) + radius,
            )

            snippet = text[begin:end].replace(
                "\r",
                " ",
            )

            print()
            print(f"--- Kontext: {term} ---")
            print(snippet)

            start = index + len(term)
            count += 1


def print_forms(parser: ResourceParser) -> None:
    """Skriver formulär och formulärfält."""

    if parser.forms:
        print()
        print("--- FORMULÄR ---")

        for index, form in enumerate(
            parser.forms,
            start=1,
        ):
            print(
                f"FORM {index}: "
                f"action={form['action']!r} "
                f"method={form['method']!r} "
                f"name={form['name']!r} "
                f"id={form['id']!r}"
            )

    if parser.inputs:
        print()
        print("--- INPUTS ---")

        for item in parser.inputs:
            print(
                "input "
                f"type={item['type']!r} "
                f"name={item['name']!r} "
                f"value={item['value']!r} "
                f"id={item['id']!r} "
                f"form={item['form']!r}"
            )

            for key in (
                "data-url",
                "data-href",
                "data-endpoint",
            ):
                if item[key]:
                    print(
                        f"    {key}={item[key]!r}"
                    )

    if parser.selects:
        print()
        print("--- SELECTS ---")

        for select in parser.selects:
            print(
                "select "
                f"name={select['name']!r} "
                f"id={select['id']!r} "
                f"form={select['form']!r} "
                f"data-url={select['data-url']!r} "
                f"data-endpoint={select['data-endpoint']!r}"
            )

    if parser.options:
        print()
        print("--- OPTIONS ---")

        for option in parser.options[:100]:
            print(
                f"option "
                f"name={option['name']!r} "
                f"value={option['value']!r} "
                f"text={option['text']!r}"
            )

        if len(parser.options) > 100:
            print(
                f"... ytterligare "
                f"{len(parser.options) - 100} options"
            )

    if parser.buttons:
        print()
        print("--- BUTTONS ---")

        for button in parser.buttons:
            print(
                "button "
                f"type={button['type']!r} "
                f"name={button['name']!r} "
                f"value={button['value']!r} "
                f"id={button['id']!r} "
                f"text={button.get('text', '')!r}"
            )

            for key in (
                "onclick",
                "data-url",
                "data-href",
                "data-endpoint",
            ):
                if button[key]:
                    print(
                        f"    {key}={button[key]!r}"
                    )


def extract_position_holder_links(
    parser: ResourceParser,
    base_url: str,
) -> list[str]:
    """Hittar unika Positionsinnehavare-länkar."""

    candidates: list[str] = []
    seen: set[str] = set()

    for link in parser.links:
        absolute = absolute_url(
            base_url,
            link,
        )

        path = urlparse(absolute).path.lower()

        if "positionsinnehavare" not in path:
            continue

        if absolute in seen:
            continue

        seen.add(absolute)
        candidates.append(absolute)

    return candidates[:MAX_POSITION_HOLDER_PAGES]


def extract_relevant_links(
    parser: ResourceParser,
    base_url: str,
) -> list[str]:
    """Hittar länkar som är relevanta för blankningsregistret."""

    candidates: list[str] = []
    seen: set[str] = set()

    keywords = (
        "blankningsregister",
        "positionsinnehavare",
        "emittent",
        "position",
        "blankning",
    )

    for link in parser.links:
        absolute = absolute_url(
            base_url,
            link,
        )

        if not same_host(
            absolute,
            FI_URL,
        ):
            continue

        lower = absolute.lower()

        if not any(
            keyword in lower
            for keyword in keywords
        ):
            continue

        if absolute in seen:
            continue

        seen.add(absolute)
        candidates.append(absolute)

    return candidates


def inspect_page(
    session: requests.Session,
    url: str,
    label: str,
) -> tuple[set[str], list[str], list[str]]:
    """Hämtar och analyserar en FI-sida."""

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
        return set(), [], []

    print(f"HTTP: {response.status_code}")
    print(
        "Content-Type:",
        response.headers.get(
            "content-type",
            "",
        ),
    )
    print(f"Length: {len(response.content)}")

    if response.status_code != 200:
        return set(), [], []

    html = response.text

    parser = ResourceParser()

    try:
        parser.feed(html)
    except Exception as exc:
        print(f"HTML-parserfel: {exc}")

    endpoints = extract_endpoints(html)

    external_scripts: list[str] = []

    for script in parser.scripts:
        script_url = absolute_url(
            url,
            script,
        )

        if same_host(
            script_url,
            FI_URL,
        ):
            external_scripts.append(
                script_url
            )

    print()
    print(f"Länkar: {len(parser.links)}")
    print(
        f"Externa script: "
        f"{len(external_scripts)}"
    )
    print(
        f"Inline script: "
        f"{len(parser.inline_scripts)}"
    )
    print(
        f"Intressanta attribut: "
        f"{len(parser.attributes)}"
    )
    print(
        f"Formulär: "
        f"{len(parser.forms)}"
    )
    print(
        f"Inputs: "
        f"{len(parser.inputs)}"
    )
    print(
        f"Selects: "
        f"{len(parser.selects)}"
    )
    print(
        f"Buttons: "
        f"{len(parser.buttons)}"
    )
    print(
        f"Identifierade endpoints: "
        f"{len(endpoints)}"
    )

    if endpoints:
        print()
        print("--- Identifierade endpoints ---")

        for endpoint in sorted(endpoints):
            print(endpoint)

    print_forms(parser)

    position_holder_links = (
        extract_position_holder_links(
            parser,
            url,
        )
    )

    if position_holder_links:
        print()
        print(
            "--- Positionsinnehavare-länkar ---"
        )

        for position_url in position_holder_links:
            print(position_url)

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
                "datum",
                "date",
                "isin",
                "issuer",
                "emittent",
            )
        )
    ]

    if relevant_attributes:
        print()
        print("--- Relevanta attribut ---")

        for tag, name, value in relevant_attributes:
            print(
                f"<{tag}> "
                f"{name}={value}"
            )

    relevant_text_terms = [
        "Positionsinnehavare",
        "position",
        "positionsdatum",
        "datum",
        "date",
        "histor",
        "emittent",
        "issuer",
        "ISIN",
        "blankning",
        "Summa procent",
    ]

    if any(
        term.lower() in html.lower()
        for term in relevant_text_terms
    ):
        print()
        print(
            "--- Relevanta HTML-kontexter ---"
        )

        print_contexts(
            html,
            relevant_text_terms,
            radius=450,
            max_contexts_per_term=3,
        )

    for script in parser.inline_scripts:
        endpoints.update(
            extract_endpoints(script)
        )

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
                "positionsinnehavare",
            )
        ):
            print()
            print(
                "--- Relevant inline JavaScript ---"
            )

            print_contexts(
                script,
                [
                    "GetBlankningsregisterAggregat",
                    "GetHistFile",
                    "GetAktuellFile",
                    "RunReport",
                    "BlankningsRegister",
                    "Positionsinnehavare",
                    "aggregat",
                    "histor",
                ],
                radius=700,
            )

    return (
        endpoints,
        sorted(set(external_scripts)),
        position_holder_links,
    )


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
    print(
        "Content-Type:",
        response.headers.get(
            "content-type",
            "",
        ),
    )
    print(
        f"Length: "
        f"{len(response.content)}"
    )

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
            "positionsinnehavare",
        )
    ):
        print_contexts(
            text,
            [
                "GetBlankningsregisterAggregat",
                "GetHistFile",
                "GetAktuellFile",
                "BlankningsRegister",
                "Positionsinnehavare",
                "aggregat",
                "histor",
            ],
            radius=600,
        )

    return endpoints


def candidate_pages(
    links: list[str],
    base_url: str,
) -> list[str]:
    """Väljer relevanta FI-sidor för vidare undersökning."""

    candidates: list[str] = []
    seen: set[str] = set()

    keywords = (
        "blankningsregister",
        "positionsinnehavare",
        "emittent",
        "position",
        "blankning",
    )

    for link in links:
        absolute = absolute_url(
            base_url,
            link,
        )

        if not same_host(
            absolute,
            FI_URL,
        ):
            continue

        path_lower = urlparse(
            absolute
        ).path.lower()

        if not any(
            keyword in path_lower
            for keyword in keywords
        ):
            continue

        if absolute in seen:
            continue

        seen.add(absolute)
        candidates.append(absolute)

    def score(
        url: str,
    ) -> tuple[int, int]:
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

    candidates.sort(
        key=score
    )

    return candidates[
        :MAX_RELEVANT_PAGES
    ]


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
        html = fetch_html()
    except FIError as exc:
        print(
            f"Kunde inte hämta FI-sidan: {exc}"
        )
        return 1
    except Exception as exc:
        print(
            "Oväntat fel vid hämtning "
            f"av FI-sidan: {exc}"
        )
        return 1

    print()
    print("=" * 80)
    print("FI HUVUDSIDA")
    print("=" * 80)
    print(
        f"Length: {len(html)}"
    )

    parser = ResourceParser()

    try:
        parser.feed(html)
    except Exception as exc:
        print(
            f"HTML-parserfel: {exc}"
        )

    print(
        f"Länkar: {len(parser.links)}"
    )
    print(
        f"Externa script: "
        f"{len(parser.scripts)}"
    )
    print(
        f"Inline script: "
        f"{len(parser.inline_scripts)}"
    )
    print(
        f"Attribut: "
        f"{len(parser.attributes)}"
    )

    # ------------------------------------------------------------------
    # 2. Alla BlankningsRegister-länkar
    # ------------------------------------------------------------------

    blank_links: set[str] = set()

    for link in parser.links:
        absolute = absolute_url(
            FI_URL,
            link,
        )

        if (
            "/BlankningsRegister/"
            in absolute
        ):
            blank_links.add(absolute)

    for _tag, _name, value in (
        parser.attributes
    ):
        absolute = absolute_url(
            FI_URL,
            value,
        )

        if (
            "/BlankningsRegister/"
            in absolute
        ):
            blank_links.add(absolute)

    print()
    print(
        "--- BlankningsRegister-länkar ---"
    )

    if blank_links:
        for link in sorted(
            blank_links
        ):
            print(link)
    else:
        print("(inga)")

    # ------------------------------------------------------------------
    # 3. Hitta endpoints direkt i huvudsidan
    # ------------------------------------------------------------------

    endpoints = extract_endpoints(
        html
    )

    print()
    print(
        "--- Endpoints identifierade "
        "från huvudsidan ---"
    )

    if endpoints:
        for endpoint in sorted(
            endpoints
        ):
            print(endpoint)
    else:
        print("(inga)")

    # ------------------------------------------------------------------
    # 4. Relevant JavaScript på huvudsidan
    # ------------------------------------------------------------------

    print()
    print("=" * 80)
    print(
        "RELEVANT JAVASCRIPT "
        "PÅ HUVUDSIDAN"
    )
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
                "positionsinnehavare",
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
                    "Positionsinnehavare",
                    "aggregat",
                ],
                radius=700,
            )

    # ------------------------------------------------------------------
    # 5. Analysera externa FI-script
    # ------------------------------------------------------------------

    external_scripts = sorted(
        {
            absolute_url(
                FI_URL,
                script,
            )
            for script in parser.scripts
            if same_host(
                absolute_url(
                    FI_URL,
                    script,
                ),
                FI_URL,
            )
        }
    )

    print()
    print("=" * 80)
    print("EXTERNA FI-SCRIPT")
    print("=" * 80)

    for script_url in external_scripts:
        discovered = (
            inspect_external_script(
                session,
                script_url,
            )
        )

        endpoints.update(
            discovered
        )

    # ------------------------------------------------------------------
    # 6. Hitta relevanta undersidor
    # ------------------------------------------------------------------

    pages = candidate_pages(
        parser.links,
        FI_URL,
    )

    # Lägg även till BlankningsRegister-länkar
    # som hittades via attribut eller JavaScript.
    for link in sorted(
        blank_links
    ):
        if link not in pages:
            pages.append(link)

    pages = list(
        dict.fromkeys(pages)
    )[:MAX_RELEVANT_PAGES]

    print()
    print("=" * 80)
    print("RELEVANTA UNDERSIDOR")
    print("=" * 80)

    for page in pages:
        print(page)

    # ------------------------------------------------------------------
    # 7. Inspektera undersidor
    # ------------------------------------------------------------------

    position_holder_pages: list[str] = []

    for page in pages:
        (
            page_endpoints,
            page_scripts,
            holder_links,
        ) = inspect_page(
            session,
            page,
            label=(
                "Blankningsregister-relaterad sida"
            ),
        )

        endpoints.update(
            page_endpoints
        )

        for holder_link in holder_links:
            if (
                holder_link
                not in position_holder_pages
            ):
                position_holder_pages.append(
                    holder_link
                )

        for script_url in page_scripts:
            script_endpoints = (
                inspect_external_script(
                    session,
                    script_url,
                )
            )

            endpoints.update(
                script_endpoints
            )

    # ------------------------------------------------------------------
    # 8. Följ Positionsinnehavare-länkar
    #
    # Detta är den viktiga nya delen.
    # Vi följer endast ett litet antal unika länkar.
    # ------------------------------------------------------------------

    position_holder_pages = list(
        dict.fromkeys(
            position_holder_pages
        )
    )[:MAX_POSITION_HOLDER_PAGES]

    print()
    print("=" * 80)
    print(
        "POSITIONSINNEHAVARE "
        "SOM SKA UNDERSÖKAS"
    )
    print("=" * 80)

    if position_holder_pages:
        for page in position_holder_pages:
            print(page)
    else:
        print("(inga)")

    for page in position_holder_pages:
        (
            page_endpoints,
            page_scripts,
            nested_holder_links,
        ) = inspect_page(
            session,
            page,
            label=(
                "Positionsinnehavare-sida"
            ),
        )

        endpoints.update(
            page_endpoints
        )

        # Skriv även ut om Positionsinnehavare-sidan
        # länkar vidare till ytterligare positioner.
        if nested_holder_links:
            print()
            print(
                "--- Positionsinnehavare "
                "länkar vidare ---"
            )

            for nested in nested_holder_links:
                print(nested)

        for script_url in page_scripts:
            script_endpoints = (
                inspect_external_script(
                    session,
                    script_url,
                )
            )

            endpoints.update(
                script_endpoints
            )

    # ------------------------------------------------------------------
    # 9. Samlad endpoint-lista
    # ------------------------------------------------------------------

    print()
    print("=" * 80)
    print("SAMLAD ENDPOINT-KARTA")
    print("=" * 80)

    for endpoint in sorted(
        endpoints
    ):
        print(endpoint)

    # ------------------------------------------------------------------
    # 10. Parameternamn
    # ------------------------------------------------------------------

    parameter_names: set[str] = set()

    for endpoint in endpoints:
        parameter_names.update(
            extract_parameter_names(
                endpoint
            )
        )

    parameter_names.update(
        extract_parameter_names(
            html
        )
    )

    print()
    print("=" * 80)
    print(
        "IDENTIFIERADE "
        "PARAMETERNAMN"
    )
    print("=" * 80)

    if parameter_names:
        for parameter in sorted(
            parameter_names
        ):
            print(parameter)
    else:
        print("(inga)")

    # ------------------------------------------------------------------
    # 11. Särskild sökning efter datum-/historikmekanismer
    # ------------------------------------------------------------------

    print()
    print("=" * 80)
    print(
        "DATUM / HISTORIK / POSITION"
    )
    print("=" * 80)

    history_terms = [
        "positionsdatum",
        "positionDate",
        "historicalDate",
        "historik",
        "historical",
        "history",
        "datum",
        "date",
        "fromDate",
        "toDate",
        "startDate",
        "endDate",
        "issuer",
        "emittent",
        "isin",
        "position",
    ]

    print_contexts(
        html,
        history_terms,
        radius=300,
        max_contexts_per_term=2,
    )

    # ------------------------------------------------------------------
    # 12. Kontrollera kända aggregate-endpointen separat
    # ------------------------------------------------------------------

    aggregate_url = absolute_url(
        FI_URL,
        "/BlankningsRegister/"
        "GetBlankningsregisterAggregat",
    )

    print()
    print("=" * 80)
    print(
        "KÄND AGGREGATE-ENDPOINT"
    )
    print("=" * 80)
    print(aggregate_url)

    try:
        response = session.get(
            aggregate_url,
            headers=HEADERS,
            timeout=REQUEST_TIMEOUT,
        )

        print(
            f"HTTP: {response.status_code}"
        )
        print(
            "Content-Type:",
            response.headers.get(
                "content-type",
                "",
            ),
        )
        print(
            "Length:",
            len(response.content),
        )

    except requests.RequestException as exc:
        print(
            f"HTTP-fel: {exc}"
        )

    # ------------------------------------------------------------------
    # 13. Kontrollera kända endpoints
    # ------------------------------------------------------------------

    known_paths = {
        "aggregate": (
            "/BlankningsRegister/"
            "GetBlankningsregisterAggregat"
        ),
        "historical": (
            "/BlankningsRegister/"
            "GetHistFile"
        ),
        "current": (
            "/BlankningsRegister/"
            "GetAktuellFile"
        ),
    }

    print()
    print("=" * 80)
    print("KÄNDA FI-ENDPOINTS")
    print("=" * 80)

    for name, path in known_paths.items():
        found = any(
            path.lower()
            in endpoint.lower()
            for endpoint in endpoints
        )

        print(
            f"{name:12} "
            f"{path:65} "
            f"{'HITTAD' if found else 'EJ HITTAD'}"
        )

    print()
    print("=" * 80)
    print("DIAGNOSTIK KLAR")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
