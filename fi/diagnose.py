"""Diagnostik för FI:s blankningsregister.
Hämtar samma HTML som den vanliga FI-klienten och undersöker
inline-JavaScript och resursreferenser för att förstå hur FI:s
rapporter genereras.
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
    "runreport",
    "ajax",
    "fetch",
    "xmlhttprequest",
)
ENDPOINT_PATTERN = re.compile(
    r"""['"](?P<endpoint>/BlankningsRegister/[^'"]+)['"]""",
    re.IGNORECASE,
)
class ResourceParser(HTMLParser):
    """Samlar resurser och inline-script från HTML."""
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []
        self.scripts: list[str] = []
        self.raw_attributes: list[tuple[str, str]] = []
        self.inline_scripts: list[str] = []
        self._in_script = False
        self._script_parts: list[str] = []
    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        if tag.lower() == "script":
            self._in_script = True
            self._script_parts = []
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
                if (
                    tag.lower() == "script"
                    and lowered_name == "src"
                ):
                    self.scripts.append(absolute)
    def handle_data(self, data: str) -> None:
        if self._in_script:
            self._script_parts.append(data)
    def handle_endtag(self, tag: str) -> None:
        if tag.lower() != "script":
            return
        if self._script_parts:
            script = "".join(self._script_parts).strip()
            if script:
                self.inline_scripts.append(script)
        self._in_script = False
        self._script_parts = []
def is_interesting(value: str) -> bool:
    lowered = value.lower()
    return any(
        term in lowered
        for term in INTERESTING_TERMS
    )
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
def extract_function(
    script: str,
    function_name: str,
) -> str | None:
    """Extraherar en JavaScript-funktion med enkel bracket matching."""
    pattern = re.compile(
        rf"""
        function
        \s+
        {re.escape(function_name)}
        \s*
        \(
        """,
        re.IGNORECASE | re.VERBOSE,
    )
    match = pattern.search(script)
    if not match:
        return None
    start = match.start()
    brace_start = script.find("{", match.end())
    if brace_start == -1:
        return None
    depth = 0
    for index in range(
        brace_start,
        len(script),
    ):
        character = script[index]
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return script[start:index + 1]
    return script[start:]
def print_contexts(
    html: str,
    search_term: str,
    radius: int = 1500,
) -> None:
    """Skriver ut unik HTML/JavaScript-kontext runt ett sökord."""
    print()
    print("=" * 72)
    print(f"KONTEXT: {search_term}")
    print("=" * 72)
    matches = list(
        re.finditer(
            re.escape(search_term),
            html,
            re.IGNORECASE,
        )
    )
    if not matches:
        print("(inga träffar)")
        return
    contexts: set[str] = set()
    for match in matches:
        start = max(
            0,
            match.start() - radius,
        )
        end = min(
            len(html),
            match.end() + radius,
        )
        context = html[start:end].strip()
        contexts.add(context)
    for number, context in enumerate(
        sorted(contexts),
        start=1,
    ):
        print()
        print(f"--- träff {number} ---")
        print(context)
def main() -> int:
    try:
        html = fetch_html()
    except FIError as exc:
        print(f"FI: FEL: {exc}")
        return 1
    parser = ResourceParser()
    parser.feed(html)
    all_urls = [
        value
        for _, value in parser.raw_attributes
    ]
    interesting_urls = [
        value
        for value in all_urls
        if is_interesting(value)
    ]
    endpoints = sorted(
        {
            match.group("endpoint")
            for match in ENDPOINT_PATTERN.finditer(html)
        }
    )
    print("FI-DIAGNOSTIK")
    print(f"URL: {FI_URL}")
    print(
        f"HTML-bytes: "
        f"{len(html.encode('utf-8'))}"
    )
    print(
        f"Alla URL-attribut: "
        f"{len(all_urls)}"
    )
    print(
        f"Script-resurser: "
        f"{len(parser.scripts)}"
    )
    print(
        f"Inline-script: "
        f"{len(parser.inline_scripts)}"
    )
    print_section(
        "BLANKNINGSREGISTER-ENDPOINTS",
        endpoints,
    )
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
    print("RUNREPORT-DEFINITION")
    print("=" * 72)
    functions: set[str] = set()
    for script in parser.inline_scripts:
        function = extract_function(
            script,
            "RunReport",
        )
        if function:
            functions.add(function)
    if functions:
        for function in sorted(functions):
            print(function)
    else:
        print("(ingen function RunReport hittades)")
    print()
    print("=" * 72)
    print("SCRIPTBLOCK MED RUNREPORT")
    print("=" * 72)
    runreport_scripts = [
        script
        for script in parser.inline_scripts
        if "runreport" in script.lower()
    ]
    if runreport_scripts:
        for number, script in enumerate(
            runreport_scripts,
            start=1,
        ):
            print()
            print(f"--- scriptblock {number} ---")
            print(script)
    else:
        print("(inga träffar)")
    print()
    print("=" * 72)
    print("SCRIPTBLOCK MED HTTP-ANROP")
    print("=" * 72)
    http_scripts = [
        script
        for script in parser.inline_scripts
        if any(
            term in script.lower()
            for term in (
                "ajax",
                "fetch(",
                "xmlhttprequest",
                "$.get",
                "$.post",
            )
        )
    ]
    if http_scripts:
        for number, script in enumerate(
            http_scripts,
            start=1,
        ):
            print()
            print(f"--- scriptblock {number} ---")
            print(script)
    else:
        print("(inga träffar)")
    print_contexts(
        html,
        "GetBlankningsregisterAggregat",
    )
    print_contexts(
        html,
        "RunReport",
    )
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
