from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from datetime import date
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen


DEFAULT_INDEX_URL = (
    "https://bolagsverket.se/apierochoppnadata/"
    "hamtaforetagsinformation/nedladdningsbarafiler.2517.html"
)


class LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[tuple[str, str]] = []
        self._text: list[str] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        if tag.lower() == "a":
            href = dict(attrs).get("href")
            if href:
                self._text = []
                self.links.append((href, ""))

    def handle_data(self, data: str) -> None:
        if self.links:
            self._text.append(data.strip())

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "a" and self.links:
            href, _ = self.links[-1]
            self.links[-1] = (
                href,
                " ".join(x for x in self._text if x),
            )


def fetch(url: str) -> bytes:
    request = Request(
        url,
        headers={
            "User-Agent": "Blankdiss/1.0 (open-data ingestion)",
            "Accept": (
                "text/html,application/xhtml+xml,"
                "application/zip,*/*"
            ),
        },
    )

    with urlopen(request, timeout=60) as response:
        return response.read()


def discover_links(index_url: str) -> list[str]:
    raw = fetch(index_url)

    parser = LinkParser()
    parser.feed(raw.decode("utf-8", errors="replace"))

    urls: list[str] = []

    for href, text in parser.links:
        absolute = urljoin(index_url, href)
        haystack = f"{absolute} {text}".lower()

        if "arsredovis" in haystack:
            urls.append(absolute)

    return list(dict.fromkeys(urls))


def parse_date_tokens(value: str) -> list[date]:
    dates: list[date] = []

    for match in re.findall(
        r"(20\d{2})[-_](\d{2})[-_](\d{2})",
        value,
    ):
        try:
            dates.append(
                date(
                    int(match[0]),
                    int(match[1]),
                    int(match[2]),
                )
            )
        except ValueError:
            pass

    return dates


def relevant(url: str, start: date, end: date) -> bool:
    dates = parse_date_tokens(url)

    if not dates:
        return True

    return any(start <= current <= end for current in dates)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def download(url: str, output: Path) -> None:
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    request = Request(
        url,
        headers={
            "User-Agent": "Blankdiss/1.0 (open-data ingestion)",
        },
    )

    with (
        urlopen(request, timeout=180) as response,
        output.open("wb") as file,
    ):
        while True:
            chunk = response.read(1024 * 1024)

            if not chunk:
                break

            file.write(chunk)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Discover and download "
            "Bolagsverket open annual-report archives."
        )
    )

    parser.add_argument(
        "--start",
        default="2020-01-01",
    )

    parser.add_argument(
        "--end",
        default="2020-06-30",
    )

    parser.add_argument(
        "--output",
        default="data/bolagsverket",
    )

    parser.add_argument(
        "--index-url",
        default=DEFAULT_INDEX_URL,
    )

    parser.add_argument(
        "--max-archives",
        type=int,
        default=30,
    )

    args = parser.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    output = Path(args.output)

    print(
        f"Bolagsverket period: "
        f"{start} -> {end}"
    )
    print(f"Output: {output}")

    try:
        links = discover_links(args.index_url)
    except Exception as exc:
        print(
            "ERROR: could not discover "
            f"Bolagsverket links: {exc}",
            file=sys.stderr,
        )
        return 2

    archives = [
        url
        for url in links
        if urlparse(url).path.lower().endswith(
            (".zip", ".zip/")
        )
        and relevant(url, start, end)
    ]

    if not archives:
        print(
            "ERROR: no dated ZIP archives "
            "were discovered. The Bolagsverket "
            "page may have changed or may require "
            "browser rendering.",
            file=sys.stderr,
        )
        return 3

    archives = archives[: args.max_archives]

    print(
        f"Archives selected: {len(archives)}"
    )

    manifest: dict[str, object] = {
        "source": "Bolagsverket",
        "source_url": args.index_url,
        "start": args.start,
        "end": args.end,
        "archives": [],
    }

    for index, url in enumerate(
        archives,
        start=1,
    ):
        filename = (
            Path(urlparse(url).path).name
            or f"archive_{index:03d}.zip"
        )

        destination = (
            output / "raw" / filename
        )

        print(
            f"[{index}/{len(archives)}] {url}"
        )

        if not destination.exists():
            download(
                url,
                destination,
            )

        entry = {
            "url": url,
            "file": str(destination),
            "bytes": destination.stat().st_size,
            "sha256": sha256(destination),
        }

        manifest["archives"].append(entry)

        time.sleep(0.25)

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_path = output / "manifest.json"

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Wrote {manifest_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
