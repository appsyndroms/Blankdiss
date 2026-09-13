"""
Bygger Blankdiss statiska webbplats.

Webben är den primära presentationen.
"""

from __future__ import annotations

import html
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]

ANALYSIS_DIR = ROOT / "data" / "analysis"
EVENT_DIR = ROOT / "data" / "events"

TEMPLATE = ROOT / "web" / "templates" / "index.html"
STATIC_DIR = ROOT / "web" / "static"

OUTPUT_DIR = ROOT / "pages"

STOCKHOLM = ZoneInfo("Europe/Stockholm")


def read_latest_analysis() -> dict:
    """Läser den senast genererade analysen."""

    files = sorted(
        ANALYSIS_DIR.glob("analysis_*.json")
    )

    if not files:
        return {
            "generated_at": None,
            "results": [],
        }

    return json.loads(
        files[-1].read_text(
            encoding="utf-8"
        )
    )


def read_events() -> list[dict]:
    """Läser samtliga genererade event-filer."""

    records: list[dict] = []

    files = sorted(
        EVENT_DIR.glob("short_events_*.jsonl")
    )

    for path in files:
        with path.open(
            encoding="utf-8"
        ) as handle:

            for line in handle:
                line = line.strip()

                if line:
                    records.append(
                        json.loads(line)
                    )

    return records


def format_pct(value) -> str:
    """Formaterar avkastning som procent."""

    if value is None:
        return "—"

    return f"{value * 100:.1f} %"


def format_pp(value) -> str:
    """Formaterar blankningsförändring i procentenheter."""

    if value is None:
        return "—"

    sign = "+" if value > 0 else ""

    return f"{sign}{value:.2f} pp"


def format_short_interest(value) -> str:
    """Formaterar aktuell blankning."""

    if value is None:
        return "—"

    return f"{value:.2f} %"


def build_event_rows(
    events: list[dict],
) -> str:
    """
    Bygger HTML-rader för katalogen Senaste händelser.

    De senaste händelserna visas först.
    """

    if not events:
        return """
        <tr>
            <td colspan="7" class="empty">
                Inga händelser ännu.
            </td>
        </tr>
        """

    sorted_events = sorted(
        events,
        key=lambda event: (
            event.get("event_date") or ""
        ),
        reverse=True,
    )

    rows: list[str] = []

    for event in sorted_events[:100]:

        issuer = html.escape(
            str(
                event.get(
                    "issuer",
                    "Okänd aktie",
                )
            )
        )

        short_interest = format_short_interest(
            event.get(
                "short_interest_pct"
            )
        )

        change = event.get(
            "change_pp"
        )

        change_class = ""

        if change is not None:
            if change > 0:
                change_class = "increase"
            elif change < 0:
                change_class = "decrease"

        change_html = (
            f'<span class="{change_class}">'
            f'{format_pp(change)}'
            f'</span>'
        )

        return_1d = format_pct(
            event.get("return_1d")
        )

        return_5d = format_pct(
            event.get("return_5d")
        )

        return_20d = format_pct(
            event.get("return_20d")
        )

        return_60d = format_pct(
            event.get("return_60d")
        )

        rows.append(
            f"""
            <tr>
                <td>
                    <strong>{issuer}</strong>
                </td>

                <td>
                    {short_interest}
                </td>

                <td>
                    {change_html}
                </td>

                <td>
                    {return_1d}
                </td>

                <td>
                    {return_5d}
                </td>

                <td>
                    {return_20d}
                </td>

                <td>
                    {return_60d}
                </td>
            </tr>
            """
        )

    return "\n".join(rows)


def build_analysis_rows(
    analysis: dict,
) -> str:
    """Bygger HTML-rader för den historiska analysen."""

    results = analysis.get(
        "results",
        [],
    )

    if not results:
        return """
        <tr>
            <td colspan="6" class="empty">
                Ingen analysdata ännu.
            </td>
        </tr>
        """

    rows: list[str] = []

    for item in results:

        bucket = html.escape(
            str(
                item.get(
                    "bucket",
                    "Okänd",
                )
            )
        )

        events = item.get(
            "events",
            0,
        )

        median_1d = format_pct(
            item.get(
                "median_1d"
            )
        )

        median_5d = format_pct(
            item.get(
                "median_5d"
            )
        )

        median_20d = format_pct(
            item.get(
                "median_20d"
            )
        )

        median_60d = format_pct(
            item.get(
                "median_60d"
            )
        )

        rows.append(
            f"""
            <tr>
                <td>
                    <strong>{bucket}</strong>
                </td>

                <td>
                    {events}
                </td>

                <td>
                    {median_1d}
                </td>

                <td>
                    {median_5d}
                </td>

                <td>
                    {median_20d}
                </td>

                <td>
                    {median_60d}
                </td>
            </tr>
            """
        )

    return "\n".join(rows)


def build_payload() -> dict:
    """Bygger den data som används av webbplatsen."""

    analysis = read_latest_analysis()
    events = read_events()

    generated_at = datetime.now(
        STOCKHOLM
    )

    analysis_results = analysis.get(
        "results",
        [],
    )

    analysis_event_count = sum(
        int(
            item.get(
                "events",
                0
            ) or 0
        )
        for item in analysis_results
    )

    return {
        "generated_at": generated_at.strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        "timezone": "Europe/Stockholm",
        "event_count": len(events),
        "analysis_event_count": analysis_event_count,
        "events": events,
        "analysis": analysis_results,
    }


def build() -> None:
    """Genererar hela den statiska webbplatsen."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = build_payload()

    template = TEMPLATE.read_text(
        encoding="utf-8"
    )

    event_rows = build_event_rows(
        payload["events"]
    )

    analysis_rows = build_analysis_rows(
        {
            "results": payload["analysis"]
        }
    )

    html_output = template

    html_output = html_output.replace(
        "{{GENERATED_AT}}",
        payload["generated_at"],
    )

    html_output = html_output.replace(
        "{{EVENT_COUNT}}",
        str(
            payload["event_count"]
        ),
    )

    html_output = html_output.replace(
        "{{ANALYSIS_EVENT_COUNT}}",
        str(
            payload[
                "analysis_event_count"
            ]
        ),
    )

    html_output = html_output.replace(
        "{{EVENT_ROWS}}",
        event_rows,
    )

    html_output = html_output.replace(
        "{{ANALYSIS_ROWS}}",
        analysis_rows,
    )

    static_source = (
        STATIC_DIR
        / "style.css"
    )

    static_target = (
        OUTPUT_DIR
        / "style.css"
    )

    static_target.write_text(
        static_source.read_text(
            encoding="utf-8"
        ),
        encoding="utf-8",
    )

    (
        OUTPUT_DIR
        / "index.html"
    ).write_text(
        html_output,
        encoding="utf-8",
    )

    (
        OUTPUT_DIR
        / "data.json"
    ).write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"Webb byggd: {OUTPUT_DIR}"
    )

    print(
        "Svensk tid:",
        payload["generated_at"],
    )


def main() -> None:
    build()


if __name__ == "__main__":
    main()
