"""
Bygger Blankdiss statiska webbplats.

Webben är den primära presentationen.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

ANALYSIS_DIR = ROOT / "data" / "analysis"
EVENT_DIR = ROOT / "data" / "events"

TEMPLATE = (
    ROOT
    / "web"
    / "templates"
    / "index.html"
)

STATIC_DIR = (
    ROOT
    / "web"
    / "static"
)

OUTPUT_DIR = ROOT / "web_site"


def read_latest_analysis() -> dict:
    files = sorted(
        ANALYSIS_DIR.glob(
            "analysis_*.json"
        )
    )

    if not files:
        return {
            "results": []
        }

    return json.loads(
        files[-1].read_text(
            encoding="utf-8"
        )
    )


def read_events() -> list[dict]:
    records = []

    for path in sorted(
        EVENT_DIR.glob(
            "short_events_*.jsonl"
        )
    ):
        with path.open(
            encoding="utf-8"
        ) as handle:
            for line in handle:
                if line.strip():
                    records.append(
                        json.loads(line)
                    )

    return records


def format_pct(value) -> str:
    if value is None:
        return "—"

    return f"{value * 100:.1f} %"


def build_payload() -> dict:
    analysis = read_latest_analysis()
    events = read_events()

    return {
        "generated_at":
            datetime.now().isoformat(),

        "event_count":
            len(events),

        "analysis":
            analysis.get(
                "results",
                [],
            ),
    }


def build() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = build_payload()

    template = TEMPLATE.read_text(
        encoding="utf-8"
    )

    html = template.replace(
        "{{GENERATED_AT}}",
        payload["generated_at"],
    )

    html = html.replace(
        "{{EVENT_COUNT}}",
        str(payload["event_count"]),
    )

    rows = []

    for item in payload["analysis"]:
        rows.append(
            f"""
            <tr>
                <td>{item["bucket"]}</td>
                <td>{item["events"]}</td>
                <td>{format_pct(item.get("median_1d"))}</td>
                <td>{format_pct(item.get("median_5d"))}</td>
                <td>{format_pct(item.get("median_20d"))}</td>
                <td>{format_pct(item.get("median_60d"))}</td>
            </tr>
            """
        )

    html = html.replace(
        "{{ANALYSIS_ROWS}}",
        "\n".join(rows),
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
        html,
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
        f"Webb: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    build()
