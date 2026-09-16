"""
Bygger Blankdiss statiska webbplats.
"""
from __future__ import annotations
import html
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
ROOT = Path(__file__).resolve().parent
ANALYSIS_DIR = (
    ROOT
    / "data"
    / "analysis"
)
EVENT_DIR = (
    ROOT
    / "data"
    / "events"
)
ML_DIR = (
    ROOT
    / "data"
    / "processed"
    / "ml"
)
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
OUTPUT_DIR = (
    ROOT
    / "pages"
)
STOCKHOLM = ZoneInfo(
    "Europe/Stockholm"
)
def read_latest_analysis() -> dict:
    files = sorted(
        ANALYSIS_DIR.glob(
            "analysis_*.json"
        )
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
    records: list[dict] = []
    files = sorted(
        EVENT_DIR.glob(
            "short_events_*.jsonl"
        )
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
def read_economic_results() -> dict:
    path = (
        ML_DIR
        / "economic_results.json"
    )
    if not path.exists():
        return {
            "created_at": None,
            "experiments": [],
            "experiment_count": 0,
        }
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )
def format_pct(
    value,
) -> str:
    if value is None:
        return "—"
    return (
        f"{value * 100:.1f} %"
    )
def format_pp(
    value,
) -> str:
    if value is None:
        return "—"
    sign = (
        "+"
        if value > 0
        else ""
    )
    return (
        f"{sign}{value:.2f} pp"
    )
def format_short_interest(
    value,
) -> str:
    if value is None:
        return "—"
    return (
        f"{value:.2f} %"
    )
def format_metric(
    value,
    decimals: int = 3,
) -> str:
    if value is None:
        return "—"
    return (
        f"{value:.{decimals}f}"
    )
def build_event_rows(
    events: list[dict],
) -> str:
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
            event.get(
                "event_date"
            )
            or ""
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
        short_interest = (
            format_short_interest(
                event.get(
                    "short_interest_pct"
                )
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
            f"{format_pp(change)}"
            f"</span>"
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
                    {format_pct(
                        event.get("return_1d")
                    )}
                </td>
                <td>
                    {format_pct(
                        event.get("return_5d")
                    )}
                </td>
                <td>
                    {format_pct(
                        event.get("return_20d")
                    )}
                </td>
                <td>
                    {format_pct(
                        event.get("return_60d")
                    )}
                </td>
            </tr>
            """
        )
    return "\n".join(rows)
def build_analysis_rows(
    analysis: dict,
) -> str:
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
        rows.append(
            f"""
            <tr>
                <td>
                    <strong>{bucket}</strong>
                </td>
                <td>
                    {item.get("events", 0)}
                </td>
                <td>
                    {format_pct(
                        item.get("median_1d")
                    )}
                </td>
                <td>
                    {format_pct(
                        item.get("median_5d")
                    )}
                </td>
                <td>
                    {format_pct(
                        item.get("median_20d")
                    )}
                </td>
                <td>
                    {format_pct(
                        item.get("median_60d")
                    )}
                </td>
            </tr>
            """
        )
    return "\n".join(rows)
def _primary_strategy(
    experiment: dict,
) -> dict | None:
    for strategy in experiment.get(
        "strategies",
        [],
    ):
        fraction = strategy.get(
            "fraction"
        )
        if fraction is not None:
            if abs(
                float(fraction)
                - 0.01
            ) < 1e-12:
                return strategy
    return None
def build_ml_rows(
    economic_results: dict,
) -> str:
    experiments = economic_results.get(
        "experiments",
        [],
    )
    if not experiments:
        return """
        <tr>
            <td colspan="9" class="empty">
                Inga ML-resultat ännu.
            </td>
        </tr>
        """
    rows: list[str] = []
    for experiment in experiments:
        strategy = _primary_strategy(
            experiment
        )
        if strategy is None:
            continue
        feature_set = html.escape(
            str(
                experiment.get(
                    "feature_set",
                    "—",
                )
            )
        )
        target = html.escape(
            str(
                experiment.get(
                    "target",
                    "—",
                )
            )
        )
        task = html.escape(
            str(
                experiment.get(
                    "task",
                    "—",
                )
            )
        )
        model = html.escape(
            str(
                experiment.get(
                    "model",
                    "—",
                )
            )
        )
        direction = html.escape(
            str(
                experiment.get(
                    "direction",
                    "—",
                )
            )
        )
        rows.append(
            f"""
            <tr>
                <td>
                    <strong>{feature_set}</strong>
                </td>
                <td>
                    {target}
                </td>
                <td>
                    {task}
                </td>
                <td>
                    {model}
                </td>
                <td>
                    {direction}
                </td>
                <td>
                    {format_pct(
                        strategy.get(
                            "net_compounded_return"
                        )
                    )}
                </td>
                <td>
                    {format_pct(
                        strategy.get(
                            "benchmark_compounded_return"
                        )
                    )}
                </td>
                <td>
                    {format_pct(
                        strategy.get(
                            "excess_return_vs_benchmark"
                        )
                    )}
                </td>
                <td>
                    {format_pct(
                        strategy.get(
                            "max_drawdown"
                        )
                    )}
                </td>
            </tr>
            """
        )
    if not rows:
        return """
        <tr>
            <td colspan="9" class="empty">
                Inga kompletta ML-strategier ännu.
            </td>
        </tr>
        """
    return "\n".join(rows)
def build_payload() -> dict:
    analysis = (
        read_latest_analysis()
    )
    events = read_events()
    economic_results = (
        read_economic_results()
    )
    generated_at = datetime.now(
        STOCKHOLM
    )
    analysis_results = (
        analysis.get(
            "results",
            [],
        )
    )
    analysis_event_count = sum(
        int(
            item.get(
                "events",
                0,
            )
            or 0
        )
        for item in analysis_results
    )
    return {
        "generated_at": (
            generated_at.strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        ),
        "timezone": (
            "Europe/Stockholm"
        ),
        "event_count": len(
            events
        ),
        "analysis_event_count": (
            analysis_event_count
        ),
        "events": events,
        "analysis": analysis_results,
        "economic_results": (
            economic_results
        ),
    }
def build() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    payload = build_payload()
    template = (
        TEMPLATE.read_text(
            encoding="utf-8"
        )
    )
    html_output = template
    replacements = {
        "{{GENERATED_AT}}": (
            payload["generated_at"]
        ),
        "{{EVENT_COUNT}}": str(
            payload["event_count"]
        ),
        "{{ANALYSIS_EVENT_COUNT}}": str(
            payload[
                "analysis_event_count"
            ]
        ),
        "{{EVENT_ROWS}}": (
            build_event_rows(
                payload["events"]
            )
        ),
        "{{ANALYSIS_ROWS}}": (
            build_analysis_rows(
                {
                    "results": (
                        payload[
                            "analysis"
                        ]
                    )
                }
            )
        ),
        "{{ML_ROWS}}": (
            build_ml_rows(
                payload[
                    "economic_results"
                ]
            )
        ),
        "{{ML_EXPERIMENT_COUNT}}": str(
            payload[
                "economic_results"
            ].get(
                "experiment_count",
                0,
            )
        ),
    }
    for placeholder, value in (
        replacements.items()
    ):
        html_output = (
            html_output.replace(
                placeholder,
                value,
            )
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
    print(
        "ML-experiment:",
        payload[
            "economic_results"
        ].get(
            "experiment_count",
            0,
        ),
    )
def main() -> None:
    build()
if __name__ == "__main__":
    main()
