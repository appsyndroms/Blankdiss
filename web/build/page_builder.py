"""
Bygger Blankdiss statiska webbplats.

Webbplatsen består av:
    index.html
    koplage.html
    events.html
    bedomning.html
    dataanalys.html

All dataläsning och presentationslogik ligger
i separata build-moduler.
"""

from __future__ import annotations

import html
import json

from datetime import datetime

from .analysis import build_analysis_rows

from .config import (
    OUTPUT_DIR,
    STATIC_DIR,
    STOCKHOLM,
    TEMPLATE_DIR,
)

from .data_loader import (
    read_economic_results,
    read_evaluation_history,
    read_events,
    read_latest_analysis,
)

from .evaluation import (
    build_evaluation_history_rows,
    build_evaluation_latest,
    build_evaluation_payload,
    build_evaluation_summary,
)

from .events import build_event_rows
from .ml import build_ml_rows


def build_payload() -> dict:
    analysis = (
        read_latest_analysis()
    )

    events = read_events()

    economic_results = (
        read_economic_results()
    )

    evaluations = (
        read_evaluation_history()
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
        "evaluations": (
            evaluations
        ),
        "evaluation_payload": (
            build_evaluation_payload(
                evaluations
            )
        ),
    }


def _common_replacements(
    payload: dict,
) -> dict[str, str]:
    evaluation_summary = (
        build_evaluation_summary(
            payload["evaluations"]
        )
    )

    latest_evaluation = (
        build_evaluation_latest(
            payload["evaluations"]
        )
    )

    matured = (
        evaluation_summary[
            "matured_observations"
        ]
    )

    if matured is None:
        matured_text = "—"
    else:
        matured_text = f"{matured:,}"

    return {
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
        "{{ML_EXPERIMENT_COUNT}}": str(
            payload[
                "economic_results"
            ].get(
                "experiment_count",
                0,
            )
        ),
        "{{EVALUATION_RUN_COUNT}}": str(
            evaluation_summary[
                "run_count"
            ]
        ),
        "{{EVALUATION_CANDIDATE}}": (
            html.escape(
                str(
                    evaluation_summary[
                        "candidate"
                    ]
                    or "—"
                )
            )
        ),
        "{{EVALUATION_FEATURE_DATE}}": (
            html.escape(
                str(
                    evaluation_summary[
                        "latest_candidate_feature_date"
                    ]
                    or "—"
                )
            )
        ),
        "{{EVALUATION_OBSERVATIONS}}": (
            f"{evaluation_summary['total_observations']:,}"
        ),
        "{{EVALUATION_MATURED}}": (
            matured_text
        ),
        "{{EVALUATION_LATEST_RUN}}": (
            html.escape(
                latest_evaluation[
                    "run_id"
                ]
            )
        ),
        "{{EVALUATION_LATEST_ROWS}}": (
            html.escape(
                latest_evaluation[
                    "feature_rows"
                ]
            )
        ),
    }


def _page_replacements(
    payload: dict,
) -> dict[str, str]:
    replacements = _common_replacements(
        payload
    )

    replacements.update(
        {
            "{{KOPLAGE_ROWS}}": (
                build_event_rows(
                    payload["events"],
                    include_ranking=True,
                )
            ),
            "{{EVENT_ROWS}}": (
                build_event_rows(
                    payload["events"],
                    include_ranking=False,
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
            "{{EVALUATION_HISTORY_ROWS}}": (
                build_evaluation_history_rows(
                    payload[
                        "evaluations"
                    ]
                )
            ),
        }
    )

    return replacements


def _render_template(
    template_name: str,
    replacements: dict[str, str],
) -> str:
    template_path = (
        TEMPLATE_DIR
        / template_name
    )

    html_output = (
        template_path.read_text(
            encoding="utf-8"
        )
    )

    for placeholder, value in (
        replacements.items()
    ):
        html_output = (
            html_output.replace(
                placeholder,
                value,
            )
        )

    return html_output


def _write_page(
    template_name: str,
    output_name: str,
    replacements: dict[str, str],
) -> None:
    html_output = _render_template(
        template_name,
        replacements,
    )

    (
        OUTPUT_DIR
        / output_name
    ).write_text(
        html_output,
        encoding="utf-8",
    )


def _copy_static() -> None:
    for filename in (
        "style.css",
        "site.js",
    ):
        static_source = (
            STATIC_DIR
            / filename
        )

        static_target = (
            OUTPUT_DIR
            / filename
        )

        static_target.write_text(
            static_source.read_text(
                encoding="utf-8"
            ),
            encoding="utf-8",
        )


def _write_data(
    payload: dict,
) -> None:
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


def build() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = build_payload()

    replacements = _page_replacements(
        payload
    )

    _write_page(
        "index.html",
        "index.html",
        replacements,
    )

    _write_page(
        "koplage.html",
        "koplage.html",
        replacements,
    )

    _write_page(
        "events.html",
        "events.html",
        replacements,
    )

    _write_page(
        "bedomning.html",
        "bedomning.html",
        replacements,
    )

    _write_page(
        "dataanalys.html",
        "dataanalys.html",
        replacements,
    )

    _copy_static()

    _write_data(
        payload
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

    print(
        "Prospektiva evaluation-körningar:",
        len(
            payload["evaluations"]
        ),
    )
