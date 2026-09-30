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

EVALUATION_DIR = (
    ML_DIR
    / "research"
    / "evaluation"
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


def read_evaluation_history() -> list[dict]:
    """
    Läser alla sparade prospektiva evaluation-körningar.

    Varje körning ligger i en egen katalog:
        data/processed/ml/research/evaluation/<run-id>/

    och innehåller evaluation.json.
    """
    if not EVALUATION_DIR.exists():
        return []

    evaluations: list[dict] = []

    for path in sorted(
        EVALUATION_DIR.glob(
            "*/evaluation.json"
        )
    ):
        try:
            payload = json.loads(
                path.read_text(
                    encoding="utf-8"
                )
            )
        except (
            OSError,
            json.JSONDecodeError,
        ):
            continue

        payload["_source_path"] = str(
            path.relative_to(ROOT)
        )

        evaluations.append(
            payload
        )

    evaluations.sort(
        key=lambda item: (
            item.get(
                "created_at_utc",
                "",
            )
        ),
        reverse=True,
    )

    return evaluations


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


def format_date(
    value,
) -> str:
    if not value:
        return "—"

    return html.escape(
        str(value)[:10]
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


def _walk_forward_result(
    evaluation: dict,
) -> dict:
    walk_forward = evaluation.get(
        "walk_forward",
        {},
    )

    if not isinstance(
        walk_forward,
        dict,
    ):
        return {}

    return walk_forward


def _result_rows(
    evaluation: dict,
) -> list[dict]:
    results = evaluation.get(
        "results",
        [],
    )

    if not isinstance(
        results,
        list,
    ):
        return []

    return [
        result
        for result in results
        if isinstance(
            result,
            dict,
        )
    ]


def _result_event_count(
    result: dict,
) -> int:
    """
    Försöker läsa antal observationer från
    de vanligaste resultatnivåerna utan att
    anta ett enda resultatformat.
    """
    candidates = [
        result.get("events"),
        result.get("event_count"),
        result.get("n"),
        result.get("observations"),
        result.get("observation_count"),
    ]

    for value in candidates:
        if isinstance(
            value,
            (int, float),
        ):
            return int(value)

    analysis = result.get(
        "analysis",
        {},
    )

    if isinstance(
        analysis,
        dict,
    ):
        candidates = [
            analysis.get("events"),
            analysis.get("event_count"),
            analysis.get("n"),
            analysis.get("observations"),
            analysis.get("observation_count"),
        ]

        for value in candidates:
            if isinstance(
                value,
                (int, float),
            ):
                return int(value)

    return 0


def _matured_observation_count(
    evaluation: dict,
) -> int | None:
    """
    Hämtar mogen observation count om fältet
    finns i evaluation-resultatet.
    """
    possible = [
        evaluation.get(
            "matured_observations"
        ),
        evaluation.get(
            "matured_observation_count"
        ),
    ]

    data = evaluation.get(
        "data",
        {},
    )

    if isinstance(
        data,
        dict,
    ):
        possible.extend(
            [
                data.get(
                    "matured_observations"
                ),
                data.get(
                    "matured_observation_count"
                ),
            ]
        )

    for value in possible:
        if isinstance(
            value,
            (int, float),
        ):
            return int(value)

    return None


def _total_observation_count(
    evaluation: dict,
) -> int:
    data = evaluation.get(
        "data",
        {},
    )

    if isinstance(
        data,
        dict,
    ):
        for key in (
            "observations",
            "observation_count",
            "feature_rows",
        ):
            value = data.get(key)

            if isinstance(
                value,
                (int, float),
            ):
                return int(value)

    results = _result_rows(
        evaluation
    )

    return sum(
        _result_event_count(result)
        for result in results
    )


def _latest_result(
    evaluation: dict,
) -> dict:
    results = _result_rows(
        evaluation
    )

    if not results:
        return {}

    return results[-1]


def _metric_from_result(
    result: dict,
    names: tuple[str, ...],
):
    """
    Letar efter ett metric-fält på både
    toppnivå och vanliga nested-nivåer.
    """
    containers = [
        result
    ]

    for key in (
        "metrics",
        "analysis",
        "statistics",
        "aggregate",
        "summary",
    ):
        value = result.get(key)

        if isinstance(
            value,
            dict,
        ):
            containers.append(
                value
            )

    for container in containers:
        for name in names:
            value = container.get(
                name
            )

            if value is not None:
                return value

    return None


def _candidate_name(
    evaluation: dict,
) -> str:
    candidate = evaluation.get(
        "candidate",
        {},
    )

    if isinstance(
        candidate,
        dict,
    ):
        return str(
            candidate.get(
                "id",
                "Okänd kandidat",
            )
        )

    return "Okänd kandidat"


def build_evaluation_summary(
    evaluations: list[dict],
) -> dict:
    if not evaluations:
        return {
            "run_count": 0,
            "latest": None,
            "candidate": None,
            "latest_feature_date": None,
            "latest_candidate_feature_date": None,
            "total_observations": 0,
            "matured_observations": None,
        }

    latest = evaluations[0]

    total_observations = _total_observation_count(
        latest
    )

    matured_observations = (
        _matured_observation_count(
            latest
        )
    )

    data = latest.get(
        "data",
        {},
    )

    latest_feature_date = None
    latest_candidate_feature_date = None

    if isinstance(
        data,
        dict,
    ):
        latest_feature_date = data.get(
            "latest_feature_date"
        )

        latest_candidate_feature_date = (
            data.get(
                "latest_candidate_feature_date"
            )
        )

    return {
        "run_count": len(
            evaluations
        ),
        "latest": latest,
        "candidate": _candidate_name(
            latest
        ),
        "latest_feature_date": (
            latest_feature_date
        ),
        "latest_candidate_feature_date": (
            latest_candidate_feature_date
        ),
        "total_observations": (
            total_observations
        ),
        "matured_observations": (
            matured_observations
        ),
    }


def build_evaluation_history_rows(
    evaluations: list[dict],
) -> str:
    if not evaluations:
        return """
        <tr>
            <td colspan="8" class="empty">
                Inga prospektiva evaluation-resultat ännu.
            </td>
        </tr>
        """

    rows: list[str] = []

    for evaluation in evaluations[:100]:
        created = evaluation.get(
            "created_at_utc"
        )

        data = evaluation.get(
            "data",
            {},
        )

        if not isinstance(
            data,
            dict,
        ):
            data = {}

        candidate = evaluation.get(
            "candidate",
            {},
        )

        if not isinstance(
            candidate,
            dict,
        ):
            candidate = {}

        candidate_id = html.escape(
            str(
                candidate.get(
                    "id",
                    "—",
                )
            )
        )

        candidate_version = (
            candidate.get(
                "version",
                "—",
            )
        )

        latest_feature_date = (
            data.get(
                "latest_candidate_feature_date"
            )
            or data.get(
                "latest_feature_date"
            )
        )

        observation_count = (
            _total_observation_count(
                evaluation
            )
        )

        matured = (
            _matured_observation_count(
                evaluation
            )
        )

        latest_result = _latest_result(
            evaluation
        )

        auc = _metric_from_result(
            latest_result,
            (
                "auc",
                "roc_auc",
                "mean_auc",
            ),
        )

        lift = _metric_from_result(
            latest_result,
            (
                "top_0_1_lift",
                "top_01_lift",
                "top_0.1_lift",
                "lift",
            ),
        )

        run_id = html.escape(
            str(
                evaluation.get(
                    "run_id",
                    "—",
                )
            )
        )

        rows.append(
            f"""
            <tr>
                <td>
                    <strong>
                        {format_date(
                            latest_feature_date
                        )}
                    </strong>
                </td>
                <td>
                    {format_date(created)}
                </td>
                <td>
                    {candidate_id}
                    <span class="subtle">
                        v{candidate_version}
                    </span>
                </td>
                <td>
                    {observation_count:,}
                </td>
                <td>
                    {
                        "—"
                        if matured is None
                        else f"{matured:,}"
                    }
                </td>
                <td>
                    {format_metric(auc)}
                </td>
                <td>
                    {
                        "—"
                        if lift is None
                        else format_metric(
                            lift,
                            2,
                        )
                    }
                </td>
                <td>
                    <span class="run-id">
                        {run_id[:16]}
                    </span>
                </td>
            </tr>
            """
        )

    return "\n".join(rows)


def build_evaluation_latest(
    evaluations: list[dict],
) -> dict:
    if not evaluations:
        return {
            "candidate": "—",
            "version": "—",
            "feature_date": "—",
            "created_at": "—",
            "run_id": "—",
            "feature_rows": "—",
            "fingerprint": "—",
        }

    latest = evaluations[0]

    candidate = latest.get(
        "candidate",
        {},
    )

    if not isinstance(
        candidate,
        dict,
    ):
        candidate = {}

    data = latest.get(
        "data",
        {},
    )

    if not isinstance(
        data,
        dict,
    ):
        data = {}

    fingerprint = str(
        data.get(
            "fingerprint",
            "—",
        )
    )

    return {
        "candidate": str(
            candidate.get(
                "id",
                "—",
            )
        ),
        "version": str(
            candidate.get(
                "version",
                "—",
            )
        ),
        "feature_date": str(
            data.get(
                "latest_candidate_feature_date",
                "—",
            )
        ),
        "created_at": str(
            latest.get(
                "created_at_utc",
                "—",
            )
        ),
        "run_id": str(
            latest.get(
                "run_id",
                "—",
            )
        ),
        "feature_rows": str(
            data.get(
                "feature_rows",
                "—",
            )
        ),
        "fingerprint": fingerprint,
    }


def build_evaluation_payload(
    evaluations: list[dict],
) -> dict:
    summary = build_evaluation_summary(
        evaluations
    )

    latest = build_evaluation_latest(
        evaluations
    )

    return {
        "run_count": summary[
            "run_count"
        ],
        "candidate": summary[
            "candidate"
        ],
        "latest_feature_date": summary[
            "latest_candidate_feature_date"
        ],
        "total_observations": summary[
            "total_observations"
        ],
        "matured_observations": summary[
            "matured_observations"
        ],
        "latest": latest,
        "history": evaluations,
    }


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

        "{{EVALUATION_HISTORY_ROWS}}": (
            build_evaluation_history_rows(
                payload["evaluations"]
            )
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
    }

    html_output = template

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

    print(
        "Prospektiva evaluation-körningar:",
        len(
            payload["evaluations"]
        ),
    )


def main() -> None:
    build()


if __name__ == "__main__":
    main()
