"""
Presentation och sammanställning av prospektiva ML-evaluationsresultat.
"""

from __future__ import annotations

import html

from .formatting import (
    format_date,
    format_metric,
)


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
