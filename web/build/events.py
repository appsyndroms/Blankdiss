"""
Presentation av historiska short-events.
"""

from __future__ import annotations

import html

from .formatting import (
    format_pct,
    format_pp,
    format_short_interest,
)


def _event_identity(
    event: dict,
) -> tuple[str, str]:
    event_date = str(
        event.get("event_date")
        or ""
    )

    identity = (
        event.get("isin")
        or event.get("lei")
        or event.get("yahoo_symbol")
        or event.get("issuer")
        or ""
    )

    return (
        event_date,
        str(identity),
    )


def build_event_rows(
    events: list[dict],
    include_ranking: bool = False,
) -> str:
    colspan = (
        8
        if include_ranking
        else 7
    )

    if not events:
        return f"""
        <tr>
            <td colspan="{colspan}" class="empty">
                Inga händelser ännu.
            </td>
        </tr>
        """

    sorted_events = sorted(
        events,
        key=lambda event: (
            event.get("event_date")
            or ""
        ),
        reverse=True,
    )

    ranking: dict[
        tuple[str, str],
        int
    ] = {}

    if include_ranking:
        # Köpläge ska vara försorterat efter
        # bästa ranking. Största minskningen
        # av blankningen får rank 1.
        ranked_events = sorted(
            sorted_events,
            key=lambda event: (
                event.get("change_pp")
                if event.get(
                    "change_pp"
                ) is not None
                else float("inf")
            ),
        )

        for index, event in enumerate(
            ranked_events,
            start=1,
        ):
            ranking[
                _event_identity(event)
            ] = index

        # Viktigt: själva tabellen ska också
        # visas i rankingordning när sidan
        # öppnas.
        sorted_events = ranked_events

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

        ranking_html = ""

        if include_ranking:
            rank = ranking.get(
                _event_identity(event)
            )

            ranking_html = (
                f'<td data-sort-value="{rank or ""}">'
                f"{rank or '—'}"
                f"</td>"
            )

        rows.append(
            f"""
            <tr>
                {ranking_html}
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
