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


def _event_change(
    event: dict,
) -> float:
    change = event.get(
        "change_pp"
    )

    if change is None:
        return float("inf")

    return float(change)


def _build_event_cells(
    event: dict,
) -> tuple[str, str, str, str, str, str]:
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

    return (
        issuer,
        short_interest,
        change_html,
        format_pct(
            event.get("return_1d")
        ),
        format_pct(
            event.get("return_5d")
        ),
        format_pct(
            event.get("return_20d")
        ),
        format_pct(
            event.get("return_60d")
        ),
    )


def build_buy_signal_rows(
    events: list[dict],
    limit: int = 10,
) -> str:
    """
    Bygger den övre Köpläge-tabellen.

    Den långsiktiga avsikten är att rankingen ska bygga
    på historisk prediktiv evidens. Innan tillräckligt
    många mogna historiska observationer finns används
    största minskning i blankning som ett transparent,
    provisoriskt urval.

    Detta gör att tabellen redan har rätt struktur för
    den framtida prediktiva rankingen utan att låtsas
    att en sådan modell finns ännu.
    """

    if not events:
        return """
        <tr>
            <td colspan="10" class="empty">
                Inga köpläge-observationer ännu.
            </td>
        </tr>
        """

    sorted_events = sorted(
        events,
        key=_event_change,
    )

    rows: list[str] = []

    for rank, event in enumerate(
        sorted_events[:limit],
        start=1,
    ):
        (
            issuer,
            short_interest,
            change_html,
            return_1d,
            return_5d,
            return_20d,
            return_60d,
        ) = _build_event_cells(event)

        # Den här kolumnen kommer senare att fyllas
        # med antalet historiskt jämförbara events.
        historical_events = "—"

        # Utan mogna historiska utfall finns ännu
        # ingen statistiskt underbyggd köpsignal.
        signal = "—"

        rows.append(
            f"""
            <tr>
                <td data-sort-value="{rank}">
                    {rank}
                </td>

                <td>
                    <strong>{issuer}</strong>
                </td>

                <td>
                    {short_interest}
                </td>

                <td>
                    {change_html}
                </td>

                <td data-sort-value="">
                    {historical_events}
                </td>

                <td data-sort-value="">
                    {return_1d}
                </td>

                <td data-sort-value="">
                    {return_5d}
                </td>

                <td data-sort-value="">
                    {return_20d}
                </td>

                <td data-sort-value="">
                    {return_60d}
                </td>

                <td data-sort-value="">
                    {signal}
                </td>
            </tr>
            """
        )

    return "\n".join(rows)


def build_event_rows(
    events: list[dict],
    include_ranking: bool = False,
) -> str:
    colspan = 7

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

    rows: list[str] = []

    for event in sorted_events[:100]:
        (
            issuer,
            short_interest,
            change_html,
            return_1d,
            return_5d,
            return_20d,
            return_60d,
        ) = _build_event_cells(event)

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
