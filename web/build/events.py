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
