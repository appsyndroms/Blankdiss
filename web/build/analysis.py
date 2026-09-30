"""
Presentation av historiska analysresultat.
"""

from __future__ import annotations

import html

from .formatting import format_pct


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
