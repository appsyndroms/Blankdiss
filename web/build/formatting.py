"""
Gemensam formatering för webbplatsens presentation.
"""

from __future__ import annotations

import html


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
