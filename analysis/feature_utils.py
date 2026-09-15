"""Gemensamma hjälpfunktioner för feature-bygget."""
from __future__ import annotations

import re
from typing import Any

import pandas as pd


def normalize_text(
    value: Any,
) -> str:
    if value is None or pd.isna(value):
        return ""

    text = str(value).strip().upper()

    text = re.sub(
        r"[^A-Z0-9ÅÄÖÉÜÆØ]+",
        " ",
        text,
    )

    return re.sub(
        r"\s+",
        " ",
        text,
    ).strip()


def security_key(
    isin: Any,
    issuer: Any,
) -> str:
    isin_text = normalize_text(
        isin
    )

    if isin_text:
        return f"ISIN:{isin_text}"

    return (
        "ISSUER:"
        f"{normalize_text(issuer)}"
    )


def clean_for_json(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    frame = frame.copy()

    date_columns = {
        "snapshot_date",
        "previous_snapshot_date",
        "price_date",
    }

    for column in date_columns:
        if column not in frame.columns:
            continue

        frame[column] = (
            pd.to_datetime(
                frame[column],
                errors="coerce",
            )
            .dt.strftime(
                "%Y-%m-%d"
            )
        )

    frame = frame.astype(object)

    return frame.where(
        pd.notna(frame),
        None,
    )


def feature_key(
    frame: pd.DataFrame,
) -> pd.Series:
    return (
        frame["security_key"].astype(str)
        + "|"
        + frame["snapshot_date"].dt.strftime(
            "%Y-%m-%d"
        )
    )
