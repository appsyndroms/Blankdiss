from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import pandas as pd


@dataclass
class ExperimentContext:
    """
    Gemensam kontext för ett diagnostiskt experiment.

    data:
        Hela feature-datasetet.

    date_column:
        Datumkolumn för temporal split.

    test_start/test_end:
        Testperiodens gränser.
    """

    data: pd.DataFrame
    date_column: str = "date"
    test_start: str | pd.Timestamp | None = None
    test_end: str | pd.Timestamp | None = None

    def __post_init__(self) -> None:
        self.data = self.data.copy()

        if self.date_column in self.data.columns:
            self.data[self.date_column] = pd.to_datetime(
                self.data[self.date_column],
                errors="coerce",
            )

    @property
    def pretest(self) -> pd.DataFrame:
        if self.test_start is None:
            raise ValueError(
                "test_start måste anges för att använda pretest-data."
            )

        start = pd.Timestamp(
            self.test_start
        )

        return self.data.loc[
            self.data[self.date_column] < start
        ].copy()

    @property
    def test(self) -> pd.DataFrame:
        if self.test_start is None:
            raise ValueError(
                "test_start måste anges för att använda test-data."
            )

        start = pd.Timestamp(
            self.test_start
        )

        mask = (
            self.data[self.date_column]
            >= start
        )

        if self.test_end is not None:
            end = pd.Timestamp(
                self.test_end
            )

            mask &= (
                self.data[self.date_column]
                <= end
            )

        return self.data.loc[mask].copy()

    def quantile(
        self,
        column: str,
        q: float,
    ) -> float:
        values = pd.to_numeric(
            self.pretest[column],
            errors="coerce",
        ).dropna()

        if values.empty:
            raise ValueError(
                f"Kunde inte beräkna quantile för "
                f"'{column}', q={q}."
            )

        value = values.quantile(q)

        if pd.isna(value):
            raise ValueError(
                f"Kunde inte beräkna quantile för "
                f"'{column}', q={q}."
            )

        return float(value)

    def thresholds(
        self,
        column: str,
        quantiles: Iterable[float],
    ) -> dict[float, float]:
        return {
            q: self.quantile(
                column,
                q,
            )
            for q in quantiles
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "rows": len(self.data),
            "pretest_rows": (
                len(self.pretest)
                if self.test_start is not None
                else None
            ),
            "test_rows": (
                len(self.test)
                if self.test_start is not None
                else None
            ),
            "date_column": self.date_column,
            "test_start": (
                str(self.test_start)
                if self.test_start is not None
                else None
            ),
            "test_end": (
                str(self.test_end)
                if self.test_end is not None
                else None
            ),
        }
