from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import pandas as pd


@dataclass
class ExperimentContext:
    """
    Context för ett enda walk-forward-fönster.

    TRAIN:
        <= train_end

    VALIDATION:
        > train_end
        <= validation_end

    PRETEST:
        TRAIN + VALIDATION

    TEST:
        > validation_end
        <= test_end
    """

    data: pd.DataFrame

    date_column: str = "snapshot_date"

    train_end: str | pd.Timestamp | None = None
    validation_end: str | pd.Timestamp | None = None
    test_end: str | pd.Timestamp | None = None

    def __post_init__(self) -> None:
        self.data = self.data.copy()

        if self.date_column not in self.data.columns:
            raise KeyError(
                f"Saknar datumkolumn: {self.date_column}"
            )

        self.data[self.date_column] = pd.to_datetime(
            self.data[self.date_column],
            errors="coerce",
        )

        self.data = self.data.loc[
            self.data[self.date_column].notna()
        ].copy()

        self.data.sort_values(
            [self.date_column],
            inplace=True,
        )

        self.data.reset_index(
            drop=True,
            inplace=True,
        )

    @property
    def train(self) -> pd.DataFrame:
        self._require("train_end")

        return self.data.loc[
            self.data[self.date_column]
            <= pd.Timestamp(self.train_end)
        ].copy()

    @property
    def validation(self) -> pd.DataFrame:
        self._require(
            "train_end",
            "validation_end",
        )

        train_end = pd.Timestamp(self.train_end)
        validation_end = pd.Timestamp(self.validation_end)

        mask = (
            (self.data[self.date_column] > train_end)
            & (self.data[self.date_column] <= validation_end)
        )

        return self.data.loc[mask].copy()

    @property
    def pretest(self) -> pd.DataFrame:
        self._require("validation_end")

        return self.data.loc[
            self.data[self.date_column]
            <= pd.Timestamp(self.validation_end)
        ].copy()

    @property
    def test(self) -> pd.DataFrame:
        self._require("validation_end")

        mask = (
            self.data[self.date_column]
            > pd.Timestamp(self.validation_end)
        )

        if self.test_end is not None:
            mask &= (
                self.data[self.date_column]
                <= pd.Timestamp(self.test_end)
            )

        return self.data.loc[mask].copy()

    def quantile(
        self,
        column: str,
        q: float,
    ) -> float:
        if column not in self.pretest.columns:
            raise KeyError(
                f"Saknar kolumn: {column}"
            )

        values = pd.to_numeric(
            self.pretest[column],
            errors="coerce",
        ).dropna()

        if values.empty:
            raise ValueError(
                f"Kan inte beräkna quantile för '{column}'."
            )

        value = values.quantile(q)

        if pd.isna(value):
            raise ValueError(
                f"Quantile blev NaN för '{column}', q={q}."
            )

        return float(value)

    def thresholds(
        self,
        column: str,
        quantiles: Iterable[float],
    ) -> dict[float, float]:
        return {
            q: self.quantile(column, q)
            for q in quantiles
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "rows": int(len(self.data)),
            "train_rows": int(len(self.train)),
            "validation_rows": int(len(self.validation)),
            "pretest_rows": int(len(self.pretest)),
            "test_rows": int(len(self.test)),
            "date_column": self.date_column,
            "train_end": str(self.train_end),
            "validation_end": str(self.validation_end),
            "test_end": (
                str(self.test_end)
                if self.test_end is not None
                else None
            ),
        }

    def _require(self, *names: str) -> None:
        for name in names:
            if getattr(self, name) is None:
                raise ValueError(
                    f"{name} måste anges."
                )
