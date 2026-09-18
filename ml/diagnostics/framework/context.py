from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import pandas as pd


@dataclass
class ExperimentContext:
    """
    Gemensam kontext för ett diagnostics-experiment.

    Context representerar ett enda walk-forward-fönster:

        TRAIN
            <= train_end

        VALIDATION
            > train_end
            <= validation_end

        TEST / OOS
            > validation_end
            <= test_end

    Alla thresholds och modellval som ska användas på TEST
    måste härledas från TRAIN/VALIDATION enligt experimentets
    definition.
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

    @property
    def train(self) -> pd.DataFrame:
        if self.train_end is None:
            raise ValueError(
                "train_end måste anges."
            )

        end = pd.Timestamp(self.train_end)

        return self.data.loc[
            self.data[self.date_column] <= end
        ].copy()

    @property
    def validation(self) -> pd.DataFrame:
        if self.train_end is None:
            raise ValueError(
                "train_end måste anges."
            )

        if self.validation_end is None:
            raise ValueError(
                "validation_end måste anges."
            )

        train_end = pd.Timestamp(
            self.train_end
        )
        validation_end = pd.Timestamp(
            self.validation_end
        )

        mask = (
            self.data[self.date_column] > train_end
        ) & (
            self.data[self.date_column]
            <= validation_end
        )

        return self.data.loc[mask].copy()

    @property
    def pretest(self) -> pd.DataFrame:
        """
        All data som får användas för att definiera
        information före OOS-testet.

        Detta är TRAIN + VALIDATION.
        """

        if self.validation_end is None:
            raise ValueError(
                "validation_end måste anges."
            )

        end = pd.Timestamp(
            self.validation_end
        )

        return self.data.loc[
            self.data[self.date_column] <= end
        ].copy()

    @property
    def test(self) -> pd.DataFrame:
        if self.validation_end is None:
            raise ValueError(
                "validation_end måste anges."
            )

        start = pd.Timestamp(
            self.validation_end
        )

        mask = (
            self.data[self.date_column] > start
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
        """
        Beräknar threshold ENBART på pre-test-data.

        Testdata får aldrig påverka thresholden.
        """

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
                f"Kan inte beräkna quantile för "
                f"'{column}'."
            )

        value = values.quantile(q)

        if pd.isna(value):
            raise ValueError(
                f"Quantile blev NaN för "
                f"'{column}', q={q}."
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
            "train_rows": int(len(self.train))
            if self.train_end is not None
            else None,
            "validation_rows": int(
                len(self.validation)
            )
            if (
                self.train_end is not None
                and self.validation_end is not None
            )
            else None,
            "pretest_rows": int(
                len(self.pretest)
            )
            if self.validation_end is not None
            else None,
            "test_rows": int(len(self.test))
            if self.validation_end is not None
            else None,
            "date_column": self.date_column,
            "train_end": (
                str(self.train_end)
                if self.train_end is not None
                else None
            ),
            "validation_end": (
                str(self.validation_end)
                if self.validation_end is not None
                else None
            ),
            "test_end": (
                str(self.test_end)
                if self.test_end is not None
                else None
            ),
        }
