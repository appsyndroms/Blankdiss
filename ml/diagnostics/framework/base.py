from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .metrics import event_summary
from .stratification import two_dimensional_stratification


@dataclass
class ExperimentResult:
    """
    Standardiserat resultat från ett diagnostiskt experiment.
    """

    name: str
    description: str = ""

    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_table(
        self,
        name: str,
        table: pd.DataFrame,
    ) -> None:
        self.tables[name] = table

    def add_metric(
        self,
        name: str,
        value: Any,
    ) -> None:
        self.metrics[name] = value

    def add_metadata(
        self,
        name: str,
        value: Any,
    ) -> None:
        self.metadata[name] = value


class DiagnosticExperiment(ABC):
    """
    Basinterface för diagnostiska experiment.

    Experimentklassen ska normalt bara definiera:

        - name
        - description
        - targets
        - analyze_window()

    Gemensam diagnostiklogik hör hemma här eller i frameworkets
    hjälpfunktioner, inte i varje experimentfil.
    """

    name: str = ""
    description: str = ""
    targets: tuple[str, ...] = ()

    def __init__(self, **kwargs: Any) -> None:
        self.options = kwargs

    @abstractmethod
    def analyze_window(
        self,
        context,
    ) -> Any:
        """
        Kör själva experimentanalysen för ett testfönster.
        """
        raise NotImplementedError

    def run(
        self,
        context,
    ) -> ExperimentResult:
        """
        Standardiserad körning.

        Ett experiment behöver normalt inte implementera denna metod.
        """

        result = ExperimentResult(
            name=self.name,
            description=self.description,
        )

        analysis = self.analyze_window(context)

        if isinstance(analysis, ExperimentResult):
            return analysis

        if isinstance(analysis, pd.DataFrame):
            result.add_table(
                "analysis",
                analysis,
            )

        elif isinstance(analysis, dict):
            self._add_analysis(
                result,
                analysis,
            )

        elif analysis is not None:
            result.metadata["analysis"] = analysis

        result.metadata.update(
            context.metadata()
        )

        return result

    def execute(
        self,
        context,
    ) -> ExperimentResult:
        result = self.run(context)

        if not isinstance(
            result,
            ExperimentResult,
        ):
            raise TypeError(
                f"Experiment '{self.name}' måste "
                f"returnera ExperimentResult, "
                f"men returnerade "
                f"{type(result).__name__}."
            )

        return result

    # ------------------------------------------------------------------
    # Pre-test stratification
    # ------------------------------------------------------------------

    def make_pretest_bins(
        self,
        test: pd.DataFrame,
        column: str,
        quantiles: Iterable[float] = (
            0.20,
            0.50,
            0.80,
        ),
    ) -> dict[str, float]:
        """
        Skapar thresholds från pre-test-data.

        OBS:
            `test` används endast för att säkerställa index/struktur.
            Själva thresholds beräknas från den aktiva contextens
            pre-test-data.

        Returnerar ett enkelt label -> threshold-format som kan
        användas av frameworkets stratifieringsfunktioner.
        """

        # `test` finns med i API:t eftersom experimentet uttryckligen
        # arbetar mot testperioden. Thresholds måste däremot alltid
        # hämtas från pre-test-data.
        del test

        context = self._context

        values = pd.to_numeric(
            context.pretest[column],
            errors="coerce",
        ).dropna()

        if values.empty:
            raise ValueError(
                f"Kan inte skapa pre-test bins för "
                f"'{column}': inga numeriska värden."
            )

        thresholds: dict[str, float] = {}

        for quantile in quantiles:
            value = values.quantile(
                quantile
            )

            if pd.isna(value):
                raise ValueError(
                    f"Kunde inte beräkna threshold för "
                    f"'{column}', q={quantile}."
                )

            thresholds[
                f"q{int(quantile * 100):02d}"
            ] = float(value)

        return thresholds

    def build_2d_analysis(
        self,
        test: pd.DataFrame,
        x_bins: dict[str, float],
        y_bins: dict[str, float],
        targets: Iterable[str] | None = None,
        x_column: str | None = None,
        y_column: str | None = None,
    ) -> pd.DataFrame:
        """
        Gemensam 2D-stratifiering.

        x_column/y_column kan utelämnas om experimentet sätter:

            x_column = "..."
            y_column = "..."

        som klassattribut.
        """

        if x_column is None:
            x_column = getattr(
                self,
                "x_column",
                None,
            )

        if y_column is None:
            y_column = getattr(
                self,
                "y_column",
                None,
            )

        if x_column is None:
            raise ValueError(
                f"{self.name}: x_column saknas."
            )

        if y_column is None:
            raise ValueError(
                f"{self.name}: y_column saknas."
            )

        if targets is None:
            targets = self.targets

        return two_dimensional_stratification(
            test,
            x_column=x_column,
            x_thresholds=x_bins,
            y_column=y_column,
            y_thresholds=y_bins,
            event_columns=targets,
        )

    def _add_analysis(
        self,
        result: ExperimentResult,
        analysis: dict[str, Any],
    ) -> None:
        """
        Konverterar vanliga analysresultat till standardformat.
        """

        for key, value in analysis.items():
            if isinstance(value, pd.DataFrame):
                result.add_table(
                    key,
                    value,
                )
            elif isinstance(value, dict):
                result.metadata[key] = value
            else:
                result.add_metric(
                    key,
                    value,
                )

    def _bind_context(
        self,
        context,
    ) -> None:
        """
        Binder aktuell context till experimentet.

        Används internt av frameworkets hjälpfunktioner.
        """
        self._context = context
