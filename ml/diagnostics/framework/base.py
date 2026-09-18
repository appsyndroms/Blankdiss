from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .context import ExperimentContext
from .stratification import (
    PretestBins,
    make_pretest_bins,
    two_dimensional_stratification,
)


@dataclass
class ExperimentResult:
    """Standardiserat resultat från ett diagnostics-experiment."""

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
    Basinterface för diagnostics-experiment.

    Experimentfilerna ska normalt bara definiera:

        name
        description
        targets / parametrar
        analyze_window()

    Gemensam logik ska ligga i frameworket.
    """

    name: str = ""
    description: str = ""
    targets: tuple[str, ...] = ()

    def __init__(self, **kwargs: Any) -> None:
        self.options = kwargs
        self._context: ExperimentContext | None = None

    def execute(
        self,
        context: ExperimentContext,
    ) -> ExperimentResult:
        self._context = context

        try:
            analysis = self.analyze_window(context)

            if isinstance(analysis, ExperimentResult):
                result = analysis

            elif isinstance(analysis, pd.DataFrame):
                result = ExperimentResult(
                    name=self.name,
                    description=self.description,
                )

                result.add_table(
                    "analysis",
                    analysis,
                )

            elif isinstance(analysis, dict):
                result = ExperimentResult(
                    name=self.name,
                    description=self.description,
                )

                for key, value in analysis.items():
                    if isinstance(value, pd.DataFrame):
                        result.add_table(
                            key,
                            value,
                        )
                    else:
                        result.add_metric(
                            key,
                            value,
                        )

            else:
                raise TypeError(
                    f"Experiment '{self.name}' måste returnera "
                    "ExperimentResult, DataFrame eller dict från "
                    "analyze_window(), men returnerade "
                    f"{type(analysis).__name__}."
                )

            result.metadata.update(
                context.metadata()
            )

            return result

        finally:
            self._context = None

    @abstractmethod
    def analyze_window(
        self,
        context: ExperimentContext,
    ):
        """
        Kör analysen för ett walk-forward-fönster.

        Experimentet ska inte själv hantera datumgränser.
        """
        raise NotImplementedError

    @property
    def context(self) -> ExperimentContext:
        if self._context is None:
            raise RuntimeError(
                "ExperimentContext är endast tillgänglig "
                "under analyze_window()."
            )

        return self._context

    # ------------------------------------------------------------------
    # Stratifiering
    # ------------------------------------------------------------------

    def make_pretest_bins(
        self,
        test: pd.DataFrame,
        column: str,
        quantiles: tuple[float, ...] = (0.80,),
    ) -> PretestBins:
        """
        Skapar bins från pretest-data.

        Argumentet test finns kvar för att experimenten ska kunna
        uttrycka analysen naturligt, men thresholds beräknas alltid
        från context.pretest.
        """

        if test is None:
            raise ValueError(
                "test får inte vara None."
            )

        return make_pretest_bins(
            self.context.pretest,
            column,
            quantiles=quantiles,
        )

    def build_2d_analysis(
        self,
        test: pd.DataFrame,
        x_bins: PretestBins,
        y_bins: PretestBins,
        targets: tuple[str, ...],
        return_column: str | None = None,
    ) -> pd.DataFrame:
        """Applicerar pretest-definierade bins på OOS-testdata."""

        if test is None:
            raise ValueError(
                "test får inte vara None."
            )

        return two_dimensional_stratification(
            test,
            x_bins,
            y_bins,
            event_columns=targets,
            return_column=return_column,
        )
