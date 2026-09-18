from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .stratification import (
    PretestBins,
    make_pretest_bins,
    two_dimensional_stratification,
)


@dataclass
class ExperimentResult:
    """
    Standardiserat resultat från ett diagnostiskt experiment.

    Experiment ska returnera data här i stället för att skriva ut
    resultat direkt till stdout.
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
    Basinterface för alla diagnostics-experiment.

    Ett experiment ska huvudsakligen definiera:

        name
        description
        targets
        analyze_window()

    Exempel:

        class MyExperiment(DiagnosticExperiment):
            name = "my_experiment"

            targets = (
                "down_5pct_5d",
                "down_10pct_5d",
            )

            def analyze_window(self, context):
                volatility_bins = self.make_pretest_bins(
                    context.test,
                    "price_volatility_20d",
                )

                si_bins = self.make_pretest_bins(
                    context.test,
                    "short_interest_level",
                )

                return self.build_2d_analysis(
                    context.test,
                    volatility_bins,
                    si_bins,
                    self.targets,
                )

    Experimentfilerna ska vara små och deklarativa.
    Gemensam logik ska ligga i framework-klasserna.
    """

    name: str = ""
    description: str = ""

    targets: tuple[str, ...] = ()

    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        self.options = kwargs
        self._context = None

    def run(
        self,
        context,
    ) -> ExperimentResult:
        """
        Kör experimentet mot en ExperimentContext.

        Experimentfiler ska normalt inte implementera run().
        De implementerar i stället analyze_window().

        analyze_window() får returnera:

            ExperimentResult
            pandas.DataFrame
            dict

        DataFrame och dict konverteras automatiskt till
        ExperimentResult.
        """

        self._context = context

        try:
            analysis = self.analyze_window(context)

            if isinstance(
                analysis,
                ExperimentResult,
            ):
                return analysis

            if isinstance(
                analysis,
                pd.DataFrame,
            ):
                result = ExperimentResult(
                    name=self.name,
                    description=self.description,
                )

                result.add_table(
                    "analysis",
                    analysis,
                )

                result.metadata.update(
                    context.metadata()
                )

                return result

            if isinstance(
                analysis,
                dict,
            ):
                result = ExperimentResult(
                    name=self.name,
                    description=self.description,
                )

                for key, value in analysis.items():
                    if isinstance(
                        value,
                        pd.DataFrame,
                    ):
                        result.add_table(
                            key,
                            value,
                        )
                    else:
                        result.add_metric(
                            key,
                            value,
                        )

                result.metadata.update(
                    context.metadata()
                )

                return result

            raise TypeError(
                f"Experiment '{self.name}' måste returnera "
                "ExperimentResult, DataFrame eller dict från "
                f"analyze_window(), men returnerade "
                f"{type(analysis).__name__}."
            )

        finally:
            self._context = None

    @abstractmethod
    def analyze_window(
        self,
        context,
    ):
        """
        Utför själva analysen för ett walk-forward-fönster.

        context innehåller bland annat:

            context.train
            context.validation
            context.pretest
            context.test

        Experimentet ska inte själv hantera datumgränser,
        walk-forward-split eller presentation.
        """
        raise NotImplementedError

    def execute(
        self,
        context,
    ) -> ExperimentResult:
        """
        Publikt körinterface som säkerställer ett standardiserat
        ExperimentResult.
        """

        result = self.run(context)

        if not isinstance(
            result,
            ExperimentResult,
        ):
            raise TypeError(
                f"Experiment '{self.name}' måste returnera "
                f"ExperimentResult, men returnerade "
                f"{type(result).__name__}."
            )

        return result

    # ------------------------------------------------------------------
    # Gemensamma stratifieringshelpers
    # ------------------------------------------------------------------

    def make_pretest_bins(
        self,
        test: pd.DataFrame,
        column: str,
        quantiles: tuple[float, ...] = (0.80,),
    ) -> PretestBins:
        """
        Skapar thresholds från pre-test-data.

        'test' finns kvar i signaturen för att experimentfilerna ska
        kunna uttrycka analysen naturligt:

            self.make_pretest_bins(
                context.test,
                "price_volatility_20d",
            )

        Men själva thresholds beräknas ALLTID från context.pretest.

        Testdata får alltså aldrig påverka gränsvärdena.
        """

        if self._context is None:
            raise RuntimeError(
                "make_pretest_bins() får endast användas "
                "under analyze_window()."
            )

        if test is None:
            raise ValueError(
                "test får inte vara None."
            )

        return make_pretest_bins(
            self._context.pretest,
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
        """
        Bygger en tvådimensionell OOS-stratifiering.

        x_bins och y_bins måste vara skapade från pre-test-data.
        Själva analysen appliceras därefter på testdata.
        """

        if self._context is None:
            raise RuntimeError(
                "build_2d_analysis() får endast användas "
                "under analyze_window()."
            )

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
