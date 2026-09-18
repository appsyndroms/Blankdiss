from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


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
                ...
                return self.build_2d_analysis(
                    context.test,
                    x_bins,
                    y_bins,
                    self.targets,
                )

    Gemensam experimentlogik hör hemma här i stället för
    att upprepas i varje experimentfil.
    """

    name: str = ""
    description: str = ""
    targets: tuple[str, ...] = ()

    def __init__(self, **kwargs: Any) -> None:
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
        """

        self._context = context

        try:
            analysis = self.analyze_window(context)

            if isinstance(analysis, ExperimentResult):
                return analysis

            if isinstance(analysis, pd.DataFrame):
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

            if isinstance(analysis, dict):
                result = ExperimentResult(
                    name=self.name,
                    description=self.description,
                )

                for key, value in analysis.items():
                    if isinstance(value, pd.DataFrame):
                        result.add_table(key, value)
                    else:
                        result.add_metric(key, value)

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
        Utför själva analysen för en test-/walk-forward-window.

        Experimentfiler ska vara små och deklarativa.
        Gemensam logik ska ligga i framework-klasserna.
        """
        raise NotImplementedError

    def execute(
        self,
        context,
    ) -> ExperimentResult:
        """
        Publikt körinterface som säkerställer ett standardiserat resultat.
        """

        result = self.run(context)

        if not isinstance(result, ExperimentResult):
            raise TypeError(
                f"Experiment '{self.name}' måste returnera "
                f"ExperimentResult, men returnerade "
                f"{type(result).__name__}."
            )

        return result
