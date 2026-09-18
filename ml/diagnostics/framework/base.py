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

    def add_table(self, name: str, table: pd.DataFrame) -> None:
        self.tables[name] = table

    def add_metric(self, name: str, value: Any) -> None:
        self.metrics[name] = value

    def add_metadata(self, name: str, value: Any) -> None:
        self.metadata[name] = value


class DiagnosticExperiment(ABC):
    """
    Basinterface för alla diagnostics-experiment.

    Ett experiment ska huvudsakligen göra tre saker:

    1. definiera sina parametrar
    2. analysera data
    3. returnera ExperimentResult

    Presentation/loggning hör hemma i reporting.py.
    """

    name: str = ""
    description: str = ""

    def __init__(self, **kwargs: Any) -> None:
        self.options = kwargs

    @abstractmethod
    def run(self, context) -> ExperimentResult:
        raise NotImplementedError

    def execute(self, context) -> ExperimentResult:
        result = self.run(context)

        if not isinstance(result, ExperimentResult):
            raise TypeError(
                f"Experiment '{self.name}' måste returnera ExperimentResult, "
                f"men returnerade {type(result).__name__}."
            )

        return result
