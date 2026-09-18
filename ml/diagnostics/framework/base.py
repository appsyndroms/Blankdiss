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
    name: str
    description: str = ""
    tables: dict[str, pd.DataFrame] = field(
        default_factory=dict
    )
    metrics: dict[str, Any] = field(
        default_factory=dict
    )
    metadata: dict[str, Any] = field(
        default_factory=dict
    )

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
    name: str = ""
    description: str = ""
    targets: tuple[str, ...] = ()

    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        self.options = kwargs
        self._context: ExperimentContext | None = None

    def execute(
        self,
        context: ExperimentContext,
    ) -> ExperimentResult:
        self._context = context

        try:
            analysis = self.analyze_window(
                context
            )

            if isinstance(
                analysis,
                ExperimentResult,
            ):
                result = analysis

            elif isinstance(
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

            elif isinstance(
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

            else:
                raise TypeError(
                    f"Experiment '{self.name}' returned "
                    f"unsupported analysis type: "
                    f"{type(analysis).__name__}"
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
        raise NotImplementedError

    @property
    def context(self) -> ExperimentContext:
        if self._context is None:
            raise RuntimeError(
                "Experiment context is only available "
                "while an experiment is executing."
            )

        return self._context

    def make_pretest_bins(
        self,
        column: str,
        quantiles: tuple[float, ...] = (0.80,),
    ) -> PretestBins:
        return make_pretest_bins(
            self.context.pretest,
            column,
            quantiles=quantiles,
        )

    def build_2d_analysis(
        self,
        frame: pd.DataFrame,
        x_bins: PretestBins,
        y_bins: PretestBins,
        event_columns: tuple[str, ...],
        return_column: str | None = None,
    ) -> pd.DataFrame:
        return two_dimensional_stratification(
            frame,
            x_bins,
            y_bins,
            event_columns,
            return_column=return_column,
        )

    def run_si_level_confirmation(
        self,
        context: ExperimentContext,
        *,
        risk_cutoffs: tuple[float, ...],
        short_interest_cutoff: float,
    ) -> ExperimentResult:
        from .event_risk import (
            run_si_level_confirmation,
        )

        return run_si_level_confirmation(
            context,
            risk_cutoffs=risk_cutoffs,
            short_interest_cutoff=short_interest_cutoff,
        )

    def run_event_risk_interaction(
        self,
        context: ExperimentContext,
        *,
        risk_cutoffs: tuple[float, ...],
        positive_change_cutoff: float,
    ) -> ExperimentResult:
        from .event_risk import (
            run_event_risk_interaction,
        )

        return run_event_risk_interaction(
            context,
            risk_cutoffs=risk_cutoffs,
            positive_change_cutoff=positive_change_cutoff,
        )

    def run_si_price_dynamics(
        self,
        context: ExperimentContext,
        *,
        horizons: tuple[int, ...],
        si_change_cutoffs: tuple[float, ...],
        event_risk_cutoffs: tuple[float, ...],
        prior_return_columns: tuple[str, ...],
    ) -> ExperimentResult:
        from .price_dynamics import (
            run_si_price_dynamics,
        )

        return run_si_price_dynamics(
            context,
            horizons=horizons,
            si_change_cutoffs=si_change_cutoffs,
            event_risk_cutoffs=event_risk_cutoffs,
            prior_return_columns=prior_return_columns,
        )
