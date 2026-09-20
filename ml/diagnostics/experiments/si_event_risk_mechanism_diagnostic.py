from __future__ import annotations
from ml.diagnostics.framework import (
    DiagnosticExperiment,
)
from ml.diagnostics.framework.si_event_risk_mechanism import (
    run_si_event_risk_mechanism,
)
class SIEventRiskMechanismExperiment(
    DiagnosticExperiment
):
    name = "si_event_risk_mechanism"
    description = (
        "Låst mekanismtest av SI/event-risk-"
        "hypotesen: timing av SI-förändringen, "
        "kontroll för volatilitet samt sektor- "
        "och marknadsrelativt utfall."
    )
    def analyze_window(
        self,
        context,
    ):
        return run_si_event_risk_mechanism(
            context
        )
