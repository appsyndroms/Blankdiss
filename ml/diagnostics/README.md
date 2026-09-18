Blankdiss Diagnostics

Diagnostics används för hypotesdrivna analyser som kräver mer specifik experimentlogik än den generella research-matrisen.

Exempel:

* volatility × short interest
* short-interest change × event risk
* volatility regime
* directional tail analysis
* report-date proximity
* sector-relative return

⸻

Arkitektur

ml/diagnostics/
│
├── framework/
│   ├── __init__.py
│   ├── base.py
│   ├── context.py
│   ├── metrics.py
│   ├── reporting.py
│   ├── runner.py
│   └── stratification.py
│
└── experiments/
    ├── *_diagnostic.py
    └── ...

Det finns en tydlig separation mellan:

framework
    = gemensam infrastruktur
experiments
    = specifika forskningsfrågor

⸻

DiagnosticExperiment

Alla nya diagnostics ska normalt ärva från:

DiagnosticExperiment

Exempel:

from ml.diagnostics.framework import DiagnosticExperiment
class MyExperiment(DiagnosticExperiment):
    name = "my_experiment"
    targets = (
        "down_5pct_5d",
    )
    def analyze_window(self, context):
        ...

Experimentet behöver normalt bara implementera:

analyze_window(context)

⸻

ExperimentContext

Runnern skapar ett:

ExperimentContext

för varje walk-forward-window.

Context innehåller:

train
validation
pretest
test

Exempel:

def analyze_window(self, context):
    train = context.train
    validation = context.validation
    test = context.test

Experimenten ska inte själva implementera walk-forward-datumfilter.

⸻

ExperimentResult

Experimentet returnerar resultat som frameworket konverterar till:

ExperimentResult

Resultatet kan innehålla:

tables
metrics
metadata

Exempel:

return {
    "analysis": analysis_table,
    "bootstrap": bootstrap_table,
}

Standardiserad reporting hanteras av frameworket.

⸻

Helpers

Gemensam analyslogik ska ligga i frameworket.

Exempel:

make_pretest_bins()
build_2d_analysis()

samt andra helpers för:

* logistic screening
* event-risk modelling
* feature-set comparison
* directional tail analysis
* bootstrap
* volatility regimes
* economic tail analysis

Målet är att experimentfilerna ska vara små och deklarativa.

⸻

Exempel

Ett enkelt experiment:

class VolatilitySIInteractionExperiment(
    DiagnosticExperiment
):
    name = "volatility_si_interaction"
    targets = (
        "down_5pct_5d",
        "down_7pct_5d",
        "down_10pct_5d",
    )
    def analyze_window(self, context):
        volatility_bins = self.make_pretest_bins(
            context.test,
            "price_volatility_20d",
        )
        si_bins = self.make_pretest_bins(
            context.test,
            "short_interest_pct",
        )
        return self.build_2d_analysis(
            context.test,
            volatility_bins,
            si_bins,
            self.targets,
        )

Poängen är att experimentet beskriver vad som ska analyseras, medan frameworket beskriver hur den gemensamma infrastrukturen fungerar.

⸻

Walk-forward

Diagnostics följer samma grundprincip som övrig ML:

TRAIN
   ↓
VALIDATION
   ↓
MODEL SELECTION
   ↓
REFIT
   ↓
OOS TEST

Testperioden får inte användas för:

* modellval
* threshold selection
* feature selection
* optimering av experimentet

Detta gäller även deskriptiva diagnostics när testpopulationer definieras med quantiles eller thresholds.

⸻

Pre-test populationer

När ett experiment behöver exempelvis:

HIGH VOL
HIGH SI
HIGH EVENT RISK

ska grupperna definieras med information som är tillgänglig före testperioden.

Exempel:

pretest
   ↓
quantile threshold
   ↓
test classification

Inte:

test outcomes
   ↓
optimera threshold

⸻

2D-interaktioner

För två faktorer används exempelvis:

                    LOW SI       HIGH SI
LOW VOL                A             B
HIGH VOL               C             D

Interaktion:

(D - C) - (B - A)

Detta skiljer en faktisk interaktion från en situation där den högsta gruppen bara råkar ha högst event rate.

⸻

Event-risk

Event-risk kan fungera som en mellanliggande dimension i flera diagnostics.

Exempel:

abs(forward_return_5d) >= 10 %

En event-riskmodell kan tränas och väljas genom:

TRAIN
   ↓
VALIDATION
   ↓
REFIT
   ↓
OOS SCORE

Andra diagnostics kan sedan analysera exempelvis:

event-risk tail
        ×
short-interest level

eller:

event-risk tail
        ×
short-interest change

⸻

Experimentfiler

Experimentfilerna ska normalt vara små.

Undvik att lägga följande direkt i experimentfilen:

* stora data-loading-block
* egna walk-forward-splitter
* duplicerad modellträning
* duplicerad threshold-logik
* duplicerad bootstrap-kod
* duplicerad reporting
* terminalorienterad output

Om samma kod behövs av flera experiment ska den normalt flyttas till frameworket.

⸻

Nya experiment

Processen är:

1. Forskningsfråga
       ↓
2. Välj diagnostic
       ↓
3. Skapa experimentklass
       ↓
4. Återanvänd framework helpers
       ↓
5. Lägg eventuell generell ny logik i framework
       ↓
6. Registrera experimentet
       ↓
7. Kör walk-forward
       ↓
8. Inspektera OOS-resultat

Ett nytt experiment ska alltså inte innebära att ett nytt stort standalone-script byggs.

⸻

Designmål

Diagnostics-systemet ska göra det möjligt att gå från:

Ny hypotes

till:

Liten experimentklass
        ↓
Gemensam framework
        ↓
Walk-forward
        ↓
OOS
        ↓
Standardiserat resultat

Det innebär att forskningslogiken blir mer återanvändbar och att nya experiment kan implementeras utan att duplicera infrastrukturen.

⸻

Relaterad dokumentation

Övergripande projekt:

README.md

ML-system:

ml/README.md

Research:

ml/research/README.md

Diagnostics:

ml/diagnostics/README.md
