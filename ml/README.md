Blankdiss ML

Detta är den övergripande dokumentationen för Blankdiss ML-system.

ML-systemet består av:

ml/
├── config.py
├── dataset.py
├── walk_forward.py
│
├── research/
│
└── diagnostics/
    ├── framework/
    │   ├── base.py
    │   ├── context.py
    │   ├── metrics.py
    │   ├── reporting.py
    │   ├── runner.py
    │   └── stratification.py
    │
    └── experiments/
        └── *_diagnostic.py

⸻

1. Övergripande arkitektur

                    Blankdiss data
                         │
                         ▼
                  Feature generation
                         │
                         ▼
              data/processed/analysis
                         │
                         ▼
                    ml/dataset.py
                         │
              ┌──────────┴──────────┐
              │                     │
              ▼                     ▼
       Generic Research       Diagnostics
              │                     │
              │              Experiment classes
              │                     │
              │              DiagnosticExperiment
              │                     │
              │              ExperimentContext
              │                     │
              └──────────┬──────────┘
                         ▼
                  OOS / Walk-forward
                         │
                         ▼
                   ExperimentResult
                         │
                         ▼
                    Result artifacts

⸻

2. Gemensam ML-konfiguration

ml/config.py innehåller gemensam konfiguration:

* targets
* target thresholds
* target directions
* walk-forward windows
* random state
* andra gemensamma ML-inställningar

Targets definieras som TargetConfig.

Exempel:

TargetConfig(
    name="down_10pct_5d",
    return_column="forward_return_5d",
    threshold=-0.10,
    direction="below",
)

⸻

3. Dataset

ml/dataset.py ansvarar för att läsa och förbereda feature-data.

Viktiga funktioner inkluderar:

load_features()
build_target()
prepare_feature_set()
prepare_ml_data()

Feature-datasetet kommer från:

data/processed/analysis/

Framtida avkastningskolumner och andra leakage-källor ska exkluderas från features.

⸻

4. Walk-forward

ml/walk_forward.py innehåller den generella walk-forward-logiken.

Principen är:

TRAIN
   ↓
VALIDATION
   ↓
MODEL SELECTION
   ↓
REFIT
   ↓
OOS TEST

Testdata får inte användas för:

* modellval
* feature selection
* threshold selection
* optimering av experiment

Walk-forward-windows definieras i ml/config.py.

⸻

5. Generic Research

Generic Research används för bred screening.

Den kan exempelvis undersöka:

signal
 ×
target
 ×
tail fraction
 ×
walk-forward window

Generic Research är avsedd för frågor som kan uttryckas med en generell experimentmodell.

⸻

6. Diagnostics

Diagnostics används när en forskningsfråga kräver egen experimentlogik.

Exempel:

volatility × short interest
short-interest change × event risk
report proximity
sector-relative return

Diagnostics finns under:

ml/diagnostics/

Frameworket finns under:

ml/diagnostics/framework/

Experimenten finns under:

ml/diagnostics/experiments/

⸻

7. Diagnostics-framework

Frameworket består av:

framework/
├── base.py
├── context.py
├── metrics.py
├── reporting.py
├── runner.py
└── stratification.py

base.py

Definierar:

DiagnosticExperiment
ExperimentResult

DiagnosticExperiment är bas-klassen för experimenten.

⸻

context.py

Definierar:

ExperimentContext

Context representerar en walk-forward-window och exponerar bland annat:

train
validation
pretest
test

Experimenten ska använda context i stället för att implementera egna datumfilter.

⸻

metrics.py

Gemensamma statistiska mått:

* event rate
* lift
* classification metrics
* return summary
* event summary

⸻

stratification.py

Gemensam logik för:

* pre-test thresholds
* quantile buckets
* threshold classification
* 2D stratification

⸻

reporting.py

Standardiserad rapportering och serialisering av experimentresultat.

⸻

runner.py

Orkestrerar diagnostics.

Runnern ansvarar för:

* experiment execution
* walk-forward windows
* context creation
* result collection
* rapportering

Forskningslogiken ska inte ligga i runnern.

⸻

8. Experimentklasser

Ett experiment ska vara så litet som möjligt.

Exempel:

from ml.diagnostics.framework import DiagnosticExperiment
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

Experimentklassen ska huvudsakligen beskriva experimentet.

Gemensam implementation ska ligga i frameworket.

Det innebär att experimentfilerna inte ska innehålla stora kopierade block för:

* data loading
* walk-forward splitting
* threshold calculation
* modellval
* standardiserad rapportering
* serialisering

⸻

9. ExperimentContext

ExperimentContext är gränssnittet mellan runnern och experimentet.

Runnern skapar context:

WalkForwardWindow
       ↓
ExperimentContext
       ↓
DiagnosticExperiment

Experimentet arbetar sedan med:

context.train
context.validation
context.pretest
context.test

Detta gör experimenten oberoende av hur runnern skapar datumintervallen.

⸻

10. ExperimentResult

Alla diagnostics ska i slutändan ge ett:

ExperimentResult

Resultatet kan innehålla:

tables
metrics
metadata

Det gör att experiment med helt olika analysmetoder ändå kan hanteras av samma reporting- och runner-infrastruktur.

⸻

11. Pre-test thresholds

Ett centralt designmål är att skilja på:

TEST

och:

information available before TEST

Exempel:

top 20 % volatility
top 20 % short interest
top 5 % event risk

Thresholds ska beräknas före testperioden.

Experimenten ska därför använda frameworkets pre-test-funktioner i stället för att beräkna thresholds från testutfallen.

⸻

12. Event-risk

Event-risk kan användas som en separat dimension i diagnostics.

Exempel:

abs(forward_return_5d) >= 10 %

En event-riskmodell kan använda:

volatility_20d
volatility_60d
volatility_20d + volatility_60d
volatility_20d + volatility_60d + term structure

Modellval:

TRAIN
   ↓
VALIDATION
   ↓
best model
   ↓
REFIT
   ↓
OOS TEST

OOS event-risk scores kan därefter användas av andra diagnostics.

⸻

13. Interaktioner

För en explicit interaktionsfråga ska analysen inte reduceras till att bara jämföra den högsta gruppen.

Exempel:

                    LOW SI       HIGH SI
LOW VOL                A             B
HIGH VOL               C             D

Direkt interaktion:

(D - C) - (B - A)

Detta kan implementeras gemensamt i frameworket så att experimenten bara definierar vilka dimensioner som ska analyseras.

⸻

14. Experiment Registry

Registry används som katalog och execution configuration för diagnostics.

Den ska beskriva:

id
status
question
experiment
priority

Registryt ska inte innehålla själva forskningslogiken.

Den nya arkitekturen skiljer tydligt mellan:

Registry
    = VAD ska köras?
Runner
    = HUR körs det?
DiagnosticExperiment
    = VAD gör analysen?
ExperimentContext
    = VILKEN walk-forward-window analyseras?
ExperimentResult
    = VAD blev resultatet?

⸻

15. Resultat

Research-resultat skrivs till:

data/processed/ml/research/

Senaste körningen:

data/processed/ml/research/latest/

Resultat ska vara maskinläsbara.

Exempel:

latest/
├── results.jsonl
├── pooled.json
├── metadata.json
├── report.md
└── diagnostics/
    └── <experiment_id>.json

⸻

16. AI-readable results

AI ska kunna läsa resultat-artifacts direkt.

Prioriteringen är:

structured result
       ↓
AI analysis

och inte:

terminal output
       ↓
manual copy/paste
       ↓
AI analysis

Actions-loggen är därför främst till för:

* status
* fel
* verifiering
* körningsinformation

Resultatfilerna är den primära statistiska källan.

⸻

17. Ny diagnostic

När en ny forskningsfråga kräver en diagnostic:

Steg 1

Definiera hypotesen.

Steg 2

Skapa experimentklassen under:

ml/diagnostics/experiments/

Steg 3

Ärv från:

DiagnosticExperiment

Steg 4

Implementera:

def analyze_window(self, context):
    ...

Steg 5

Återanvänd frameworkets helpers.

Steg 6

Om en helper saknas och samma logik kan användas av flera experiment ska den läggas i frameworket.

Steg 7

Registrera experimentet.

Steg 8

Kör walk-forward/OOS.

⸻

18. Designprincip

Den centrala arkitekturen är:

config.py
    = gemensam ML-konfiguration
dataset.py
    = data → ML-data
walk_forward.py
    = generisk walk-forward-logik
research/
    = bred screening
diagnostics/framework/
    = gemensam diagnostic-infrastruktur
diagnostics/experiments/
    = små hypotesdrivna experimentklasser
registry
    = vilka experiment som ska köras
runner
    = orchestration
ExperimentResult
    = standardiserat resultat
artifacts
    = maskinläsbar output

Det viktiga är att nya experiment ska växa horisontellt genom små klasser, inte genom att varje nytt experiment blir ännu ett stort standalone-script.
