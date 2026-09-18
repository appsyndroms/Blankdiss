Blankdiss

Blankdiss är ett forskningsprojekt för att undersöka om offentlig information om bolag, blankning, prisrörelser, rapporter och andra marknadsvariabler innehåller statistiskt och ekonomiskt användbara mönster.

Projektet kombinerar:

* datainsamling
* feature engineering
* datakvalitet
* ML
* walk-forward evaluation
* hypotesdriven research
* automatiserade experiment
* reproducerbara resultat

Målet är inte att bygga en samling enskilda analyser, utan en återanvändbar forskningspipeline där nya hypoteser kan testas systematiskt.

⸻

Översikt

                         Blankdiss
                            │
              ┌─────────────┴─────────────┐
              │                           │
        Data collection                Research
              │                           │
       ┌──────┴──────┐             ┌──────┴──────┐
       │             │             │             │
      FI           Prices       Generic       Diagnostics
       │             │          Research          │
       └──────┬──────┘             │             │
              ▼                    │             │
       Feature generation           │             │
              │                    │             │
              ▼                    ▼             ▼
       Feature QC            ml/research/   ml/diagnostics/
              │                    │             │
              └──────────────┬─────┴─────────────┘
                             ▼
                       Research results
                             │
                             ▼
                       GitHub Actions

⸻

Data → Features → ML → Research

Den övergripande pipelinen är:

FI data
   │
   ├──────────────┐
   │              │
   ▼              ▼
FI aggregation   Prices
   │              │
   └──────┬───────┘
          ▼
   Feature generation
          │
          ▼
   Feature dataset
          │
          ▼
      Feature QC
          │
          ▼
         ML
          │
      ┌───┴────┐
      │        │
      ▼        ▼
   Generic   Diagnostics
   Research
      │        │
      └───┬────┘
          ▼
      OOS results

⸻

ML-systemet

ML-delen finns under:

ml/

Viktiga komponenter:

ml/
├── config.py
├── dataset.py
├── walk_forward.py
│
├── research/
│   └── ...
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

config.py innehåller gemensamma targets och walk-forward-windows.

dataset.py ansvarar för att läsa och förbereda feature-datasetet.

walk_forward.py innehåller den generella walk-forward-logiken.

research/ används för bred generell screening.

diagnostics/ används för specifika hypotesdrivna experiment.

⸻

Generic Research vs Diagnostics

Blankdiss har två kompletterande forskningslägen.

Generic Research

Generic Research används för bred screening.

Exempel:

signal
  ×
target
  ×
tail fraction
  ×
walk-forward window

Detta passar frågor där analysen kan uttryckas generellt som:

feature → target

⸻

Diagnostics

Diagnostics används när forskningsfrågan kräver en specifik experimentdesign.

Exempel:

volatility × short interest
short-interest change × event risk
report-date proximity
sector-relative return

En diagnostic kan innehålla:

* explicit interaktion
* event-riskmodell
* specialiserad modell
* bootstrap
* pre-test thresholds
* relativa jämförelser
* flera targets
* flera analysdimensioner

Diagnostics ligger under:

ml/diagnostics/experiments/

⸻

Diagnostic-arkitekturen

Den nya diagnostics-arkitekturen är klassbaserad.

En diagnostic är en liten klass som ärver från:

DiagnosticExperiment

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

Experimentfilen ska framför allt beskriva:

1. vilken hypotes som testas
2. vilka features/targets som används
3. vilken analysmetod som ska köras

Gemensam ML- och analyslogik ska ligga i frameworket, inte dupliceras i varje experiment.

⸻

ExperimentContext

Varje diagnostics-körning får ett:

ExperimentContext

Context representerar en walk-forward-window och innehåller:

data
train
validation
pretest
test

Det gör att experimenten inte behöver implementera egna datumfilter.

Exempel:

def analyze_window(self, context):
    train = context.train
    validation = context.validation
    test = context.test

Context ansvarar även för gemensamma operationer som quantiles och thresholds.

⸻

DiagnosticExperiment

Bas-klassen:

DiagnosticExperiment
        │
        ├── context
        ├── targets
        ├── execute()
        ├── analyze_window()
        ├── make_pretest_bins()
        └── build_2d_analysis()

Det viktiga gränssnittet är:

def analyze_window(self, context):
    ...

Experimentet behöver alltså inte själv:

* läsa dataset
* skapa walk-forward masks
* hantera runnern
* skriva resultatfiler
* implementera standardiserad rapportering

Det hanteras av frameworket.

⸻

ExperimentResult

Diagnostics returnerar ett standardiserat:

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

Frameworket konverterar resultatet till ett standardiserat ExperimentResult.

Det gör att olika experiment kan använda olika analysmetoder men ändå producerar samma typ av output.

⸻

Runner

Runnern ansvarar för orchestration.

Förenklat:

experiment
     │
     ▼
runner
     │
     ├── window 1
     │     └── ExperimentContext
     │             └── analyze_window()
     │
     ├── window 2
     │     └── ExperimentContext
     │             └── analyze_window()
     │
     └── ...

Det innebär att experimentklasserna inte behöver känna till hela forskningskörningen.

Runnern ansvarar för:

* walk-forward windows
* context creation
* experiment execution
* result collection
* reporting
* felhantering

⸻

Experiment Registry

Registry kan användas som katalog över vilka experiment som ska köras, men registryt ska inte innehålla själva forskningslogiken.

Registryt beskriver exempelvis:

experiment id
status
question
experiment
priority

Den tidigare modellen där registryt pekade direkt på en modul med:

main()

är inte längre den centrala experimentmodellen.

Den nya kedjan är:

Registry
    ↓
Experiment class
    ↓
DiagnosticExperiment
    ↓
ExperimentContext
    ↓
analyze_window()
    ↓
ExperimentResult

⸻

OOS och walk-forward

ML-forskningen använder walk-forward evaluation.

Grundprincipen är:

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
* optimering av experimentet

Det gäller även diagnostics.

Om en diagnostic exempelvis använder:

top 20 % volatility
top 20 % short interest
top 5 % event risk

ska trösklarna definieras utifrån information som finns före testperioden.

⸻

Event-risk

Event-risk är ett centralt forskningsspår i Blankdiss.

Ett exempel på event-definition är:

abs(forward_return_5d) >= 10 %

Event-risk kan modelleras med bland annat:

volatility_20d
volatility_60d
volatility_20d + volatility_60d
volatility_20d + volatility_60d + term structure

Modellval sker på train/validation.

Modellen refittas därefter före OOS-testet.

Detta gör event-risk till en separat dimension som kan användas av flera diagnostics.

⸻

Interaktionsexperiment

När forskningsfrågan gäller två faktorer ska Blankdiss skilja mellan:

hög nivå

och:

interaktion

Exempel:

                    LOW SI       HIGH SI
LOW VOL                A             B
HIGH VOL               C             D

Direkt interaktion:

(D - C) - (B - A)

Det testar om effekten av SI förändras beroende på volatilitet.

Samma princip kan användas för andra forskningsdimensioner.

⸻

Resultat

Research-resultat skrivs till:

data/processed/ml/research/

Den aktuella körningen finns under:

data/processed/ml/research/latest/

Timestampade körningar sparas separat.

Resultaten ska vara maskinläsbara.

Exempel:

latest/
├── results.jsonl
├── pooled.json
├── metadata.json
├── report.md
└── diagnostics/
    └── <experiment_id>.json

Den exakta resultatstrukturen styrs av research-runnern och diagnostics-frameworket.

⸻

AI-readable research

Result-artifacts är den primära kommunikationskanalen mellan forskningskörningen och efterföljande analys.

Den avsedda kedjan är:

GitHub Actions
      ↓
Research
      ↓
Machine-readable artifacts
      ↓
AI analysis
      ↓
Nästa forskningsfråga

Actions-loggen används främst för:

* pipeline-status
* fel
* verifiering
* körningssammanfattning

Den ska inte vara den primära källan för statistisk analys.

⸻

GitHub Actions

Den automatiserade forskningspipelinen körs via:

.github/workflows/ml-research.yml

Övergripande:

Fetch FI
   ↓
Fetch prices
   ↓
Build features
   ↓
Feature QC
   ↓
Generic Research
   ↓
Diagnostics
   ↓
Verify results
   ↓
Upload artifacts

Nya diagnostics ska normalt inte kräva ändringar i workflow-filen.

⸻

Nya experiment

När en ny forskningsfråga uppstår:

1. Definiera hypotesen

Exempel:

Förstärks effekten av förändrad blankning
när volatiliteten ökar?

2. Avgör om det är generic research eller diagnostic

Om frågan är en vanlig:

feature → target

kan generic research räcka.

Om den kräver exempelvis interaktion, event-risk eller specialiserad analys används diagnostic.

3. Skapa experimentklassen

Skapa:

ml/diagnostics/experiments/<name>_diagnostic.py

med:

class MyExperiment(DiagnosticExperiment):
    name = "my_experiment"
    def analyze_window(self, context):
        ...

4. Lägg gemensam logik i frameworket

Om flera experiment behöver samma analyslogik ska den normalt flyttas till frameworket i stället för att kopieras.

5. Registrera experimentet

Lägg till experimentet i den mekanism som används för att starta diagnostics.

6. Kör walk-forward/OOS

Verifiera att testperioden inte används för modellval eller threshold selection.

⸻

Forskningsprinciper

Blankdiss research följer några centrala principer:

1. Hypotes före resultat
2. OOS före slutsats
3. Walk-forward evaluation
4. Ingen test leakage
5. Interaktion ska testas som interaktion
6. Screening ska skiljas från hypotesdriven analys
7. Intressanta resultat ska följas av robusthetstester
8. Statistiskt intressanta resultat är inte automatiskt ekonomiskt användbara

⸻

Automatiseringsmål

Blankdiss ska kunna gå från:

Ny forskningsidé
        ↓
Generic Research eller Diagnostic
        ↓
Automatisk walk-forward-körning
        ↓
OOS-resultat
        ↓
Machine-readable artifacts
        ↓
AI analysis
        ↓
Nästa forskningsfråga

Målet är att nya experiment ska kunna läggas till utan att varje experiment kräver en ny specialbyggd pipeline.

⸻

Dokumentation

ML-översikt:

ml/README.md

Research-system:

ml/research/README.md

Diagnostics-framework och experiment:

ml/diagnostics/README.md

Detta README beskriver projektets övergripande arkitektur.
