Blankdiss ML

Detta är den övergripande dokumentationen för Blankdiss ML-system.

ML-delen är uppdelad i:

ml/
├── config.py
├── dataset.py
├── experiment_registry.json
├── experiment_registry_runner.py
│
├── research/
│   ├── README.md
│   ├── runner.py
│   ├── experiments.py
│   ├── evaluator.py
│   └── cache.py
│
└── diagnostics/
    ├── ...
    └── *_diagnostic.py

ML-systemet har två huvudsakliga användningsområden:

1. Bred research/screening av många signaler och targets.
2. Hypotesdrivna diagnostics för specifika forskningsfrågor.

⸻

1. Övergripande arkitektur

                    Blankdiss data
                         │
                         ▼
              analysis/build_features
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
       Generic Research       Specific Diagnostics
              │                     │
              ▼                     ▼
       ml/research/           ml/diagnostics/
              │                     │
              └──────────┬──────────┘
                         │
                         ▼
                 Research Results

GitHub Actions binder ihop hela kedjan:

.github/workflows/ml-research.yml
              │
              ▼
       data collection
              │
              ▼
       feature generation
              │
              ▼
         Feature QC
              │
        ┌─────┴─────┐
        ▼           ▼
    Research     Diagnostics
      Matrix      Registry
        │           │
        └─────┬─────┘
              ▼
          Artifacts

⸻

2. ml/config.py

config.py innehåller gemensam konfiguration för ML-systemet.

Bland annat:

* targets
* target thresholds
* target directions
* walk-forward windows
* random state
* annan gemensam ML-konfiguration

Targets definieras som TargetConfig.

Exempel:

TargetConfig(
    name="down_10pct_5d",
    return_column="forward_return_5d",
    threshold=-0.10,
    direction="below",
)

Det innebär:

forward_return_5d <= -10 %

är en positiv target-event för down_10pct_5d.

⸻

3. Walk-forward

ML-systemet använder walk-forward evaluation.

Nuvarande windows definieras i config.py.

Principen är:

TRAIN
   │
   ▼
VALIDATION
   │
   ▼
MODEL SELECTION / THRESHOLDS
   │
   ▼
REFIT
   │
   ▼
OOS TEST

Testdata får inte användas för:

* modellval
* feature selection
* threshold selection
* optimering av experimentet

Testperioden ska representera verklig out-of-sample-användning.

⸻

4. ml/dataset.py

dataset.py ansvarar för att göra feature-data användbar för ML.

Viktiga funktioner inkluderar bland annat:

load_features()
build_target()
prepare_feature_set()

load_features() läser feature-datasetet från:

data/processed/analysis/

Features innehåller bland annat:

* snapshot_date
* security_key
* price features
* short-interest features
* forward returns
* targets/underlag för targets

Framtida avkastningskolumner måste hanteras korrekt så att de inte läcker in som features.

⸻

5. Generic Research

Den generella research-motorn finns under:

ml/research/

Se:

ml/research/README.md

för detaljerad dokumentation.

Huvudkomponenterna är:

ml/research/runner.py
ml/research/experiments.py
ml/research/evaluator.py
ml/research/cache.py

experiments.py

Definierar experimentmatrisen.

Den kombinerar bland annat:

signal
target
tail fraction
tail direction

Nuvarande matrix innehåller ett stort antal kombinationer som screenas systematiskt.

⸻

evaluator.py

Ansvarar för själva utvärderingen.

Resultat kan bland annat innehålla:

* AUC
* antal observationer
* antal events
* event rate
* baseline event rate
* lift
* return metrics
* window
* split

⸻

cache.py

Bygger återanvändbara datamängder för research-körning.

Det minskar onödigt arbete när samma:

* signals
* targets
* returns
* tails
* window masks

behöver användas av många experiment.

⸻

runner.py

Orkestrerar den generella research matrixen.

Den:

1. laddar features
2. bygger experimentmatris
3. bygger cache
4. kör walk-forward evaluation
5. poolar resultaten
6. skriver resultatfiler
7. genererar rapport

Resultaten skrivs bland annat till:

data/processed/ml/research/latest/

med:

results.jsonl
pooled.json
metadata.json
report.md

⸻

6. Hypotesdrivna Diagnostics

Specifika forskningsfrågor ligger under:

ml/diagnostics/

En diagnostic används när frågan inte bara är:

signal → target

utan kräver mer specifik analyslogik.

Exempel:

volatility × short-interest level
short-interest change × event risk
report-date proximity
sector-relative return

Diagnostics kan innehålla:

* interaktioner
* specialiserade modeller
* event-risk
* bootstrap
* specifika populationer
* relativa jämförelser
* andra experimentdesigner

⸻

7. Experiment Registry

Alla registrerade forskningsfrågor definieras i:

ml/experiment_registry.json

Registryt beskriver:

VAD ska köras?

En experiment-post innehåller:

{
  "id": "example_experiment",
  "status": "active",
  "question": "Forskningsfrågan",
  "module": "ml.diagnostics.example_experiment_diagnostic",
  "priority": 1
}

status

active

betyder att experimentet körs.

planned

betyder att experimentet finns dokumenterat men ännu inte körs.

module

Binder registry-posten till Python-koden.

Exempel:

ml.diagnostics.volatility_si_level_interaction_diagnostic

motsvarar:

ml/diagnostics/volatility_si_level_interaction_diagnostic.py

priority

Bestämmer körordningen mellan aktiva experiment.

⸻

8. experiment_registry_runner.py

Registry-runnern är orkestratorn för diagnostics.

Den:

1. läser experiment_registry.json
2. hittar active experiment
3. sorterar efter priority
4. importerar respektive diagnostic
5. kör main()
6. fortsätter med nästa experiment även om ett experiment misslyckas
7. returnerar felstatus om något experiment misslyckades

Förenklat:

experiment_registry.json
          │
          ▼
experiment_registry_runner.py
          │
          ├── diagnostic A → main()
          │
          ├── diagnostic B → main()
          │
          └── diagnostic C → main()

Det gör att nya experiment inte behöver hårdkodas i runnern.

⸻

9. Hur ett nytt experiment läggs till

När en ny forskningsfråga ska testas:

Steg 1

Definiera frågan.

Exempel:

Förstärks effekten av förändrad blankning när volatiliteten ökar?

Steg 2

Avgör om den passar:

ml/research/

eller kräver:

ml/diagnostics/

Steg 3

Om diagnostic behövs, skapa:

ml/diagnostics/<name>_diagnostic.py

med:

def main():
    ...

Steg 4

Implementera korrekt walk-forward/OOS-logik.

Steg 5

Lägg till experimentet i:

ml/experiment_registry.json

Steg 6

Sätt:

"status": "active"

när experimentet är färdigt för körning.

Steg 7

Registry-runnern kör automatiskt experimentet.

Ingen ny hårdkodning i runnern ska behövas.

⸻

10. Aktuella registrerade experiment

Registryt innehåller för närvarande:

Volatility × SI level

volatility_si_level_interaction

Forskningsfråga:

Är risken för en extrem nedgång särskilt hög när både volatiliteten och blankningsnivån är hög?

Diagnostic:

ml/diagnostics/volatility_si_level_interaction_diagnostic.py

⸻

SI change × event risk

si_event_risk_interaction

Forskningsfråga:

Är effekten av förändrad blankning starkare vid extrem event-risk?

Diagnostic:

ml/diagnostics/fi_short_interest_event_risk_interaction_diagnostic.py

⸻

Report proximity

report_proximity

Forskningsfråga:

Förstärks sambandet nära rapportdatum?

Status:

planned

⸻

Sector relative return

sector_relative_return

Forskningsfråga:

Är effekten specifik för aktien relativt sektor och marknad?

Status:

planned

⸻

11. Event-risk

Ett viktigt befintligt forskningsspår är event-risk.

Event definieras i den befintliga event-riskdiagnostiken utifrån:

abs(forward_return_5d) >= 10 %

Event-riskmodellen använder bland annat volatilitet som prediktor.

Feature sets omfattar bland annat:

volatility_20d
volatility_60d
volatility_20d_plus_60d
volatility_20d_plus_60d_plus_term_structure

Event-riskmodellen väljs på train/validation.

Efter modellval refittas modellen före OOS-testet.

Detta gör event-risk till en separat dimension som sedan kan användas i andra diagnostics.

⸻

12. Interaktionsexperiment

Blankdiss ska i första hand skilja mellan:

Hög nivå

och:

Interaktion

Exempelvis räcker det inte att observera:

HIGH VOL + HIGH SI → hög event rate

för att visa att volatilitet och blankning interagerar.

För en 2×2-analys:

                    LOW SI       HIGH SI
LOW VOL                A             B
HIGH VOL               C             D

är den direkta interaktionen:

(D - C) - (B - A)

Detta mäter om skillnaden mellan hög och låg SI förändras beroende på volatilitet.

Samma princip kan användas för andra forskningsdimensioner.

⸻

13. OOS och leakage

Detta är en central princip i hela ML-systemet.

Information från testperioden får inte användas för att optimera modellen eller experimentet.

Exempel på korrekt flöde:

TRAIN
  │
  ├── model fitting
  │
  ▼
VALIDATION
  │
  ├── model selection
  │
  ▼
TRAIN + VALIDATION
  │
  ├── refit
  │
  ├── pre-test thresholds
  │
  ▼
TEST
  │
  └── OOS evaluation

Thresholds som används för exempelvis:

top 20 % volatility
top 20 % SI
top 5 % event risk

ska definieras från information som är tillgänglig före testperioden.

⸻

14. Statistik

Diagnostics kan använda mer detaljerade statistiska analyser än den generella research matrixen.

Exempel:

* event rate
* baseline event rate
* lift
* AUC
* interaction effect
* bootstrap confidence interval
* P(interaction > 0)
* cell N
* walk-forward consistency
* pooled OOS results

Ett enskilt positivt resultat ska inte automatiskt betraktas som en etablerad effekt.

Forskningsflödet är:

Screening
   ↓
Hypotes
   ↓
Diagnostic
   ↓
OOS validation
   ↓
Robustness checks
   ↓
Potentially useful signal

⸻

15. Research vs Production ML

Detta ML-system är i första hand ett research-system.

Syftet är att hitta och validera strukturer i data.

Det betyder att:

research result

inte automatiskt betyder:

trading signal

och:

statistically interesting

inte automatiskt betyder:

economically useful

Nästa steg efter en intressant effekt kan därför vara:

* längre period
* fler walk-forward windows
* annan target
* sektorjustering
* marknadsjustering
* transaktionskostnader
* regimanalys
* rapportdatum
* ytterligare out-of-sample-period
* oberoende validering

⸻

16. Automatisering

Den avsedda användningen är att Blankdiss själv ska kunna köra stora delar av forskningsprocessen.

Den normala vägen är:

Forskningsidé
     ↓
Diagnostic
     ↓
Registry
     ↓
GitHub Actions
     ↓
Automatisk datainsamling
     ↓
Feature generation
     ↓
Research
     ↓
Diagnostics
     ↓
Artifacts

Målet är att minimera behovet av att manuellt:

* kopiera data
* köra enskilda scripts
* kopiera loggar
* flytta resultat mellan experiment
* ändra workflow för varje nytt test

När infrastrukturen är på plats ska en ny forskningsfråga i huvudsak kräva:

1. Ny diagnostic
2. Ny registry-post
3. Kör pipeline

⸻

17. Relaterad dokumentation

Detaljerad dokumentation för research-systemet finns i:

ml/research/README.md

Det dokumentet beskriver särskilt:

* research runner
* experiment registry
* registry runner
* diagnostics
* hur nya experiment skapas
* GitHub Actions-kopplingen
* aktuella forskningsfrågor

Detta dokument (ml/README.md) är den övergripande ML-kartan.

⸻

18. Designprincip

Den viktigaste uppdelningen är:

ml/config.py
    = gemensam ML-konfiguration
ml/dataset.py
    = data → ML-data
ml/research/
    = bred, generell research
ml/diagnostics/
    = specifika hypoteser och analyser
ml/experiment_registry.json
    = vilka diagnostics som finns och vilka som är aktiva
ml/experiment_registry_runner.py
    = startar aktiva diagnostics
.github/workflows/ml-research.yml
    = kör hela automatiserade research-pipelinen

Kort sagt:

CONFIG
  ↓
DATASET
  ↓
RESEARCH / DIAGNOSTICS
  ↓
REGISTRY
  ↓
RUNNER
  ↓
GITHUB ACTIONS
  ↓
RESULTS

Detta är den avsedda arkitekturen för Blankdiss ML.
