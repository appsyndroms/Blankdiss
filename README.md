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
         Data collection              Research
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

Projektets grundidé

Blankdiss försöker gå från:

"Det här verkar intressant"

till:

hypotes
   ↓
mätbar definition
   ↓
historisk analys
   ↓
train / validation / test
   ↓
OOS-resultat
   ↓
robusthetstest
   ↓
eventuell vidare forskning

Det är viktigt att skilja mellan:

* ett observerat samband
* en statistiskt intressant effekt
* en robust OOS-effekt
* en ekonomiskt användbar signal

Ett positivt resultat i en enskild analys är därför början på fortsatt forskning, inte automatiskt ett färdigt trading-case.

⸻

Repository-struktur

De viktigaste delarna är:

.
├── analysis/
│   └── ...
│
├── data/
│   ├── raw/
│   └── processed/
│
├── ml/
│   ├── README.md
│   ├── config.py
│   ├── dataset.py
│   ├── experiment_registry.json
│   ├── experiment_registry_runner.py
│   │
│   ├── research/
│   │   ├── README.md
│   │   ├── runner.py
│   │   ├── experiments.py
│   │   ├── evaluator.py
│   │   └── cache.py
│   │
│   └── diagnostics/
│       └── *_diagnostic.py
│
├── prices/
│   └── ...
│
├── fi/
│   └── ...
│
└── .github/
    └── workflows/
        └── ml-research.yml

ML-systemet dokumenteras mer detaljerat i:

ml/README.md

Research- och experimentarkitekturen dokumenteras i:

ml/research/README.md

⸻

Data → Features → ML → Research

Den övergripande pipeline är:

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

ML

ML-delen finns under:

ml/

Den centrala dokumentationen finns i:

ml/README.md

ML-systemet består huvudsakligen av:

ml/config.py
ml/dataset.py
ml/research/
ml/diagnostics/
ml/experiment_registry.json
ml/experiment_registry_runner.py

config.py

Gemensam ML-konfiguration, bland annat:

* targets
* thresholds
* target directions
* walk-forward windows
* random state

dataset.py

Förbereder feature-data för ML.

research/

Generell research och bred screening.

diagnostics/

Specifika hypotesdrivna experiment.

experiment_registry.json

Register över forskningsfrågor och aktiva experiment.

experiment_registry_runner.py

Kör de diagnostics som är markerade som active.

⸻

Research

Blankdiss har två kompletterande forskningslägen.

Generic Research

Den generella research-motorn testar många kombinationer systematiskt.

Exempel:

signal
   ×
target
   ×
tail fraction
   ×
tail direction

Detta används för bred screening.

⸻

Diagnostics

Diagnostics används när forskningsfrågan kräver en mer specifik experimentdesign.

Exempel:

volatility × short-interest level
short-interest change × event risk
report-date proximity
sector-relative return

En diagnostic kan exempelvis innehålla:

* explicit interaktion
* specialiserad modell
* event-risk
* bootstrap
* relativa jämförelser
* särskilda populationsdefinitioner

Detaljer finns i:

ml/research/README.md

⸻

Experiment Registry

Forskningsfrågor som ska kunna köras automatiskt registreras i:

ml/experiment_registry.json

Exempel:

{
  "id": "volatility_si_level_interaction",
  "status": "active",
  "question": "Är risken för en extrem nedgång särskilt hög när både volatiliteten och blankningsnivån är hög?",
  "module": "ml.diagnostics.volatility_si_level_interaction_diagnostic",
  "priority": 1
}

Kopplingen är:

experiment_registry.json
        │
        ▼
experiment_registry_runner.py
        │
        ▼
ml.diagnostics.volatility_si_level_interaction_diagnostic
        │
        ▼
main()

Status

active

innebär att experimentet körs.

planned

innebär att forskningsfrågan är registrerad men inte körs ännu.

Priority

Lägre nummer körs först.

⸻

Nya experiment

Ett nytt experiment ska normalt skapas enligt:

1. Definiera forskningsfrågan
2. Avgör generic research eller diagnostic
3. Implementera diagnostic vid behov
4. Säkerställ korrekt OOS/walk-forward-design
5. Lägg till experimentet i experiment_registry.json
6. Sätt status = active
7. Kör research-workflowen

En diagnostic ska normalt ligga här:

ml/diagnostics/<experiment_name>_diagnostic.py

och exponera:

def main():
    ...

Registryt pekar sedan på modulen.

Workflowen behöver normalt inte ändras för varje nytt experiment.

⸻

OOS och walk-forward

ML-forskningen använder walk-forward evaluation.

Grundprincip:

TRAIN
   ↓
VALIDATION
   ↓
MODEL SELECTION
   ↓
REFIT
   ↓
OOS TEST

Testdata ska inte användas för:

* modellval
* threshold selection
* feature selection
* optimering av experimentet

Thresholds som definierar testpopulationer ska också vara baserade på information som är tillgänglig före testperioden.

Detta är särskilt viktigt i diagnostics som använder exempelvis:

top 20 % volatility
top 20 % short interest
top 5 % event risk

⸻

Exempel på aktuell forskning

Volatility × SI level

Fråga:

Är risken för en extrem nedgång särskilt hög när både volatiliteten och blankningsnivån är hög?

2×2-design:

                    LOW SI       HIGH SI
LOW VOL                A             B
HIGH VOL               C             D

Direkt interaktion:

(D - C) - (B - A)

Det testar om effekten av hög blankning förändras beroende på volatiliteten.

⸻

SI change × event risk

Fråga:

Är effekten av förändrad blankning starkare vid extrem event-risk?

Event-risk definieras bland annat utifrån extrema femdagarsrörelser.

Detta experiment använder OOS event-risk för att undersöka om short-interest-effekten skiljer sig mellan olika risknivåer.

⸻

Report proximity

Fråga:

Förstärks sambandet nära rapportdatum?

Detta är ett framtida diagnostic-spår där rapportdatum används som tidsdimension.

⸻

Sector relative return

Fråga:

Är effekten specifik för aktien relativt sektor och marknad?

Detta ska skilja aktiespecifika effekter från bredare marknads- och sektorrörelser.

⸻

Event-risk

Event-risk är ett separat viktigt forskningsspår.

En event definieras i den befintliga analysen som:

abs(forward_return_5d) >= 10 %

Event-riskmodellen använder bland annat volatilitet som feature.

Exempel på feature sets:

volatility_20d
volatility_60d
volatility_20d + volatility_60d
volatility_20d + volatility_60d + term structure

Modellval sker på train/validation.

Modellen refittas därefter innan OOS-testet.

⸻

GitHub Actions

Den automatiserade ML-pipelinen körs via:

.github/workflows/ml-research.yml

Workflowen ansvarar bland annat för:

Fetch FI aggregate
        ↓
Fetch prices
        ↓
Build features
        ↓
Inspect features
        ↓
Verify features
        ↓
Feature QC
        ↓
Generic Research Matrix
        ↓
Registered Experiments
        ↓
Verify results
        ↓
Upload artifacts

Inför varje research-körning rensas filerna i:

data/processed/ml/research/latest/

Själva katalogstrukturen bevaras, inklusive:

data/processed/ml/research/latest/diagnostic/

Därefter skrivs den nya körningens resultat dit.

Registry-baserade diagnostics startas genom:

python -u -m ml.experiment_registry_runner

Det innebär att nya aktiva diagnostics kan köras utan att workflow-filen behöver byggas om.

⸻

Resultat

Research-resultat skrivs till data-katalogen efter varje körning:

data/processed/ml/research/

Den aktuella körningens resultat finns alltid under:

data/processed/ml/research/latest/

`latest/` är den stabila sökvägen till den senaste research-körningen.

Inför varje ny körning tas befintliga filer i `latest/` bort. Katalogstrukturen behålls och fylls sedan med resultaten från den nya körningen.

Den aktuella resultatstrukturen är:

data/processed/ml/research/latest/
├── results.jsonl
├── pooled.json
├── metadata.json
├── report.md
├── diagnostics.json
└── diagnostic/
    └── <experiment_id>.json

Viktiga filer för generisk research är:

results.jsonl
pooled.json
metadata.json
report.md

Registry-baserade diagnostics producerar dessutom:

diagnostics.json
diagnostic/<experiment_id>.json

`diagnostics.json` är manifestet för diagnostic-körningen.

De individuella diagnostic-resultaten ligger i:

diagnostic/<experiment_id>.json

En timestampad körning sparas dessutom historiskt under:

data/processed/ml/research/<run_timestamp>/

Den historiska körningen innehåller samma resultatstruktur:

data/processed/ml/research/<run_timestamp>/
├── results.jsonl
├── pooled.json
├── metadata.json
├── report.md
├── diagnostics.json
└── diagnostic/
    └── <experiment_id>.json

Det innebär att:

* `latest/` alltid representerar den senaste körningen.
* Filerna i `latest/` ersätts vid varje ny körning.
* Katalogstrukturen i `latest/` bevaras.
* Timestampade körningar sparas separat som historik.
* Resultaten skrivs till `data/` som en del av varje research-körning.
* Diagnostic-resultat är maskinläsbara JSON-filer.
* GitHub Actions laddar upp resultatfilerna som artifacts.

Resultatfilerna är den primära outputen från research-körningen.

AI ska läsa resultat-artifacts direkt efter en genomförd körning när de är tillgängliga. Actions-loggen är inte den primära källan för statistisk analys.

Prioriterad läsordning är:

1. diagnostic/<experiment_id>.json
2. diagnostics.json
3. pooled.json
4. results.jsonl
5. report.md

Actions-loggen används främst för:

* pipeline-status
* fel
* verifiering
* korta körningssammanfattningar

Användaren ska normalt inte behöva kopiera forskningsresultat från Actions-loggen till chatten.

Den avsedda kedjan är:

GitHub Actions
      ↓
Research
      ↓
Result artifacts i data/
      ↓
AI reads result files
      ↓
Research analysis
      ↓
Nästa forskningsfråga

⸻

AI-readable research contract

Resultat från Blankdiss research ska vara maskinläsbara och möjliga att konsumera utan att terminaloutput behöver tolkas manuellt.

För generisk research är huvudresultaten:

results.jsonl
pooled.json
metadata.json

För registry-baserade diagnostics är:

diagnostics.json
diagnostic/<experiment_id>.json

den primära resultatvägen.

En diagnostic-resultatfil ska identifieras med experimentets stabila registry-id.

Exempel:

data/processed/ml/research/latest/diagnostic/si_event_risk_interaction.json

Resultatfilen ska innehålla experimentets strukturerade resultat när diagnosticen stödjer detta.

Fri terminaloutput får användas som kompletterande information, men ska inte vara den enda representationen av forskningsresultatet.

Det innebär att Blankdiss researcharkitektur ska gå mot:

research
   ↓
machine-readable artifact
   ↓
AI-readable result
   ↓
analysis

och inte:

research
   ↓
terminal log
   ↓
manual copy/paste
   ↓
analysis

⸻

Lokal körning

För att köra den generella research-motorn:

python -u -m ml.research.runner

För att köra alla aktiva registrerade diagnostics:

python -u -m ml.experiment_registry_runner

Normalt ska hela kedjan köras genom:

.github/workflows/ml-research.yml

så att data, features, QC och research använder samma pipeline.

⸻

Forskningsprinciper

Blankdiss research ska följa några grundprinciper.

1. Hypotes före resultat

Forskningsfrågan ska definieras innan resultatet analyseras.

2. OOS före slutsats

Ett samband ska helst verifieras out-of-sample.

3. Walk-forward

Historiska modeller ska testas på efterföljande perioder.

4. Undvik leakage

Information från framtiden får inte påverka modell eller populationströsklar.

5. Interaktion framför bara korrelation

När frågan gäller en kombination av faktorer ska interaktionen testas direkt.

6. Robusthet

Intressanta resultat ska följas av relevanta robusthetstester.

7. Separera screening från hypotesprövning

Bred screening och specifika diagnostics har olika roller.

⸻

Automatiseringsmål

Blankdiss ska kunna gå från:

Ny forskningsidé

till:

Automatisk historisk analys
        ↓
OOS-resultat
        ↓
Maskinläsbara artifacts
        ↓
AI-analys
        ↓
Rapport
        ↓
Nästa forskningsfråga

utan att varje experiment kräver en ny specialbyggd pipeline eller manuell överföring av loggar.

Den önskade arkitekturen är:

                 RESEARCH IDEA
                       │
                       ▼
              ┌─────────────────┐
              │ Generic Research│
              │       eller     │
              │    Diagnostic   │
              └────────┬────────┘
                       │
                       ▼
                Experiment Registry
                       │
                       ▼
                Registry Runner
                       │
                       ▼
                 GitHub Actions
                       │
                       ▼
                  OOS Results
                       │
                       ▼
                   Artifacts
                       │
                       ▼
                  AI Analysis
                       │
                       ▼
              Nästa forskningsfråga

Detta gör ML-delen till ett växande forskningssystem snarare än en samling fristående scripts.

⸻

Dokumentation

För ML-översikten:

ml/README.md

För detaljer om research, registry, runner och diagnostics:

ml/research/README.md

För specifika experiment:

ml/diagnostics/

Rot-README:n beskriver projektets övergripande struktur; ML-README:n beskriver ML-arkitekturen; research-README:n beskriver själva experimentinfrastrukturen.
