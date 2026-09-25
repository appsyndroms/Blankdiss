Blankdiss ML

Detta är den övergripande dokumentationen för Blankdiss ML-system.

Arkitektur

ml/
├── config.py
├── dataset.py
├── walk_forward.py
│
├── research/
│   ├── spec.py
│   ├── session.py
│   ├── cache.py
│   ├── signals.py
│   ├── engine.py
│   ├── runner.py
│   ├── reporting.py
│   ├── bootstrap.py
│   ├── specs/
│   └── custom/
│
└── diagnostics/
    ├── framework/
    └── experiments/

ML-systemet har två forskningsvägar:

                    ML
                     │
             ┌───────┴───────┐
             │               │
          Research       Diagnostics
             │               │
        deklarativt       specialiserat
             │               │
             └───────┬───────┘
                     ▼
                  OOS

⸻

1. config.py

Innehåller gemensam ML-konfiguration:

* targets
* target thresholds
* target directions
* walk-forward windows
* gemensamma konstanter

Konfiguration som delas av flera delar av systemet ska ligga här.

⸻

2. dataset.py

Ansvarar för datasetet.

Viktiga funktioner inkluderar:

* load_features()
* build_target()
* feature preparation
* ML-data preparation

Feature-data kommer från:

data/processed/analysis/

Datasetlagret ska inte innehålla forskningsspecifik analyslogik.

⸻

3. walk_forward.py

Innehåller den gemensamma tidsmässiga uppdelningen.

Grundprincip:

TRAIN
   ↓
VALIDATION
   ↓
MODEL SELECTION
   ↓
REFIT
   ↓
TEST / OOS

Testperioden är den slutliga out-of-sample-gränsen.

⸻

4. Research Engine

ml/research/ är den generiska forskningsmotorn.

Den är optimerad för:

idé
 ↓
YAML
 ↓
SCAN
 ↓
DEEP
 ↓
resultat

Den ska minimera behovet av ny Python-kod för vanliga hypoteser.

⸻

Research Engine-komponenter

spec.py

Definierar YAML-formatet.

En research spec beskriver bland annat:

* forskningsfråga
* signaler
* riktning
* tail fractions
* targets
* analysis type
* mode
* windows
* splits
* metadata

⸻

session.py

Skapar en gemensam ResearchSession.

Sessionen laddar feature-datasetet en gång och identifierar vilka komponenter som behövs för alla specs.

⸻

cache.py

Bygger gemensamma cache-komponenter.

Exempel:

* signaler
* targets
* returns
* tail masks
* window masks

Samma data ska inte beräknas om för varje hypotes.

⸻

signals.py

Definierar standardiserade research-signaler.

Exempel:

* short interest level
* short interest change
* short interest acceleration
* price momentum
* volatility
* distance from highs

⸻

engine.py

Kör den generiska analysen.

Engine innehåller återanvändbar analyslogik.

Exempel på analysis types:

* tail
* interaction
* incremental_model

En analysis type beskriver en generell typ av forskningsanalys, inte en specifik hypotes.

Exempel:

incremental_model kan användas för att jämföra:

M0 = baseline signal

M1 = baseline signal + incremental signal

M2 = baseline signal + incremental signal + interaction

Den konkreta hypotesen ska sedan beskrivas i YAML.

Engine beräknar standardiserade resultat såsom:

* antal observationer
* event count
* event rate
* baseline
* lift
* mean return
* median return
* return difference
* modellmetrics
* bootstrap-resultat i DEEP

Gemensam analyslogik ska ligga centralt i Engine.

⸻

runner.py

Orkestrerar research.

load specs
    ↓
build session
    ↓
build shared cache
    ↓
run specs
    ↓
write results
    ↓
write manifest

Alla specs i samma körning delar session och cache.

⸻

reporting.py

Ansvarar för standardiserad output.

Resultaten ska vara maskinläsbara och lämpade för vidare analys.

⸻

bootstrap.py

Gemensam bootstrap-funktionalitet för DEEP-analyser.

Bootstrap ska inte implementeras separat i varje hypotes.

⸻

5. SCAN och DEEP

SCAN

SCAN är billig screening.

Syftet är:

Finns det någonting här som är värt att undersöka vidare?

SCAN kan testa:

* flera signaler
* flera tail fractions
* flera targets
* flera windows

Dyra analyser ska normalt inte ligga här.

DEEP

DEEP används efter ett intressant SCAN-resultat.

DEEP kan innehålla:

* bootstrap
* confidence intervals
* alternativa cutoffs
* robusthetskontroller
* placebo-/kontrollanalyser
* fler tidsperioder
* specialiserad analys

⸻

6. När ska Python skrivas?

Börja alltid med YAML.

Innan ny Python skrivs ska följande kontrolleras:

1. Kan hypotesen uttryckas med befintlig Research Engine?
2. Finns redan nödvändiga signaler?
3. Finns redan nödvändiga targets?
4. Finns redan nödvändiga cache-komponenter?
5. Finns redan en lämplig analysis type?

Om svaret är JA:

Hypotes
   ↓
YAML
   ↓
Research Engine

Ingen ny experimentklass ska skapas.

Om svaret är NEJ ska nästa fråga vara:

Är den saknade funktionaliteten generell?

Om JA:

Hypotes
   ↓
ny generell Engine-funktionalitet
   ↓
YAML
   ↓
Research Engine

Exempel:

Om Research Engine saknar incremental_model ska man inte skapa:

momentum_incremental_si.py

för den första hypotesen.

I stället ska incremental_model implementeras generellt i Engine.

Sedan ska:

momentum + SI change

beskrivas som en YAML-spec.

Om funktionaliteten däremot är unik och inte rimligen ska bli en generell analysis type kan den placeras i:

ml/research/custom/

eller, om den kräver ett separat specialiserat experimentframework:

ml/diagnostics/

Grundregel:

Ny hypotes
    → YAML

Ny generell analysform
    → Engine + YAML

Unik specialanalys
    → custom / Diagnostics

⸻

7. Research kontra Diagnostics

Diagnostics är den specialiserade forskningsvägen.

Den används när generisk Research Engine inte räcker efter att det har bedömts att den saknade logiken inte bör vara en generell Engine-funktion.

Diagnostics ska inte användas som wrapper för analyser som enkelt kan uttryckas i YAML.

Diagnostics ska inte heller användas enbart för att Research Engine ännu inte råkar stödja en viss generell analysform.

⸻

8. Resultat

Research-resultat:

data/processed/ml/research/spec_runs/

En körning innehåller exempelvis:

spec_runs/
└── <timestamp>/
    ├── manifest.json
    ├── spec_a.json
    └── spec_b.json

Resultatfilerna är den primära källan för statistisk analys.

⸻

9. Forskningsloop

Den normala arbetsprocessen är:

Hypotes
   ↓
Läs README
   ↓
Kontrollera befintlig Engine
   ↓
Kan YAML beskriva den?
   │
   ├── JA → Research spec
   │          ↓
   │        SCAN
   │          ↓
   │        DEEP
   │
   └── NEJ
        ↓
   Är den nya analysformen generell?
        │
        ├── JA → utöka Engine
        │          ↓
        │        Research spec
        │
        └── NEJ → custom / diagnostics

Efter resultat:

resultat
   ↓
tolkning
   ↓
nästa hypotes

Målet är kortast möjliga väg mellan idé och information.

⸻

10. Viktiga principer

* Gemensam logik ska ligga centralt.
* Hypoteser ska vara små.
* Nya konkreta hypoteser ska normalt vara YAML.
* SCAN ska vara billig.
* DEEP ska användas selektivt.
* OOS är den slutliga kontrollen.
* Test leakage ska undvikas.
* Resultat ska vara maskinläsbara.
* Duplicerad kod ska undvikas.
* Speciallogik ska vara explicit.
* Död kod ska tas bort.
* En ny generell analysform ska implementeras i Engine.
* Diagnostics ska inte användas som fallback för saknad generell Engine-funktionalitet.
* Legacy-implementationer ska tas bort efter verifierad migrering.

⸻

Relaterad dokumentation

* README.md
* ml/research/README.md
* ml/diagnostics/README.md
* ml/research/specs/README.md
* ml/research/custom/README.md
