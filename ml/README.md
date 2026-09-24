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

Nuvarande analysformer:

* tail
* interaction

Engine beräknar bland annat:

* antal observationer
* event count
* event rate
* baseline
* lift
* mean return
* median return
* return difference
* bootstrap-resultat i DEEP

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

Börja med YAML.

Om hypotesen kan beskrivas som:

signal
×
tail
×
target
×
window

ska ingen ny experimentklass behövas.

Python används när hypotesen kräver exempelvis:

* specialiserad regression
* permutationstest
* komplex interaktion
* event-sekvens
* path dependence
* specialiserad gruppering
* mekanismanalys

Sådan kod hör normalt hemma i:

ml/research/custom/

eller, om den är mer specialiserad:

ml/diagnostics/

⸻

7. Diagnostics

Diagnostics är den specialiserade forskningsvägen.

Den används när generisk Research Engine inte räcker.

Diagnostics ska inte användas som wrapper för analyser som enkelt kan uttryckas i YAML.

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
Kan YAML beskriva den?
   │
   ├── JA → Research spec
   │          ↓
   │        SCAN
   │          ↓
   │        DEEP
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
* SCAN ska vara billig.
* DEEP ska användas selektivt.
* OOS är den slutliga kontrollen.
* Test leakage ska undvikas.
* Resultat ska vara maskinläsbara.
* Duplicerad kod ska undvikas.
* Speciallogik ska vara explicit.
* Död kod ska tas bort.

⸻

Relaterad dokumentation

* README.md
* ml/research/README.md
* ml/diagnostics/README.md
* ml/research/specs/README.md
* ml/research/custom/README.md
