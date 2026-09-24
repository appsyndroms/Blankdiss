Blankdiss Research Engine

ml/research/ är Blankdiss generiska forskningsmotor.

Målet är att göra nya forskningshypoteser billiga att formulera, köra och iterera.

Grundflödet är:

hypotes
   ↓
YAML research spec
   ↓
SCAN
   ↓
intressant resultat
   ↓
DEEP
   ↓
robusthet / specialanalys
   ↓
OOS-resultat

⸻

Arkitektur

ml/research/
├── spec.py
├── session.py
├── cache.py
├── signals.py
├── engine.py
├── runner.py
├── reporting.py
├── bootstrap.py
│
├── specs/
│   └── *.yaml
│
└── custom/
    └── *.py

⸻

spec.py

Definierar det deklarativa research-formatet.

En spec beskriver:

* forskningsfråga
* signaler
* signalriktning
* tail fractions
* targets
* analys
* mode
* windows
* splits
* metadata

Vanliga hypoteser ska normalt börja här.

⸻

session.py

Skapar en gemensam ResearchSession.

Sessionen:

1. laddar feature-datasetet
2. identifierar gemensamma krav från alla specs
3. bygger ResearchCache
4. återanvänder samma data mellan analyser

Det gör att flera hypoteser kan köras utan att samma dyra förberedelser upprepas.

⸻

cache.py

Cachelagret innehåller återanvändbara komponenter:

* signaler
* targets
* forward returns
* tail masks
* walk-forward masks

Princip:

1 × data preparation
N × research hypotheses

inte:

N × data preparation
N × research hypotheses

⸻

signals.py

Innehåller standardiserade signaldefinitioner.

Exempel:

* short_interest_level
* short_interest_change
* short_interest_acceleration
* price_momentum_5d
* price_momentum_20d
* price_momentum_60d
* price_volatility_20d
* distance_from_20d_high
* distance_from_60d_high

Nya återanvändbara signaler ska läggas här.

⸻

engine.py

Den generiska analysmotorn.

Nuvarande analysformer:

tail
interaction

Engine ansvarar för standardiserade mått såsom:

* observation count
* event count
* event rate
* baseline event rate
* lift
* mean return
* median return
* return difference
* bootstrap CI i DEEP

Analyslogik som återkommer mellan hypoteser ska flyttas hit.

⸻

runner.py

Runnern orkestrerar hela research-körningen.

YAML specs
    ↓
ResearchSession
    ↓
ResearchCache
    ↓
Engine
    ↓
Results
    ↓
Manifest

Alla specs i samma körning delar session och cache.

Exempel:

python -m ml.research.runner

En specifik spec:

python -m ml.research.runner \
  ml/research/specs/my_hypothesis.yaml

⸻

SCAN

SCAN är första filtret.

Syftet är:

Hitta hypoteser som förtjänar mer analys.

SCAN ska vara:

* bred
* snabb
* billig
* reproducerbar

Typiskt:

många signaler
×
flera tails
×
flera targets
×
OOS windows

Dyra analyser ska normalt vänta.

⸻

DEEP

DEEP används när SCAN visar något som är värt att undersöka.

Exempel:

* bootstrap
* confidence intervals
* alternativa cutoffs
* robusthetskontroller
* placebo
* kontrollgrupper
* fler tidsperioder
* specialiserad analys

Princip:

Gör inte en dyr analys av något som först borde ha screenats bort.

⸻

Research specs

Research specs finns under:

ml/research/specs/

En vanlig spec kan exempelvis definiera:

id: example_scan
question: >
  Hypotesen som ska testas.
mode: scan
signals:
  - name: short_interest_change
    direction: upper
    bins: [0.20, 0.10, 0.05, 0.01]
targets:
  - down_5pct_5d
analysis:
  type: tail
  bootstrap: false
windows:
  - window_1
  - window_2
splits:
  - test

Specen beskriver vad som ska testas.

Engine beskriver hur den generiska analysen genomförs.

⸻

Custom research

All research passar inte YAML.

Specialiserad research som fortfarande hör hemma i Research-lagret placeras under:

ml/research/custom/

Exempel:

* permutationstest
* path dependence
* komplexa event-sekvenser
* specialiserad regression
* ovanlig gruppering
* avancerad mekanismanalys

Custom-kod ska fortfarande återanvända Research Engine där det är möjligt.

⸻

Research kontra Diagnostics

Tumregel:

Kan frågan beskrivas deklarativt?
        │
       JA
        ↓
     Research
        │
       NEJ
        ↓
 custom / Diagnostics

Diagnostics används när analysen kräver ett mer specialiserat experimentframework.

⸻

Walk-forward och OOS

Research ska respektera walk-forward-gränser.

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

Testperioden får inte användas för att optimera hypotesen.

Det gäller även:

* thresholds
* feature selection
* modellval
* cutoffs
* population definitions

⸻

Resultat

Resultat skrivs under:

data/processed/ml/research/spec_runs/

Exempel:

spec_runs/
└── 20260924T123456Z/
    ├── manifest.json
    ├── hypothesis_a.json
    └── hypothesis_b.json

Resultat ska vara maskinläsbara.

Den normala kedjan är:

Research
   ↓
JSON / artifacts
   ↓
AI / human analysis
   ↓
Next hypothesis

⸻

När ny Python behövs

Skriv inte en ny experimentklass bara för att testa en vanlig hypotes.

Börja med YAML.

Ny Python är motiverad när analysen kräver verkligt ny logik.

Om samma logik sedan används av flera analyser ska den flyttas från custom till Engine.

⸻

Designprincip

Research Engine är optimerad för:

idé → information

inte:

idé
 ↓
ny Python-fil
 ↓
ny klass
 ↓
registry
 ↓
wrapper
 ↓
workflow
 ↓
körning

Den deklarativa vägen är standard.

⸻

Migration status

Den gamla experiment-/registry-arkitekturen är inte en del av den slutliga Research Engine.

Efter migreringen ska:

* generiska experiment vara YAML
* gemensam logik finnas i Engine
* specialiserad Research finnas i custom/
* Diagnostics endast innehålla verkligt specialiserade analyser
* legacy-filer vara borttagna
* död kod vara borttagen

Det finns ingen anledning att behålla en gammal implementation parallellt när den nya funktionaliteten är verifierad.
