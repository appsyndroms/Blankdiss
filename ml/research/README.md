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

Engine ska innehålla återanvändbara analysis types.

Exempel:

tail
interaction
incremental_model

En analysis type är en generell analysförmåga.

Den ska inte vara namngiven efter en specifik forskningshypotes.

Exempel:

Rätt:

incremental_model

Fel:

momentum_incremental_si

Det konkreta experimentet ska beskrivas av YAML-specen.

Engine ansvarar för standardiserade mått såsom:

* observation count
* event count
* event rate
* baseline event rate
* lift
* mean return
* median return
* return difference
* modellmetrics
* bootstrap CI i DEEP

Analyslogik som återkommer mellan hypoteser ska flyttas hit.

⸻

Införande av nya forskningshypoteser

När en ny hypotes ska införas ska följande ordning alltid användas:

1. Formulera forskningsfrågan.
2. Läs denna README.
3. Kontrollera spec.py.
4. Kontrollera befintliga analysis types i engine.py.
5. Kontrollera signals.py.
6. Kontrollera cache.py.
7. Avgör om hypotesen kan uttryckas med befintlig Engine.
8. Om JA: skapa YAML-spec.
9. Om NEJ: avgör om den saknade analysformen är generell.
10. Om generell: implementera den i Engine.
11. Skapa därefter YAML-specen.
12. Kör research runner.
13. Verifiera resultat och OOS.
14. Ta bort eventuell legacy-implementation när migreringen är verifierad.

Den viktiga distinktionen är:

Ny hypotes
    → YAML

Ny generell analysis type
    → Engine

Ny hypotes som använder den nya analysis typen
    → YAML

En första hypotes som kräver ny Engine-funktionalitet ska alltså inte implementeras som standalone Python.

Exempel:

Hypotes:

M0 = momentum

M1 = momentum + SI change

M2 = momentum + SI change + momentum × SI change

Om Engine saknar incremental_model ska man inte skapa:

ml/research/momentum_incremental_si_analysis.py

för den nya hypotesen.

I stället:

incremental_model
    ↓
engine.py

och:

momentum_incremental_si.yaml
    ↓
Research Engine

Det gör att nästa liknande hypotes kan använda samma analysis type utan ny specialkod.

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

Custom ska inte användas enbart för att en generell analysis type ännu saknas i Engine.

Om samma analyslogik kan användas av flera framtida hypoteser ska den normalt införas som en generell Engine-funktion i stället.

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

Men innan en fråga flyttas till custom eller Diagnostics ska det kontrolleras om den egentligen representerar en generell analysis type som saknas i Engine.

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

Om hypotesen kan uttryckas med befintlig analysis type ska endast YAML-specen behöva skapas.

Om hypotesen inte kan uttryckas med befintlig analysis type:

1. Identifiera vilken funktionalitet som saknas.
2. Avgör om den är generell.
3. Om generell: lägg den i Engine.
4. Skapa därefter YAML-specen.
5. Om unik: överväg custom/ eller Diagnostics.

Exempel:

Ny hypotes
    ↓
befintlig Engine?
    │
    ├── JA → YAML
    │
    └── NEJ
         ↓
    generell analysform?
         │
         ├── JA → Engine → YAML
         │
         └── NEJ → custom / Diagnostics

Om samma logik sedan används av flera analyser ska den finnas centralt i Engine.

⸻

Legacy och migration

När en äldre standalone-analys ersätts av Research Engine ska migreringen ske i följande ordning:

1. Identifiera vilken generell analysförmåga legacy-koden representerar.
2. Implementera den generellt i Engine.
3. Skapa en YAML-spec som reproducerar hypotesen.
4. Kör gammal och ny implementation parallellt under verifieringen.
5. Jämför resultat.
6. Verifiera OOS och output.
7. När den nya vägen är verifierad: ta bort legacy-koden.
8. Ta bort eventuell duplicerad registry-/workflow-logik.

Det ska inte finnas två permanenta implementationsvägar för samma analys.

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

Den generiska Engine-funktionen ska byggas först när en ny analysform behövs.

Den konkreta forskningshypotesen ska därefter beskrivas deklarativt.

⸻

Migration status

Den gamla experiment-/registry-arkitekturen är inte en del av den slutliga Research Engine.

Efter migreringen ska:

* generiska experiment vara YAML
* gemensam logik finnas i Engine
* nya generella analysis types finnas i Engine
* specialiserad Research finnas i custom/
* Diagnostics endast innehålla verkligt specialiserade analyser
* legacy-filer vara borttagna
* död kod vara borttagen
* duplicerad analyslogik vara borttagen

Det finns ingen anledning att behålla en gammal implementation parallellt när den nya funktionaliteten är verifierad.
