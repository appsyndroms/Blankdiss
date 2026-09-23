Blankdiss Research Engine

ml/research/ är Blankdiss generiska forskningsmotor.

Målet är att göra nya researchhypoteser billiga att formulera och köra.

Grundprincipen är:

hypotes
   ↓
YAML research spec
   ↓
SCAN
   ↓
intressant signal/interaktion
   ↓
DEEP
   ↓
robusthetsanalys / specialanalys

Arkitektur

ml/
└── research/
    ├── README.md
    ├── bootstrap.py
    ├── cache.py
    ├── engine.py
    ├── reporting.py
    ├── runner.py
    ├── session.py
    ├── signals.py
    ├── spec.py
    │
    ├── specs/
    │   ├── README.md
    │   └── *.yaml
    │
    └── custom/
        └── README.md

spec.py

Definierar det deklarativa research-formatet.

En spec beskriver:

* forskningsfråga
* signaler
* tail-riktning
* tail-fraktioner
* targets
* analysis-typ
* walk-forward-fönster
* SCAN/DEEP
* metadata

Nya vanliga hypoteser ska normalt börja som YAML.

session.py

Bygger en gemensam ResearchSession.

Sessionen:

1. läser feature-datasetet en gång
2. identifierar vilka signaler/targets som behövs
3. bygger en gemensam ResearchCache

Alla specs som körs i samma runner delar denna cache.

Detta är centralt för prestandan.

cache.py

Innehåller återanvändbara NumPy-arrayer:

* signaler
* targets
* forward returns
* tail masks
* walk-forward masks
* target-konfigurationer

Dyra operationer ska göras här en gång per session, inte en gång per hypotes.

engine.py

Den generiska analysmotorn.

Nuvarande generiska analysformer:

* tail
* interaction

tail analyserar en signal ensam.

interaction analyserar kombinationen av två signaler.

Motorn beräknar bland annat:

* antal observationer
* event count
* event rate
* baseline event rate
* lift
* mean return
* median return
* return difference
* bootstrap CI i DEEP-läge

runner.py

Kör en eller flera YAML-specar.

Utan argument körs alla specs i:

ml/research/specs/

En session byggs först och därefter körs alla specs mot samma cache.

Exempel:

python -m ml.research.runner

En specifik spec:

python -m ml.research.runner \
  ml/research/specs/si_momentum_scan.yaml

Flera specs:

python -m ml.research.runner \
  ml/research/specs/si_momentum_scan.yaml \
  ml/research/specs/tail_signal_scan.yaml

SCAN och DEEP

SCAN

SCAN ska vara billigt.

Syftet är att svara på:

Finns det någonting här som är värt att undersöka vidare?

SCAN använder därför normalt:

* hela tillgängliga signalmatrisen
* flera tail-fraktioner
* flera targets
* walk-forward-test
* enkla effektmått

Bootstrap och andra dyra analyser ska normalt vara avstängda.

DEEP

DEEP körs först när en SCAN producerat en intressant hypotes.

DEEP kan aktivera:

* bootstrap
* robusthetskontroller
* alternativa cutoffs
* fler tidsperioder
* placebo-/kontrollanalyser
* specialiserad analys

Principen är:

Gör inte en dyr analys av något som först borde ha screenats bort.

När ska Python skrivas?

Python ska normalt inte behövas för en ny vanlig hypotes.

Börja med YAML om frågan kan uttryckas som:

* en signal
* en tail
* två signaler
* ett target
* ett antal cutoffs
* ett walk-forward-fönster

Python läggs till först när frågan kräver något som den generiska motorn inte kan uttrycka.

Exempel:

* specialiserad 2×2-interaktion
* conditional regression
* permutationstest
* event-sekvensanalys
* path dependence
* ovanlig gruppering
* komplex mekanismanalys

Sådana analyser hör hemma i:

ml/research/custom/

Resultat

Research-resultat skrivs under:

data/processed/ml/research/spec_runs/

Varje körning får en timestamp:

spec_runs/
└── 20260923T183000Z/
    ├── manifest.json
    ├── si_momentum_downside_scan.json
    └── tail_signal_scan.json

manifest.json beskriver hela körningen.

Migration från legacy

Den gamla experimentarkitekturen tas inte bort direkt.

Följande komponenter betraktas som legacy:

* ml/research/experiments.py
* ml/experiment_registry.json
* ml/experiment_registry_runner.py
* tunna wrappers under ml/diagnostics/experiments/

De får finnas kvar under migrationen.

En legacy-komponent tas bort först när:

1. dess relevanta analys är reproducerad i den nya motorn eller custom/
2. resultat har jämförts
3. ingen workflow längre behöver komponenten
4. inga andra moduler importerar den

Designprincip

Research-koden ska optimeras för:

idé → test

inte:

idé
→ ny Python-fil
→ ny experimentklass
→ registry
→ wrapper
→ workflow
→ körning

Det deklarativa formatet är därför standardvägen för nya researchfrågor.
