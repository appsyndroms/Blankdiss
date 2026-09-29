README.md

# Blankdiss
Blankdiss är ett forskningsprojekt för att undersöka om offentlig information om bolag, blankning, prisrörelser, rapporter och andra marknadsvariabler innehåller statistiskt och ekonomiskt användbara mönster.
Projektet kombinerar:
- datainsamling
- feature engineering
- datakvalitet
- ML
- hypotesdriven research
- walk-forward evaluation
- prospective evaluation
- automatiserade analyser
- reproducerbara resultat
- AI-assisterad forskning
Målet är inte att bygga en samling fristående analyser.
Målet är att bygga en sammanhängande forskningskedja där en hypotes kan gå från:
    hypotes
        ↓
    deklarativ research-spec
        ↓
    SCAN
        ↓
    DEEP / specialanalys
        ↓
    kontrollerad hypotes
        ↓
    frozen candidate
        ↓
    prospective evaluation
        ↓
    walk-forward / OOS
        ↓
    verification
        ↓
    resultat
        ↓
    nästa hypotes
med så lite specialkod som möjligt.
---
## Översikt
```text
                         Blankdiss
                            │
          ┌─────────────────┴─────────────────┐
          │                                   │
       Data layer                         Research layer
          │                                   │
    ┌─────┼─────┐                       ┌─────┴─────┐
    │     │     │                       │           │
   FI   Prices Market                 Research   Diagnostics
    │     │     │                       │           │
    └─────┼─────┘                       │           │
          ▼                             │           │
   Feature generation                  │           │
          │                             │           │
          ▼                             ▼           ▼
    Feature dataset              Research Engine
          │                             │
          ▼                             │
      Feature QC                       │
          │                             │
          └──────────────┬──────────────┘
                         ▼
                    Verification
                         │
                         ▼
                  OOS / Evaluation
                         │
                         ▼
                  AI / analysis

⸻

Dataflöde

Den aktiva data- och forskningskedjan är:

FI
 ↓
Prices
 ↓
OMXSPI
 ↓
Feature generation
 ↓
Feature QC
 ↓
Research
 ↓
Evaluation

Feature generation är en aktiv upstream-komponent.

Research Engine ska i första hand använda det gemensamma feature-datasetet i stället för att bygga om features för varje hypotes.

⸻

Research Engine

Den generiska research-motorn finns under:

ml/research/

Den centrala modellen är:

Research question
      ↓
YAML spec
      ↓
ResearchSpec
      ↓
ResearchSession
      ↓
shared cache
      ↓
Research Engine
      ↓
machine-readable result

En ny konkret hypotes ska normalt beskrivas som YAML.

Om en generell analysförmåga saknas ska Research Engine utökas generellt.

En enskild hypotes ska inte få en egen specialgren i Engine.

⸻

SCAN

SCAN är den breda och relativt billiga forskningsfasen.

Syftet är:

Finns det något här som är värt att undersöka vidare?

SCAN kan exempelvis undersöka flera:

* signaler
* targets
* tail fractions
* tidsfönster
* kombinationer

Resultatet används för att identifiera strukturer som kan motivera DEEP eller en mer kontrollerad hypotes.

⸻

DEEP

DEEP används för selektiv fördjupning.

Exempel:

* bootstrap
* confidence intervals
* robusthetskontroller
* alternativa cutoffs
* placeboanalyser
* kontrollanalyser
* längre tidsperioder
* ytterligare stratifiering

DEEP ska inte automatiskt bli en optimeringsloop som letar efter den mest attraktiva kombinationen.

⸻

Candidates

När en forskningsidé har blivit tillräckligt preciserad kan den representeras som en candidate.

Grundflödet är:

Research
   ↓
Controlled hypothesis
   ↓
Candidate
   ↓
Freeze
   ↓
Prospective evaluation

Efter freeze ska candidate-definitionen vara immutable.

Om en definition ändras ska det betraktas som en ny candidate.

⸻

Prospective evaluation

Prospective evaluation testar en frozen candidate på framtida data.

FROZEN CANDIDATE
        +
FUTURE DATA
        ↓
PROSPECTIVE EVALUATION

Evaluation får inte ändra candidate-definitionen efter att resultatet blivit känt.

⸻

Walk-forward och OOS

Forskningskedjan ska respektera tidsordningen:

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

* feature selection
* parameter selection
* threshold selection
* modellval
* optimering av hypotesen

⸻

Verification

Verification ska vara ett gemensamt kontrollager.

Det ska bland annat kunna kontrollera:

* temporal leakage
* feature/target leakage
* future-data access
* missing data
* candidate mutation
* evaluation contamination
* duplicate runs
* distributionsförändringar
* provenance

Kontrollerna ska så långt möjligt vara generella och återanvändbara.

⸻

Central entrypoint

Den centrala research-entrypointen är:

ml/research/entrypoint.py

Avsedd CLI:

python -m ml.research.entrypoint freeze ...
python -m ml.research.entrypoint scan ...
python -m ml.research.entrypoint deep ...
python -m ml.research.entrypoint evaluate ...
python -m ml.research.entrypoint pipeline ...

GitHub Actions ska använda denna entrypoint i stället för att känna till interna detaljer i Research Engine.

⸻

Research-resultat

Deklarativa research-körningar använder:

data/processed/ml/research/spec_runs/

En körning har normalt formen:

spec_runs/
└── <timestamp>/
    ├── manifest.json
    ├── <spec-id>.json
    └── ...

Resultaten ska vara maskinläsbara och kunna kopplas tillbaka till den spec som kördes.

⸻

AI Lab

AI Lab är orchestration-lagret för iterativ och adaptiv forskning.

AI Lab
  ├── discovery
  ├── orchestration
  ├── state
  └── adaptive research
          ↓
    Research Engine

Research Engine ansvarar för den generiska analysen.

AI Lab ansvarar för forskningsflödet.

YAML är forskningskontraktet.

⸻

Runtime-spårning

Blankdiss ska kunna identifiera vilka projektfiler som faktiskt används under centrala körningar.

Princip:

central entrypoint
      ↓
runtime tracking
      ↓
faktiskt använda filer
      ↓
statisk referensanalys
      ↓
verifierad filkarta

Planerat resultat:

data/filborttag/anvandfiler.jsonl

Runtime-användning är evidens för att en fil används, men är inte ensam tillräckligt bevis för att en fil kan tas bort.

⸻

GitHub Actions

Den centrala dagliga workflowen ska följa forskningskedjan:

FI
 ↓
Prices
 ↓
OMXSPI
 ↓
Feature generation / QC
 ↓
Research SCAN
 ↓
Research DEEP
 ↓
Prospective evaluation
 ↓
Artifacts

Workflowet ska använda central entrypoint och inte duplicera Research Engine-logik.

⸻

Legacy

Legacy ska inte tas bort enbart för att en fil ser gammal ut.

Rätt process är:

Ny implementation
      ↓
Tester
      ↓
Runtime verification
      ↓
Statisk referensanalys
      ↓
CI verification
      ↓
Legacy removal

Legacy får tas bort först efter verifierad ersättning.

⸻

Designprinciper

1. Hypotes före implementation.
2. YAML före specialkod när hypotesen kan uttryckas deklarativt.
3. Generisk analys före experiment-specifik implementation.
4. SCAN före dyra analyser.
5. DEEP används selektivt.
6. Frozen candidates ska vara immutabla.
7. Prospective evaluation ska vara tidsmässigt separerad.
8. OOS ska respektera tidsordningen.
9. Test leakage ska undvikas.
10. Gemensam logik ska återanvändas.
11. Resultat ska vara maskinläsbara.
12. Provenance ska bevaras.
13. Runtime-spårning ska vara generell för hela Blankdiss.
14. Legacy ska tas bort först efter verifierad ersättning.
15. Feature generation ska förbli separat från hypoteslogiken.
16. AI Lab ska orkestrera forskning, inte innehålla en samling hårdkodade experiment.

---
## `analysis/README.md`
```markdown
# Analysis – Feature Pipeline
`analysis/` innehåller den del av Blankdiss som bygger och kvalitetssäkrar det feature-dataset som används av ML och Research Engine.
Grundprincip:
```text
Rådata
  ↓
Feature generation
  ↓
Feature dataset
  ↓
Feature QC
  ↓
ML / Research Engine

Feature-lagret ansvarar för att skapa generella, återanvändbara mätvariabler.

Research Engine ansvarar för att formulera och testa hypoteser med dessa variabler.

⸻

Ansvar

Feature-pipelinen kombinerar bland annat:

* FI-data
* historiska aktiekurser
* OMXSPI
* sektortillhörighet

Feature-lagret ska vara generellt.

En konkret research-hypotes ska normalt inte kräva en specialfeature.

⸻

Feature generation

Feature-datasetet byggs från projektets upstream-data.

Feature-pipelinen ska:

1. läsa rådata
2. skapa tidsmässigt korrekta features
3. skapa targets där det behövs
4. skriva ett reproducerbart feature-dataset
5. köra Feature QC

Feature-datasetet används sedan av nedströms ML och Research Engine.

⸻

Prisfeatures

Exempel på generella prisfeatures:

price_return_5d
price_return_20d
price_return_60d
price_volatility_20d
price_distance_from_20d_high
price_distance_from_60d_high

Forward returns används som targets/utfall och ska inte förväxlas med features som är tillgängliga vid signalögonblicket.

⸻

Relativa features

Feature-lagret innehåller både absoluta och relativa mått.

Marknad:

market_return_5d
market_return_20d
market_return_60d

Sektor:

sector_return_5d
sector_return_20d
sector_return_60d

Aktien relativt marknaden:

price_return_5d_relative_market
price_return_20d_relative_market
price_return_60d_relative_market

Aktien relativt sektorn:

price_return_5d_relative_sector
price_return_20d_relative_sector
price_return_60d_relative_sector

Syftet är att kunna skilja mellan:

aktierörelse

och:

aktierörelse relativt marknad/sektor

utan att feature-pipelinen behöver känna till den specifika hypotesen.

⸻

Horisonter

Relativa momentumfeatures byggs för flera generella horisonter:

5d
20d
60d

Det gör att Research Engine kan kombinera olika horisonter utan att feature-pipelinen behöver byggas om.

⸻

Marknadsdata

Marknadsdata kommer från OMXSPI.

analysis.update_market ansvarar för att hålla historiken uppdaterad.

Marknadsdata är en upstream-datakälla.

Den är inte en del av Research Engine.

⸻

Sektordata

Sektortillhörighet används för att skapa sektorrelaterade features.

Research Engine ska inte behöva känna till hur sektorn hämtades eller beräknades.

Den ska endast behöva kunna använda färdiga features.

⸻

Feature QC

Efter feature generation körs Feature QC.

QC ska bland annat kontrollera:

* obligatoriska kolumner
* datumintegritet
* FI/prismatchning
* instrumentidentitet
* dubbletter
* numeriska värden
* forward returns
* forward-return alignment
* price leakage
* threshold-logik

QC kan ge:

PASS
WARN
FAIL

FAIL innebär att datasetet inte ska betraktas som giltigt för nedströms analys.

Warnings ska förstås innan de eventuellt hanteras.

⸻

Leakage

Historiska features får endast använda information som var tillgänglig vid observationen.

Forward returns och targets får använda framtida data eftersom de representerar utfallet som ska analyseras.

Det är därför viktigt att skilja mellan:

information tillgänglig vid signal

och:

framtida utfall

Feature QC ska kontrollera denna gräns.

⸻

Reproducerbarhet

Feature-datasetet ska kunna byggas om från rådata.

Grundflödet är:

rådata
  ↓
build features
  ↓
QC
  ↓
feature dataset

En ändring i feature-logiken ska kunna återskapas genom att bygga om datasetet.

⸻

Relation till Research Engine

Feature generation ansvarar för:

vad som kan mätas

Research Engine ansvarar för:

vad som ska undersökas

Exempel:

Feature-lagret kan skapa:

price_return_5d
price_return_20d
price_return_60d
price_return_5d_relative_market
price_return_20d_relative_market
price_return_60d_relative_market

Research Engine kan sedan formulera en YAML-spec som undersöker en specifik kombination.

Feature-lagret behöver inte känna till hypotesen.

⸻

Vad som inte hör hemma här

Research-specifik logik ska normalt inte läggas i analysis/.

Exempel:

* en specifik momentumkombination
* en specifik eventdefinition
* en specifik research-hypotes
* en viss bootstrap-analys
* en viss train/test-split
* ranking av research-resultat
* optimering av signaltrösklar

Om flera framtida analyser behöver en ny typ av feature ska feature-lagret i stället utökas generellt.

⸻

Legacy

Äldre featurevägar ska inte betraktas som primär implementation om den nya feature-pipelinen ersatt dem.

Legacy ska tas bort först efter:

1. referenskartläggning
2. runtime-verifiering
3. CI-verifiering
4. testverifiering
5. bekräftad ersättning

⸻

Designprincip

Feature-lagrets uppgift är att göra bättre frågor möjliga.

Det ska inte försöka svara på frågorna självt.

Feature layer
      ↓
mätbara variabler
      ↓
Research Engine
      ↓
hypotes

Samma feature-dataset ska kunna användas av många olika forskningsfrågor.

---
## `ml/README.md`
```markdown
# Blankdiss ML
Detta är den övergripande dokumentationen för Blankdiss ML-system.
ML-lagret innehåller gemensam dataset-, walk-forward-, research- och diagnostics-infrastruktur.
Arkitekturen är:
```text
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
│   ├── verification.py
│   ├── candidates/
│   ├── evaluation/
│   ├── specs/
│   └── custom/
│
└── diagnostics/
    ├── framework/
    └── experiments/

⸻

Dataset

ml/dataset.py ansvarar för datasetberedning.

Feature-data kommer från analysis/.

Datasetlagret ska inte innehålla forskningsspecifik hypoteslogik.

⸻

Walk-forward

ml/walk_forward.py innehåller gemensam tidsmässig uppdelning.

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

Testdata får inte användas för modellval eller parameteroptimering.

⸻

Research Engine

ml/research/ är Blankdiss generiska forskningsmotor.

Den aktiva modellen är:

idé
 ↓
YAML
 ↓
ResearchSpec
 ↓
ResearchSession
 ↓
shared cache
 ↓
Research Engine
 ↓
resultat

Research Engine ska minimera behovet av ny Python-kod för vanliga forskningsfrågor.

⸻

ResearchSpec

spec.py definierar YAML-formatet.

En spec beskriver bland annat:

* forskningsfråga
* signaler
* targets
* analysis type
* mode
* windows
* splits
* metadata

En konkret hypotes ska normalt uttryckas deklarativt.

⸻

ResearchSession

session.py skapar en gemensam ResearchSession.

Flera specs kan använda samma session:

spec A ─┐
spec B ─┼──→ ResearchSession
spec C ─┘
              ↓
        load feature data
              ↓
        build shared cache

Samma upstream-data ska inte laddas och beräknas om separat för varje hypotes när det kan undvikas.

⸻

Cache

cache.py innehåller gemensamma beräkningar.

Exempel:

* signaler
* targets
* returns
* masks
* windows

Cachelagret är en del av infrastrukturen och ska inte innehålla enskild hypoteslogik.

⸻

Signals

signals.py definierar standardiserade research-signaler.

Exempel:

* short interest level
* short interest change
* short interest acceleration
* price momentum
* volatility
* distance from highs
* relativa momentumfeatures

Research Engine ska kunna kombinera dessa utan att en ny Python-modul behöver skapas för varje hypotes.

⸻

Engine

engine.py kör generiska analysis types.

Exempel:

* tail
* interaction
* incremental_model
* regime_comparison
* multi_regime_comparison
* nested_regime_comparison

En analysis type beskriver en generell analysform.

Den ska inte beskriva en specifik hypotes.

⸻

Runner

runner.py orkestrerar research-körningen:

find specs
   ↓
load specs
   ↓
validate
   ↓
build session
   ↓
run specs
   ↓
write results
   ↓
write manifest

Den återanvändbara funktionen är:

run_research(...)

Runnern ska kunna användas både av CLI och central entrypoint.

⸻

Central entrypoint

Den centrala research-entrypointen är:

ml/research/entrypoint.py

CLI:

python -m ml.research.entrypoint freeze ...
python -m ml.research.entrypoint scan ...
python -m ml.research.entrypoint deep ...
python -m ml.research.entrypoint evaluate ...
python -m ml.research.entrypoint pipeline ...

Workflowet ska använda entrypointen i stället för att känna till interna implementationer.

⸻

SCAN

SCAN är den breda forskningsfasen.

Exempel:

signals
targets
tail fractions
windows
interactions

SCAN svarar primärt på:

Finns det något här som motiverar fördjupning?

⸻

DEEP

DEEP används för selektiv fördjupning.

Exempel:

* bootstrap
* confidence intervals
* robusthet
* placebo
* alternativa cutoffs
* kontrollanalyser
* längre perioder
* ytterligare stratifiering

DEEP ska inte användas som en dold parameteroptimerare.

⸻

Candidates

Candidate lifecycle:

Research
   ↓
Controlled hypothesis
   ↓
Candidate
   ↓
Freeze
   ↓
Prospective evaluation

Efter freeze ska candidate-definitionen vara immutable.

⸻

Prospective evaluation

Evaluation sker på data efter candidate cutoff.

candidate
    +
future data
    ↓
evaluation

Evaluation får inte ändra candidate-definitionen efter resultatet.

⸻

Verification

Gemensam verification ska kontrollera bland annat:

* temporal leakage
* feature/target leakage
* future-data access
* missing data
* candidate mutation
* evaluation contamination
* duplicate runs
* distributionsförändringar
* provenance

Verification ska vara återanvändbar mellan forskningsfrågor.

⸻

Diagnostics

Diagnostics används när analysen är verkligt specialiserad.

Beslutsregel:

Ny hypotes
    ↓
Kan Research Engine uttrycka den?
    │
    ├── JA → YAML
    │
    └── NEJ
         ↓
    Är den saknade funktionen generell?
         │
         ├── JA → Engine + YAML
         │
         └── NEJ → Diagnostics

Diagnostics ska alltså inte användas som en genväg runt Research Engine.

⸻

Resultat

Research-resultat skrivs under:

data/processed/ml/research/spec_runs/

Typisk struktur:

spec_runs/
└── <timestamp>/
    ├── manifest.json
    ├── <spec-id>.json
    └── ...

⸻

Designprinciper

1. Hypoteser ska vara deklarativa när möjligt.
2. Research Engine ska vara generell.
3. Runner ska vara återanvändbar.
4. Entry-point ska vara central.
5. Research-resultat ska vara maskinläsbara.
6. Frozen candidates ska vara immutabla.
7. Evaluation ska vara prospektiv.
8. OOS ska respektera tidsordningen.
9. Test leakage ska undvikas.
10. Generella problem ska lösas generellt.
11. Spec-specifika workarounds ska undvikas.
12. Duplicerad forskningsinfrastruktur ska undvikas.
13. Diagnostics ska reserveras för verkligt specialiserad forskning.
14. Legacy ska tas bort först efter verifierad ersättning.

---
## `ml/research/README.md`
```markdown
# Blankdiss Research Engine
`ml/research/` är Blankdiss generiska forskningsmotor.
Målet är att en konkret forskningshypotes normalt ska kunna beskrivas som YAML och köras genom samma generiska infrastruktur.
Grundflöde:
```text
Forskningsfråga
      ↓
YAML
      ↓
ResearchSpec
      ↓
ResearchSession
      ↓
Research Engine
      ↓
maskinläsbart resultat

⸻

ResearchSpec

spec.py definierar YAML-kontraktet.

En spec beskriver bland annat:

* id
* question
* mode
* signals
* targets
* analysis
* windows
* splits
* metadata

En konkret forskningshypotes ska normalt vara deklarativ.

⸻

ResearchSession

session.py bygger en gemensam session.

spec A ─┐
spec B ─┼──→ session
spec C ─┘
           ↓
      feature data
           ↓
      shared cache

Feature-datasetet ska laddas en gång när flera specs delar samma research-körning.

⸻

Cache

cache.py innehåller gemensamma beräkningar.

Exempel:

* signals
* targets
* returns
* masks
* windows

Syftet är att undvika att samma data behöver beräknas separat för varje hypotes.

⸻

Engine

engine.py innehåller generisk analyslogik.

Exempel på analysis types:

tail
interaction
incremental_model
regime_comparison
multi_regime_comparison
nested_regime_comparison

En analysis type beskriver en generell analysform.

Den konkreta hypotesen ska därefter beskrivas i YAML.

⸻

Runner

runner.py orkestrerar:

find specs
   ↓
load specs
   ↓
validate
   ↓
build session
   ↓
run specs
   ↓
write results
   ↓
write manifest

Den återanvändbara funktionen är:

run_research(...)

Runnern ska vara det gemensamma körlagret för research, inte ett alternativt system vid sidan av entrypointen.

⸻

Central entrypoint

Den centrala entrypointen är:

ml/research/entrypoint.py

CLI:

python -m ml.research.entrypoint freeze ...
python -m ml.research.entrypoint scan ...
python -m ml.research.entrypoint deep ...
python -m ml.research.entrypoint evaluate ...
python -m ml.research.entrypoint pipeline ...

Extern orchestration ska använda entrypointen.

Interna implementationer ska inte behöva exponeras för GitHub Actions.

⸻

SCAN

SCAN är bred screening.

Exempel:

signals
targets
tail fractions
windows
interactions

SCAN ska vara tillräckligt billig för att kunna användas för discovery.

Syftet är:

Finns det en struktur som motiverar fördjupning?

⸻

DEEP

DEEP används för selektiv fördjupning.

Exempel:

* bootstrap
* confidence intervals
* robusthetskontroller
* placebo
* alternativa cutoffs
* kontrollanalyser
* längre perioder
* ytterligare stratifiering

DEEP är inte automatiskt samma sak som parameteroptimering.

⸻

Candidate lifecycle

Research Engine ingår i en större candidate lifecycle:

Research
   ↓
Controlled hypothesis
   ↓
Candidate
   ↓
Freeze
   ↓
Prospective evaluation

Efter freeze ska candidate-definitionen vara immutable.

⸻

Prospective evaluation

Evaluation sker på data efter candidate cutoff.

candidate
    +
future data
    ↓
evaluation

Evaluation får inte ändra candidate-definitionen efter resultatet.

⸻

Verification

Research Engine ska kunna användas tillsammans med gemensam verification.

Kontroller omfattar bland annat:

* temporal leakage
* feature/target leakage
* future-data access
* missing data
* candidate mutation
* evaluation contamination
* duplicate runs
* distributionsförändringar
* provenance

⸻

Research kontra Diagnostics

Grundregel:

Generisk analys
    → Research Engine + YAML
Specialiserad analys
    → Diagnostics

Om en analys saknas ska man först fråga:

Är analysformen generell?

Om svaret är ja ska Engine utökas generellt.

En specifik hypotes ska inte få en egen specialimplementation bara för att den är ny.

⸻

När ny Python ska skrivas

Innan ny kod skrivs:

1. Kontrollera befintliga analysis types.
2. Kontrollera signals.
3. Kontrollera targets.
4. Kontrollera cache.
5. Kontrollera YAML-kontraktet.
6. Kontrollera om Research Engine redan kan uttrycka hypotesen.

Om hypotesen kan uttryckas med YAML ska ingen ny Python skrivas.

Om generell funktionalitet saknas ska Engine utökas.

Endast verkligt unik speciallogik ska gå till Diagnostics/custom.

⸻

Resultat

Research-resultat skrivs under:

data/processed/ml/research/spec_runs/

Typisk struktur:

spec_runs/
└── <timestamp>/
    ├── manifest.json
    ├── <spec-id>.json
    └── ...

Research Engine-resultatet innehåller bland annat:

spec_id
question
mode
analysis
results

spec_id är den identifierare som kopplar resultatet till den körda ResearchSpec.

⸻

Designprinciper

1. YAML är forskningskontraktet.
2. Engine är generell.
3. Runner är gemensam.
4. Entry-point är central.
5. SCAN är discovery.
6. DEEP är selektiv fördjupning.
7. Frozen candidates är immutabla.
8. Evaluation är prospektiv.
9. OOS respekterar tidsordningen.
10. Verification är gemensam.
11. Spec-specifika workarounds ska undvikas.
12. Resultat ska vara maskinläsbara.
13. Provenance ska bevaras.
14. Legacy ska tas bort först efter verifierad ersättning.

---
## `ml/diagnostics/README.md`
```markdown
# Blankdiss Diagnostics
Diagnostics används för verkligt specialiserade analyser som inte lämpar sig för den generiska Research Engine.
Diagnostics är inte en genväg runt Research Engine.
Grundregel:
```text
Generisk analys
    → Research Engine + YAML
Specialiserad analys
    → Diagnostics

⸻

När ska Diagnostics användas?

Innan ett nytt experiment skapas:

1. Kan frågan uttryckas med Research Engine?
2. Om inte, är den saknade funktionen generell?
3. Kan samma analysform användas av framtida hypoteser?

Om generell:

Research Engine
      ↓
YAML

Om verkligt unik:

Diagnostics

⸻

Arkitektur

ml/diagnostics/
│
├── framework/
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

⸻

DiagnosticExperiment

Nya diagnostics ska normalt använda den gemensamma experimentinfrastrukturen.

Experimentet ska beskriva den specifika forskningsfrågan.

Gemensam logik ska ligga i frameworket.

⸻

ExperimentContext

Runnern skapar ett ExperimentContext för varje walk-forward-window.

Context kan innehålla:

TRAIN
VALIDATION
PRETEST
TEST

Experimentet ska inte duplicera den gemensamma tidsuppdelningen.

⸻

ExperimentResult

Experiment ska returnera strukturerade resultat.

Exempel:

return {
    "analysis": analysis_table,
    "bootstrap": bootstrap_table,
}

Reporting ska vara gemensam när det är möjligt.

⸻

Walk-forward

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

Testperioden får inte användas för:

* modellval
* threshold selection
* feature selection
* parameteroptimering

⸻

Pre-test populationer

När populationer definieras ska klassificeringen använda information som fanns före testperioden.

Rätt:

PRETEST
   ↓
threshold
   ↓
TEST classification

Inte:

TEST outcomes
   ↓
optimera threshold

Detta gäller även deskriptiva diagnostics.

⸻

2D-interaktioner

Två faktorer kan exempelvis bilda:

                 LOW B      HIGH B
LOW A               A           B
HIGH A              C           D

Interaktionen kan sedan analyseras som skillnaden mellan den dubbla effekten och de individuella effekterna.

Syftet är att skilja faktisk interaktion från att den högsta gruppen bara råkar ha högst event rate.

⸻

Event-risk

Event-risk kan användas som en mellanliggande dimension.

Exempel:

abs(forward_return_5d) >= 10 %

En event-riskmodell ska tränas och väljas med tidsordning:

TRAIN
   ↓
VALIDATION
   ↓
REFIT
   ↓
OOS SCORE

Därefter kan diagnostics analysera exempelvis:

event-risk tail
      ×
short-interest level

⸻

Experimentfiler

Experimentfilerna ska normalt vara små.

Undvik att lägga följande direkt i ett experiment:

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

Process:

1. Formulera forskningsfrågan.
2. Kontrollera Research Engine.
3. Kontrollera om frågan kan uttryckas deklarativt.
4. Kontrollera om eventuell saknad funktion är generell.
5. Om generell: utöka Research Engine.
6. Om unik: skapa Diagnostics-experiment.
7. Återanvänd framework helpers.
8. Kör walk-forward.
9. Kontrollera OOS.
10. Använd gemensam verification där det är relevant.

Ett nytt Diagnostics-experiment ska alltså inte skapas bara för att Research Engine ännu saknar en generell analysis type.

⸻

Designmål

Målet är:

Ny specialiserad hypotes
        ↓
Liten experimentklass
        ↓
Gemensamt framework
        ↓
Walk-forward
        ↓
Verification
        ↓
OOS
        ↓
Standardiserat resultat

Samtidigt ska generiska analyser stanna i Research Engine.

---
## `ml/ai-lab/README.md`
```markdown
# Blankdiss AI Lab
AI Lab är Blankdiss orchestration-lager för kontrollerad och iterativ forskning.
Syftet är att låta tidigare experiment ge information om vilket forskningssteg som ska tas härnäst, samtidigt som forskningen förblir:
- reproducerbar
- spårbar
- deklarativ
- skyddad mot data leakage
- skyddad mot resultatjakt
AI Lab är ett orchestration-lager.
Det är inte en separat analysmotor.
---
# Grundprincip
AI Lab optimerar:
```text
idé → information

inte:

idé → mer kod

Ett negativt resultat är ett giltigt forskningsresultat.

AI Lab ska inte vrida parametrar tills ett positivt resultat uppstår.

⸻

Arkitektur

GitHub Actions
      │
      ▼
    AI Lab
      │
      ├── discovery
      ├── orchestration
      ├── state
      └── adaptive research
              │
              ▼
       Research Engine
              │
              ▼
         YAML specs
              │
              ▼
       persisted results

Workflowet ska starta forskningskedjan.

Workflowet ska inte innehålla enskilda forskningshypoteser.

⸻

YAML är forskningskontraktet

Forskningsspecifikationerna finns under:

ml/research/specs/*.yaml

En spec beskriver exempelvis:

* id
* forskningsfråga
* signaler
* targets
* analysis type
* bins/trösklar
* windows
* splits
* metadata

Normalfallet för en ny hypotes är:

ny hypotes
   ↓
ny YAML-spec

inte:

ny hypotes
   ↓
ny Python-specialgren

⸻

Kontrollerade forskningsspecar

En spec som ska köras av AI Lab ska uttryckligen opta in.

Exempel:

metadata:
  ai_lab_execution:
    enabled: true
    mode: once

AI Lab ska upptäcka sådana specs generellt.

Det ska inte krävas att ett specifikt spec_id läggs in i Python-kod.

⸻

Once-semantik

once innebär att specen körs en gång och därefter betraktas som genomförd baserat på persisterad state/resultat.

Resultat finns under:

data/processed/ml/research/spec_runs/

Om processen avbryts innan resultatet persisterats ska specen kunna köras igen.

Python-processens minne är alltså inte den enda sanningskällan för om en once-spec är genomförd.

⸻

Delad research-session

AI Lab ska kunna samla requirements från flera forskningsspecar och bygga en gemensam ResearchSession.

adaptive research ─────┐
                       │
controlled specs ─────┼──→ ResearchSession
                       │
                       ▼
                 load features
                       │
                       ▼
                 build cache

Feature-data och gemensamma cache-komponenter ska inte byggas om för varje spec när de kan delas.

⸻

Adaptiv forskning

Den adaptiva forskningen följer principen:

OBSERVE
   ↓
ANALYZE
   ↓
ADAPT
   ↓
EXPERIMENT
   ↓
ANALYZE
   │
   └────→ OBSERVE

Parameterutrymmet ska vara deklarerat.

AI Lab ska inte fritt uppfinna nya parametrar eller ny Python-kod baserat på ett attraktivt resultat.

⸻

Kontrollerad forskning kontra adaptiv forskning

De två mekanismerna har olika syften.

Kontrollerad spec:

fördefinierad forskningsfråga
        ↓
fördefinierad YAML
        ↓
exakt körning

Adaptiv forskning:

observerat resultat
        ↓
fördefinierad urvalslogik
        ↓
nästa kandidat inom deklarerat utrymme

De får inte blandas ihop.

⸻

Research Engine

AI Lab implementerar inte analysalgoritmerna.

Research Engine finns under:

ml/research/

Research Engine ansvarar för:

* signals
* targets
* analysis types
* shared cache
* research execution
* standardiserade resultat

AI Lab ansvarar för:

* orchestration
* discovery
* state
* adaptive loop

⸻

Resultat och spårbarhet

Research-resultat skrivs under:

data/processed/ml/research/spec_runs/

Typisk struktur:

<run>/
├── <spec-id>.json
└── manifest.json

Research Engine-resultatet innehåller bland annat:

spec_id
question
mode
analysis
results

Read-back och verifiering ska använda spec_id för att säkerställa att resultatet hör till rätt ResearchSpec.

⸻

Data leakage

AI Lab ska upprätthålla:

* test-data får inte användas för parameterurval
* låsta confirmation-specar får inte modifieras
* resultatbaserad kandidat-rankning får inte ersätta fördefinierad urvalslogik
* experiment ska vara reproducerbara
* genererad forskning ska vara spårbar
* negativa resultat ska få förbli negativa

⸻

Frozen candidates

När en hypotes går vidare till candidate-lagret:

Research
   ↓
Controlled hypothesis
   ↓
Frozen candidate
   ↓
Prospective evaluation

Efter freeze får AI Lab inte ändra candidate-definitionen.

En ändrad parameterkombination är en ny candidate.

⸻

Vad behöver ändras när en ny hypotes läggs till?

Normalfallet:

1. Skapa YAML-spec.
2. Deklarera forskningsfrågan.
3. Deklarera signaler och targets.
4. Välj befintlig analysis type.
5. Opta vid behov in specen till AI Lab.
6. Kör samma pipeline.

Inte:

1. Ändra workflow.
2. Ändra AI Lab-koden för spec-id.
3. Skriva ny experimentkod.
4. Skriva ny analysmotor.
5. Skapa en specialgren.

⸻

Arkitekturregel

Efter ett problem ska första frågan vara:

Är detta ett YAML-problem,
ett orchestration-problem
eller ett generellt Engine-problem?

Om discovery inte hittar en spec:

1. kontrollera metadata
2. kontrollera enabled
3. kontrollera mode

Om specen upptäcks men inte kan köras:

1. kontrollera analysis.type
2. kontrollera YAML-kontraktet
3. kontrollera Research Engine

Generella problem ska lösas generellt.

⸻

Designmål

AI Lab ska vara:

orchestration
+ discovery
+ state
+ adaptive research

Research Engine ska vara:

generisk analys
+ execution
+ resultat

YAML ska vara:

forskningskontrakt

Den avsedda modellen är:

GitHub Actions
      ↓
    AI Lab
      ↓
 YAML specs
      ↓
Research Engine
      ↓
persisted artifacts

Målet är att Blankdiss ska kunna växa med fler forskningsfrågor utan att AI Lab eller workflowet behöver byggas om för varje ny hypotes.

---
## `data/ai_lab/results/README.md`
```markdown
# AI Lab Results
Den här katalogen innehåller resultat från AI Lab-specifika experiment.
Varje experiment får normalt en egen katalog:
```text
data/ai_lab/results/<experiment-id>/

Typisk struktur:

<experiment-id>/
├── results.json
└── report.md

⸻

Resultat kontra research artifacts

AI Lab-resultat är output från AI Lab.

Generella Research Engine-resultat skrivs under:

data/processed/ml/research/spec_runs/

Det är viktigt att skilja på:

Research spec
    ↓
Research Engine
    ↓
research artifact

och:

AI Lab
    ↓
orchestration / experiment
    ↓
AI Lab result

⸻

Reproducerbarhet

Resultat bör kunna kopplas till:

* experiment/spec
* version
* relevant data
* run metadata
* Research Engine-version när relevant

Maskinläsbara resultat ska prioriteras.

⸻

Frozen candidates

AI Lab-resultat får inte användas för att mutera en frozen candidate.

Rätt flöde:

Research
   ↓
result
   ↓
analysis
   ↓
new hypothesis

Om ett resultat leder till en ny parameterkombination ska detta behandlas som en ny hypotes eller candidate.

⸻

GitHub Actions

GitHub Actions kan skapa research artifacts.

Workflowet ska inte innehålla experiment-specifik logik.

Resultat är output från forskningskedjan.

Konfigurationen för forskningen ska i första hand finnas i YAML-specifikationerna och den gemensamma Research Engine.
