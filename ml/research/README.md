Research Engine

ml/research/ är Blankdiss generella forskningsmotor.

Den används för att definiera, köra, cacha, rapportera och verifiera forskningsförsök på feature-datasetet utan att varje ny hypotes behöver implementeras som separat analyskod.

Roll i arkitekturen

analysis/
    │
    ▼
feature dataset
    │
    ▼
Research Engine
    │
    ├── specs
    ├── scan
    ├── deep
    ├── analysis
    ├── candidates
    ├── evaluation
    └── verification

Research Engine ligger mellan den gemensamma datagrunden och kandidat-/valideringsprocessen.

Grundprincip

Research Engine ska skilja mellan:

* vad som ska undersökas
* hur experimentet genomförs
* vilket resultat experimentet gav
* om resultatet är tillräckligt intressant för att bli kandidat
* om kandidaten senare klarar prospektiv utvärdering

Det gör forskningsprocessen mer reproducerbar och minskar risken för att enskilda experiment får egen, duplicerad logik.

Centrala komponenter

ml/research/
├── spec.py
├── session.py
├── cache.py
├── signals.py
├── engine.py
├── runner.py
├── reporting.py
├── verification.py
├── candidates/
├── evaluation/
└── specs/

spec.py

Definierar strukturen för research specs.

En spec beskriver ett experiment på ett deklarativt sätt.

session.py

Hanterar kontexten för en research-körning.

cache.py

Hanterar återanvändning av beräknade resultat där samma analys kan återanvändas utan att köras om i onödan.

signals.py

Innehåller signalrelaterad logik som används av forskningsmotorn.

engine.py

Är den centrala motorn som genomför research-analyser.

runner.py

Ansvarar för att driva research-körningar.

reporting.py

Sammanställer och skriver forskningsresultat.

verification.py

Innehåller verifieringsrelaterad logik för att kontrollera forskningsresultat och kandidatflöden.

Research specs

Research specs gör experiment deklarativa.

I stället för att skapa en ny Python-implementation för varje hypotes kan forskaren beskriva experimentets parametrar i en spec.

Specifikationen kan bland annat ange:

* dataset
* target
* features eller feature-grupper
* analys
* tidsperiod
* parametrar
* körläge
* begränsningar

Exakt schema styrs av implementationen i spec.py och de aktuella specs som finns i:

ml/research/specs/

Körlägen

Research Engine har två centrala körlägen:

scan
deep

Scan

scan används för bredare sökningar.

Syftet är att snabbt kunna undersöka många potentiella samband och identifiera områden som förtjänar djupare analys.

Deep

deep används när ett område eller en signal redan är intressant och behöver undersökas mer omfattande.

Det kan exempelvis innebära fler analyser, fler parametrar eller djupare uppdelningar av resultatet.

Entrypoint

Research Engine kan köras via:

ml.research.entrypoint

Exempel:

python -m ml.research.entrypoint scan
python -m ml.research.entrypoint deep

Kandidat- och evaluation-flöden kan exempelvis startas med:

python -m ml.research.entrypoint freeze <candidate.yaml>
python -m ml.research.entrypoint evaluate <candidate.yaml> <evaluation.yaml>

Det finns även ett pipeline-läge:

python -m ml.research.entrypoint pipeline

Research results

Research Engine-resultat skrivs normalt till:

data/processed/ml/research/spec_runs/

En körning har ett eget run-directory.

En typisk struktur är:

spec_runs/
└── <timestamp>/
    ├── manifest.json
    └── <spec-resultat>.json

Manifestet gör det möjligt att identifiera vilka specs och förutsättningar som hör till körningen.

Session och reproducerbarhet

En research session representerar kontexten för ett experiment eller en grupp experiment.

Det är viktigt att en research-körning kan kopplas till:

* vilken spec som användes
* vilket dataset som användes
* vilken tidsperiod som användes
* vilka parametrar som användes
* vilket run-id som skapades
* vilka resultat som genererades

Detta är särskilt viktigt när ett resultat senare ska bli kandidat.

Cache

Research Engine använder cache för att undvika att samma dyra beräkning behöver göras flera gånger när indata och analys är oförändrade.

Cache får dock inte förändra den logiska betydelsen av ett experiment.

Ett cache-hit ska motsvara samma beräkning som om experimentet hade körts från början.

Candidates

Research Engine har ett separat kandidatflöde:

ml/research/candidates/

En kandidat är en fryst representation av en forskningshypotes som bedöms vara tillräckligt konkret för att testas vidare.

Kandidaten ska kunna beskriva exakt vad som ska utvärderas.

Det är viktigt att skilja mellan:

Research result
      │
      ▼
Candidate
      │
      ▼
Evaluation
      │
      ▼
Verification

Ett historiskt resultat är alltså inte automatiskt en kandidat, och en kandidat är inte automatiskt verifierad.

Prospective evaluation

Prospektiv utvärdering finns under:

ml/research/evaluation/

Syftet är att testa frysta kandidater på data som inte användes för att skapa hypotesen.

Exempel:

python -m ml.research.entrypoint freeze <candidate.yaml>
python -m ml.research.entrypoint evaluate <candidate.yaml> <evaluation.yaml>

Resultat skrivs normalt till:

data/processed/ml/research/evaluation/

En evaluation-körning ska vara tydligt kopplad till den kandidat som utvärderades.

Verification

Verifiering är det steg där forskningsresultat och kandidater kontrolleras mot de krav som gäller för nästa steg.

Verifiering kan bland annat hjälpa till att upptäcka:

* felaktiga antaganden
* bristande reproducerbarhet
* tidsmässig läckage
* problem i kandidatdefinitionen
* skillnader mellan historiskt och prospektivt resultat

Verifiering är därför en separat del av forskningsprocessen.

Relation till AI Lab

AI Lab finns under:

ml/ai-lab/

AI Lab kan använda Research Engine som sin experimentmotor.

AI Lab
   │
   │ planerar
   ▼
Research spec
   │
   ▼
Research Engine
   │
   ▼
Resultat
   │
   ▼
AI Lab

Research Engine behöver alltså inte känna till hela AI Labs forskningsstrategi.

Relation till Diagnostics

ml/diagnostics/ innehåller specialiserade analyser.

Research Engine bör användas för generella, återanvändbara experiment.

Diagnostics används när en specifik frågeställning kräver särskild analyslogik.

Om en diagnostic senare visar sig vara generell kan funktionaliteten abstraheras till Research Engine.

Dataflöde

Research Engine läser den feature-grund som byggs av analysis/.

analysis/build_features.py
          │
          ▼
data/processed/analysis/
          │
          ▼
     Research Engine
          │
          ├── scan
          ├── deep
          └── other research analyses
                  │
                  ▼
             spec_runs
                  │
                  ▼
              candidates
                  │
                  ▼
             evaluation
                  │
                  ▼
             verification

Research Engine ska inte själv vara ansvarig för att bygga om den centrala feature-grunden.

Tidsmässig separation

Research Engine måste hantera tidsdimensionen korrekt.

När en signal undersöks måste det vara möjligt att skilja mellan:

* information som fanns tillgänglig vid observationstidpunkten
* framtida information
* historisk träningsperiod
* testperiod
* eventuell OOS-period

Detta är centralt för att undvika look-ahead bias.

Forskningslivscykel

En typisk Blankdiss-forskningsprocess kan beskrivas som:

1. Feature dataset
       │
       ▼
2. Scan
       │
       ▼
3. Intressant observation
       │
       ▼
4. Deep analysis
       │
       ▼
5. Hypotes
       │
       ▼
6. Candidate freeze
       │
       ▼
7. Prospective evaluation
       │
       ▼
8. Verification

Det är denna separation som gör det möjligt att skilja mellan upptäckt och faktisk validering.

Grundprincip

Research Engine ska vara Blankdiss generella experimentmotor.

När en ny forskningsidé kan uttryckas som:

“Kör denna analys på dessa features och targets under dessa förutsättningar”

bör den i första hand implementeras som en Research Engine-spec eller som en generell utökning av Research Engine.

En ny separat experimentimplementation bör först skapas när problemet faktiskt kräver specialiserad logik.
