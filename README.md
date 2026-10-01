Blankdiss

Blankdiss är ett system för att analysera blankningsdata, marknadsdata och aktiekurser, bygga ett gemensamt feature-dataset och använda det för statistisk analys, maskininlärning och systematisk forskning.

Projektet är uppbyggt så att data → features → analys/forskning → kandidater → prospektiv utvärdering → verifiering hålls separerade. Det gör att nya hypoteser kan undersökas utan att själva datagrunden eller produktionslogiken behöver byggas om.

Övergripande arkitektur

FI / priser / marknad / sektor
              │
              ▼
          analysis/
              │
              ▼
    data/processed/analysis/
       feature-dataset
              │
       ┌──────┴──────┐
       ▼             ▼
      ml/          analysis/
       │
       ├── conventional ML
       ├── research/
       │      │
       │      ├── specs
       │      ├── scan/deep
       │      ├── candidates
       │      └── evaluation
       │
       ├── diagnostics/
       │
       └── ai-lab/
              │
              ▼
       forskningsresultat
              │
              ▼
       prospektiv utvärdering
              │
              ▼
          verification
              │
              ▼
            web/
              │
              ▼
           pages/

Data och feature-byggande

analysis/ ansvarar för att bygga det centrala feature-datasetet.

Data kommer bland annat från:

* Finansinspektionens blankningsdata
* aktiekurser
* marknadsdata
* sektor-/branschdata
* härledda pris- och volatilitetsegenskaper

Den huvudsakliga feature-byggaren är:

analysis/build_features.py

Resultatet skrivs till:

data/processed/analysis/

Datasetet är chunkat i JSONL-filer och kompletteras med metadata och kvalitetskontroller.

Feature-byggandet är en separat fas från själva forskningen. Research Engine ska inte bygga om feature-datasetet för varje hypotes.

Machine Learning

ml/ innehåller maskininlärning och forskningsrelaterad analys.

Det finns tre huvudsakliga delar:

Conventional ML

Traditionella modeller och ML-relaterad analys som arbetar på det byggda feature-datasetet.

Research Engine

ml/research/ är Blankdiss generella forskningsmotor.

Den är deklarativ och bygger på research-specifikationer snarare än att varje experiment ska implementeras som egen Python-kod.

Centrala delar är bland annat:

ml/research/spec.py
ml/research/session.py
ml/research/cache.py
ml/research/signals.py
ml/research/engine.py
ml/research/runner.py
ml/research/reporting.py
ml/research/verification.py

Research Engine stödjer bland annat:

* scan
* deep
* signalanalys
* featureanalys
* targetanalys
* olika experimenttyper
* caching
* rapportering
* kandidatgenerering
* prospektiv utvärdering
* verifiering

Exempel:

python -m ml.research.entrypoint scan
python -m ml.research.entrypoint deep
python -m ml.research.entrypoint freeze <candidate.yaml>
python -m ml.research.entrypoint evaluate <candidate.yaml> <evaluation.yaml>
python -m ml.research.entrypoint pipeline

Research-resultat skrivs normalt till:

data/processed/ml/research/spec_runs/

Varje körning får ett eget run-directory med manifest och resultat per spec.

Candidates och prospective evaluation

En viktig princip i Blankdiss är att skilja mellan:

1. en intressant upptäckt
2. en fryst kandidat
3. en prospektiv utvärdering
4. verifierad evidens

Kandidater hanteras under:

ml/research/candidates/

Prospektiv utvärdering hanteras under:

ml/research/evaluation/

Resultaten hamnar under:

data/processed/ml/research/evaluation/

Det gör att en signal inte automatiskt betraktas som validerad bara för att den fungerar i ett historiskt experiment.

AI Lab

ml/ai-lab/ innehåller den mer kontrollerade och adaptiva forskningsorkestreringen.

AI Lab kan:

* hålla forskningsstate
* planera experiment
* välja nästa experiment
* hantera kandidater
* styra forskningsloopar
* använda Research Engine som underliggande analysmotor

AI Lab ska alltså inte duplicera Research Engines generella analyslogik. Research Engine är den generella experimentmotorn medan AI Lab står för högre nivåns forskningsstyrning.

AI Lab-relaterade resultat och artefakter kan även förekomma under:

data/ai_lab/results/

Diagnostics

ml/diagnostics/ innehåller specialiserade diagnostiska analyser.

Diagnostics används när en analys inte naturligt passar in i den generella Research Engine-modellen.

Strukturen är bland annat:

ml/diagnostics/framework/
ml/diagnostics/experiments/

Research Engine används för återanvändbara generella experiment.

Diagnostics används för mer specifika undersökningar, exempelvis när man behöver analysera ett särskilt fenomen i datasetet eller kontrollera en specifik metodfråga.

Web

web/ är Blankdiss presentationslager.

Det genererar en statisk webbplats som publiceras under:

pages/

Bygget startas via:

python web_build.py

och använder:

web/build/page_builder.py

Web-lagret läser färdiga analys- och ML-resultat och presenterar dem. Det ska inte vara platsen där forskningslogik eller feature-beräkningar implementeras.

Webbplatsen innehåller bland annat:

* startsida
* köpläge
* bedömning
* dataanalys

Källan finns under:

web/templates/

och bygglogiken under:

web/build/

Scripts

scripts/ innehåller tunna shell-wrapperar för återkommande kommandon.

Exempel:

build_features.sh
fetch_fi.sh
fetch_prices.sh
git_push_with_retry.sh
update_market.sh

De ska framför allt göra återkommande körningar enklare. Den egentliga affärs- och analyslogiken ska ligga i Python-koden.

Testning

Testerna ligger nära den kod de verifierar och används för att skydda framför allt:

* dataflöden
* feature-byggande
* Research Engine
* kandidathantering
* utvärdering
* rapportering
* webbyggande

När arkitekturen förändras ska tester uppdateras tillsammans med koden.

Grundprinciper

Blankdiss bygger på några centrala principer:

Separera data från forskning

Feature-datasetet ska vara en stabil grund som flera olika forskningsmetoder kan använda.

Separera upptäckt från validering

En historiskt intressant signal är inte automatiskt en validerad signal.

Undvik duplicerad forskningslogik

Generiska experiment ska implementeras i Research Engine i stället för att varje nytt experiment får en egen fristående implementation.

Behåll reproducerbarhet

Research runs, specs, kandidater och utvärderingar ska kunna identifieras och återskapas.

OOS och prospektiv testning

Historiska resultat används för hypotesgenerering. Frysta kandidater ska kunna testas på data som inte användes för att skapa hypotesen.

Presentation är ett separat lager

Webbplatsen presenterar resultat från analys- och ML-lagren. Den ska inte innehålla den centrala forskningslogiken.

Projektstruktur

Blankdiss/
│
├── analysis/
│   └── feature pipeline
│
├── data/
│   ├── analysis/
│   ├── events/
│   ├── processed/
│   │   ├── analysis/
│   │   └── ml/
│   └── ai_lab/
│
├── ml/
│   ├── ai-lab/
│   ├── diagnostics/
│   └── research/
│
├── scripts/
│
├── tests/
│
├── web/
│   ├── build/
│   └── templates/
│
├── pages/
│
└── web_build.py

Blankdiss är därmed inte ett enskilt ML-script utan en pipeline där datainsamling, feature engineering, forskning, kandidatvalidering och presentation är separerade delar av samma system.
