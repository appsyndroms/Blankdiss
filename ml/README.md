Machine Learning

ml/ innehåller Blankdiss maskininlärning, forskningsmotorer, diagnostik och AI-baserad forskningsorkestrering.

ML-lagret använder i huvudsak feature-datasetet som byggs av analysis/.

Arkitektur

analysis/
    │
    ▼
data/processed/analysis/
    │
    ├──────────────┬──────────────┬──────────────┐
    ▼              ▼              ▼              ▼
conventional ML  research/    diagnostics/    ai-lab/
                    │
                    ▼
               candidates
                    │
                    ▼
               evaluation
                    │
                    ▼
               verification

De olika delarna har olika ansvar och ska inte duplicera varandra.

Conventional ML

Den vanliga ML-koden används för modeller och analyser som inte behöver Research Engines deklarativa forskningsflöde.

Den kan exempelvis användas för:

* modellträning
* prediktion
* feature importance
* modellutvärdering
* jämförelser mellan modeller

Den gemensamma datagrunden kommer från:

data/processed/analysis/

Research Engine

ml/research/ är Blankdiss generella forskningsmotor.

Research Engine är byggd för att göra experiment reproducerbara och deklarativa.

Centrala komponenter är bland annat:

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

Research Engine kan köra olika typer av forskning utan att varje ny hypotes behöver få en separat implementation.

Research modes

De huvudsakliga körlägena är:

scan
deep

scan används för bredare och snabbare undersökningar.

deep används för mer omfattande analys av intressanta områden.

Exempel:

python -m ml.research.entrypoint scan
python -m ml.research.entrypoint deep

Research specs

Experiment beskrivs med Research Engine-specifikationer.

Specifikationen anger bland annat vad som ska analyseras, vilka targets som ska användas och vilka parametrar som gäller för körningen.

Det innebär att experimentdefinitionen kan separeras från själva engine-koden.

Research results

Research runs skrivs normalt till:

data/processed/ml/research/spec_runs/

En körning identifieras av sitt run-directory och innehåller bland annat manifest och resultat per spec.

Candidates

Research Engine kan generera kandidater från intressanta historiska resultat.

Kandidatkod finns under:

ml/research/candidates/

En kandidat representerar en konkretiserad hypotes som kan tas vidare till utvärdering.

Det är viktigt att skilja på:

historiskt forskningsresultat
        ≠
fryst kandidat
        ≠
prospektivt verifierat resultat

Prospective evaluation

Prospektiv utvärdering finns under:

ml/research/evaluation/

Evaluation används för att testa frysta kandidater på data som inte användes för att skapa hypotesen.

Exempel:

python -m ml.research.entrypoint freeze <candidate.yaml>
python -m ml.research.entrypoint evaluate <candidate.yaml> <evaluation.yaml>

Resultaten skrivs normalt till:

data/processed/ml/research/evaluation/

Diagnostics

ml/diagnostics/ innehåller specialiserade diagnostiska analyser.

Diagnostics används när en undersökning inte naturligt passar Research Engines generella experimentmodell.

Strukturen är:

ml/diagnostics/
├── framework/
└── experiments/

framework/ innehåller gemensam diagnostisk infrastruktur.

experiments/ innehåller de specifika diagnostiska analyserna.

Research Engine bör användas när analysen är en generell forskningsoperation som kan beskrivas deklarativt.

Diagnostics används när analysen är mer specialiserad.

AI Lab

ml/ai-lab/ innehåller AI Labs forskningsorkestrering.

AI Lab kan:

* hålla state
* planera experiment
* välja nästa experiment
* arbeta adaptivt
* hantera kandidater
* styra forskningsloopar

AI Lab ska använda Research Engine som experimentmotor när det är lämpligt i stället för att skapa en parallell implementation av samma analyslogik.

AI Lab
   │
   │ planerar
   ▼
Research Engine
   │
   │ kör
   ▼
Research result

Verification

Verifieringslogiken finns i Research Engine och används för att kontrollera forskningsresultat och kandidatflöden.

Verifiering ska hjälpa till att säkerställa att ett resultat inte betraktas som starkare än vad experimentet faktiskt stödjer.

Reproducerbarhet

ML-lagret bygger på att experiment ska kunna återköras och förstås i efterhand.

Därför ska experiment i möjligaste mån kunna kopplas till:

* feature-dataset
* research spec
* target
* tidsperiod
* parametrar
* run-id
* kandidat
* evaluation
* resultat

Separation av ansvar

En förenklad tumregel:

Fråga	Placering
Hur byggs feature-datasetet?	analysis/
Hur tränas en vanlig ML-modell?	ml/
Hur definieras ett generellt forskningsförsök?	ml/research/
Hur körs research specs?	ml/research/
Hur hanteras kandidater?	ml/research/candidates/
Hur görs prospektiv evaluation?	ml/research/evaluation/
Hur görs specialiserad diagnostik?	ml/diagnostics/
Hur styrs adaptiv AI-forskning?	ml/ai-lab/
Var sparas Research Engine-resultat?	data/processed/ml/research/
Var sparas AI Lab-specifika artefakter?	data/ai_lab/results/

Grundprincip

ml/ är alltså inte en enda monolitisk ML-modul.

Det är ett lager där olika typer av maskininlärning och forskning hålls separerade:

                  ml/
                   │
       ┌───────────┼───────────┐
       ▼           ▼           ▼
 conventional   research   diagnostics
 ML                │
                   ▼
                AI Lab

Separationen gör att den generella forskningsmotorn kan utvecklas utan att bli beroende av specifika experiment, samtidigt som AI Lab kan utvecklas som ett högre lager ovanpå den.
