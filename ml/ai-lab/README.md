AI Lab

ml/ai-lab/ innehåller Blankdiss AI Lab – den del av systemet som används för kontrollerad och adaptiv forskningsorkestrering.

AI Lab ligger på en högre abstraktionsnivå än Research Engine. Research Engine genomför experimenten; AI Lab kan planera vilka experiment som ska genomföras och använda resultaten för att bestämma nästa steg.

Roll i arkitekturen

                 feature dataset
                       │
                       ▼
                Research Engine
                       ▲
                       │
                ┌──────┴──────┐
                │   AI Lab    │
                │             │
                │ state       │
                │ planning    │
                │ orchestration
                │ candidates  │
                └─────────────┘

Den grundläggande principen är:

AI Lab styr forskningen på hög nivå. Research Engine utför de generella experimenten.

AI Lab ska därför inte duplicera generisk analyslogik som redan finns i Research Engine.

Vad AI Lab gör

AI Lab kan användas för forskningsloopar där nästa experiment beror på vad tidigare experiment visade.

En förenklad loop är:

Start
  │
  ▼
Läs state
  │
  ▼
Planera experiment
  │
  ▼
Kör experiment
  │
  ▼
Analysera resultat
  │
  ▼
Uppdatera state
  │
  ▼
Välj nästa steg
  │
  └───────────────►

Det gör AI Lab lämpligt för forskning där man inte vill definiera hela experimentsekvensen i förväg.

State

AI Lab kan hålla state mellan forskningssteg.

State kan bland annat beskriva:

* vilka experiment som genomförts
* vilka hypoteser som undersökts
* vilka resultat som är intressanta
* vilka kandidater som skapats
* vilka områden som bör undersökas vidare
* vilken forskningsfas systemet befinner sig i

State är viktigt för att AI Lab ska kunna fungera som en iterativ forskningsprocess i stället för en serie helt fristående experiment.

Experiment planning

Experiment planner används för att bestämma vilka experiment som ska genomföras.

Planeringen kan utgå från:

* tidigare resultat
* forskningsmål
* befintliga specs
* hypoteser
* kandidater
* begränsningar
* resultat från tidigare körningar

Planeringen ska leda till konkreta experiment som sedan kan utföras av Research Engine.

Research Engine som underliggande motor

AI Lab ska normalt inte implementera en ny version av Research Engines analysfunktioner.

I stället:

AI Lab
  │
  ├── väljer vad som ska testas
  │
  ▼
Research spec
  │
  ▼
Research Engine
  │
  ├── läser feature-data
  ├── kör analys
  ├── beräknar resultat
  └── skriver research-resultat

Det ger en tydlig separation mellan forskningsstrategi och experimentgenomförande.

Kandidater

AI Lab kan arbeta med kandidater under forskningsprocessen.

En kandidat är en konkretiserad hypotes som kan tas vidare efter att ett historiskt resultat bedömts som tillräckligt intressant för fortsatt testning.

Kandidatflödet bör därför hållas separat från själva AI Lab-state:

Experiment
   │
   ▼
Resultat
   │
   ▼
Hypotes
   │
   ▼
Candidate
   │
   ▼
Prospective evaluation
   │
   ▼
Verification

Att AI Lab hittar en kandidat innebär inte i sig att kandidaten är verifierad.

Resultat

AI Lab-specifika resultat och artefakter kan sparas under:

data/ai_lab/results/

Research Engine-resultat sparas däremot normalt under:

data/processed/ml/research/

Evaluation-resultat sparas normalt under:

data/processed/ml/research/evaluation/

Det är viktigt att inte blanda dessa tre typer av resultat.

Kontrollerad forskning

AI Lab ska inte bara maximera antalet experiment.

Forskningsorkestreringen behöver kunna ta hänsyn till exempelvis:

* experimentbudget
* redan testade hypoteser
* duplicerade experiment
* datatillgänglighet
* OOS-krav
* kandidatstatus
* behov av verifiering

Målet är att skapa en kontrollerad forskningsprocess där nya experiment bygger vidare på tidigare information.

Adaptiv forskning

Den adaptiva delen innebär att forskningsprocessen kan ändra riktning baserat på observerade resultat.

Exempel:

Bred scan
    │
    ▼
Intressant signal
    │
    ▼
Djupare analys
    │
    ▼
Variantanalys
    │
    ▼
Kandidat
    │
    ▼
Prospektiv evaluation

Vilket steg som kommer härnäst behöver inte vara identiskt för alla forskningsområden.

Separation från diagnostics

ml/diagnostics/ används för specialiserade diagnostiska analyser.

AI Lab kan använda diagnostiska resultat som information i forskningsprocessen, men AI Lab är inte själva diagnostics-lagret.

Förenklat:

Research Engine
    └── generell experimentmotor
Diagnostics
    └── specialiserade analyser
AI Lab
    └── forskningsstyrning och adaptiv orkestrering

Reproducerbarhet

Även om AI Lab är adaptivt måste processen vara spårbar.

En AI Lab-körning bör därför kunna kopplas till:

* state
* planerade experiment
* Research Engine runs
* specs
* resultat
* kandidater
* evaluation
* verifiering

Det ska vara möjligt att i efterhand förstå varför ett visst experiment genomfördes och vilket tidigare resultat som ledde fram till det.

Grundprincip

AI Lab är alltså inte ytterligare en generell analysmotor.

Dess huvudsakliga funktion är att lägga ett intelligent och kontrollerat besluts-/orkestreringslager ovanpå den befintliga forskningsinfrastrukturen:

                  AI Lab
                     │
              forskningsstrategi
                     │
                     ▼
              Research Engine
                     │
               experiment
                     │
                     ▼
                resultat
                     │
                     ▼
                 AI Lab
                     │
              nästa beslut
                     │
                     └──────►

Det gör att Blankdiss kan utvecklas från manuellt definierade experiment mot mer adaptiv forskning utan att den centrala experimentmotorn behöver dupliceras.
