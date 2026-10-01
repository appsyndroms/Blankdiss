Diagnostics

ml/diagnostics/ innehåller specialiserade diagnostiska analyser som används för att undersöka egenskaper, problem och fenomen i Blankdiss data och forskningsresultat.

Diagnostics är ett komplement till Research Engine, inte en alternativ generell forskningsmotor.

Roll i arkitekturen

                    analysis/
                        │
                        ▼
                feature dataset
                        │
          ┌─────────────┴─────────────┐
          ▼                           ▼
   Research Engine              Diagnostics
          │                           │
          ▼                           ▼
   forskningsresultat          diagnostiska resultat

Research Engine används när en analys kan beskrivas som ett återanvändbart generellt experiment.

Diagnostics används när analysen är mer specialiserad och framför allt syftar till att förstå eller kontrollera ett specifikt fenomen.

När ska Diagnostics användas?

Diagnostics passar exempelvis när man behöver undersöka:

* varför ett resultat ser ut som det gör
* om ett dataset innehåller ett särskilt problem
* om en viss feature beter sig oväntat
* om en modell eller signal har en specifik egenskap
* om ett experiment behöver kompletteras med en specialiserad kontroll
* om en forskningshypotes kräver en analys som inte passar Research Engines generella spec-modell

Det behöver alltså inte vara en ny prediktionsmodell för att motivera en diagnostic.

När ska Research Engine användas?

Om samma typ av analys kan beskrivas generellt och återanvändas för många olika features, targets eller hypoteser bör den normalt placeras i Research Engine.

Exempel:

"Testa feature X mot target Y"

är typiskt en Research Engine-uppgift.

Medan:

"Undersök varför signalen X endast fungerar
under ett specifikt marknadsfenomen"

kan vara en diagnostisk analys om den kräver särskild logik.

Struktur

Diagnostics är uppdelat i två huvudsakliga delar:

ml/diagnostics/
├── framework/
└── experiments/

framework/

Innehåller gemensam infrastruktur som kan användas av flera diagnostiska analyser.

Syftet är att undvika att samma tekniska stöd implementeras flera gånger.

experiments/

Innehåller de konkreta diagnostiska analyserna.

Varje experiment ska ha ett tydligt syfte och beskriva vilket fenomen eller vilken frågeställning det undersöker.

Förhållande till Research Engine

Research Engine:

generell
återanvändbar
deklarativ
experimentorienterad

Diagnostics:

specialiserad
frågestyrd
diagnostisk
fenomenorienterad

Gränsen behöver inte vara absolut. Om en diagnostisk analys visar sig vara återanvändbar på ett generellt sätt kan logiken senare flyttas eller abstraheras in i Research Engine.

Förhållande till AI Lab

AI Lab kan använda diagnostiska resultat som underlag för nästa forskningssteg.

Exempel:

AI Lab
   │
   ▼
Research Engine
   │
   ▼
Intressant resultat
   │
   ▼
Diagnostics
   │
   ▼
Förklaring / ytterligare information
   │
   ▼
AI Lab
   │
   ▼
Nästa experiment

Diagnostics är därmed ett verktyg som kan ge mer information till forskningsprocessen, men är inte själva forskningsorkestreringen.

OOS och tidsmässig separation

Diagnostiska analyser måste följa samma grundläggande tidsmässiga principer som övrig forskning.

När analysen används för att bedöma en hypotes ska man vara tydlig med:

* vilken period som analyseras
* om data användes vid hypotesgenerering
* om analysen är in-sample eller out-of-sample
* vilka observationer som är tillgängliga vid den aktuella tidpunkten
* om analysen kan introducera look-ahead bias

En diagnostic får alltså inte omedvetet göra en historisk signal starkare genom att använda framtida information.

Resultat

Diagnostiska resultat ska vara spårbara till:

* det experiment som kördes
* aktuell datamängd
* relevant tidsperiod
* parametrar
* eventuella forskningsresultat som analyserades

Om ett diagnostiskt resultat leder till en ny kandidat ska kandidaten hanteras via det ordinarie kandidatflödet i Research Engine.

Vad Diagnostics inte ska göra

Diagnostics ska inte bli en alternativ plats för generell forskningslogik.

Undvik exempelvis att skapa separata implementationer för:

* samma signaltest som Research Engine redan stöder
* samma targetanalys
* samma feature-ranking
* samma standardiserade evaluation
* samma kandidatflöde

Om en funktion blir generell bör den i stället övervägas för Research Engine.

Grundprincip

Diagnostics ska hjälpa Blankdiss att förstå varför ett resultat ser ut som det gör, inte bara konstatera att ett resultat finns.

Research
   │
   ▼
Observation
   │
   ▼
Diagnostic question
   │
   ▼
Specialized analysis
   │
   ▼
Better understanding

Det gör ml/diagnostics/ till ett kompletterande analyslager mellan generell forskning och djupare förståelse av specifika resultat.
