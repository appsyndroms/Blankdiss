Jag har gått igenom den aktuella web/-strukturen och byggflödet i Blankdiss. 

Jag skapade README-innehållet utifrån den faktiska implementationen, inklusive:

* web/build/ och dess ansvarsfördelning
* templates och genererade sidor
* dataflödet från data/
* prospektiv evaluation
* pages/ som build-output
* python web_build.py
* separationen mellan forskningslogik och presentationslager

Jag försökte även skriva web/README.md direkt till GitHub, men GitHub-anslutningen saknar skrivbehörighet och returnerade 403 Resource not accessible by integration. Så filen är inte committad till repot ännu.

Här är den färdiga README:n:

Blankdiss Web

Den här katalogen innehåller den statiska webbplatsen för Blankdiss.

Webb-delen presenterar resultat från Blankdiss data- och ML-pipelines på ett sammanhållet sätt. Webbplatsen byggs från datafiler i repositoryt och består av HTML-sidor, en gemensam CSS-fil och en genererad JSON-payload.

Syfte

Webbplatsen är presentationslagret för Blankdiss. Den ska göra det möjligt att följa hela kedjan:

1. observerade förändringar i blankning,
2. historiska samband mellan blankningshändelser och aktieutveckling,
3. ML-experiment och deras ekonomiska resultat,
4. prospektiva evaluation-körningar,
5. utvecklingen från historisk analys till testning på framtida, tidigare okända observationer.

Webb-koden ska i första hand presentera data. Datainsamling, feature engineering, modellträning och själva forskningslogiken ligger på andra ställen i repositoryt.

Struktur

web/
├── build/
│   ├── __init__.py
│   ├── analysis.py
│   ├── config.py
│   ├── data_loader.py
│   ├── evaluation.py
│   ├── events.py
│   ├── formatting.py
│   ├── ml.py
│   └── page_builder.py
├── static/
│   └── style.css
└── templates/
    ├── index.html
    ├── koplage.html
    ├── bedomning.html
    └── dataanalys.html

Den färdiga webbplatsen skrivs till repositoryts pages/-katalog.

Byggsystem

Byggningen startar via repositoryts web_build.py:

python web_build.py

Den filen anropar:

from web.build.page_builder import build

och page_builder.py ansvarar för att:

* läsa in aktuell data,
* skapa den gemensamma payloaden,
* rendera HTML-mallarna,
* kopiera CSS,
* skriva data.json,
* skriva de fyra färdiga HTML-sidorna.

Byggningen använder endast Python-standardbibliotek i web-lagret.

Dataflöde

Webbbyggaren läser data från följande delar av repositoryt:

Källa	Användning
data/analysis/analysis_*.json	Senaste historiska analysresultatet
data/events/short_events_*.jsonl	Historiska short-events
data/processed/ml/economic_results.json	Ekonomiska ML-resultat
data/processed/ml/research/evaluation/*/evaluation.json	Prospektiva evaluation-resultat

Data laddas av web/build/data_loader.py. Presentationslogiken är uppdelad efter datatyp:

* analysis.py – historiska analysresultat
* events.py – short-events
* ml.py – ekonomiska ML-resultat
* evaluation.py – prospektiva evaluation-resultat
* formatting.py – gemensam formatering
* page_builder.py – samordnar hela byggningen

Sidor

Hem

index.html

Översikt över Blankdiss med:

* antal händelser,
* antal ML-experiment,
* antal evaluation-körningar,
* senaste uppdatering,
* kort introduktion till Blankdiss analysflöde.

Köpläge

koplage.html

Presenterar aktuella observationer och de senaste förändringarna i blankning.

Bedömning

bedomning.html

Presenterar var projektet befinner sig i övergången från historisk analys till prospektiv testning.

Dataanalys

dataanalys.html

Presenterar den underliggande historiska analysen, ML-resultaten och evaluation-historiken.

Templates

HTML-mallarna ligger i web/templates/.

De innehåller placeholders som ersätts av page_builder.py vid byggning. Exempel:

{{EVENT_COUNT}}
{{ML_EXPERIMENT_COUNT}}
{{EVALUATION_RUN_COUNT}}
{{GENERATED_AT}}
{{EVENT_ROWS}}
{{ANALYSIS_ROWS}}
{{ML_ROWS}}
{{EVALUATION_HISTORY_ROWS}}

Gemensamma ersättningar och sid-specifika ersättningar hanteras centralt i page_builder.py.

Det innebär att templates inte ska läsa rådata direkt.

Genererad output

Efter en lyckad byggning skapas bland annat:

pages/
├── index.html
├── koplage.html
├── bedomning.html
├── dataanalys.html
├── style.css
└── data.json

data.json innehåller den samlade payload som byggningen använder, inklusive:

* genereringstid,
* tidszon,
* eventdata,
* historiska analysresultat,
* ekonomiska ML-resultat,
* evaluation-historik,
* sammanställd evaluation-information.

Genererad output ska betraktas som build artifacts. Källan är datafilerna, templates och Python-koden under web/.

Tidszon

Webbbyggaren använder:

Europe/Stockholm

Genereringstiden som visas på webbplatsen är därför svensk lokal tid.

Prospektiv evaluation

Evaluation-delen är särskilt viktig för Blankdiss eftersom den skiljer mellan historisk analys och framtida testning.

Evaluation-resultaten läses från:

data/processed/ml/research/evaluation/<run-id>/evaluation.json

Webblagret försöker läsa information om bland annat:

* kandidat och kandidatversion,
* feature-datum,
* antal observationer,
* mogna observationer,
* AUC,
* top-0,1 %-lift,
* run-id,
* feature rows,
* fingerprint.

Evaluation-presentationskoden är avsiktligt tolerant mot mindre skillnader i resultatformat och letar efter metrics på flera vanliga nivåer i evaluation-resultatet.

Designprinciper

Web-lagret följer några enkla principer.

Separation av ansvar

Data ska läsas i data_loader.py, bearbetas till presentationsformat i respektive modul och renderas av page_builder.py.

Templates ska vara enkla

HTML-filerna ska främst beskriva struktur och presentation. De ska inte innehålla Python-logik eller direkt läsa repositorydata.

Ingen forskningslogik i webben

Webben ska inte träna modeller, skapa features eller förändra forskningsresultat. Den ska visa resultat som redan har producerats av övriga delar av Blankdiss.

Build utifrån aktuell data

När byggningen körs läses de datafiler som finns vid byggtillfället och en ny statisk webb skapas.

Robust presentation

Om en datakälla saknas ska webbbyggaren i möjligaste mån kunna bygga sidan ändå och visa ett tomt eller neutralt läge i stället för att anta att data alltid finns.

Ändra webben

Vid ändringar av presentationen:

* ändra HTML-strukturen i web/templates/,
* ändra styling i web/static/style.css,
* ändra dataläsning i web/build/data_loader.py,
* ändra presentationslogik i rätt modul under web/build/,
* ändra byggflödet endast i page_builder.py när det faktiskt behövs.

Efter ändringar bör webbbyggningen köras från repositoryts rot:

python web_build.py

Kontrollera därefter innehållet i pages/.

Viktigt vid framtida ändringar

web/ är ett presentationslager, inte en separat datamodell.

Om ett nytt fält eller en ny forskningskälla ska visas bör flödet normalt vara:

forsknings-/datapipeline
        ↓
datafil
        ↓
web/build/data_loader.py
        ↓
presentationsmodul
        ↓
template
        ↓
pages/

Det gör det möjligt att hålla forskningslogiken och webbens presentationslogik separerade och minskar risken att webbändringar påverkar själva Blankdiss-analysen.

Relaterat

Repositoryts huvudprojekt:

https://github.com/appsyndroms/Blankdiss
