# Blankdiss Web
Den här katalogen innehåller den statiska webbplatsen för Blankdiss.
Webb-delen är presentationslagret för Blankdiss. Webbplatsen byggs från datafiler i repositoryt och presenterar aktuella short-observationer, historiska events, analyser, ML-resultat och prospektiva evaluation-körningar.
Webben ska i första hand presentera resultat. Datainsamling, feature engineering, modellträning och övrig forskningslogik ligger på andra ställen i repositoryt.
https://github.com/appsyndroms/Blankdiss
## Syfte
Webbplatsen ska göra det möjligt att följa Blankdiss från rå observation till analyserad signal:
1. observerade förändringar i blankning,
2. historiska short-events,
3. aktiens efterföljande utveckling efter dessa events,
4. historiska samband mellan blankningsförändringar och aktieutveckling,
5. ML-experiment och deras ekonomiska resultat,
6. prospektiva evaluation-körningar,
7. utvecklingen mot en historiskt validerad köpsignal.
Webben är alltså ett presentationslager för forskningen och ska inte själv skapa forskningsresultat som inte finns i underliggande data.
## Struktur
```text
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
│   ├── site.js
│   └── style.css
├── templates/
│   ├── index.html
│   ├── koplage.html
│   ├── events.html
│   ├── bedomning.html
│   └── dataanalys.html
└── README.md

Den färdiga webbplatsen skrivs till repositoryts pages/-katalog.

Byggsystem

Byggningen startar via repositoryts web_build.py:

python web_build.py

web_build.py anropar:

from web.build.page_builder import build

page_builder.py ansvarar för att:

* läsa in aktuell data,
* skapa den gemensamma payloaden,
* skapa presentationsrader,
* rendera HTML-mallarna,
* kopiera CSS och JavaScript,
* skriva data.json,
* skriva de färdiga HTML-sidorna.

Byggningen använder endast Python-standardbibliotek i web-lagret.

Dataflöde

Webbbyggaren läser data från repositoryts datalager.

Exempel:

data/analysis/analysis_*.json
        ↓
historiska analysresultat
data/events/short_events_*.jsonl
        ↓
historiska short-events
data/processed/ml/economic_results.json
        ↓
ekonomiska ML-resultat
data/processed/ml/research/evaluation/*/evaluation.json
        ↓
prospektiva evaluation-resultat

Data laddas av:

web/build/data_loader.py

Presentationslogiken är uppdelad efter datatyp:

* analysis.py – historiska analysresultat
* events.py – short-events och köpläge
* ml.py – ekonomiska ML-resultat
* evaluation.py – prospektiva evaluation-resultat
* formatting.py – gemensam formatering
* page_builder.py – samordnar hela byggningen

Sidor

Hem

index.html

Översikt över Blankdiss med bland annat:

* antal händelser,
* antal ML-experiment,
* antal evaluation-körningar,
* senaste uppdatering,
* introduktion till Blankdiss analysflöde.

Köpläge

koplage.html

Köpläge består av två separata tabeller.

Topp 10

Den övre tabellen visar de tio aktuella observationer som för närvarande är kandidater till köpläge.

Tabellen innehåller:

* Ranking
* Aktie
* Blankning
* Förändring
* Historiska events
* 1d
* 5d
* 20d
* 60d
* Köpsignal

Den långsiktiga definitionen av ranking är inte att största minskningen i blankning automatiskt är bäst.

Tanken är i stället:

Aktuell observation
        ↓
identifiera jämförbara historiska events
        ↓
studera efterföljande aktieutveckling
        ↓
1d / 5d / 20d / 60d
        ↓
sammanväg historisk evidens
        ↓
köpsignal
        ↓
ranking

Det innebär att en stor minskning av blankningen bara är en observation. Den blir intressant som köpsignal först när historiska data visar vad liknande förändringar normalt har följts av.

När tillräckligt många historiska events och mogna efterföljande avkastningar finns ska ranking därför baseras på den historiska evidensen.

Webben ska inte fabricera en signal eller sannolikhet när underlaget saknas. Saknade eller omogna resultat visas som —.

Placeholder:

{{KOPLAGE_TOP_ROWS}}

Aktuella observationer

Den nedre tabellen visar de aktuella observationerna utan ranking.

Kolumner:

* Aktie
* Blankning
* Förändring
* 1d
* 5d
* 20d
* 60d

Placeholder:

{{KOPLAGE_ROWS}}

Den här tabellen är avsedd att visa själva observationerna, medan den övre tabellen ska representera den analyserade köplägesbedömningen.

Händelser

events.html

Visar historiska short-events.

Kolumner:

* Aktie
* Blankning
* Förändring
* 1d
* 5d
* 20d
* 60d

Events dedupliceras innan presentation så att samma event inte visas flera gånger om det förekommer i flera eventfiler.

Bedömning

bedomning.html

Presenterar projektets övergång från historisk analys till prospektiv testning.

Sidan används för att följa hur mycket av evaluation-resultaten som faktiskt har hunnit mogna.

Dataanalys

dataanalys.html

Presenterar bland annat:

* historiska analysresultat,
* ML-resultat,
* evaluation-historik,
* underliggande mätvärden.

Templates

HTML-mallarna ligger i:

web/templates/

De innehåller placeholders som ersätts av page_builder.py vid byggning.

Exempel:

{{EVENT_COUNT}}
{{ML_EXPERIMENT_COUNT}}
{{EVALUATION_RUN_COUNT}}
{{GENERATED_AT}}
{{KOPLAGE_TOP_ROWS}}
{{KOPLAGE_ROWS}}
{{EVENT_ROWS}}
{{ANALYSIS_ROWS}}
{{ML_ROWS}}
{{EVALUATION_HISTORY_ROWS}}

Templates ska inte läsa rådata direkt.

All data ska först gå via bygglagret och omvandlas till presentationsformat.

Sortering

Tabellerna på webbplatsen kan sorteras genom att klicka på kolumnrubrikerna.

Sorteringen hanteras av:

web/static/site.js

JavaScript-filen:

* gör tabellrubriker klickbara,
* hanterar stigande och fallande sortering,
* använder data-sort-value när ett numeriskt sorteringsvärde finns,
* fungerar även med tangentbord via Enter och mellanslag.

Sorteringen är en presentationsfunktion och förändrar inte underliggande data.

Genererad output

Efter en lyckad byggning skapas bland annat:

pages/
├── index.html
├── koplage.html
├── events.html
├── bedomning.html
├── dataanalys.html
├── style.css
├── site.js
└── data.json

data.json innehåller den samlade payload som byggningen använder, inklusive:

* genereringstid,
* tidszon,
* eventdata,
* historiska analysresultat,
* ekonomiska ML-resultat,
* evaluation-historik,
* sammanställd evaluation-information.

Genererad output ska betraktas som build artifacts.

Källan är datafilerna, templates och Python-koden under web/.

Tidszon

Webbbyggaren använder:

Europe/Stockholm

Genereringstiden som visas på webbplatsen är därför svensk lokal tid.

Prospektiv evaluation

Evaluation-delen är viktig eftersom den skiljer mellan historisk analys och framtida testning.

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

Evaluation-presentationskoden är tolerant mot mindre skillnader i resultatformat och letar efter metrics på flera vanliga nivåer i evaluation-resultatet.

Köpläge och historisk evidens

Köpläget ska utvecklas från en enkel observationslista till en empiriskt underbyggd signal.

Den centrala principen är:

En förändring i blankning är inte i sig en köpsignal. Det intressanta är vad som historiskt har hänt efter jämförbara förändringar.

För en aktuell observation kan framtida analys därför exempelvis använda:

aktuell förändring i blankning
+ aktuell blankningsnivå
+ aktie
+ historiska jämförbara events
        ↓
efterföljande avkastning
        ↓
1d / 5d / 20d / 60d
        ↓
historisk evidens
        ↓
köpsignal

När datamängden är tillräckligt mogen kan den övre Topp 10-tabellen använda denna information för att rangordna kandidaterna.

Fram till dess ska webbplatsen tydligt visa när information saknas.

Det är viktigt att presentationen inte ersätter saknad historisk evidens med påhittade sannolikheter eller signalvärden.

Separation av ansvar

Web-lagret följer några grundprinciper.

Data ska laddas centralt

data_loader.py ansvarar för att läsa datafilerna.

Presentation ska hanteras av byggmoduler

Exempel:

events.py
analysis.py
ml.py
evaluation.py
formatting.py

Dessa omvandlar rådata till HTML-kompatibla presentationsvärden.

Templates ska vara enkla

HTML-filerna ska främst beskriva struktur och presentation.

De ska inte innehålla Python-logik eller läsa repositorydata direkt.

Ingen modellträning i webben

Webben ska inte:

* träna modeller,
* skapa ML-features,
* ändra forskningsresultat,
* köra forskningspipeline,
* anta resultat som inte finns i data.

Webben ska presentera resultat som redan har producerats av övriga delar av Blankdiss.

Build utifrån aktuell data

När byggningen körs läses de datafiler som finns vid byggtillfället och en ny statisk webb skapas.

Robust presentation

Om en datakälla saknas ska webbbyggaren i möjligaste mån kunna bygga sidan ändå och visa ett tomt eller neutralt läge i stället för att anta att data alltid finns.

Ändra webben

Vid ändringar av presentationen:

* ändra HTML-strukturen i web/templates/,
* ändra styling i web/static/style.css,
* ändra tabellbeteende i web/static/site.js,
* ändra dataläsning i web/build/data_loader.py,
* ändra presentationslogik i rätt modul under web/build/,
* ändra byggflödet i page_builder.py endast när det faktiskt behövs.

Efter ändringar bör webbbyggningen köras från repositoryts rot:

python web_build.py

Kontrollera därefter innehållet i:

pages/

Viktigt vid framtida ändringar

web/ är ett presentationslager, inte en separat datamodell.

Om ett nytt fält eller en ny forskningskälla ska visas bör flödet normalt vara:

forsknings-/datapipeline
        ↓
datafil
        ↓
web/build/data_loader.py
        ↓
presentationslogik
        ↓
template
        ↓
pages/

Om ny analyslogik behövs bör den i första hand placeras i Blankdiss centrala analys-/forskningslager och inte gömmas i HTML eller JavaScript.

Det gör det möjligt att hålla forskningslogiken och webbens presentationslogik separerade.

Relaterat

Repositoryts huvudprojekt:
https://github.com/appsyndroms/Blankdiss
