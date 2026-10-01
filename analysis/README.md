Analysis

analysis/ ansvarar för Blankdiss centrala feature-pipeline.

Syftet är att bygga ett gemensamt, reproducerbart feature-dataset från rådata som sedan kan användas av ML, Research Engine, diagnostics och andra analyser.

Ansvarsområde

Analysis-lagret ansvarar för:

* inläsning av rådata
* synkronisering av FI-data och prisdata
* sammanfogning med marknads- och sektordata
* beräkning av features
* skapande av targets
* kvalitetskontroller
* skrivning av det kanoniska feature-datasetet

Det är viktigt att skilja detta från forskningslagret.

analysis/ svarar på:

Vilka datapunkter och features ska finnas som grund för forskningen?

ml/research/ svarar på:

Vilka samband, signaler och hypoteser kan vi hitta i dessa data?

Huvudflöde

Rådata
  │
  ├── FI blankningsdata
  ├── aktiekurser
  ├── marknadsdata
  └── sektor-/branschdata
          │
          ▼
    analysis/build_features.py
          │
          ├── tidsmässig synkronisering
          ├── feature engineering
          ├── targets
          └── kvalitetskontroller
          │
          ▼
data/processed/analysis/
          │
          ├── features_0001.jsonl
          ├── features_0002.jsonl
          ├── ...
          └── metadata / QC

Det resulterande datasetet är den gemensamma datagrunden för efterföljande analys.

Feature builder

Den centrala implementationen finns i:

analysis/build_features.py

Den kan bygga om feature-datasetet från underliggande data.

Full ombyggnad kan göras med:

python analysis/build_features.py --force

Exakta parametrar och ytterligare CLI-alternativ ska alltid hämtas från den aktuella implementationen.

Feature-dataset

Resultatet skrivs till:

data/processed/analysis/

Datasetet är uppdelat i JSONL-chunks för att kunna hantera stora datamängder utan att behöva läsa hela datasetet som ett enda objekt.

Varje rad representerar en datapunkt för en aktie och ett relevant datum och innehåller bland annat:

* identifierande information
* datum
* blankningsrelaterade värden
* prisrelaterade features
* volatilitet
* marknadsrelaterade features
* sektor-/branschinformation
* targets

Vilka features som faktiskt finns ska betraktas som en implementationdetalj och läsas från den aktuella feature-builden och dess metadata, inte från en statisk lista i denna README.

Dataflöde

Analysis-lagret kombinerar flera datakällor.

FI-data

Blankningsdata från Finansinspektionen används som central källa för blankningsrelaterade observationer.

Prisdata

Prisdata används bland annat för att skapa:

* avkastningsmått
* prisförändringar
* volatilitet
* tidsserierelaterade features
* framtida targets

Marknadsdata

Marknadsdata används för att sätta individuella aktier i ett bredare marknadssammanhang.

Det gör exempelvis att en akties utveckling kan analyseras relativt marknaden i stället för isolerat.

Sektordata

Sektor- och branschinformation gör det möjligt att kontrollera eller analysera skillnader mellan olika delar av marknaden.

Targets

Targets definieras som en del av feature-pipelinen eftersom forskningen behöver ett konsekvent sätt att beskriva vad som ska förutsägas eller analyseras.

Ett target kan exempelvis representera framtida prisutveckling under en viss horisont.

Research Engine ska använda dessa definierade targets i stället för att varje experiment skapar sin egen grundläggande target-definition.

Kvalitetskontroll

Feature-builden innehåller kontroller för att upptäcka problem i datasetet.

Exempel på sådant som behöver kontrolleras är:

* saknade värden
* ogiltiga datum
* otillräckliga prisserier
* felaktiga joinar
* duplicerade observationer
* orimliga värden
* bristande tidsmässig täckning

QC-information sparas tillsammans med resultatet av feature-builden.

Chunking

Feature-datasetet skrivs i flera JSONL-filer.

Det gör att:

* stora dataset kan bearbetas stegvis
* ML- och analysjobb kan läsa data chunkvis
* resultatet blir enklare att hantera i Git/LFS och lokala arbetsflöden
* en komplett rebuild inte behöver representeras som en enda mycket stor fil

Chunkningen är en lagringsdetalj. Den ska inte påverka den logiska betydelsen av datasetet.

Förhållande till ML

ML-lagret använder feature-datasetet som input.

analysis/
    │
    ▼
data/processed/analysis/
    │
    ├── conventional ML
    ├── Research Engine
    ├── diagnostics
    └── AI Lab

Analysis ska därför inte innehålla experimentlogik som egentligen hör hemma i ml/research/.

Om en ny idé handlar om att testa om en befintlig feature har prediktiv kraft är det normalt en Research Engine-fråga.

Om idén däremot kräver att en ny generell feature skapas från rådata hör implementationen hemma i analysis-lagret.

Reproducerbarhet

Feature-datasetet är en central del av Blankdiss reproducerbarhet.

En forskningskörning ska kunna hänvisa till den feature-grund som användes när experimentet genomfördes.

Det är därför viktigt att inte manuellt modifiera feature-filer utan att gå via feature-pipelinen när data behöver byggas om.

Vad som inte hör hemma här

Följande ska normalt inte implementeras i analysis/:

* kandidatbedömning
* prospektiv utvärdering
* Research Engine-specifikationer
* AI Lab-state
* webbpresentation
* rapportlogik för enskilda forskningskörningar
* beslut om vilka hypoteser som ska testas

Dessa hör till andra delar av projektet.

Relation till övriga projektet

                  analysis/
                      │
                      ▼
            feature-dataset
                      │
          ┌───────────┼───────────┐
          ▼           ▼           ▼
     ml/research   diagnostics   ai-lab
          │
          ▼
      candidates
          │
          ▼
     evaluation
          │
          ▼
      verification

På detta sätt fungerar analysis/ som projektets gemensamma datagrund medan forsknings- och ML-lagren kan utvecklas oberoende av själva databyggandet.
