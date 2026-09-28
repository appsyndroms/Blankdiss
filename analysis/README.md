# Analysis – Feature Pipeline

Det här katalogen innehåller den del av Blankdiss som bygger och kvalitetssäkrar det
feature-dataset som används av ML och Research Engine.

Målet är att hålla feature-bygget generellt och reproducerbart.

Grundprincipen är:

    Rådata
      ↓
    Feature generation
      ↓
    Feature dataset
      ↓
    Feature QC
      ↓
    ML / Research Engine

Research Engine ska i första hand välja och kombinera befintliga features.
Ny forskning ska därför normalt inte kräva att feature-pipelinen ändras.

---

## 1. Vad feature-lagret gör

Feature-pipelinen kombinerar framför allt:

- FI-data
- historiska aktiekurser
- OMXSPI
- sektortillhörighet

och bygger ett gemensamt dataset där varje rad representerar en FI-observation
kopplad till ett prisdatum och historiska/framåtblickande features.

Exempel på befintliga prisfeatures:

- `price_return_5d`
- `price_return_20d`
- `price_return_60d`
- `price_volatility_20d`
- `price_distance_from_20d_high`
- `price_distance_from_60d_high`

Forward returns och targets byggs från den matchade signalprispunkten.

Feature-datasetet är alltså den gemensamma datagrunden för senare analyser.

---

## 2. Varför relativa features byggs

En akties egen utveckling säger inte alltid om den är svag eller stark i ett
bredare marknadsläge.

Exempel:

En aktie har fallit 8 % på 20 dagar.

Det kan betyda helt olika saker om:

- OMXSPI samtidigt har fallit 10 %
- OMXSPI är oförändrat
- OMXSPI har stigit 8 %

På samma sätt kan en akties utveckling behöva jämföras med utvecklingen i dess
egen sektor.

Därför bygger feature-lagret både absoluta och relativa mått.

Det gör att Research Engine senare kan testa frågor som:

- Är absolut momentum viktigare än relativt momentum?
- Är en aktie svag relativt marknaden?
- Är den svag relativt sin sektor?
- Uppstår FI-/volatilitetseffekten framför allt när aktien underpresterar
  marknaden eller sektorn?
- Är ett observerat samband egentligen bara ett uttryck för börsklimatet?

Detta är en generell utökning av feature-lagret, inte en speciallösning för
en enskild research-hypotes.

---

## 3. Relativa features

Relativa features byggs för tre horisonter:

- 5 dagar
- 20 dagar
- 60 dagar

### Marknad

Marknadsavkastning:

- `market_return_5d`
- `market_return_20d`
- `market_return_60d`

Dessa beskriver utvecklingen för OMXSPI under motsvarande period.

### Sektor

Sektoravkastning:

- `sector_return_5d`
- `sector_return_20d`
- `sector_return_60d`

Dessa beskriver utvecklingen för den sektor som aktien tillhör.

### Aktien relativt marknaden

- `price_return_5d_relative_market`
- `price_return_20d_relative_market`
- `price_return_60d_relative_market`

Dessa beskriver aktiens utveckling i relation till OMXSPI.

### Aktien relativt sektorn

- `price_return_5d_relative_sector`
- `price_return_20d_relative_sector`
- `price_return_60d_relative_sector`

Dessa beskriver aktiens utveckling i relation till den egna sektorn.

---

## 4. Varför 5d, 20d och 60d byggs samtidigt

Research Engine ska inte behöva ändra feature-pipelinen varje gång en ny
hypotes undersöker en annan tidshorisont.

Därför byggs 5d, 20d och 60d som en generell featurefamilj.

Det är viktigt även när den aktuella hypotesen bara använder två av
horisonterna.

Exempel:

En första analys kan undersöka:

    5d × 60d

En senare analys kan undersöka:

    5d × 20d

eller:

    20d × 60d

eller kombinera absolut och relativ momentum.

Feature-lagret behöver då inte byggas om för varje forskningsfråga.

Research-specifik selektion ska i stället ske i YAML-specen för Research Engine.

---

## 5. Marknadsdata

Marknadsdata kommer från OMXSPI och lagras som rådata under:

    data/raw/market/omxspi.jsonl

Marknadsdatan innehåller i grunden:

- `market_date`
- `market_close`

`analysis.update_market` ansvarar för att hålla denna historik uppdaterad.

Feature-pipelinen använder därefter marknadsserien för att beräkna
marknadsrelaterade features.

Marknadsdata är alltså en källa till features, inte en del av själva
Research Engine.

---

## 6. Sektordata

Sektortillhörighet finns som en separat mappning under:

    data/analysis/sector_map.json

Sektormappningen är en generell metadata-/datakälla som används för att kunna
beräkna sektorrelaterade features.

Det är viktigt att sektortillhörigheten hålls separat från själva
forskningsspecifikationerna.

Research Engine ska exempelvis inte behöva känna till hur en akties sektor
hämtades eller beräknades.

Den ska bara kunna använda:

    sector_return_20d

eller:

    price_return_20d_relative_sector

som vanliga features.

---

## 7. Absolut momentum kontra relativt momentum

Det finns en viktig skillnad mellan:

    price_return_20d

och:

    price_return_20d_relative_market

Den första frågar:

> Hur mycket har aktien rört sig?

Den andra frågar:

> Hur mycket har aktien rört sig relativt marknaden?

På motsvarande sätt skiljer sig:

    price_return_20d_relative_market

från:

    price_return_20d_relative_sector

Den senare frågar om aktien under- eller överpresterat sin egen sektor.

Det gör det möjligt att separera flera tänkbara mekanismer.

Exempel:

    Aktien faller
        ↓
    hela marknaden faller
        ↓
    aktien är kanske inte särskilt svag relativt marknaden

kontra:

    Aktien faller
        ↓
    marknaden är stabil
        ↓
    aktien är tydligt svag relativt marknaden

och:

    Aktien faller
        ↓
    sektorn faller lika mycket
        ↓
    rörelsen kan vara sektordriven

Detta är särskilt relevant när man försöker förstå om ett samband i
FI-/prisdata verkligen är aktiespecifikt.

---

## 8. Relation till den aktuella momentumforskningen

En aktuell forskningslinje undersöker samspelet mellan:

- kortsiktigt momentum
- längre momentum
- 20-dagars volatilitet
- framtida downside events

De första analyserna har bland annat jämfört:

    5d × 20d
    5d × 60d
    20d × 60d

Resultaten har gjort det relevant att undersöka om den observerade effekten
fortfarande finns när momentum sätts i relation till:

- den breda marknaden
- den egna sektorn

Det är viktigt att skilja mellan två frågor:

1. Finns ett samband mellan aktiens egna momentum/volatilitet och framtida
   downside?

2. Finns sambandet fortfarande när man kontrollerar för att aktien samtidigt
   rör sig tillsammans med marknaden eller sektorn?

Den första frågan kan besvaras med absoluta features.

Den andra kräver relativa features.

Feature-lagret ska därför tillhandahålla båda.

---

## 9. Feature generation och Research Engine ska hållas separata

Feature generation ansvarar för att skapa mätbara variabler.

Research Engine ansvarar för att formulera och testa hypoteser med dessa
variabler.

Exempel:

Feature-lagret skapar:

    price_return_5d
    price_return_20d
    price_return_60d
    price_return_5d_relative_market
    price_return_20d_relative_market
    price_return_60d_relative_market

Research Engine kan sedan få en YAML-spec som testar:

    5d momentum
        ×
    60d relative-to-market momentum
        ×
    high 20d volatility

utan att feature-pipelinen behöver känna till denna specifika hypotes.

Detta är en central arkitekturprincip i Blankdiss.

---

## 10. Feature QC

Efter feature generation körs Feature QC.

QC kontrollerar bland annat:

- obligatoriska kolumner
- datumintegritet
- matchning mellan FI och prisdata
- mapping/instrumentidentitet
- dubbletter
- numeriska featurevärden
- forward returns
- forward-return alignment
- price leakage
- threshold-logik

QC kan returnera:

    PASS
    WARN
    FAIL

`WARN` betyder att datasetet innehåller en känd avvikelse som inte
nödvändigtvis gör datasetet oanvändbart.

`FAIL` betyder att feature-datasetet inte ska betraktas som giltigt för
nedströms analys.

Warnings ska inte automatiskt "fixas" bara för att få en PASS-status.
Först måste man förstå vad varningen representerar.

---

## 11. Leakage

Feature-lagret måste skilja mellan information som är känd vid
signalögonblicket och information som inträffar efter signalögonblicket.

Historiska features får endast använda information som var tillgänglig vid
den aktuella observationen.

Forward returns och event targets får däremot använda framtida prisdata,
eftersom de representerar det som ska förklaras/predikteras.

Feature QC kontrollerar därför bland annat:

    price_leakage
    forward_return_alignment

Detta är en grundläggande del av reproducerbarheten i hela forskningskedjan.

---

## 12. Reproducerbarhet

Feature-datasetet ska kunna byggas om från rådata.

Det innebär att en ändring i feature-logiken inte ska kräva manuell redigering
av gamla featurefiler.

Build Features-workflowen bygger hela feature-datasetet på nytt från de
aktuella rådatakällorna.

Det gör processen:

    rådata
      ↓
    build features
      ↓
    QC
      ↓
    nytt feature-dataset

Det minskar risken för att gamla och nya features blandas ihop.

---

## 13. Build Features-workflow

Feature-bygget körs separat från Research Engine.

Workflowen ansvarar för:

1. Hämta/utgå från aktuella rådata.
2. Bygga feature-datasetet.
3. Kontrollera storleken på feature-chunks.
4. Köra Feature QC.
5. Stoppa vid QC-status `FAIL`.
6. Tillåta `PASS` och `WARN`.
7. Spara feature-datasetet som artifact.
8. Uppdatera de versionshanterade featurefilerna.

Det är medvetet separerat från ML/Research Engine.

Research Engine ska kunna köras flera gånger mot samma feature-dataset utan
att behöva bygga om features varje gång.

---

## 14. Filstruktur

De viktigaste delarna är:

    analysis/
        build_features.py
        feature_config.py
        feature_prices.py
        feature_source.py
        feature_relative.py
        features_qc.py
        update_market.py
        build_sector_map.py
        README.md

Rådata:

    data/raw/market/omxspi.jsonl
    data/raw/prices/...

Sektormetadata:

    data/analysis/sector_map.json

Genererat feature-dataset:

    data/processed/analysis/features_*.jsonl
    data/processed/analysis/features_metadata.json
    data/processed/analysis/features_qc.json

Det äldre:

    data/processed/analysis/fi_price_features.jsonl

är en legacy feature-vy och ska inte vara den primära versionshanterade
featuremodellen.

---

## 15. Vad som inte hör hemma här

Research-specifik logik ska normalt inte läggas i `analysis/`.

Exempel på sådant som inte bör byggas som specialkod här:

- en specifik momentumkombination
- ett specifikt event
- en specifik hypotes
- en viss bootstrap-analys
- en viss train/test-split
- ranking av forskningsresultat
- automatisk optimering av signaltrösklar

Sådant hör hemma i Research Engine och dess YAML-specifikationer.

Om en ny hypotes kräver en feature som saknas kan feature-lagret utökas med en
generell featurefamilj.

Exempel:

Om flera framtida analyser kan behöva relativ momentum är det rimligt att
bygga hela den generella relativa momentumfamiljen.

Däremot ska vi inte bygga:

    price_return_17d_relative_market_for_this_one_experiment

bara för att en enskild hypotes råkar behöva 17 dagar.

---

## 16. Forskningsprincip

Feature-lagrets uppgift är att göra det möjligt att ställa bättre frågor.

Det ska inte försöka svara på frågorna självt.

Exempel:

    Feature layer
        ↓
    "Vad hände med aktien relativt marknaden?"

    Research Engine
        ↓
    "Förändras FI/volatilitets-effekten när relativt momentum är negativt?"

På så sätt kan samma feature-dataset användas för många olika analyser.

---

## 17. Nästa steg

När de relativa featuresen är byggda och QC är godkänd kan Research Engine
använda dem för kontroller av de observerade momentum-/volatilitetseffekterna.

En naturlig forskningsordning är exempelvis:

1. Absoluta momentumfeatures.
2. Marknadsrelativa momentumfeatures.
3. Sektorrelativa momentumfeatures.
4. Kontrollera om effekten överlever dessa relativa perspektiv.
5. Därefter undersöka mer specifika mekanismer, om resultaten motiverar det.

Det viktiga är att varje steg är en ny, explicit forskningsfråga.

Feature-lagret ska förbli generellt och återanvändbart.
Research Engine ska bära hypoteserna.
