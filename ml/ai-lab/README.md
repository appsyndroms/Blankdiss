Du har rätt. Problemet är att gränssnittet verkar lägga metadata på kodblocken när jag försöker ge hela dokumentet som ett enda block. Jag ska inte göra det.

Här är README:n som vanlig text, med kodexempel indragna med fyra mellanslag. Då blir det ren Markdown utan konstig blockformatering:

Blankdiss AI Lab

AI Lab är Blankdiss mekanism för kontrollerad, iterativ forskning.

Syftet är inte att hitta ett positivt resultat så snabbt som möjligt.

Syftet är att låta tidigare experiment ge information om vilket forskningssteg som ska tas härnäst, samtidigt som forskningen förblir reproducerbar, spårbar och skyddad mot data leakage och resultatjakt.

⸻

Grundprincip

AI Lab optimerar:

idé → information

inte:

idé → mer kod

Ett negativt resultat är därför ett giltigt forskningsresultat.

Om en hypotes inte fungerar ska AI Lab inte försöka vrida parametrarna tills ett positivt resultat uppstår. När ett fördefinierat parameterutrymme är uttömt ska systemet i stället kunna konstatera att den aktuella experimentfamiljen inte längre ger tillräcklig information och gå vidare till en ny forskningsfråga eller experimentfamilj.

⸻

Arkitektur

AI Lab är ett generellt exekveringslager ovanpå den generiska Research Engine.

Den viktiga principen är:

GitHub Actions workflow
        │
        ▼
ml/ai-lab/adaptive_research.py
        │
        ├───────────────► kontrollerade YAML-specar
        │                       │
        │                       ▼
        │                 Research Engine
        │
        └───────────────► adaptiv forskningsloop
                                │
                                ▼
                          Research Engine

Workflowet känner alltså inte till enskilda forskningsfrågor.

Workflowet startar endast AI Lab:

Workflow → AI Lab → Research Engine → spec

En ny forskningsfråga ska därför normalt kunna läggas till genom att skapa eller ändra en YAML-specifikation. AI Lab ska inte behöva ändras för varje ny hypotes.

⸻

Forskningsspecifikationer

Forskningslogiken deklareras i ml/research/specs/*.yaml.

En spec beskriver exempelvis:

* id
* forskningsfråga
* signaler
* targets
* analysmetod
* bins/trösklar
* windows
* data splits
* metadata

Research Engine tolkar sedan specifikationen och kör den generiska analysmotorn.

Exempel:

ml/research/specs/
└── momentum_tail_threshold_analysis.yaml

Specen behöver inte ha någon motsvarande specialgren i AI Lab.

⸻

Kontrollerade forskningsspecar

För experiment som ska köras som en del av AI Lab kan en spec uttryckligen opta in genom metadata:

metadata:
  stage: hypothesis_test
  purpose: momentum_tail_threshold_analysis
  ai_lab_execution:
    enabled: true
    mode: once

AI Lab upptäcker automatiskt YAML-specar som uttryckligen har:

metadata.ai_lab_execution.enabled = true

och en stödd exekveringsmodell.

Det innebär att en ny kontrollerad forskningsfråga kan läggas till utan att:

* ändra GitHub Actions-workflowet
* lägga till ett specifikt spec-id i AI Lab
* lägga till en särskild if-sats för experimentet
* skriva ny experimentkod i ml/ai-lab

Det räcker att lägga till den deklarativa specen.

⸻

mode: once

Kontrollerade specar kan köras med:

ai_lab_execution:
  enabled: true
  mode: once

once betyder att AI Lab kör specen en gång och sedan betraktar den som genomförd.

Exekveringen görs inte enbart utifrån minnet i den aktuella Python-processen. AI Lab kontrollerar även redan persisterade forskningsresultat och manifest under:

data/processed/ml/research/spec_runs/

Det gör att en ny workflow-körning inte automatiskt kör om samma once-spec.

Om en körning däremot avbryts innan resultatet har persisterats betraktas specen inte som färdig och kan köras igen vid nästa försök.

⸻

Delad research-session

AI Lab bygger en gemensam research-session för den aktuella körningen.

I sessionen samlas requirements från både:

1. den befintliga adaptiva forskningen
2. kontrollerade forskningsspecar som ska köras

Feature-data laddas en gång och research-cachen byggs en gång.

adaptiv forskning ─────┐
                       │
kontrollerade specar ─┼──► gemensam ResearchSession
                       │
                       ▼
                 load features
                       │
                       ▼
                 build cache
                       │
                ┌──────┴──────┐
                ▼             ▼
         adaptive loop   controlled specs

Detta undviker att varje ny YAML-spec behöver implementera sin egen dataladdning eller cachehantering.

⸻

Den adaptiva forskningsloopen

Den befintliga adaptiva forskningen arbetar fortfarande enligt:

OBSERVE
   │
   ▼
ANALYZE
   │
   ▼
ADAPT
   │
   ▼
EXPERIMENT
   │
   ▼
ANALYZE
   │
   ├──────────────► OBSERVE
   │
   └─ parameterutrymme slut
                     │
                     ▼
                   EXTEND
                     │
                     ▼
              ny experimentfamilj

Den adaptiva delen har ett fördefinierat parameterutrymme.

AI Lab får alltså inte fritt uppfinna nya parametrar eller Python-kod baserat på ett attraktivt resultat. Kandidater väljs från det deklarerade utrymmet och resultaten används för att avgöra nästa forskningssteg.

⸻

Kontrollerad forskning kontra adaptiv forskning

De två mekanismerna har olika syften.

Kontrollerad spec

En kontrollerad spec används när forskningsfrågan redan är definierad och ska genomföras exakt enligt sin YAML-spec.

Exempel:

momentum_tail_threshold_analysis

Den kan exempelvis testa:

price_momentum_5d
        │
        ├─ 20 %
        ├─ 10 %
        ├─ 5 %
        └─ 2.5 %
        │
        ▼
down_5pct_5d
down_7pct_5d
down_10pct_5d

AI Lab väljer inte dessa trösklar utifrån resultatet. De är deklarerade i förväg.

Adaptiv forskning

Den adaptiva delen kan däremot gå vidare genom ett fördefinierat parameterutrymme och välja nästa kandidat enligt den fastställda urvalslogiken.

Detta gör att hypotesprövning och adaptiv utforskning kan existera samtidigt utan att den ena behöver känna till detaljerna i den andra.

⸻

Research Engine

AI Lab ska inte implementera själva analysalgoritmerna.

Den generiska motorn finns under:

ml/research/

Exempel på generiska analysformer är:

* tail
* interaction
* regime_comparison

En AI Lab-spec beskriver vad som ska undersökas.

Research Engine ansvarar för hur den deklarerade analysen genomförs.

Detta ger följande ansvarsfördelning:

AI Lab
  = orchestration, discovery, state och forskningsloop
Research spec
  = deklarerad forskningsfråga
Research Engine
  = generisk exekvering
Research artifacts
  = resultat, manifest och state

⸻

Spårbarhet

Varje körning av en forskningsspec får ett eget run-directory under:

data/processed/ml/research/spec_runs/

Där sparas bland annat:

<run>/
├── <spec-id>.json
└── manifest.json

Manifestet innehåller information om den körda specen och resultatets sökväg.

AI Lab läser dessutom tillbaka det persisterade resultatet efter körningen och verifierar att resultatets id motsvarar den körda specen.

Det gör forskningskedjan kontrollerbar även efter att Python-processen avslutats.

⸻

Data leakage och forskningsdisciplin

Forskningsarkitekturen ska hålla följande principer:

* test-data får inte användas för parameterurval
* låsta confirmation-specar får inte modifieras av den adaptiva loopen
* resultatbaserad kandidat-rankning ska inte ersätta den fördefinierade urvalsregeln
* experiment ska vara reproducerbara
* genererad forskning ska vara spårbar
* ett negativt resultat ska få förbli negativt

När ett experiment kräver adaptiv förfining ska förfiningen ske inom ett deklarerat och förutbestämt parameterutrymme.

⸻

Vad behöver ändras när en ny hypotes läggs till?

Normalfallet ska vara:

1. Skapa YAML-spec
2. Deklarera forskningsfrågan
3. Deklarera signaler/targets/analys
4. Opta vid behov in specen till AI Lab
5. Kör samma workflow

Inte:

1. Ändra workflow
2. Ändra adaptive_research.py
3. Lägg till ett nytt experiment-id i Python
4. Skriva ny specialkod
5. Ändra Research Engine för varje hypotes

Detta är en central arkitekturprincip för Blankdiss AI Lab.

⸻

Exempel: momentum-tail

En kontrollerad momentum-studie kan därför definieras helt deklarativt:

ml/research/specs/
└── momentum_tail_threshold_analysis.yaml

Specen deklarerar:

signal:
    price_momentum_5d
trösklar:
    20 %
    10 %
    5 %
    2.5 %
targets:
    down_5pct_5d
    down_7pct_5d
    down_10pct_5d
splits:
    validation
    test
execution:
    once

AI Lab upptäcker specen automatiskt, inkluderar dess krav i den gemensamma research-cachen och kör den genom Research Engine.

Själva forskningsfrågan behöver därmed inte byggas in i AI Lab-koden.

⸻

Sammanfattning

AI Lab är ett orkestreringslager, inte en samling hårdkodade experiment.

Den avsedda modellen är:

GitHub Actions
      │
      ▼
    AI Lab
      │
      ├── discovery
      ├── orchestration
      ├── state
      └── adaptive loop
             │
      ┌──────┴──────┐
      ▼             ▼
 YAML specs   predefined
              search space
      │             │
      └──────┬──────┘
             ▼
      Research Engine
             │
             ▼
   persisted artifacts

Målet är att Blankdiss ska kunna växa med fler forskningsfrågor utan att AI Lab eller workflowet behöver byggas om för varje ny hypotes.
