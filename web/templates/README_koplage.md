# Köpläge
Sidan Köpläge visar aktuella förändringar i blankning och försöker koppla dessa observationer till historiska utfall i aktiekursen.
Den centrala idén är:
```text
Aktuell blankningsförändring
        ↓
jämförbara historiska events
        ↓
efterföljande aktieutveckling
        ↓
1d / 5d / 20d / 60d
        ↓
historisk evidens
        ↓
köpsignal
        ↓
ranking

Det innebär att en förändring i blankning inte automatiskt är en köpsignal.

Topp 10

Den övre tabellen visar Blankdiss aktuella topp 10-kandidater för köpläge.

Kolumnerna är:

Kolumn	Vad den visar	Hur den ska tolkas
Ranking	Aktiens placering i köpläget	Den framtida rankingen ska baseras på historisk evidens för hur jämförbara blankningsförändringar följts av aktieutveckling.
Aktie	Bolaget/aktien	Vilket bolag observationen gäller.
Blankning	Aktuell blankningsnivå	Hur stor andel av aktien som är blankad, uttryckt i procent.
Förändring	Förändringen i blankning	Förändringen i procentenheter. Negativt värde betyder att blankningen har minskat.
Historiska events	Antal jämförbara historiska observationer	Hur många historiska events som kan användas som underlag för bedömningen.
1d	Efterföljande avkastning efter 1 handelsdag	Vad som historiskt hänt efter jämförbara events.
5d	Efterföljande avkastning efter 5 handelsdagar	Historiskt utfall på kort sikt.
20d	Efterföljande avkastning efter 20 handelsdagar	Historiskt utfall på ungefär en handelsmånad.
60d	Efterföljande avkastning efter 60 handelsdagar	Historiskt utfall på ungefär tre månader.
Köpsignal	Sammanvägd historisk signal	Ska på sikt beskriva hur starkt det historiska underlaget stödjer ett positivt framtida utfall.

Vad rankingen betyder

Rankingen ska inte betyda:

“Störst minskning av blankningen = bäst aktie.”

Det är bara en observerad egenskap hos ett event.

Den långsiktiga tanken är i stället:

“Hur har aktier historiskt utvecklats efter liknande blankningsförändringar?”

Exempel:

Aktuell observation
Blankning:     8,2 %
Förändring:   -1,4 pp

Blankningen har då minskat med 1,4 procentenheter, exempelvis från 9,6 % till 8,2 %.

Nästa steg är att hitta historiska observationer som liknar denna.

Aktuell:
-1,4 pp
8,2 % blankning
        ↓
Historiska liknande events
Event 1 → +2,1 % efter 5d
Event 2 → +4,3 % efter 5d
Event 3 → -0,8 % efter 5d
Event 4 → +1,7 % efter 5d
...

Därefter kan utfallet på olika tidshorisonter analyseras.

Historisk evidens

De fyra avkastningshorisonterna är viktiga eftersom en blankningsförändring kan följas av olika beteenden på olika tidshorisonter.

1d   → mycket kortsiktigt utfall
5d   → kortsiktigt utfall
20d  → medellångt utfall
60d  → längre utfall

Det kan exempelvis vara möjligt att ett historiskt mönster är positivt på 20 dagar men inte på 1 dag.

Därför bör inte en enskild tidshorisont ensam definiera köpsignalen.

Historiska events

Historiska events anger hur mycket historiskt underlag som finns för en aktuell observation.

Det är viktigt eftersom två aktier annars kan se likadana ut trots att den ena har hundratals jämförbara historiska observationer medan den andra bara har ett fåtal.

Få observationer innebär större osäkerhet.

När data saknas eller ännu inte har hunnit mogna visas:

—

Webben ska inte fylla sådana värden med påhittade resultat.

Köpsignal

Köpsignal är tänkt att bli en sammanfattning av den historiska evidensen.

Den ska inte vara en subjektiv bedömning.

Den ska bygga på observerade historiska utfall och exempelvis kunna ta hänsyn till:

* antal jämförbara events,
* riktning på efterföljande avkastning,
* genomsnittlig avkastning,
* median,
* andel positiva utfall,
* 1d / 5d / 20d / 60d,
* hur lik den aktuella observationen är de historiska eventsen.

Exakt metod för signalen ska bestämmas av forskningslogiken och inte av presentationen.

Aktuella observationer

Den nedre tabellen visar de aktuella observationerna utan ranking.

Kolumnerna är:

Kolumn	Vad den visar
Aktie	Bolaget/aktien
Blankning	Aktuell blankningsnivå
Förändring	Förändring i procentenheter
1d	Efterföljande avkastning efter 1 handelsdag
5d	Efterföljande avkastning efter 5 handelsdagar
20d	Efterföljande avkastning efter 20 handelsdagar
60d	Efterföljande avkastning efter 60 handelsdagar

Denna tabell är observationslagret.

Den ska inte blandas ihop med den analyserade Topp 10-rankingen.

Förändring i procentenheter

Det är viktigt att skilja mellan procent och procentenheter.

Exempel:

Blankning före: 9,6 %
Blankning efter: 8,2 %
Förändring: −1,4 pp

Det betyder att blankningsnivån har minskat med 1,4 procentenheter.

Det betyder inte att aktiekursen har minskat med 1,4 %.

Vad sidan försöker hitta

Kärnan i Blankdiss Köpläge är därför:

                    AKTUELLT
                       │
                       ▼
             Blankningsförändring
                       │
                       ▼
              Historiska events
                       │
                       ▼
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
       1d             5d             20d
        │              │              │
        └──────────────┼──────────────┘
                       ▼
                      60d
                       │
                       ▼
             Historisk evidens
                       │
                       ▼
                 Köpsignal
                       │
                       ▼
                   Ranking

Det är alltså sambandet mellan blankningsförändring och efterföljande aktieutveckling som är det centrala forskningsproblemet.

Nuvarande dataläge

I början kan data saknas för delar av analysen.

Exempelvis kan en ny observation ha:

1d   —
5d   —
20d  —
60d  —

Det beror på att tillräckligt många handelsdagar ännu inte har passerat.

På samma sätt kan det saknas tillräckligt många historiska events för att skapa en robust jämförelse.

Detta är förväntat i en prospektiv analys.

Därför ska Blankdiss skilja mellan:

1. observationen som finns nu,
2. historiska events som redan finns,
3. utfall som har hunnit mogna,
4. signaler som faktiskt har tillräckligt historiskt underlag.

Viktig princip

En stor minskning av blankningen är intressant, men den är inte i sig ett bevis på framtida positiv aktieutveckling.

Blankdiss ska i stället försöka mäta detta empiriskt:

Liknande blankningsförändring
        ↓
Vad hände därefter historiskt?
