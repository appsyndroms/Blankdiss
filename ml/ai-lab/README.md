# Blankdiss AI Lab

AI Lab är Blankdiss mekanism för kontrollerad, iterativ forskning.

Syftet är inte att hitta ett positivt resultat så snabbt som möjligt.

Syftet är att låta tidigare experiment ge information om vilket forskningssteg som ska tas härnäst, samtidigt som forskningen förblir reproducerbar, spårbar och skyddad mot data leakage och resultatjakt.

## Grundprincip

AI Lab optimerar:

    idé → information

inte:

    idé → mer kod

Ett negativt resultat är därför ett giltigt forskningsresultat.

Om en hypotes inte fungerar ska AI Lab inte försöka vrida parametrarna tills ett positivt resultat uppstår. När ett fördefinierat parameterutrymme är uttömt ska systemet i stället kunna konstatera att den aktuella experimentfamiljen inte längre ger tillräcklig information och gå vidare till en ny forskningsfråga eller experimentfamilj.

---

## Forskningsloopen

AI Lab arbetar i följande steg:

```text
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
