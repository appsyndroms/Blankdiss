# Blankdiss

Blankdiss är ett forskningsprojekt som undersöker sambandet mellan
förändringar i rapporterad blankning och efterföljande aktieavkastning.

## Grundfråga

> Vad händer med en aktie efter att den rapporterade blankningen
> har ökat eller minskat?

Projektet börjar med en enkel event-study.

Vi försöker inte från början skapa en tradingmodell.

Först ska vi ta reda på om det faktiskt finns ett robust statistiskt
samband.

## Datakällor

### Finansinspektionen

Blankdiss använder Finansinspektionens blankningsregister.

Den primära serien är:

`Summa blankning %`

FI anger att denna summa omfattar rapporterade blankningspositioner
över 0,1 procent av emitterat aktiekapital.

### Prisdata

V0.1 använder Yahoo Finance via `yfinance` för prototypens historiska
dagliga priser.

Om projektet senare publiceras som en tjänst behöver datalicenserna
ses över.

## Datamodell

Rådata sparas som daterade JSONL-filer.

```text
data/
├── raw/
├── events/
└── analysis/
