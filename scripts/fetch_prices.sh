#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"

cd "$ROOT"

FORCE=false

if [[ "${1:-}" == "--force" ]]; then
    FORCE=true
fi

if [[ "${FORCE_SYNC:-false}" == "true" ]]; then
    FORCE=true
fi

echo "=========================================="
echo "FETCH PRICES"
echo "=========================================="

SWEDISH_WEEKDAY="$(TZ=Europe/Stockholm date '+%u')"
SWEDISH_TIME="$(TZ=Europe/Stockholm date '+%H:%M')"

if [ "$FORCE" = true ]; then
    echo
    echo "FORCE SYNC AKTIVERAD."
    echo "Normal tids- och helgspärr kringgås."
    echo "Svensk tid: $SWEDISH_TIME"
else
    if [ "$SWEDISH_WEEKDAY" -ge 6 ]; then
        echo
        echo "Yahoo-hämtning hoppas över."
        echo "Orsak: hämtning är endast tillåten måndag–fredag kl. 17:00–23:59 svensk tid."
        echo "Idag är lördag eller söndag enligt svensk tid."
        echo "Svensk tid: $SWEDISH_TIME"
        exit 0
    fi

    if [[ "$SWEDISH_TIME" < "17:00" ]]; then
        echo
        echo "Yahoo-hämtning hoppas över."
        echo "Orsak: hämtning är endast tillåten måndag–fredag kl. 17:00–23:59 svensk tid."
        echo "Klockan är före 17:00 svensk tid."
        echo "Svensk tid: $SWEDISH_TIME"
        exit 0
    fi

    echo
    echo "Yahoo-hämtning tillåten."
    echo "Tillåtet tidsfönster: måndag–fredag kl. 17:00–23:59 svensk tid."
    echo "Svensk tid: $SWEDISH_TIME"
fi

echo
echo "Hämtar Yahoo-priser från 2022-01-01..."

python -u -m prices --start 2022-01-01

echo
echo "=========================================="
echo "KONTROLLERA PRICE/MAPPING-DATA"
echo "=========================================="

if [ ! -f prices/known_yahoo_symbols.jsonl ]; then
    echo "Ingen prices/known_yahoo_symbols.jsonl hittades."
    exit 1
fi

if [ ! -f data/analysis/instrument_map.json ]; then
    echo "Ingen data/analysis/instrument_map.json hittades."
    exit 1
fi

if ! compgen -G "data/raw/prices/*.jsonl" > /dev/null; then
    echo "Inga prisfiler hittades i data/raw/prices/."
    exit 1
fi

echo
echo "=========================================="
echo "FÖRBERED PRICE/MAPPING/IDENTITY-COMMIT"
echo "=========================================="

git config user.name "github-actions[bot]"
git config user.email "41898282+github-actions[bot]@users.noreply.github.com"

git add \
  data/raw/prices/*.jsonl \
  prices/known_yahoo_symbols.jsonl \
  data/analysis/instrument_map.json \
  shared/data/instrument_aliases.jsonl \
  shared/data/instrument_entities.jsonl

echo
echo "Price-, Yahoo-, instrument- och identitymappningar före commit:"
git status --short

if git diff --cached --quiet; then
    echo
    echo "Inga price-, mapping- eller identityförändringar att committa."
    exit 0
fi

echo
echo "Filer som kommer att committas:"
git diff --cached --name-status

echo
echo "Förändringsstatistik:"
git diff --cached --stat -- \
  data/raw/prices \
  prices/known_yahoo_symbols.jsonl \
  data/analysis/instrument_map.json \
  shared/data/instrument_aliases.jsonl \
  shared/data/instrument_entities.jsonl

git commit \
  -m "Update price data and instrument identity"

echo
echo "=========================================="
echo "PRICE-DATA, MAPPNINGAR OCH IDENTITY COMMITTADE"
echo "=========================================="

git status --short

echo
echo "Pushar price/mapping/identity-data..."

bash scripts/git_push_with_retry.sh

echo
echo "=========================================="
echo "PRICES KLAR"
echo "=========================================="

git status --short
