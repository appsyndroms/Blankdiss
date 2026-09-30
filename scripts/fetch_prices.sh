#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"

cd "$ROOT"

echo "=========================================="
echo "FETCH PRICES"
echo "=========================================="

SWEDISH_WEEKDAY="$(TZ=Europe/Stockholm date '+%u')"
SWEDISH_TIME="$(TZ=Europe/Stockholm date '+%H:%M')"

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

echo
echo "=========================================="
echo "FÖRBERED PRICE/MAPPING-COMMIT"
echo "=========================================="

git config user.name "github-actions[bot]"
git config user.email "41898282+github-actions[bot]@users.noreply.github.com"

git add \
  prices/known_yahoo_symbols.jsonl \
  data/analysis/instrument_map.json

echo
echo "Yahoo- och instrumentmappningar före commit:"
git status --short

if git diff --cached --quiet; then
    echo
    echo "Inga Yahoo-mappningar eller instrumentmappningar att committa."
    exit 0
fi

echo
echo "Filer som kommer att committas:"
git diff --cached --name-status

echo
echo "Mappingförändringar:"
git diff --cached --stat -- \
  prices/known_yahoo_symbols.jsonl \
  data/analysis/instrument_map.json

git commit \
  -m "Update Yahoo symbol mappings"

echo
echo "=========================================="
echo "MAPPNINGAR COMMITTADE"
echo "=========================================="

git status --short

echo
echo "Pussar price/mapping-data..."

bash scripts/git_push_with_retry.sh

echo
echo "=========================================="
echo "PRICES KLAR"
echo "=========================================="

git status --short
