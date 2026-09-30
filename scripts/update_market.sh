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
echo "UPDATE OMXSPI"
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
        echo "OMXSPI-uppdatering hoppas över."
        echo "Orsak: hämtning är endast tillåten måndag–fredag kl. 17:00–23:59 svensk tid."
        echo "Idag är lördag eller söndag enligt svensk tid."
        echo "Svensk tid: $SWEDISH_TIME"
        exit 0
    fi

    if [[ "$SWEDISH_TIME" < "17:00" ]]; then
        echo
        echo "OMXSPI-uppdatering hoppas över."
        echo "Orsak: hämtning är endast tillåten måndag–fredag kl. 17:00–23:59 svensk tid."
        echo "Klockan är före 17:00 svensk tid."
        echo "Svensk tid: $SWEDISH_TIME"
        exit 0
    fi

    echo
    echo "OMXSPI-uppdatering tillåten."
    echo "Tillåtet tidsfönster: måndag–fredag kl. 17:00–23:59 svensk tid."
    echo "Svensk tid: $SWEDISH_TIME"
fi

echo
echo "Uppdaterar OMXSPI..."

python -u -m analysis.update_market

echo
echo "=========================================="
echo "OMXSPI KLAR"
echo "=========================================="
