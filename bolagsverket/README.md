# Bolagsverket

Ingestion of Bolagsverket open data for Blankdiss.

## Purpose

This module downloads the publicly available digital annual-report
archives from Bolagsverket.

The downloaded data is stored under:

    data/bolagsverket/raw/

A manifest is written to:

    data/bolagsverket/manifest.json

The manifest contains:

- source URL
- requested date range
- downloaded archive URLs
- local file paths
- file sizes
- SHA-256 checksums

## Usage

From the repository root:

    python -m bolagsverket.download \
      --start 2020-01-01 \
      --end 2020-06-30

Limit the number of downloaded archives:

    python -m bolagsverket.download \
      --start 2020-01-01 \
      --end 2020-06-30 \
      --max-archives 10

## Data layout

    data/bolagsverket/
    ├── manifest.json
    └── raw/
        ├── *.zip
        └── ...

The raw archives should not normally be committed to Git.

GitHub Actions stores the generated data as an artifact.

## Next step

The downloaded annual reports are iXBRL.

A separate processing stage should transform the reports into a
normalized dataset containing, where available:

- organisationsnummer
- rapportperiod
- inlämningsdatum
- omsättning
- rörelseresultat / EBIT
- resultat
- tillgångar
- skulder
- eget kapital

That processed dataset can then be joined with Blankdiss market and
FI short-position data.
