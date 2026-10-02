name: Felsökning - Yahoo Price

on:
  workflow_dispatch:

jobs:
  debug-fingerprint-price:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt

      - name: Run Fingerprint price diagnostic
        run: |
          python felsokning/debug_yahoo_price.py
