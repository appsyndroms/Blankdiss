# Blankdiss Frozen Hypothesis

## Hypothesis

Är extremt hög förändring i short interest kombinerat med extremt hög 20-dagars prisvolatilitet associerat med förhöjd risk för minst 10 procents nedgång inom 5 handelsdagar i en helt separat OOS-period?

## Frozen candidate

- Candidate: `short_interest_change__1pct__price_volatility_20d__upper__2_5pct__down_10pct_5d`
- Target: `down_10pct_5d`
- Signal: `short_interest_change`
- Signal tail: `0.01`
- Stress feature: `price_volatility_20d`
- Stress tail: `0.025`
- Stress direction: `upper`

## Discovery boundary

- Discovery run: `20260919T192154Z`
- Discovery end: `2025-12-19`

## OOS evaluation

- Start: `2025-12-20`
- End: `2026-09-18`
- Rows: 26
- Events: 4
- Event rate: 0.15384615384615385
- Baseline event rate: 0.03759398496240601
- Lift: 4.092307692307693
- Mean return: -0.025010095201932536
- Rest mean return: 0.0015005140605138603
- Return difference: -0.026510609262446395

## Frozen null test

- Metric: `lift`
- Permutations: 1,000
- Valid permutations: 1,000
- Seed: 42
- Observed: 4.092307692307693
- Null mean: 0.9923846153846154
- Null std: 0.9903786228347458
- Null 95th percentile: 3.0692307692307694
- Null 99th percentile: 4.092307692307693
- Empirical p-value: 0.015984015984015984

## Method

The candidate was frozen before the OOS evaluation. The null test keeps the candidate, signal and stress definition fixed and permutes only the outcome within the OOS period.

No candidate search or parameter selection is performed during the frozen null test.
