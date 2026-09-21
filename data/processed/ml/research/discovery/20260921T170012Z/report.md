# Blankdiss Discovery

Generated: 20260921T170012Z

## Summary

- Candidates: 900
- OOS results: 1,800
- Pooled candidates: 900
- Findings: 0

## Findings

No candidates passed the configured discovery thresholds.

## Candidate Space

| Dimension | Values |
|---|---|
| Targets | down_5pct_5d, down_7pct_5d, down_10pct_5d |
| FI signals | short_interest_change, short_interest_level |
| Stress features | price_volatility_20d, price_momentum_5d, price_momentum_20d, price_momentum_60d, distance_from_20d_high, distance_from_60d_high |
| Tail fractions | 0.2, 0.1, 0.05, 0.025, 0.01 |

## Walk-forward Windows

- window_1: train <= 2023-12-31, validation <= 2024-12-31, test <= 2025-12-31
- window_2: train <= 2024-12-31, validation <= 2025-12-31, test <= 2026-12-31

## Interpretation

Discovery is exploratory. Findings are candidates for subsequent diagnostics, not confirmed trading signals.
