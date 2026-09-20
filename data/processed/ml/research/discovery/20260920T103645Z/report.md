# Blankdiss Discovery

Generated: 20260920T103645Z

## Summary

- Candidates: 16
- OOS results: 32
- Pooled candidates: 16
- Findings: 0

## Findings

No candidates passed the configured discovery thresholds.

## Candidate Space

| Dimension | Values |
|---|---|
| Targets | down_10pct_5d |
| FI signals | short_interest_change |
| Stress features | price_volatility_20d |
| Tail fractions | 0.01, 0.025, 0.05, 0.1 |

## Walk-forward Windows

- window_1: train <= 2023-12-31, validation <= 2024-12-31, test <= 2025-12-31
- window_2: train <= 2024-12-31, validation <= 2025-12-31, test <= 2026-12-31

## Interpretation

Discovery is exploratory. Findings are candidates for subsequent diagnostics, not confirmed trading signals.
