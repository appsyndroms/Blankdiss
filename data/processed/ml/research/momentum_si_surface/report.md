# Momentum × SI 10×10 Surface

- Independent test period: `(2024-12-31, 2025-12-19]`
- Rows: `47,958`
- Permutations: `1,000`
- Minimum cell N for max-statistic: `50`

## Summary

| Target | Max abs incremental pp | Surface-wide permutation p | High-high incremental pp | High-high one-sided p | High-high two-sided p |
|---|---:|---:|---:|---:|---:|
| down_5pct_5d | 10.4794 | 0.0010 | 2.8176 | 0.1119 | 0.2018 |
| down_7pct_5d | 8.3438 | 0.0010 | 4.8158 | 0.0100 | 0.0130 |
| down_10pct_5d | 7.3058 | 0.0010 | 2.7326 | 0.0340 | 0.0549 |

## What the test does

The 10×10 surface describes the relationship between cross-sectional momentum and SI.

For each momentum decile, the SI effect is measured relative to the complete momentum decile row. This removes the simple effect of momentum itself from the cell comparison.

The permutation test shuffles SI decile labels within each snapshot date. Momentum and the target remain unchanged.

The surface-wide p-value compares the observed largest absolute cell effect against the largest absolute cell effect from each permutation. This accounts for the fact that 100 cells are being searched.

The high-high p-values are a separate focused test of the extreme momentum × extreme SI corner.

Only the 2025 independent test period is used. Window_2 validation is the same 2025 calendar period and is not counted as another independent test.

## Important limitation

This is an inference test of the pre-specified momentum × SI surface. It does not select a trading rule, optimize entry or exit thresholds, or establish economic profitability.
