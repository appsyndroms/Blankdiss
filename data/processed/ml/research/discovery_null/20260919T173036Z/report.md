# Blankdiss Discovery Null Test

Generated: 20260919T173036Z

## Summary

- Candidates per run: 900
- Permutations: 1,000
- Seed: 42
- Observed max discovery score: 7.40495345744643
- Empirical p-value: 0.000999000999000999

## Observed best candidate

- Candidate: short_interest_level__5pct__distance_from_20d_high__lower__2_5pct__down_10pct_5d
- Target: down_10pct_5d
- Signal: short_interest_level
- Signal tail: 0.05
- Stress feature: distance_from_20d_high
- Stress tail: 0.025
- Stress direction: lower
- Lift: 8.304920941295745
- Return difference: -0.10003251615068505
- Valid windows: 2
- Discovery score: 7.40495345744643

## Null distribution

- Valid null scores: 1,000
- Median: 1.178607150649462
- 95th percentile: 2.10920465475833
- 99th percentile: 2.6175498877206347
- Maximum: 2.9584191323018634

## Interpretation

The null test repeats the complete discovery candidate search after randomly permuting the outcome series while keeping signals, stress features and candidate definitions unchanged.

For each permutation, the maximum discovery score across the full candidate space is retained. The empirical p-value therefore measures how often a discovery at least as strong as the observed one appears under the permutation null.

This is a multiple-testing/null-distribution diagnostic, not confirmation that a discovered candidate is a causal or tradable signal.
