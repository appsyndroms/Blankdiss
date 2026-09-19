# Blankdiss Discovery Null Test

Generated: 20260919T192154Z

## Summary

- Candidates per run: 900
- Permutations: 1,000
- Seed: 42
- Observed max discovery score: 7.543177723199898
- Empirical p-value: 0.000999000999000999

## Observed best candidate

- Candidate: short_interest_change__1pct__price_volatility_20d__upper__2_5pct__down_10pct_5d
- Target: down_10pct_5d
- Signal: short_interest_change
- Signal tail: 0.01
- Stress feature: price_volatility_20d
- Stress tail: 0.025
- Stress direction: upper
- Lift: 8.454548629299808
- Return difference: -0.08862909390009042
- Valid windows: 1
- Discovery score: 7.543177723199898

## Null distribution

- Valid null scores: 1,000
- Median: 1.907072145162182
- 95th percentile: 3.2481044786181874
- 99th percentile: 4.009085136525066
- Maximum: 5.674320275519441

## Interpretation

The null test repeats the complete discovery candidate search after randomly permuting the outcome series while keeping signals, stress features and candidate definitions unchanged.

For each permutation, the maximum discovery score across the full candidate space is retained. The empirical p-value therefore measures how often a discovery at least as strong as the observed one appears under the permutation null.

This is a multiple-testing/null-distribution diagnostic, not confirmation that a discovered candidate is a causal or tradable signal.
