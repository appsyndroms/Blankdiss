# Incremental Short-Interest Analysis

- Target: `down_10pct_5d`
- SI signal: `short_interest_change`
- Conditioning feature: `price_volatility_20d`
- Frozen cutoff: `2025-12-19`

## Conditional SI analysis

| Volatility | SI decile | N | Events | Event rate | Mean return |
|---|---:|---:|---:|---:|---:|
| low | 1 | 2839 | 46 | 0.016203 | 0.000816 |
| low | 2 | 3233 | 58 | 0.017940 | 0.001089 |
| low | 3 | 3460 | 44 | 0.012717 | 0.001314 |
| low | 4 | 3217 | 46 | 0.014299 | 0.002285 |
| low | 5 | 3275 | 48 | 0.014656 | 0.001548 |
| low | 6 | 3414 | 44 | 0.012888 | 0.001484 |
| low | 7 | 3455 | 50 | 0.014472 | 0.000054 |
| low | 8 | 3222 | 35 | 0.010863 | 0.001988 |
| low | 9 | 3471 | 50 | 0.014405 | 0.001297 |
| low | 10 | 3805 | 109 | 0.028647 | -0.000516 |
| middle | 1 | 2868 | 84 | 0.029289 | 0.001345 |
| middle | 2 | 3392 | 78 | 0.022995 | 0.001984 |
| middle | 3 | 3456 | 81 | 0.023438 | 0.001687 |
| middle | 4 | 3293 | 92 | 0.027938 | 0.000866 |
| middle | 5 | 3271 | 91 | 0.027820 | 0.001108 |
| middle | 6 | 3530 | 119 | 0.033711 | 0.001814 |
| middle | 7 | 3476 | 131 | 0.037687 | 0.000013 |
| middle | 8 | 3273 | 142 | 0.043385 | -0.000749 |
| middle | 9 | 3575 | 123 | 0.034406 | 0.002051 |
| middle | 10 | 3829 | 162 | 0.042309 | -0.000910 |
| high | 1 | 2897 | 255 | 0.088022 | 0.002989 |
| high | 2 | 3518 | 194 | 0.055145 | 0.000648 |
| high | 3 | 3527 | 186 | 0.052736 | 0.003164 |
| high | 4 | 3316 | 294 | 0.088661 | 0.002738 |
| high | 5 | 3340 | 376 | 0.112575 | -0.003935 |
| high | 6 | 3599 | 273 | 0.075854 | 0.003872 |
| high | 7 | 3464 | 248 | 0.071594 | 0.002136 |
| high | 8 | 3379 | 273 | 0.080793 | 0.000208 |
| high | 9 | 3666 | 311 | 0.084834 | 0.001720 |
| high | 10 | 3856 | 334 | 0.086618 | 0.000526 |

## Conditional SI monotonicity

| Volatility | N | Events | Event rate | Spearman SI vs target |
|---|---:|---:|---:|---:|
| low | 33391 | 530 | 0.015873 | 0.016393 |
| middle | 33963 | 1103 | 0.032477 | 0.004925 |
| high | 34562 | 2744 | 0.079394 | -0.001671 |

## Walk-forward model comparison

| Window | Evaluation | Model | N | Events | AUC | Δ AUC | Log loss | Δ Log loss | Brier | Δ Brier |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| window_1 | validation | volatility_only | 28264 | 1006 | 0.748408 | 0.000000 | 0.147046 | 0.000000 | 0.033520 | 0.000000 |
| window_1 | validation | volatility_plus_si | 28245 | 1005 | 0.748111 | -0.000297 | 0.146975 | -0.000071 | 0.033511 | -0.000008 |
| window_1 | validation | volatility_plus_si_interaction | 28245 | 1005 | 0.746575 | -0.001833 | 0.147129 | 0.000083 | 0.033528 | 0.000008 |
| window_1 | test | volatility_only | 30259 | 1245 | 0.702667 | 0.000000 | 0.166992 | 0.000000 | 0.038777 | 0.000000 |
| window_1 | test | volatility_plus_si | 30227 | 1245 | 0.702857 | 0.000189 | 0.166871 | -0.000121 | 0.038772 | -0.000005 |
| window_1 | test | volatility_plus_si_interaction | 30227 | 1245 | 0.701931 | -0.000736 | 0.166973 | -0.000020 | 0.038812 | 0.000035 |
| window_2 | validation | volatility_only | 30259 | 1245 | 0.702667 | 0.000000 | 0.166650 | 0.000000 | 0.038688 | 0.000000 |
| window_2 | validation | volatility_plus_si | 30227 | 1245 | 0.702944 | 0.000276 | 0.166489 | -0.000161 | 0.038685 | -0.000002 |
| window_2 | validation | volatility_plus_si_interaction | 30227 | 1245 | 0.702646 | -0.000021 | 0.166495 | -0.000155 | 0.038686 | -0.000001 |
| window_2 | test | volatility_only |  |  |  |  |  |  |  |  |
| window_2 | test | volatility_plus_si |  |  |  |  |  |  |  |  |
| window_2 | test | volatility_plus_si_interaction |  |  |  |  |  |  |  |  |
