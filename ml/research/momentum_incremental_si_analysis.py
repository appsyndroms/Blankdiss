id: momentum_incremental_si
question: >
  Tillför förändring i short interest information om 5-dagars
  downside-risk utöver 20-dagars momentum, och tillför
  interaktionen mellan momentum och short-interest-förändring
  ytterligare information?
mode: scan
signals:
  - name: price_momentum_20d
    direction: upper
    bins:
      - 0.20
  - name: short_interest_change
    direction: upper
    bins:
      - 0.20
targets:
  - down_5pct_5d
  - down_7pct_5d
  - down_10pct_5d
analysis:
  type: incremental_model
  bootstrap: false
windows:
  - window_1
  - window_2
splits:
  - validation
  - test
metadata:
  stage: hypothesis_test
  purpose: momentum_incremental_si
  baseline_model: M0
  incremental_model: M1
  interaction_model: M2
  baseline_signal: price_momentum_20d
  incremental_signal: short_interest_change
  interaction: price_momentum_20d_x_short_interest_change
  evaluation:
    - auc
    - log_loss
    - brier
  comparison:
    - delta_auc_vs_M0
    - delta_log_loss_vs_M0
    - delta_brier_vs_M0
