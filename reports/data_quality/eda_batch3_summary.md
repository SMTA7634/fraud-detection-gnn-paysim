# EDA Batch 3 Summary — Phases 10-13

## Feature Register

See: `reports/data_quality/feature_register.csv`

## Confirmed Model 1 Features

- `step`
- `hour_of_day`
- `day_of_sim`
- `type_encoded`
- `amount`
- `oldbalanceOrg`
- `oldbalanceDest`
- `pct_balance_used`

## Confirmed Model 2 Additional Features

- `recipient_total_received` (historical aggregate — see construction notes)
- `recipient_distinct_senders` (historical aggregate — see construction notes)
- `is_new_recipient` (historical aggregate — see construction notes)
- `pair_prior_interactions` (historical aggregate — see construction notes)

## Key Corrections from Batch 3

- `scale_pos_weight` corrected from 773.7 to **1030.0** (training split)
- `balance_discrepancy` confirmed LEAKY — excluded from Models 1 and 2 (current txn)
- Sender-side RFM features dropped (99%+ of senders transact exactly once)
- Recipient-side features confirmed as primary graph feature set

## Validation Strategy

- Chronological splits: train ≤ 534 | val ≤ 638 | test > 638
- Primary metric: PR-AUC
- Threshold selected on val set, applied once to test set
- Preprocessing fit on train only

## Next Step

Phase 2 — Model 1: XGBoost Baseline pipeline.
