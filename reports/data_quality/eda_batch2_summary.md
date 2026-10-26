# EDA Batch 2 Summary — Phases 5-9

## Key Findings

- Fraud only in TRANSFER and CASH_OUT (confirmed)  
- Fraud per split -> train: 5,935 | val: 1,106 | test: 1,172  
- Total unique accounts: 9,073,900  
- Max txns by one sender: 3  | Max txns to one recipient: 113  
- balance_discrepancy: ~0 in fraud (drain-to-zero), non-zero in 90%+ legit -> top feature candidate  
- Fraud rate by split: train / val / test (see output)  
- Super-node destinations (>50 txns): 1,033  
- Graph feature candidates identified: 8  

## Split Boundary Recommendation

- Train: step 1-534  |  fraud: 5,935
- Val  : step 535-638  |  fraud: 1,106
- Test : step 639-743  |  fraud: 1,172

## Graph Feature Candidates (for Model 2)

- `sender_txn_count`: Number of prior transactions by nameOrig (frequency)  
- `sender_total_sent`: Total amount sent by nameOrig historically (monetary)  
- `sender_distinct_recipients`: Number of distinct nameDest accounts used by sender (diversity)  
- `recipient_txn_count`: Number of prior transactions received by nameDest (frequency)  
- `pair_prior_interactions`: Prior transactions between this exact nameOrig-nameDest pair  
- `time_since_last_txn_orig`: Steps since nameOrigs previous transaction (recency)  
- `pct_balance_used`: amount / oldbalanceOrg — drain rate (domain feature)  
- `balance_discrepancy`: Abs(oldbalanceOrg - newbalanceOrig - amount) — drain anomaly  

## Confirmed Feature List for Model 1 (XGBoost Baseline)

**Include:** `step`, `type` (one-hot), `amount` (log1p), `oldbalanceOrg`, `oldbalanceDest`  
**Exclude:** `nameOrig`, `nameDest`, `isFlaggedFraud`, `newbalanceOrig`, `newbalanceDest`  

## Next Step

EDA Batch 3 (Phases 10-13): RFM feasibility, feature register, validation strategy, handover.
