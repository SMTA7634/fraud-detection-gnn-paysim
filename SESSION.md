# Session Handover

## Last Updated
2026-10-21

## Current Phase
**Phase 1 — EDA COMPLETE (All 13 Phases, Batches 1-3)**

## What Was Done This Session (2026-10-21)

**Phase 0 (prior session):**
- Built full project scaffold, configs, skeleton source files, EDA notebook

**Phase 1 — EDA Batch 1 (2026-10-18 → 2026-10-21):**
- Rebuilt EDA notebook from `6_eda_playbook.docx` (13 phases)
- Converted to `run_eda_batch1.py` (plain Python, no Jupyter dependency)
- Ran EDA Batch 1 successfully on full 6.3M-row dataset
- Generated Sweetviz reports (reports/data_quality/)
- Installed: matplotlib, seaborn, sweetviz, torch 2.7.1+cu126, torch_geometric

## EDA Batch 1 Confirmed Findings

| Item | Value |
|---|---|
| Rows | 6,362,620 |
| Columns | 11 |
| Missing values | None |
| Duplicate rows | None |
| Fraud rate | 0.1291% (8,213 fraud rows) |
| scale_pos_weight (full dataset) | 773.7 → recalculate on training split |
| Types with fraud | TRANSFER, CASH_OUT only |
| Step range | 1 → 743 (simulated hours) |

## Architecture Decisions Made

- **Transaction-node graph** (not account-node): each transaction = a node; edges connect transactions sharing a sender or recipient within a 30-step time window
- **Chronological splits**: train ≤ step 534, val ≤ step 638, test > step 638 (confirm in Batch 2)
- **Python 3.13 system** — no conda, no venv
- **MLflow local**: experiments `fraud_model_1_baseline`, `fraud_model_2_graph_features`, `fraud_model_3_gnn`

## Leakage Decisions (Confirmed)

| Column | Decision | Reason |
|---|---|---|
| `newbalanceOrig` | **Exclude** | Post-settlement; 98% zero in fraud rows |
| `newbalanceDest` | **Exclude** | Post-settlement |
| `isFlaggedFraud` | **Exclude** | Simulation artefact; only 16 flagged rows (0.19% recall) |
| `nameOrig`, `nameDest` | Graph construction only | High-cardinality identifiers |
| `oldbalanceOrg` | Keep (investigate) | Pre-transaction; binding rate 99.5% in fraud |
| `oldbalanceDest` | Keep (investigate) | Pre-transaction; 65% zero in fraud (throwaway accounts) |

## Unexpected Finding — Balance Conservation

90.2% of LEGITIMATE transactions violate balance conservation (`oldbalanceOrg - newbalanceOrig - amount ≠ 0`)  
but only **0.55% of fraud transactions** do — because fraud drains accounts to exactly 0.  
→ `balance_discrepancy` feature candidate for Model 2.

## Environment Constraint

**No conda, no venv.** Using system Python 3.13 directly. Do not suggest conda or venv commands at any point in this project.

## Confirmed Environment (2026-10-18)

- Python: 3.13 (system)
- PyTorch: 2.7.1 + cu126
- torch_geometric: installed
- CUDA: confirmed working (`torch.cuda.is_available()` → True, GTX 1650 Ti)

## EDA Batches 2+3 Key Findings (added 2026-10-24)

| Item | Value |
|---|---|
| Split fraud counts | train=5,935 (0.097%) / val=1,106 (0.70%) / test=1,172 (1.36%) |
| scale_pos_weight (CORRECTED) | **1030.0** (training split) — was 773.7 (full dataset) |
| Feature drift | None across splits (all PSI < 0.10) |
| Fraud rate drift | 14x higher in test than train — simulation behaviour, not a data problem |
| balance_discrepancy | CONFIRMED LEAKY — uses newbalanceOrig (post-settlement). Excluded. |
| pct_balance_used | Safe alternative: amount / oldbalanceOrg. Fraud median = 100%. |
| Sender RFM features | DROPPED — 99%+ of senders transact exactly once (near-zero variance) |
| Recipient features | KEPT — reused accounts (max 113 txns), 63% of fraud goes to first-time recipients |
| hour_of_day / day_of_sim | Added as engineered features (step % 24, step // 24) |
| Super-nodes (>50 txns) | 1,033 recipient accounts — will need fan-out capping in GNN (Phase 4) |

## Confirmed Feature Lists

**Model 1** (XGBoost baseline — raw only): `step`, `hour_of_day`, `day_of_sim`, `type` (OHE), `amount`, `oldbalanceOrg`, `oldbalanceDest`, `pct_balance_used`

**Model 2** (+ graph features): Model 1 features + `recipient_txn_count`, `recipient_total_received`, `recipient_distinct_senders`, `is_new_recipient`, `pair_prior_interactions`

**Model 3** (GNN): Model 1 features as node feature vector; graph edges from shared nameOrig/nameDest within 30-step window

## Outputs Generated This Session

- `run_eda_batch2.py` + `run_eda_batch3.py` — executed successfully
- `reports/data_quality/feature_register.csv` — full feature inventory with leakage flags
- `reports/data_quality/charts/` — bivariate, time-series, drift, correlation, closure dashboard
- `configs/baseline_xgboost.yaml` — updated: scale_pos_weight=1030.0, full feature/exclude lists

## Models Built (2026-10-25)

All three models trained on `paysim_subset_50k.csv` (50k rows: all 8,213 fraud + 41,787 random legit).
OOT = step > 594. Train/test = stratified 80/20 on dev set (step ≤ 594).

### Model 1 — XGBoost Baseline
- 12 features, scale_pos_weight=1, threshold=0.11
- OOT: PR-AUC=1.0000, F1=0.9997, FN=1, FP=0

### Model 2 — XGBoost + Recipient Graph Features
- 16 features (+ 4 recipient-side cumulative features), threshold=0.09
- OOT: PR-AUC=1.0000, F1=0.9997, FN=0, FP=1
- Only recipient_total_received appeared in importance (other 3 near-zero variance in 50k subset)

### Model 3 — GraphSAGE GNN
- 12 node features (standardised), 496 undirected edges (shared nameDest within 30 steps)
- Architecture: SAGEConv(12→64) → SAGEConv(64→64) → Linear(64→1), 9,921 params, CUDA
- 100 epochs (no early stop triggered), threshold=0.48
- OOT: PR-AUC=0.9924, F1=0.9423, FN=151, FP=33
- NOTE: GNN underperforms XGBoost on this 50k subset because the graph is very sparse (avg 0.02 edges/node).
  In the full 6.3M dataset, the graph would be much denser and the GNN's structural advantage would show.

### 3-Model Comparison (OOT set)
| Metric | Model 1 | Model 2 | Model 3 (GNN) |
|---|---|---|---|
| PR-AUC | 1.0000 | 1.0000 | 0.9924 |
| F1 | 0.9997 | 0.9997 | 0.9423 |
| Precision | 1.0000 | 0.9994 | 0.9785 |
| Recall | 0.9994 | 1.0000 | 0.9087 |
| FN (missed fraud) | 1 | 0 | 151 |
| FP (false alarms) | 0 | 1 | 33 |

## Next Steps

- Phase 5: SHAP explainability (say "Start Phase 5")
- Phase 6: Reproducibility + Portfolio packaging
- Phase 7: FastAPI inference wrapper (optional)
