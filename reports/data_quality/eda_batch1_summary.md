# EDA Batch 1 Summary — Phases 1-4

**Dataset:** PaySim_kaggle_raw_data.csv  
**Rows:** 6,362,620  |  **Columns:** 11  
**Missing values:** None  
**Duplicate rows:** 0  

## Key Findings
- Fraud rate: 0.1291%  |  scale_pos_weight (full dataset): 773.7  
- Class 0: 6,354,407  |  Class 1: 8,213  
- Sweetviz reports generated: sweetviz_overall_report.html, sweetviz_class1_vs_class0_report.html  
- Duplicate rows: 0  |  Duplicate signatures: 0  
- Sender balance violations: 2,499,086 (90.2%)  
- isFlaggedFraud: exclude from all models (simulation artefact)  

## Leakage Risk Register
| Column | Risk | Decision |
|---|---|---|
| `newbalanceOrig` | Post-settlement balance | **Exclude** |
| `newbalanceDest` | Post-settlement balance | **Exclude** |
| `isFlaggedFraud` | Simulation artefact — near-perfect proxy for target | **Exclude** |
| `nameOrig` | High-cardinality identifier | Graph construction only |
| `nameDest` | High-cardinality identifier | Graph construction only |
| `oldbalanceOrg` | Pre-transaction — investigate further | **Investigate in Batch 2** |
| `oldbalanceDest` | Pre-transaction — zero-heavy for merchants | **Investigate in Batch 2** |

## Confirmed for Phase 2 (Model 1 preprocessing)
- Exclude: `newbalanceOrig`, `newbalanceDest`, `isFlaggedFraud`, `nameOrig`, `nameDest`
- Apply log1p to `amount`
- Use `scale_pos_weight` from training split (not full dataset)
- Chronological split: train ≤ step 534 | val ≤ 638 | test > 638  (confirm in Batch 2)

## Next Step
Run EDA Batch 2 (Phases 5-9): univariate, bivariate, time-series, drift, correlation.
