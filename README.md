# Graph-Based Fraud Detection on PaySim

Binary fraud classification on the PaySim synthetic mobile-money dataset,
progressing from an XGBoost baseline to Graph Neural Networks. Built as a
learning project and interview portfolio piece.

---

## Problem

Detect fraudulent mobile-money transactions **before they settle** — the model
scores each transaction at submission time, before the money moves.

**Dataset:** PaySim — 6.36M synthetic transactions calibrated on real M-Pesa data.
Only `TRANSFER` and `CASH_OUT` types contain fraud (0.13% fraud rate overall).

**Working subset:** 50k rows for Models 1–2, 250k rows for Models 3–4.
All 8,213 fraud rows are kept; legit rows are randomly sampled.

**Primary metric:** PR-AUC (not accuracy — the dataset is highly imbalanced).

---

## Models

| Model | Type | Features | Graph | OOT PR-AUC | OOT F1 |
|---|---|---|---|---|---|
| **Model 1** | XGBoost baseline | 12 (raw + engineered) | None | 1.0000 | 0.9997 |
| **Model 2** | XGBoost + graph features | 16 (+4 recipient history) | Flat aggregation | 1.0000 | 0.9997 |
| **Model 3** | GraphSAGE (transaction-node) | 12 (standardised) | 13,400 edges | 0.9706 | 0.8868 |
| **Model 4** | GraphSAGE (account-node) | 10 (aggregated per account) | 250,000 edges | 0.8051 | 0.7457 |

> Models 1–3 score **transactions**. Model 4 scores **accounts**.

---

## Key Design Decisions

**Temporal OOT split** — `step > 594` (last 20% of simulation time) is the
out-of-time holdout. Never a random split — fraud patterns evolve over time.

**No data leakage** — `newbalanceOrig` and `newbalanceDest` are post-settlement
columns and are excluded. `pct_balance_used = amount / oldbalanceOrg` is the
safe pre-settlement drain signal.

**`scale_pos_weight` / `pos_weight`** — Set to 1 for 50k subset (16% fraud rate
is mild). Set to 36 for 250k Model 3 and 68 for Model 4 to handle lower fraud rates.

**Graph design:**
- Model 3 (transaction-node): edges connect transactions sharing `nameDest` within 30 steps
- Model 4 (account-node): each transaction is an edge between the sender and recipient account

---

## Project Structure

```
fraud-gnn-project/
├── data/
│   └── shared/raw/          paysim_subset_50k.csv, paysim_subset_250k.csv
├── artifacts/
│   ├── model_1_baseline/    xgb_model.json, metrics_report.txt, charts
│   ├── model_2_graph/       xgb_model.json, charts
│   ├── model_3_gnn/         gnn_model.pt, charts
│   ├── model_4_account_gnn/ gnn_model.pt, charts
│   ├── shap/                SHAP explanation charts
│   └── final_report/        model_comparison.csv, model_comparison_chart.png
├── docs/
│   ├── graph_structure.html         Model 3 node/edge diagram
│   └── account_graph_structure.html Model 4 node/edge diagram
├── reports/
│   └── data_quality/        EDA charts, feature_register.csv
├── configs/
│   └── baseline_xgboost.yaml
├── create_subset.py
├── run_eda_batch1.py        EDA phases 1–4
├── run_eda_batch2.py        EDA phases 5–9
├── run_eda_batch3.py        EDA phases 10–13
├── run_model1.py            XGBoost baseline
├── run_model2.py            XGBoost + graph features
├── run_model3.py            GraphSAGE transaction-node GNN
├── run_model4.py            GraphSAGE account-node GNN
├── run_shap.py              SHAP explainability for Models 1 & 2
├── run_final_report.py      Final comparison chart and CSV
├── PROJECT.md               Architecture decisions and data flow diagram
└── SESSION.md               Session handover notes
```

---

## How to Run

```bash
# 1. Create subset from full PaySim data
python create_subset.py

# 2. EDA
python run_eda_batch1.py
python run_eda_batch2.py
python run_eda_batch3.py

# 3. Models
python run_model1.py
python run_model2.py
python run_model3.py      # uses 250k subset
python run_model4.py      # uses 250k subset

# 4. Explainability + report
python run_shap.py
python run_final_report.py
```

**Environment:** Python 3.13 (system), no conda/venv.

```bash
pip install xgboost scikit-learn pandas numpy matplotlib seaborn sweetviz mlflow shap
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install torch_geometric
```

---

## SHAP Key Findings

1. **`pct_balance_used`** is the dominant fraud signal — draining an account to zero
   is the clearest indicator. Fraud median = 100% drain.
2. **`type_TRANSFER`** is the second strongest — only TRANSFER and CASH_OUT contain fraud.
3. **`oldbalanceDest = 0`** (empty recipient account) signals a newly created mule account.
4. Graph features in Model 2 added very small SHAP values on the 250k subset — the graph
   is too sparse. On the full 6.3M dataset these would be far more impactful.

---


- Why the GNN underperforms XGBoost here: the fraud signal is in individual features, not network
  structure. The graph is too sparse (0.11 edges/node) for message passing to dominate.
- Model 3 vs Model 4: different business questions — transaction blocking vs account risk scoring.
  Both are needed in a real fraud system.
- Leakage in Model 4: node features aggregate all transactions including post-OOT ones. Production
  fix: event-time feature store computing features up to the scoring timestamp only.
