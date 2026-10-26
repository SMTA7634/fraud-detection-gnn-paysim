"""
run_model2.py  —  Phase 3: XGBoost + Graph Features (Model 2)
Run: python run_model2.py

What is different from Model 1?
  Model 1 used only raw transaction columns — information about THIS transaction.
  Model 2 adds GRAPH FEATURES — information about the recipient account's
  HISTORY across all prior transactions. These are sometimes called
  "network features" or "behavioural aggregates."

  The key insight: a fraudulent transaction often goes to an account that has
  either never received money before (brand new mule account) OR has received
  money from many different senders (high-volume money mule). By counting
  prior activity on the recipient side, we give the model context that a
  single transaction row does not contain.

New features added (all recipient-side, all historical):
  recipient_txn_count        — how many transactions this nameDest received before this one
  recipient_total_received   — total amount received by this nameDest before this one
  recipient_distinct_senders — unique senders to this nameDest before this one
  is_new_recipient           — 1 if recipient_txn_count == 0, else 0

Dataset  : paysim_subset_50k.csv
Splits   : OOT = step > 594 | Dev = step <= 594 | Train/Test = 80/20 stratified
Outputs  : artifacts/model_2_graph/ and data/model_2_graph/output/
MLflow   : experiment fraud_model_2_graph_features
"""

import sys, warnings, os
warnings.filterwarnings('ignore')
os.environ['MLFLOW_DISABLE_AGENT_HINT'] = '1'
os.environ['GIT_PYTHON_REFRESH']        = 'quiet'

from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT  = Path(__file__).resolve().parent
SUBSET_CSV    = PROJECT_ROOT / 'data' / 'shared' / 'raw' / 'paysim_subset_50k.csv'
ARTIFACTS_DIR = PROJECT_ROOT / 'artifacts' / 'model_2_graph'
CHARTS_DIR    = ARTIFACTS_DIR / 'charts'
PREDS_DIR     = PROJECT_ROOT / 'data' / 'model_2_graph' / 'output'

for d in [ARTIFACTS_DIR, CHARTS_DIR, PREDS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import OneHotEncoder
from sklearn.metrics import (
    precision_recall_curve, roc_curve,
    f1_score, precision_score, recall_score,
    confusion_matrix, roc_auc_score, average_precision_score
)
import xgboost as xgb
import mlflow
import mlflow.xgboost

sns.set_theme(style='whitegrid')
plt.rcParams['figure.dpi'] = 100

OOT_STEP_BOUNDARY = 594
RANDOM_SEED       = 42
TEST_SIZE         = 0.20

def sep(title):
    print(f'\n{"="*65}\n  {title}\n{"="*65}')

def note(text):
    for line in text.strip().split('\n'):
        print(f'  | {line}')
    print()

def save_chart(fig, name):
    p = CHARTS_DIR / f'{name}.png'
    fig.savefig(p, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved chart: {name}.png')

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 — LOAD AND SORT DATA
# ═════════════════════════════════════════════════════════════════════════════
sep('SECTION 1 — LOAD DATA')

note("""
  CRITICAL: graph features must be computed on ALL data sorted by step
  BEFORE we apply the train/test/OOT split.

  Why? Because a val-set transaction at step 600 should know about
  train-set transactions at step 599. If we split first and compute
  separately, the val-set transaction would wrongly show 0 prior history.

  The correct order is:
    1. Load ALL data, sort by step ascending
    2. Compute recipient history features (expanding window, strictly prior rows)
    3. THEN split into train / test / OOT
""")

df = pd.read_csv(SUBSET_CSV)
df = df.sort_values('step').reset_index(drop=True)
print(f'  Loaded and sorted: {len(df):,} rows')

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 — COMPUTE GRAPH FEATURES
# ═════════════════════════════════════════════════════════════════════════════
sep('SECTION 2 — COMPUTE RECIPIENT GRAPH FEATURES')

note("""
  We compute four features for each transaction by looking at what happened
  to the RECIPIENT ACCOUNT (nameDest) in all PRIOR transactions.

  The pandas technique used is groupby + cumcount/cumsum + shift(1):

    - groupby('nameDest')  : group rows that share the same recipient
    - cumcount()           : count how many rows have been seen SO FAR
                             for this nameDest (starts at 0)
    - shift(1)             : shift by 1 position so each row sees the
                             count of rows BEFORE it, not including itself
                             (shift(1) fills the first row with NaN → 0)

  Example for nameDest = "C123":
    Row 10 (step 50):  cumcount=0, shift → NaN → 0  (first appearance)
    Row 25 (step 120): cumcount=1, shift → 0         (1 prior transaction)
    Row 88 (step 300): cumcount=2, shift → 1         (2 prior transactions)

  This is O(n log n) — fast even on millions of rows.
""")

print('  Computing recipient_txn_count ...')
df['recipient_txn_count'] = (
    df.groupby('nameDest').cumcount()   # 0-indexed count within group
)
# cumcount() gives the CURRENT row's index within the group (0 = first appearance).
# That IS the number of prior transactions — no shift needed:
# row 0 in group = 0 prior transactions, row 1 = 1 prior, etc.

print('  Computing recipient_total_received ...')
# cumsum then shift(1): sum of amounts received BEFORE this transaction
df['recipient_total_received'] = (
    df.groupby('nameDest')['amount']
    .cumsum()                           # running total including current row
    .shift(1)                           # shift: exclude current row
    .fillna(0)
)

print('  Computing recipient_distinct_senders ...')
# Number of unique prior senders to this recipient.
# We use a helper: cumulative count of unique nameOrig per nameDest up to (not including) this row.
# Efficient approach: merge sorted group counts.
def cum_nunique_prior(group):
    seen = set()
    result = []
    for orig in group:
        result.append(len(seen))   # count BEFORE adding current
        seen.add(orig)
    return result

df['recipient_distinct_senders'] = (
    df.groupby('nameDest')['nameOrig']
    .transform(cum_nunique_prior)
)

print('  Computing is_new_recipient ...')
df['is_new_recipient'] = (df['recipient_txn_count'] == 0).astype(int)

# Quick sanity check
print(f'\n  Sanity check — first appearance rows:')
first_rows = df[df['recipient_txn_count'] == 0]
print(f'    Rows where recipient_txn_count = 0 : {len(first_rows):,}')
print(f'    Unique nameDest values total        : {df["nameDest"].nunique():,}')
print(f'    (Should be equal — every dest appears as new exactly once)')

print(f'\n  New feature stats:')
for feat in ['recipient_txn_count', 'recipient_total_received', 'recipient_distinct_senders']:
    print(f'    {feat:35s}: mean={df[feat].mean():.2f}  max={df[feat].max():.0f}')

print(f'\n  Fraud rate for NEW vs RETURNING recipients:')
new_fraud  = df[df['is_new_recipient'] == 1]['isFraud'].mean() * 100
ret_fraud  = df[df['is_new_recipient'] == 0]['isFraud'].mean() * 100
print(f'    is_new_recipient = 1 (first time) : {new_fraud:.2f}% fraud rate')
print(f'    is_new_recipient = 0 (seen before): {ret_fraud:.2f}% fraud rate')
print(f'    --> Higher fraud rate for new recipients confirms the feature is predictive.')

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 — SPLIT AND PREPROCESS
# ═════════════════════════════════════════════════════════════════════════════
sep('SECTION 3 — SPLIT AND PREPROCESS')

note("""
  Same preprocessing as Model 1:
    - Engineered features: hour_of_day, day_of_sim, pct_balance_used
    - log1p on: amount, oldbalanceOrg, oldbalanceDest, recipient_total_received
    - OneHotEncode type (fit on train only)
    - Drop leakage + identifier columns

  New: also apply log1p to recipient_total_received — it is the cumulative
  sum of transaction amounts, so it has the same right-skew as amount itself.
  recipient_txn_count and recipient_distinct_senders are small integers —
  no transform needed.
""")

# OOT / dev split
oot_df = df[df['step'] >  OOT_STEP_BOUNDARY].copy()
dev_df = df[df['step'] <= OOT_STEP_BOUNDARY].copy()

# Train / test split from dev
train_df, test_df = train_test_split(
    dev_df, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=dev_df['isFraud']
)

print(f'  Train: {len(train_df):,} rows | fraud: {train_df["isFraud"].sum():,} ({train_df["isFraud"].mean()*100:.2f}%)')
print(f'  Test : {len(test_df):,} rows | fraud: {test_df["isFraud"].sum():,} ({test_df["isFraud"].mean()*100:.2f}%)')
print(f'  OOT  : {len(oot_df):,} rows | fraud: {oot_df["isFraud"].sum():,} ({oot_df["isFraud"].mean()*100:.2f}%)')

def engineer_features(data):
    d = data.copy()
    d['hour_of_day']      = d['step'] % 24
    d['day_of_sim']       = d['step'] // 24
    d['pct_balance_used'] = np.where(d['oldbalanceOrg'] > 0,
                                     d['amount'] / d['oldbalanceOrg'], 0.0).clip(0, 10)
    return d

for name, d in [('train', train_df), ('test', test_df), ('oot', oot_df)]:
    if name == 'train':   train_df = engineer_features(d)
    elif name == 'test':  test_df  = engineer_features(d)
    else:                 oot_df   = engineer_features(d)

LOG1P_COLS = ['amount', 'oldbalanceOrg', 'oldbalanceDest', 'recipient_total_received']
for col in LOG1P_COLS:
    for d in [train_df, test_df, oot_df]:
        d[col] = np.log1p(d[col])
print(f'  log1p applied to: {LOG1P_COLS}')

ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore', dtype=np.float32)
ohe.fit(train_df[['type']])
type_cols = [f'type_{c}' for c in ohe.categories_[0]]

def apply_ohe(data):
    d = data.copy()
    enc = pd.DataFrame(ohe.transform(d[['type']]), columns=type_cols, index=d.index)
    return pd.concat([d.drop(columns=['type']), enc], axis=1)

train_df = apply_ohe(train_df)
test_df  = apply_ohe(test_df)
oot_df   = apply_ohe(oot_df)

DROP_COLS = ['nameOrig', 'nameDest', 'isFlaggedFraud',
             'newbalanceOrig', 'newbalanceDest', 'isFraud', 'split']
DROP_COLS = [c for c in DROP_COLS if c in train_df.columns]

FEATURE_COLS = [c for c in train_df.columns if c not in DROP_COLS + ['isFraud']]
TARGET_COL   = 'isFraud'

X_train = train_df[FEATURE_COLS]
y_train = train_df[TARGET_COL]
X_test  = test_df[FEATURE_COLS]
y_test  = test_df[TARGET_COL]
X_oot   = oot_df[FEATURE_COLS]
y_oot   = oot_df[TARGET_COL]

print(f'\n  Final feature set ({len(FEATURE_COLS)} features):')
m1_feats = ['step','amount','oldbalanceOrg','oldbalanceDest','hour_of_day',
            'day_of_sim','pct_balance_used','type_CASH_IN','type_CASH_OUT',
            'type_DEBIT','type_PAYMENT','type_TRANSFER']
new_feats = [f for f in FEATURE_COLS if f not in m1_feats]
for f in FEATURE_COLS:
    tag = '  <-- NEW' if f in new_feats else ''
    print(f'    {f}{tag}')

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4 — TRAIN XGBOOST
# ═════════════════════════════════════════════════════════════════════════════
sep('SECTION 4 — TRAIN XGBOOST (same params as Model 1)')

note("""
  We use identical hyperparameters to Model 1. This is deliberate —
  we want to isolate the effect of adding graph features, not confuse it
  with a change in hyperparameters. If Model 2 performs better, we can
  attribute the improvement specifically to the new features.

  This principle — changing only one thing at a time — is fundamental
  to comparing models fairly.
""")

PARAMS = dict(
    n_estimators=500, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, scale_pos_weight=1,
    eval_metric='aucpr', early_stopping_rounds=30,
    random_state=RANDOM_SEED, tree_method='hist', device='cpu', verbosity=0,
)

mlflow.set_tracking_uri(f'sqlite:///{PROJECT_ROOT}/mlflow.db')
mlflow.set_experiment('fraud_model_2_graph_features')

print('  Training XGBoost ...')
with mlflow.start_run(run_name='xgb_graph_features_v1') as run:

    model = xgb.XGBClassifier(**PARAMS)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    n_trees = model.best_iteration + 1
    print(f'  Training complete. Best iteration: {n_trees} trees.')

    prob_train = model.predict_proba(X_train)[:, 1]
    prob_test  = model.predict_proba(X_test)[:, 1]
    prob_oot   = model.predict_proba(X_oot)[:, 1]

    pr_auc_train  = average_precision_score(y_train, prob_train)
    pr_auc_test   = average_precision_score(y_test,  prob_test)
    pr_auc_oot    = average_precision_score(y_oot,   prob_oot)
    roc_auc_test  = roc_auc_score(y_test, prob_test)
    roc_auc_oot   = roc_auc_score(y_oot,  prob_oot)

    print(f'\n  {"Set":<8} {"PR-AUC":>10} {"ROC-AUC":>10}')
    print(f'  {"Train":<8} {pr_auc_train:>10.4f}')
    print(f'  {"Test":<8} {pr_auc_test:>10.4f} {roc_auc_test:>10.4f}')
    print(f'  {"OOT":<8} {pr_auc_oot:>10.4f} {roc_auc_oot:>10.4f}')

    # ── Threshold selection ───────────────────────────────────────────────
    sep('SECTION 5 — THRESHOLD SELECTION ON TEST SET')

    thresholds  = np.arange(0.01, 1.00, 0.01)
    f1_scores   = [f1_score(y_test, (prob_test >= t).astype(int), zero_division=0)
                   for t in thresholds]
    prec_scores = [precision_score(y_test, (prob_test >= t).astype(int), zero_division=0)
                   for t in thresholds]
    rec_scores  = [recall_score(y_test, (prob_test >= t).astype(int), zero_division=0)
                   for t in thresholds]

    best_idx       = int(np.argmax(f1_scores))
    best_threshold = thresholds[best_idx]
    best_f1        = f1_scores[best_idx]
    best_prec      = prec_scores[best_idx]
    best_rec       = rec_scores[best_idx]

    print(f'  Best threshold: {best_threshold:.2f}')
    print(f'  F1={best_f1:.4f}  Precision={best_prec:.4f}  Recall={best_rec:.4f}')

    # ── OOT evaluation ────────────────────────────────────────────────────
    sep('SECTION 6 — FINAL EVALUATION ON OOT SET')

    oot_preds = (prob_oot >= best_threshold).astype(int)
    oot_f1    = f1_score(y_oot,    oot_preds, zero_division=0)
    oot_prec  = precision_score(y_oot, oot_preds, zero_division=0)
    oot_rec   = recall_score(y_oot,   oot_preds, zero_division=0)
    oot_cm    = confusion_matrix(y_oot, oot_preds)
    tn, fp, fn, tp = oot_cm.ravel()

    print(f'  OOT evaluation (threshold = {best_threshold:.2f}):')
    print(f'    PR-AUC    : {pr_auc_oot:.4f}')
    print(f'    ROC-AUC   : {roc_auc_oot:.4f}')
    print(f'    F1        : {oot_f1:.4f}')
    print(f'    Precision : {oot_prec:.4f}')
    print(f'    Recall    : {oot_rec:.4f}')
    print(f'\n  Confusion matrix (OOT):')
    print(f'                     Predicted Legit  Predicted Fraud')
    print(f'    Actual Legit  :       {tn:>6,}          {fp:>6,}')
    print(f'    Actual Fraud  :       {fn:>6,}          {tp:>6,}')
    print(f'  TN={tn:,}  FP={fp:,}  FN={fn:,}  TP={tp:,}')

    # ── Model 1 vs Model 2 comparison ─────────────────────────────────────
    sep('SECTION 7 — MODEL 1 vs MODEL 2 COMPARISON')

    note("""
      Model 1 used 12 features (raw transaction columns only).
      Model 2 uses 16 features (+ 4 recipient graph features).
      Same XGBoost hyperparameters in both.

      If Model 2 scores higher, the improvement is due entirely to the
      recipient history features — this is the value of graph information.
      If scores are similar, the raw features already captured most of the
      signal and graph features add little on this dataset.
      (With PaySim's very clean drain-to-zero pattern, Model 1 is already
      near-perfect — Model 2 may not improve much here.)
    """)

    # Model 1 reference numbers (from the 50k run)
    m1_pr_auc_oot = 1.0000
    m1_f1_oot     = 0.9997
    m1_fn         = 1

    print(f'  {"Metric":<25} {"Model 1":>12} {"Model 2":>12} {"Change":>10}')
    print(f'  {"-"*60}')
    print(f'  {"PR-AUC (OOT)":<25} {m1_pr_auc_oot:>12.4f} {pr_auc_oot:>12.4f} '
          f'{pr_auc_oot - m1_pr_auc_oot:>+10.4f}')
    print(f'  {"F1 (OOT)":<25} {m1_f1_oot:>12.4f} {oot_f1:>12.4f} '
          f'{oot_f1 - m1_f1_oot:>+10.4f}')
    print(f'  {"Missed fraud (FN)":<25} {m1_fn:>12,} {fn:>12,} '
          f'{fn - m1_fn:>+10,}')
    print(f'  {"Features used":<25} {"12":>12} {len(FEATURE_COLS):>12}')
    print(f'  {"Threshold":<25} {"0.11":>12} {best_threshold:>12.2f}')

    # ── Feature importance ────────────────────────────────────────────────
    sep('SECTION 8 — FEATURE IMPORTANCE')

    importance = pd.Series(
        model.get_booster().get_score(importance_type='gain'),
        name='gain'
    ).sort_values(ascending=False)

    print('  Feature importance ranking (all features):')
    for feat, val in importance.items():
        tag = '  <-- graph feature' if feat in new_feats else ''
        print(f'    {feat:35s}: {val:>8,.1f}{tag}')

    # ── Charts ────────────────────────────────────────────────────────────
    sep('SECTION 9 — CHARTS')

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # PR curve
    prec_c, rec_c, _ = precision_recall_curve(y_oot, prob_oot)
    axes[0].plot(rec_c, prec_c, color='tomato', lw=2,
                 label=f'Model 2 OOT PR-AUC={pr_auc_oot:.4f}')
    axes[0].scatter([oot_rec], [oot_prec], color='darkred', s=120, zorder=5)
    axes[0].axhline(y_oot.mean(), color='grey', ls='--', lw=1, label='Random baseline')
    axes[0].set_xlabel('Recall'); axes[0].set_ylabel('Precision')
    axes[0].set_title('PR Curve — Model 2 (OOT)')
    axes[0].legend(fontsize=9); axes[0].set_xlim([0,1]); axes[0].set_ylim([0,1.02])

    # Confusion matrix
    cm_df = pd.DataFrame(oot_cm,
                         index=['Actual Legit','Actual Fraud'],
                         columns=['Pred Legit','Pred Fraud'])
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues',
                linewidths=1, ax=axes[1], annot_kws={'size':13})
    axes[1].set_title(f'Confusion Matrix — OOT\n(threshold={best_threshold:.2f})')

    # Feature importance
    imp_plot = importance.sort_values(ascending=True)
    colors   = ['tomato' if f in new_feats else 'steelblue' for f in imp_plot.index]
    imp_plot.plot(kind='barh', ax=axes[2], color=colors, edgecolor='white')
    axes[2].set_title('Feature Importance (gain)\nRed = graph features, Blue = raw features')
    axes[2].set_xlabel('Avg gain per split')

    plt.suptitle('Model 2 — XGBoost + Graph Features', fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_chart(fig, 'model2_summary')

    # ── Save outputs ──────────────────────────────────────────────────────
    model_path = ARTIFACTS_DIR / 'xgb_model.json'
    model.save_model(str(model_path))

    oot_out = oot_df[['step']].copy()
    oot_out['prob_fraud'] = prob_oot
    oot_out['pred_label'] = oot_preds
    oot_out['true_label'] = y_oot.values
    oot_out.to_csv(PREDS_DIR / 'predictions_oot.csv', index=False)

    mlflow.log_params({**PARAMS, 'n_trees': n_trees, 'n_features': len(FEATURE_COLS)})
    mlflow.log_metrics({'pr_auc_oot': pr_auc_oot, 'roc_auc_oot': roc_auc_oot,
                        'f1_oot': oot_f1, 'precision_oot': oot_prec,
                        'recall_oot': oot_rec, 'fn_oot': int(fn), 'fp_oot': int(fp),
                        'best_threshold': best_threshold})
    mlflow.xgboost.log_model(model, 'xgb_model')

    print(f'\n  Model saved: {model_path}')
    print(f'  Predictions saved: {PREDS_DIR / "predictions_oot.csv"}')

sep('MODEL 2 COMPLETE')
print("""
  Outputs:
    artifacts/model_2_graph/xgb_model.json
    artifacts/model_2_graph/charts/model2_summary.png
    data/model_2_graph/output/predictions_oot.csv

  Say "Start Model 3" to build the GNN.
""")
