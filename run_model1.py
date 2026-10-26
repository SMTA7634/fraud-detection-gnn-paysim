"""
run_model1.py  —  Phase 2: XGBoost Baseline (Model 1)
Run: python run_model1.py

Dataset  : paysim_subset_50k.csv  (50,000 rows, 16.43% fraud)
Strategy :
  OOT set  = step > 594  (last 20% of simulation time, ~6 days)
             Held out completely until final evaluation.
  Dev set  = step 1-594
             Split 80/20 STRATIFIED on isFraud -> Train / Test
  scale_pos_weight = 1  (no class weighting — fraud rate is 16.43%, mild enough)

Outputs:
  artifacts/model_1_baseline/xgb_model.json
  artifacts/model_1_baseline/metrics_report.txt
  artifacts/model_1_baseline/charts/  (PR curve, ROC curve, feature importance, confusion matrix)
  data/model_1_baseline/output/predictions_oot.csv
  MLflow experiment: fraud_model_1_baseline  (local, no server needed)
"""

import sys, warnings, os
warnings.filterwarnings('ignore')
os.environ['MLFLOW_DISABLE_AGENT_HINT'] = '1'
os.environ['GIT_PYTHON_REFRESH']       = 'quiet'

from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT  = Path(__file__).resolve().parent
SUBSET_CSV    = PROJECT_ROOT / 'data' / 'shared' / 'raw' / 'paysim_subset_50k.csv'
ARTIFACTS_DIR = PROJECT_ROOT / 'artifacts' / 'model_1_baseline'
CHARTS_DIR    = ARTIFACTS_DIR / 'charts'
PREDS_DIR     = PROJECT_ROOT / 'data' / 'model_1_baseline' / 'output'
MLFLOW_DIR    = PROJECT_ROOT / 'mlruns'

for d in [ARTIFACTS_DIR, CHARTS_DIR, PREDS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import OneHotEncoder
from sklearn.metrics import (
    precision_recall_curve, roc_curve, auc,
    f1_score, precision_score, recall_score,
    confusion_matrix, roc_auc_score, average_precision_score
)
import xgboost as xgb
import mlflow
import mlflow.xgboost

sns.set_theme(style='whitegrid')
plt.rcParams['figure.dpi'] = 100

# ── constants ─────────────────────────────────────────────────────────────────
OOT_STEP_BOUNDARY = 594   # step > 594  →  OOT set  (last 20% of 743 steps)
RANDOM_SEED       = 42
TEST_SIZE         = 0.20  # 20% of dev data → internal test set

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
# SECTION 1 — LOAD DATA AND DEFINE SPLITS
# ═════════════════════════════════════════════════════════════════════════════
sep('SECTION 1 — LOAD DATA AND DEFINE SPLITS')

note("""
  We use three distinct data pools, each with a different role:

  OOT (Out-Of-Time) set — step > 594
    The most recent 20% of the simulation timeline. Set aside completely
    before any model development begins. Simulates the real world: you
    train on historical data, then evaluate on genuinely future transactions.
    This gives an unbiased estimate of how the model would perform in deployment.

  Train set — 80% of step 1-594, stratified on isFraud
    The model learns its decision boundaries here. 'Stratified' means we
    ensure the fraud rate is the same in train and test (16.43%) — no
    accidental split where one set has unusually many or few fraud cases.

  Test set — 20% of step 1-594, stratified on isFraud
    Used during development to select the best decision threshold. Because
    we peek at it to choose the threshold, it is no longer fully independent.
    That is why we keep the OOT set separately for the final verdict.

  Why stratified for train/test but time-based for OOT?
    Stratified split maximises fraud examples in both train and test so
    the model and threshold are well-calibrated. The OOT set must be
    time-based so we genuinely test on data the model never saw — mirroring
    a real deployment where new transactions arrive after training is complete.
""")

df = pd.read_csv(SUBSET_CSV)
print(f'  Loaded: {len(df):,} rows  |  fraud: {df["isFraud"].sum():,}  ({df["isFraud"].mean()*100:.2f}%)')

# OOT split (time-based)
oot_df = df[df['step'] >  OOT_STEP_BOUNDARY].copy()
dev_df = df[df['step'] <= OOT_STEP_BOUNDARY].copy()

print(f'\n  OOT set  (step > {OOT_STEP_BOUNDARY}): {len(oot_df):,} rows  '
      f'| fraud: {oot_df["isFraud"].sum():,}  ({oot_df["isFraud"].mean()*100:.2f}%)')
print(f'  Dev set  (step <= {OOT_STEP_BOUNDARY}): {len(dev_df):,} rows  '
      f'| fraud: {dev_df["isFraud"].sum():,}  ({dev_df["isFraud"].mean()*100:.2f}%)')

# Stratified train/test split from dev
train_df, test_df = train_test_split(
    dev_df,
    test_size    = TEST_SIZE,
    random_state = RANDOM_SEED,
    stratify     = dev_df['isFraud']   # ensures same fraud rate in both
)

print(f'\n  Train set: {len(train_df):,} rows  '
      f'| fraud: {train_df["isFraud"].sum():,}  ({train_df["isFraud"].mean()*100:.2f}%)')
print(f'  Test set : {len(test_df):,} rows   '
      f'| fraud: {test_df["isFraud"].sum():,}  ({test_df["isFraud"].mean()*100:.2f}%)')
print(f'  OOT set  : {len(oot_df):,} rows   '
      f'| fraud: {oot_df["isFraud"].sum():,}  ({oot_df["isFraud"].mean()*100:.2f}%)')

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 — FEATURE ENGINEERING AND PREPROCESSING
# ═════════════════════════════════════════════════════════════════════════════
sep('SECTION 2 — FEATURE ENGINEERING AND PREPROCESSING')

note("""
  We engineer three new features from the raw columns, all using only
  information available at the moment the transaction is initiated
  (pre-settlement, no leakage):

    hour_of_day  = step % 24
      Which hour of the simulated day (0-23). Fraud may cluster at certain
      hours — e.g., overnight when account holders are less likely to notice.

    day_of_sim   = step // 24
      Which simulated day (0-30). Captures ramp-up patterns in fraud activity
      over the simulation timeline.

    pct_balance_used = amount / oldbalanceOrg
      What fraction of the sender's available balance was used.
      Fraud drains accounts to exactly zero — median pct_balance_used in fraud
      rows is 100%. This is a very strong signal and uses only pre-settlement
      columns (amount and oldbalanceOrg), so it is NOT leaky.
      Division-by-zero (oldbalanceOrg = 0) is set to 0.

  LOG1P TRANSFORMS for skewed numerics:
    amount, oldbalanceOrg, oldbalanceDest are all extremely right-skewed
    (a few huge transactions dominate). log1p(x) = log(1+x) compresses the
    scale so large values don't dominate XGBoost's split search.
    log1p handles zeros safely (log1p(0) = 0).

  ONE-HOT ENCODING for 'type':
    XGBoost can handle raw integer labels but not string categories.
    OHE creates one binary column per category (CASH_IN, CASH_OUT, DEBIT,
    PAYMENT, TRANSFER). Fit the encoder on the TRAINING SET ONLY, then
    transform test and OOT using those same categories. This prevents
    leakage of test/OOT category frequencies into the model.
""")

def engineer_features(data):
    """Add derived features. All use only pre-settlement columns."""
    d = data.copy()
    d['hour_of_day']      = d['step'] % 24
    d['day_of_sim']       = d['step'] // 24
    d['pct_balance_used'] = np.where(
        d['oldbalanceOrg'] > 0,
        d['amount'] / d['oldbalanceOrg'],
        0.0
    ).clip(0, 10)   # cap extreme ratios (div by near-zero balance)
    return d

train_df = engineer_features(train_df)
test_df  = engineer_features(test_df)
oot_df   = engineer_features(oot_df)
print('  Engineered: hour_of_day, day_of_sim, pct_balance_used')

# log1p transforms — deterministic, no fitting needed
LOG1P_COLS = ['amount', 'oldbalanceOrg', 'oldbalanceDest']
for col in LOG1P_COLS:
    for d in [train_df, test_df, oot_df]:
        d[col] = np.log1p(d[col])
print(f'  log1p applied to: {LOG1P_COLS}')

# One-hot encode 'type' — FIT ON TRAIN ONLY
note("""
  Why fit OneHotEncoder on training data only?
  If we fit on the full dataset, the encoder learns that certain categories
  exist in val/test. More importantly, if we ever normalise or scale encoded
  values, fitting on all data leaks statistics from future rows. The habit of
  fitting transformers only on train data is a fundamental discipline — even
  when the practical effect is small, breaking it is how subtle leakage enters.
""")

ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore', dtype=np.float32)
ohe.fit(train_df[['type']])

type_cols = [f'type_{c}' for c in ohe.categories_[0]]

def apply_ohe(data, ohe, type_cols):
    d = data.copy()
    encoded = ohe.transform(d[['type']])
    enc_df  = pd.DataFrame(encoded, columns=type_cols, index=d.index)
    d = pd.concat([d.drop(columns=['type']), enc_df], axis=1)
    return d

train_df = apply_ohe(train_df, ohe, type_cols)
test_df  = apply_ohe(test_df,  ohe, type_cols)
oot_df   = apply_ohe(oot_df,   ohe, type_cols)
print(f'  OHE applied: type -> {type_cols}')

# Define final feature columns
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
for f in FEATURE_COLS:
    print(f'    {f}')

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 — TRAIN XGBOOST
# ═════════════════════════════════════════════════════════════════════════════
sep('SECTION 3 — TRAIN XGBOOST')

note("""
  XGBoost (Extreme Gradient Boosting) builds an ensemble of decision trees
  sequentially. Each new tree corrects the mistakes of all previous trees
  by fitting on the residual errors (gradients of the loss function).

  Key hyperparameters used here:

    n_estimators=500      Maximum number of trees. Early stopping will likely
                          stop before 500 if the model stops improving.

    max_depth=6           Each tree can ask at most 6 yes/no questions
                          (splits) before making a prediction. Deeper trees
                          can capture more complex patterns but risk overfitting.

    learning_rate=0.05    How much each new tree contributes to the final
                          prediction. Lower rate = more trees needed, but
                          more robust generalisation.

    subsample=0.8         Each tree is trained on a random 80% of the rows.
                          Prevents any single unusual row from dominating.

    colsample_bytree=0.8  Each tree uses a random 80% of the features.
                          Forces diversity across trees, reduces overfitting.

    scale_pos_weight=1    No class weighting. With 16.43% fraud in the
                          training set, the imbalance is mild enough that
                          the model can learn without explicit correction.
                          (Would be 1030 if using the full 6.3M dataset.)

    eval_metric='aucpr'   Monitors PR-AUC on the validation set during
                          training, not accuracy. Each new tree is judged
                          on how much it improves fraud detection.

    early_stopping_rounds=30
                          If PR-AUC on the test set does not improve for
                          30 consecutive trees, stop training. Prevents
                          overfitting and wasted computation.
""")

PARAMS = dict(
    n_estimators          = 500,
    max_depth             = 6,
    learning_rate         = 0.05,
    subsample             = 0.8,
    colsample_bytree      = 0.8,
    scale_pos_weight      = 1,
    eval_metric           = 'aucpr',
    early_stopping_rounds = 30,
    random_state          = RANDOM_SEED,
    tree_method           = 'hist',   # fast histogram-based algorithm
    device                = 'cpu',    # change to 'cuda' to use GPU
    verbosity             = 0,
)

mlflow.set_tracking_uri(f'sqlite:///{PROJECT_ROOT}/mlflow.db')
mlflow.set_experiment('fraud_model_1_baseline')

print('  Training XGBoost ...')
with mlflow.start_run(run_name='xgb_baseline_v1') as run:

    model = xgb.XGBClassifier(**PARAMS)
    model.fit(
        X_train, y_train,
        eval_set        = [(X_test, y_test)],
        verbose         = False
    )

    n_trees_used = model.best_iteration + 1
    print(f'  Training complete. Best iteration: {n_trees_used} trees (early stopping applied).')

    # ── Score all three sets ──────────────────────────────────────────────
    prob_train = model.predict_proba(X_train)[:, 1]
    prob_test  = model.predict_proba(X_test)[:, 1]
    prob_oot   = model.predict_proba(X_oot)[:, 1]

    pr_auc_train = average_precision_score(y_train, prob_train)
    pr_auc_test  = average_precision_score(y_test,  prob_test)
    pr_auc_oot   = average_precision_score(y_oot,   prob_oot)

    roc_auc_train = roc_auc_score(y_train, prob_train)
    roc_auc_test  = roc_auc_score(y_test,  prob_test)
    roc_auc_oot   = roc_auc_score(y_oot,   prob_oot)

    print(f'\n  {"Set":<8} {"PR-AUC":>10} {"ROC-AUC":>10}')
    print(f'  {"Train":<8} {pr_auc_train:>10.4f} {roc_auc_train:>10.4f}')
    print(f'  {"Test":<8} {pr_auc_test:>10.4f} {roc_auc_test:>10.4f}')
    print(f'  {"OOT":<8} {pr_auc_oot:>10.4f} {roc_auc_oot:>10.4f}')

    # MLflow log params and base metrics
    mlflow.log_params({**PARAMS, 'n_trees_actual': n_trees_used,
                       'oot_boundary': OOT_STEP_BOUNDARY,
                       'features': len(FEATURE_COLS)})
    mlflow.log_metrics({'pr_auc_train': pr_auc_train, 'pr_auc_test': pr_auc_test,
                        'pr_auc_oot':   pr_auc_oot,   'roc_auc_test': roc_auc_test,
                        'roc_auc_oot':  roc_auc_oot})

    # ═════════════════════════════════════════════════════════════════════════
    # SECTION 4 — THRESHOLD SELECTION ON TEST SET
    # ═════════════════════════════════════════════════════════════════════════
    sep('SECTION 4 — THRESHOLD SELECTION ON TEST SET')

    note("""
      XGBoost outputs a probability between 0 and 1 for each transaction.
      To make a binary decision (fraud / not fraud) we need a THRESHOLD:
        if probability >= threshold  →  flag as fraud
        if probability <  threshold  →  pass as legitimate

      The default threshold of 0.5 is almost always wrong for fraud detection:
        - With 16.43% fraud in training, the model rarely outputs prob > 0.5
          for anything. At 0.5 you would catch very few fraud cases.

      We sweep all thresholds from 0.01 to 0.99 and measure:
        Precision = of all transactions we flagged as fraud, what % truly are?
        Recall    = of all actual fraud, what % did we catch?
        F1        = harmonic mean of precision and recall.
                    F1 = 2 * (precision * recall) / (precision + recall)
                    F1 is 1.0 only when both precision AND recall are perfect.

      We pick the threshold that maximises F1 on the TEST set.
      This threshold is then applied (once, finally) to the OOT set.

      WHY NOT CHOOSE THRESHOLD ON OOT?
        If we swept thresholds on the OOT set, we would be optimising
        for that specific set of future transactions. In real deployment,
        you never get to tune your threshold on tomorrow's data — you tune
        it on recent history and hope it generalises forward.
    """)

    thresholds = np.arange(0.01, 1.00, 0.01)
    f1_scores  = []
    prec_scores = []
    rec_scores  = []

    for t in thresholds:
        preds = (prob_test >= t).astype(int)
        f1_scores.append(f1_score(y_test, preds, zero_division=0))
        prec_scores.append(precision_score(y_test, preds, zero_division=0))
        rec_scores.append(recall_score(y_test, preds, zero_division=0))

    best_idx       = int(np.argmax(f1_scores))
    best_threshold = thresholds[best_idx]
    best_f1        = f1_scores[best_idx]
    best_prec      = prec_scores[best_idx]
    best_rec       = rec_scores[best_idx]

    print(f'  Best threshold (max F1 on test set): {best_threshold:.2f}')
    print(f'  At this threshold — Test set:')
    print(f'    F1        : {best_f1:.4f}')
    print(f'    Precision : {best_prec:.4f}  '
          f'(of transactions flagged fraud, {best_prec*100:.1f}% are truly fraud)')
    print(f'    Recall    : {best_rec:.4f}  '
          f'({best_rec*100:.1f}% of all actual fraud was caught)')

    mlflow.log_metric('best_threshold', best_threshold)
    mlflow.log_metric('f1_test_at_best_threshold', best_f1)

    # ═════════════════════════════════════════════════════════════════════════
    # SECTION 5 — FINAL EVALUATION ON OOT SET
    # ═════════════════════════════════════════════════════════════════════════
    sep('SECTION 5 — FINAL EVALUATION ON OOT SET')

    note("""
      We now apply the chosen threshold to the OOT set.
      This is the FIRST AND ONLY time we use the threshold on OOT.
      These numbers are our honest estimate of real-world performance.

      The OOT set covers step 595-743 (the last ~6 days of simulation).
      Fraud rate in OOT is higher (because fraud ramped up over time in
      the simulation), so absolute recall numbers may look different from
      the test set — this is expected and realistic.
    """)

    oot_preds = (prob_oot >= best_threshold).astype(int)

    oot_f1    = f1_score(y_oot,    oot_preds, zero_division=0)
    oot_prec  = precision_score(y_oot, oot_preds, zero_division=0)
    oot_rec   = recall_score(y_oot,   oot_preds, zero_division=0)
    oot_cm    = confusion_matrix(y_oot, oot_preds)

    tn, fp, fn, tp = oot_cm.ravel()

    print(f'  OOT set evaluation (threshold = {best_threshold:.2f}):')
    print(f'    PR-AUC    : {pr_auc_oot:.4f}')
    print(f'    ROC-AUC   : {roc_auc_oot:.4f}')
    print(f'    F1        : {oot_f1:.4f}')
    print(f'    Precision : {oot_prec:.4f}')
    print(f'    Recall    : {oot_rec:.4f}')
    print(f'\n  Confusion matrix (OOT):')
    print(f'                     Predicted Legit  Predicted Fraud')
    print(f'    Actual Legit  :       {tn:>6,}          {fp:>6,}')
    print(f'    Actual Fraud  :       {fn:>6,}          {tp:>6,}')
    print(f'\n  TN={tn:,}  FP={fp:,}  FN={fn:,}  TP={tp:,}')
    print(f'  False Positive Rate : {fp/(fp+tn)*100:.2f}%  '
          f'(legitimate transactions wrongly flagged)')
    print(f'  False Negative Rate : {fn/(fn+tp)*100:.2f}%  '
          f'(fraud transactions missed)')

    note("""
      INTERPRETING THE CONFUSION MATRIX:
        True  Positive (TP) = caught fraud.        GOOD. Want this high.
        True  Negative (TN) = correct legit.        GOOD. Want this high.
        False Positive (FP) = wrongly flagged legit. BAD. Annoys real customers.
        False Negative (FN) = missed fraud.          BAD. The model failed.

      Precision = TP / (TP + FP)  — of everything we flagged, what % was real?
      Recall    = TP / (TP + FN)  — of all real fraud, what % did we catch?

      There is always a tension: lowering the threshold catches more fraud
      (higher recall) but also flags more legitimate transactions (lower precision).
      The F1 score balances both. The right business trade-off depends on costs:
        - A fraud investigation team has limited capacity → prioritise precision
        - A high-value transaction system → prioritise recall (catch everything)
    """)

    mlflow.log_metrics({
        'pr_auc_oot_final': pr_auc_oot, 'roc_auc_oot_final': roc_auc_oot,
        'f1_oot': oot_f1, 'precision_oot': oot_prec, 'recall_oot': oot_rec,
        'tp_oot': int(tp), 'fp_oot': int(fp), 'fn_oot': int(fn), 'tn_oot': int(tn)
    })

    # ═════════════════════════════════════════════════════════════════════════
    # SECTION 6 — CHARTS
    # ═════════════════════════════════════════════════════════════════════════
    sep('SECTION 6 — CHARTS')

    # Chart 1 — PR curve + threshold sweep
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    prec_curve, rec_curve, _ = precision_recall_curve(y_oot, prob_oot)
    axes[0].plot(rec_curve, prec_curve, color='tomato', lw=2,
                 label=f'OOT PR-AUC = {pr_auc_oot:.4f}')
    axes[0].scatter([oot_rec], [oot_prec], color='darkred', s=120, zorder=5,
                    label=f'Operating point (t={best_threshold:.2f})')
    axes[0].axhline(y_oot.mean(), color='grey', ls='--', lw=1,
                    label=f'Random baseline ({y_oot.mean():.3f})')
    axes[0].set_xlabel('Recall')
    axes[0].set_ylabel('Precision')
    axes[0].set_title('Precision-Recall Curve (OOT set)')
    axes[0].legend(fontsize=9)
    axes[0].set_xlim([0, 1]); axes[0].set_ylim([0, 1.02])

    axes[1].plot(thresholds, f1_scores,   color='steelblue', lw=2, label='F1')
    axes[1].plot(thresholds, prec_scores, color='seagreen',  lw=1.5, ls='--', label='Precision')
    axes[1].plot(thresholds, rec_scores,  color='tomato',    lw=1.5, ls='--', label='Recall')
    axes[1].axvline(best_threshold, color='black', ls=':', lw=1.5,
                    label=f'Best threshold = {best_threshold:.2f}')
    axes[1].set_xlabel('Threshold')
    axes[1].set_title('Precision / Recall / F1 vs Threshold (test set)')
    axes[1].legend(fontsize=9)

    plt.suptitle('Model 1 — XGBoost Baseline: PR Analysis', fontweight='bold')
    plt.tight_layout()
    save_chart(fig, 'pr_curve_and_threshold')

    # Chart 2 — ROC curve
    fig, ax = plt.subplots(figsize=(7, 6))
    fpr_c, tpr_c, _ = roc_curve(y_oot, prob_oot)
    ax.plot(fpr_c, tpr_c, color='steelblue', lw=2,
            label=f'OOT ROC-AUC = {roc_auc_oot:.4f}')
    ax.plot([0,1],[0,1], color='grey', ls='--', lw=1, label='Random baseline')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate (Recall)')
    ax.set_title('ROC Curve (OOT set)\nModel 1 — XGBoost Baseline')
    ax.legend(fontsize=10)
    plt.tight_layout()
    save_chart(fig, 'roc_curve')

    # Chart 3 — Confusion matrix heatmap
    fig, ax = plt.subplots(figsize=(6, 5))
    cm_display = pd.DataFrame(
        oot_cm,
        index   = ['Actual Legit', 'Actual Fraud'],
        columns = ['Pred Legit',   'Pred Fraud']
    )
    sns.heatmap(cm_display, annot=True, fmt='d', cmap='Blues',
                linewidths=1, ax=ax, annot_kws={'size': 14})
    ax.set_title(f'Confusion Matrix — OOT set\n(threshold = {best_threshold:.2f})',
                 fontsize=12)
    plt.tight_layout()
    save_chart(fig, 'confusion_matrix')

    # Chart 4 — Feature importance
    note("""
      XGBoost records how many times each feature was used to split a node
      across all trees (called 'weight' or 'gain'). Features used more often
      are generally more important to the model's decisions.

      We use 'gain' importance: the average improvement in the loss function
      each time a feature is used for a split. 'Gain' is more informative
      than pure split count because it weights by how much each split helps.
    """)

    importance = pd.Series(
        model.get_booster().get_score(importance_type='gain'),
        name='gain_importance'
    ).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8, max(5, len(importance) * 0.35)))
    importance.plot(kind='barh', ax=ax, color='steelblue', edgecolor='white')
    ax.set_title('Feature Importance — Gain (Model 1 XGBoost Baseline)')
    ax.set_xlabel('Average gain per split')
    plt.tight_layout()
    save_chart(fig, 'feature_importance')

    print('\n  Feature importance ranking (top features):')
    top = importance.sort_values(ascending=False).head(10)
    for feat, val in top.items():
        print(f'    {feat:30s}: {val:,.1f}')

    # Save model artifact
    model_path = ARTIFACTS_DIR / 'xgb_model.json'
    model.save_model(str(model_path))
    mlflow.xgboost.log_model(model, 'xgb_model')
    print(f'\n  Model saved: {model_path}')

    # ═════════════════════════════════════════════════════════════════════════
    # SECTION 7 — SAVE PREDICTIONS AND METRICS REPORT
    # ═════════════════════════════════════════════════════════════════════════
    sep('SECTION 7 — SAVE OUTPUTS')

    # OOT predictions CSV
    oot_preds_df = oot_df[['step']].copy()
    oot_preds_df['prob_fraud']   = prob_oot
    oot_preds_df['pred_label']   = oot_preds
    oot_preds_df['true_label']   = y_oot.values
    oot_preds_df['correct']      = (oot_preds_df['pred_label'] == oot_preds_df['true_label']).astype(int)
    oot_preds_path = PREDS_DIR / 'predictions_oot.csv'
    oot_preds_df.to_csv(oot_preds_path, index=False)
    print(f'  OOT predictions saved: {oot_preds_path}')

    # Metrics report text file
    report_lines = [
        'MODEL 1 — XGBOOST BASELINE — METRICS REPORT\n',
        '='*50 + '\n\n',
        f'Dataset        : paysim_subset_50k.csv\n',
        f'OOT boundary   : step > {OOT_STEP_BOUNDARY}\n',
        f'Features used  : {len(FEATURE_COLS)}\n',
        f'Trees trained  : {n_trees_used}\n',
        f'Best threshold : {best_threshold:.2f} (selected on test set)\n\n',
        'METRICS\n' + '-'*30 + '\n',
        f'{"Metric":<20} {"Test":>10} {"OOT":>10}\n',
        f'{"PR-AUC":<20} {pr_auc_test:>10.4f} {pr_auc_oot:>10.4f}\n',
        f'{"ROC-AUC":<20} {roc_auc_test:>10.4f} {roc_auc_oot:>10.4f}\n',
        f'{"F1":<20} {best_f1:>10.4f} {oot_f1:>10.4f}\n',
        f'{"Precision":<20} {best_prec:>10.4f} {oot_prec:>10.4f}\n',
        f'{"Recall":<20} {best_rec:>10.4f} {oot_rec:>10.4f}\n\n',
        'CONFUSION MATRIX (OOT)\n' + '-'*30 + '\n',
        f'  TP (caught fraud)       : {tp:,}\n',
        f'  TN (correct legit)      : {tn:,}\n',
        f'  FP (wrongly flagged)    : {fp:,}\n',
        f'  FN (missed fraud)       : {fn:,}\n\n',
        f'MLflow run ID: {run.info.run_id}\n',
    ]
    report_path = ARTIFACTS_DIR / 'metrics_report.txt'
    with open(report_path, 'w') as fh:
        fh.writelines(report_lines)
    print(f'  Metrics report saved: {report_path}')
    mlflow.log_artifact(str(report_path))

sep('MODEL 1 COMPLETE')
print(f"""
  Outputs:
    artifacts/model_1_baseline/xgb_model.json
    artifacts/model_1_baseline/metrics_report.txt
    artifacts/model_1_baseline/charts/pr_curve_and_threshold.png
    artifacts/model_1_baseline/charts/roc_curve.png
    artifacts/model_1_baseline/charts/confusion_matrix.png
    artifacts/model_1_baseline/charts/feature_importance.png
    data/model_1_baseline/output/predictions_oot.csv

  To view MLflow UI (optional):
    Run in terminal:  mlflow ui --backend-store-uri mlruns
    Then open: http://localhost:5000

  Say "Start Model 2" to build XGBoost + graph features.
""")
