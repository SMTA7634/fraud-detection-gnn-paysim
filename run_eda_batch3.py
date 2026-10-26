"""
PaySim EDA - Batch 3 (Phases 10-13)
Run: python run_eda_batch3.py

Covers:
  Phase 10 — RFM / Behavioural Feature Candidates
  Phase 11 — Feature Engineering Register
  Phase 12 — Validation Strategy
  Phase 13 — EDA Closure and Handover

LEARNING NOTE — WHY THREE BATCHES?
  We split the EDA into three batches to stay disciplined:
  Batch 1 = WHAT is in the data (shape, quality, basics)
  Batch 2 = HOW the data behaves (distributions, correlations, time trends)
  Batch 3 = WHAT WE WILL BUILD (features, splits, validation plan)
  Each batch ends before you start building models, ensuring you understand
  the data deeply before writing a single line of training code.

COMPREHENSION DECISIONS (answered in this script):
  Q1. Why recalculate scale_pos_weight on training split only?
      -> Answered in Phase 12.
  Q2. Why are sender-side graph features nearly useless here?
      -> Answered in Phase 10.
  Q3. Is balance_discrepancy leaky in Model 1?
      -> Answered in Phase 11.
"""

import sys, warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import pandas as pd
import numpy as np

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

PAYSIM_CSV  = PROJECT_ROOT / 'data' / 'shared' / 'raw' / 'PaySim_kaggle_raw_data.csv'
REPORTS_DIR = PROJECT_ROOT / 'reports' / 'data_quality'
CHARTS_DIR  = REPORTS_DIR / 'charts'
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams['figure.dpi'] = 100
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', '{:.4f}'.format)

# ── Split boundaries (confirmed from Batch 2) ─────────────────────────────────
TRAIN_END = 534
VAL_END   = 638

def sep(title):
    print(f'\n{"="*70}\n  {title}\n{"="*70}')

def note(text):
    """Print a learning note in a visible way."""
    print(f'\n  [LEARNING NOTE]\n  {text}\n')

def decision(text):
    """Print a design decision and rationale."""
    print(f'\n  [DECISION] {text}\n')

def save_chart(fig, name):
    path = CHARTS_DIR / f'{name}.png'
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'  Chart saved: {name}.png')

# ── Load data ─────────────────────────────────────────────────────────────────
sep('LOADING DATA')
print(f'Reading {PAYSIM_CSV.name} ...')
df = pd.read_csv(PAYSIM_CSV)
df['split'] = pd.cut(
    df['step'],
    bins=[0, TRAIN_END, VAL_END, df['step'].max() + 1],
    labels=['train', 'val', 'test'],
    right=True
)
print(f'Loaded: {df.shape[0]:,} rows  |  splits assigned.')

train_df = df[df['split'] == 'train'].copy()
val_df   = df[df['split'] == 'val'].copy()
test_df  = df[df['split'] == 'test'].copy()

# ═════════════════════════════════════════════════════════════════════════════
# PHASE 10 — RFM / BEHAVIOURAL FEATURE CANDIDATES
# ═════════════════════════════════════════════════════════════════════════════
sep('PHASE 10 — WHAT IS RFM AND WHY DOES IT MATTER FOR FRAUD DETECTION?')

note("""
  RFM = Recency, Frequency, Monetary. It is a classic behavioural profiling
  framework from marketing analytics, but it is extremely powerful for fraud
  because fraudsters behave differently from normal users:

    RECENCY   — When did this account last transact?
                A dormant account suddenly sending a large transfer is suspicious.

    FREQUENCY — How many transactions has this account made?
                A brand-new account making its first transaction is higher risk.

    MONETARY  — How much has this account sent/received in total?
                A spike in amount relative to the account's history is a red flag.

  In fraud detection, we compute RFM features as HISTORICAL AGGREGATES:
  for each transaction, we look back at all PRIOR transactions to summarise
  the account's behaviour up to that moment. We never include the current
  transaction in its own aggregate — that would be circular.
""")

sep('PHASE 10 — SENDER (nameOrig) RFM FEASIBILITY')

note("""
  From Batch 2 entity profiling:
    - 6,353,307 unique senders
    - Max transactions by one sender: 3
    - 99th percentile transactions per sender: 1

  This means nearly every sender in PaySim makes EXACTLY ONE transaction in
  their entire simulated lifetime. This is by design — PaySim simulates a
  mobile money ecosystem where most users are occasional. The implication:

  SENDER-SIDE FREQUENCY FEATURES HAVE NEAR-ZERO VARIANCE.
  If 99%+ of senders have exactly 1 prior transaction, then
  sender_txn_count = 0 for almost every row (0 prior transactions when they
  first appear, which is the only time they appear).

  Decision: sender_txn_count, sender_total_sent, sender_distinct_recipients,
  and time_since_last_txn_orig are NOT worth building for Model 2.
  They will have near-constant value (0 or 1) and contribute nothing.
""")

# Verify: compute sender transaction counts on training data
print('Computing sender transaction frequency on training data ...')
sender_counts_train = train_df['nameOrig'].value_counts()
print(f'  Senders in training set          : {len(sender_counts_train):,}')
print(f'  Senders with exactly 1 transaction: {(sender_counts_train == 1).sum():,} '
      f'({(sender_counts_train == 1).mean()*100:.1f}%)')
print(f'  Senders with 2+ transactions      : {(sender_counts_train >= 2).sum():,}')
print(f'  Senders with 3 transactions       : {(sender_counts_train == 3).sum():,}')

decision("""
  SENDER FEATURES DROPPED from Model 2 candidate list.
  Reason: 99%+ of senders appear exactly once → near-zero variance → no
  predictive signal. Adding zero-variance features wastes model capacity and
  can introduce numerical instability in some tree-growing routines.
""")

sep('PHASE 10 — RECIPIENT (nameDest) RFM FEASIBILITY')

note("""
  Recipient accounts look very different:
    - 2,722,362 unique recipients (vs 6.3M senders)
    - Max transactions received: 113
    - 99th percentile: 25 transactions
    - 130,286 recipients receive more than 10 transactions
    - 1,033 'super-nodes' receive more than 50 transactions

  Recipients are REUSED. This makes recipient-side features informative:
  - A recipient that has received many prior transactions (high frequency)
    may be a known merchant (low risk) OR a money mule draining many accounts
    (high risk). The model can learn which pattern correlates with fraud.
  - A recipient that has never received a transaction before (first appearance)
    is an unknown entity — often fraudulent throwaway accounts.
""")

print('Computing recipient transaction frequency on training data ...')
dest_counts_train = train_df['nameDest'].value_counts()
print(f'  Unique recipients in training set : {len(dest_counts_train):,}')
print(f'  Recipients with 1 transaction     : {(dest_counts_train == 1).sum():,} '
      f'({(dest_counts_train == 1).mean()*100:.1f}%)')
print(f'  Recipients with 5+ transactions   : {(dest_counts_train >= 5).sum():,}')
print(f'  Recipients with 20+ transactions  : {(dest_counts_train >= 20).sum():,}')
print(f'  Recipients with 50+ transactions  : {(dest_counts_train >= 50).sum():,}')

# What fraction of fraud goes to first-time recipients?
note("""
  'First-time recipient' = the destination account has never received a
  transaction before this one. We compute this using the training set:
  sort by step, then flag whether it's the recipient's first appearance.
""")

train_sorted = train_df.sort_values('step').copy()
first_appearance = train_sorted.groupby('nameDest')['step'].transform('min')
train_sorted['is_new_recipient'] = (train_sorted['step'] == first_appearance).astype(int)

new_rec_fraud = train_sorted[train_sorted['isFraud'] == 1]['is_new_recipient'].mean()
new_rec_legit = train_sorted[train_sorted['isFraud'] == 0]['is_new_recipient'].mean()
print(f'\n  Transactions going to a FIRST-TIME recipient:')
print(f'    In fraud rows : {new_rec_fraud*100:.1f}%')
print(f'    In legit rows : {new_rec_legit*100:.1f}%')

decision("""
  RECIPIENT FEATURES ARE VALUABLE for Model 2.
  Useful candidates:
    - recipient_txn_count      : prior transactions received by nameDest
    - recipient_total_received : total amount received by nameDest historically
    - recipient_distinct_senders: number of unique senders to this recipient
    - is_new_recipient         : binary flag, first appearance of this dest
  These will be computed as time-ordered expanding windows in Phase 3
  (feature engineering), using strictly historical data.
""")

sep('PHASE 10 — TEMPORAL FEATURES (step-derived)')

note("""
  PaySim simulates 743 hours (about 31 days). The 'step' column is
  1 step = 1 simulated hour. This gives us temporal features:

    hour_of_day  = step % 24  (simulated time of day, 0-23)
    day_of_sim   = step // 24 (which simulated day, 0-30)

  These are worth including because:
    - Fraud may concentrate at certain hours (e.g., overnight when humans
      are less likely to notice and cancel)
    - Day-of-simulation may capture ramp-up patterns in the simulation

  IMPORTANT: step is NOT real time. Do NOT interpret these as real calendar
  hours or days. The pattern is synthetic but may still be learnable by XGBoost.
""")

df['hour_of_day'] = df['step'] % 24
df['day_of_sim']  = df['step'] // 24

print('Fraud rate by hour_of_day:')
hourly = df.groupby('hour_of_day')['isFraud'].mean() * 100
print(hourly.describe().round(4).to_string())
print(f'  Hours with highest fraud rate: {hourly.nlargest(3).index.tolist()}')
print(f'  Hours with lowest fraud rate : {hourly.nsmallest(3).index.tolist()}')

print('\nFraud rate by day_of_sim (first 5 days and last 5 days):')
daily = df.groupby('day_of_sim')['isFraud'].mean() * 100
print('  First 5 days:'); print(daily.head(5).to_string())
print('  Last  5 days:'); print(daily.tail(5).to_string())

decision("""
  INCLUDE hour_of_day and day_of_sim as candidate features for Model 1 and 2.
  They are derived purely from 'step' (which is available at transaction time,
  no post-settlement data), so there is NO leakage risk.
  Final decision on inclusion based on feature importance after Model 1 run.
""")

# ═════════════════════════════════════════════════════════════════════════════
# PHASE 11 — FEATURE ENGINEERING REGISTER
# ═════════════════════════════════════════════════════════════════════════════
sep('PHASE 11 — WHY DO WE NEED A FEATURE REGISTER?')

note("""
  A feature register is a formal inventory of every feature we plan to use,
  with metadata: where it comes from, how to compute it, and whether it
  carries leakage risk. It serves three purposes:

  1. LEAKAGE AUDIT — forces you to ask "would this value be available at
     the moment of transaction?" for every single feature. Missing even one
     leaky feature can inflate model metrics to near-perfect and give you
     false confidence before deployment.

  2. REPRODUCIBILITY — anyone reading the register can reproduce the feature
     pipeline without reverse-engineering the code.

  3. MODEL COMPARISON — by listing which features go into Model 1 vs Model 2
     vs Model 3, we make the comparison fair and explicit.
""")

sep('PHASE 11 — LEAKAGE DECISION: balance_discrepancy')

note("""
  COMPREHENSION Q3 ANSWER:
  balance_discrepancy = abs(oldbalanceOrg - newbalanceOrig - amount)

  This feature uses newbalanceOrig — the sender's balance AFTER the transaction.
  We confirmed in EDA Batch 1 that newbalanceOrig is POST-SETTLEMENT data:
  it reflects what happened after the transaction completed.

  At the moment we need to decide "is this transaction fraud?", we do NOT yet
  know newbalanceOrig — the settlement hasn't happened. Using it would be like
  a bank approving a payment and then checking the post-approval balance to
  decide whether to approve it. The feature has already seen the outcome.

  CONCLUSION: balance_discrepancy IS LEAKY for Model 1 (raw transaction features).
  It CANNOT be included in Model 1.

  CAN IT BE USED IN MODEL 2?
  Only as a HISTORICAL feature: for each transaction, compute
  balance_discrepancy from *prior* transactions of the same recipient or sender.
  The historical aggregate (e.g., 'did this recipient previously drain accounts
  to zero?') does NOT use current transaction's newbalanceOrig. That is safe.

  SAFE ALTERNATIVE: pct_balance_used = amount / oldbalanceOrg
  This uses only amount (the requested transfer size) and oldbalanceOrg
  (the sender's balance BEFORE the transaction) — both available at
  decision time. It is NOT leaky and is a strong feature.
""")

sep('PHASE 11 — COMPLETE FEATURE REGISTER')

# Build the feature register as a DataFrame
register_rows = [
    # ── Raw transaction features ──────────────────────────────────────────
    dict(
        feature_name   = 'step',
        source         = 'raw',
        derivation     = 'Direct column',
        availability   = 'pre_settlement',
        leakage_risk   = 'none',
        leakage_reason = 'Simulation time — available when transaction is initiated',
        model_1        = 'yes',
        model_2        = 'yes',
        model_3        = 'yes (node feature)',
        notes          = 'Raw integer. Consider log or leaving as-is.'
    ),
    dict(
        feature_name   = 'hour_of_day',
        source         = 'engineered_raw',
        derivation     = 'step % 24',
        availability   = 'pre_settlement',
        leakage_risk   = 'none',
        leakage_reason = 'Derived from step only',
        model_1        = 'yes',
        model_2        = 'yes',
        model_3        = 'yes',
        notes          = 'Cyclic encoding (sin/cos) optional for neural nets; XGBoost handles raw ordinal well.'
    ),
    dict(
        feature_name   = 'day_of_sim',
        source         = 'engineered_raw',
        derivation     = 'step // 24',
        availability   = 'pre_settlement',
        leakage_risk   = 'none',
        leakage_reason = 'Derived from step only',
        model_1        = 'yes',
        model_2        = 'yes',
        model_3        = 'yes',
        notes          = 'Captures day-level simulation patterns.'
    ),
    dict(
        feature_name   = 'type_encoded',
        source         = 'engineered_raw',
        derivation     = 'One-hot of type column',
        availability   = 'pre_settlement',
        leakage_risk   = 'none',
        leakage_reason = 'Transaction type is set at initiation',
        model_1        = 'yes',
        model_2        = 'yes',
        model_3        = 'yes',
        notes          = 'Only TRANSFER and CASH_OUT have fraud. OHE creates 5 columns.'
    ),
    dict(
        feature_name   = 'amount',
        source         = 'raw',
        derivation     = 'Direct column (apply log1p at training time)',
        availability   = 'pre_settlement',
        leakage_risk   = 'none',
        leakage_reason = 'Amount is specified by the user before the transaction executes',
        model_1        = 'yes',
        model_2        = 'yes',
        model_3        = 'yes',
        notes          = 'Highly right-skewed. log1p transform recommended for XGBoost (reduces dominance of very large values).'
    ),
    dict(
        feature_name   = 'oldbalanceOrg',
        source         = 'raw',
        derivation     = 'Direct column',
        availability   = 'pre_settlement',
        leakage_risk   = 'investigate',
        leakage_reason = 'PRE-transaction sender balance — safe. But 33% zero for legit, 0.5% for fraud — strong signal worth monitoring.',
        model_1        = 'yes',
        model_2        = 'yes',
        model_3        = 'yes',
        notes          = 'Strong fraud signal. Apply log1p. Zero = account with no money (unusual for fraud targets).'
    ),
    dict(
        feature_name   = 'oldbalanceDest',
        source         = 'raw',
        derivation     = 'Direct column',
        availability   = 'pre_settlement',
        leakage_risk   = 'investigate',
        leakage_reason = 'PRE-transaction recipient balance — safe. 65% zero for fraud suggests throwaway accounts.',
        model_1        = 'yes',
        model_2        = 'yes',
        model_3        = 'yes',
        notes          = 'May distinguish merchants (high balance) from mule accounts (zero balance). Apply log1p.'
    ),
    dict(
        feature_name   = 'pct_balance_used',
        source         = 'engineered_raw',
        derivation     = 'amount / oldbalanceOrg (handle div-by-zero: set to 0 or cap at 100)',
        availability   = 'pre_settlement',
        leakage_risk   = 'none',
        leakage_reason = 'Uses only amount and oldbalanceOrg, both pre-settlement. Fraud median = 100% (full drain).',
        model_1        = 'yes',
        model_2        = 'yes',
        model_3        = 'yes',
        notes          = 'STRONG SIGNAL. Fraud drains exactly 100% of balance. Cap at 100 or clip outliers from div-by-small.'
    ),
    # ── Excluded columns ─────────────────────────────────────────────────────
    dict(
        feature_name   = 'newbalanceOrig',
        source         = 'raw',
        derivation     = 'Direct column — EXCLUDED',
        availability   = 'post_settlement',
        leakage_risk   = 'CONFIRMED_LEAKAGE',
        leakage_reason = 'Post-settlement sender balance. 98% zero in fraud (drain confirmation). Available only AFTER transaction completes.',
        model_1        = 'NO',
        model_2        = 'NO (current txn); historical aggregate safe',
        model_3        = 'NO',
        notes          = 'Excluded from all models for current transaction. See balance_discrepancy_hist for safe alternative.'
    ),
    dict(
        feature_name   = 'newbalanceDest',
        source         = 'raw',
        derivation     = 'Direct column — EXCLUDED',
        availability   = 'post_settlement',
        leakage_risk   = 'CONFIRMED_LEAKAGE',
        leakage_reason = 'Post-settlement recipient balance. Only available after transaction completes.',
        model_1        = 'NO',
        model_2        = 'NO',
        model_3        = 'NO',
        notes          = 'Excluded from all models.'
    ),
    dict(
        feature_name   = 'isFlaggedFraud',
        source         = 'raw',
        derivation     = 'Direct column — EXCLUDED',
        availability   = 'pre_settlement',
        leakage_risk   = 'SIMULATION_ARTEFACT',
        leakage_reason = '16 flagged rows total. 0.19% recall. Simulation rule-based flag — not a real bank system output.',
        model_1        = 'NO',
        model_2        = 'NO',
        model_3        = 'NO',
        notes          = 'Would add noise, not signal. Excluded permanently.'
    ),
    dict(
        feature_name   = 'nameOrig',
        source         = 'raw',
        derivation     = 'IDENTIFIER — not used as feature',
        availability   = 'pre_settlement',
        leakage_risk   = 'high_cardinality',
        leakage_reason = '6.3M unique values. Cannot be OHE. Used only for graph construction and historical aggregation.',
        model_1        = 'NO (identifier only)',
        model_2        = 'used to compute graph features',
        model_3        = 'graph construction',
        notes          = 'Drop before model training. Use for edge creation in GNN.'
    ),
    dict(
        feature_name   = 'nameDest',
        source         = 'raw',
        derivation     = 'IDENTIFIER — not used as feature',
        availability   = 'pre_settlement',
        leakage_risk   = 'high_cardinality',
        leakage_reason = '2.7M unique values. Cannot be OHE.',
        model_1        = 'NO (identifier only)',
        model_2        = 'used to compute graph features',
        model_3        = 'graph construction',
        notes          = 'Drop before model training. Key for recipient-side features in Model 2.'
    ),
    # ── Graph / historical features (Model 2 only) ───────────────────────────
    dict(
        feature_name   = 'balance_discrepancy',
        source         = 'engineered_raw',
        derivation     = 'abs(oldbalanceOrg - newbalanceOrig - amount) — LEAKY for current txn',
        availability   = 'post_settlement',
        leakage_risk   = 'CONFIRMED_LEAKAGE (current transaction)',
        leakage_reason = 'Requires newbalanceOrig which is post-settlement. 99.5% zero in fraud.',
        model_1        = 'NO — leakage',
        model_2        = 'NO (current txn). balance_discrepancy_hist safe (see below)',
        model_3        = 'NO',
        notes          = 'Cannot use current txn value. Historical aggregate (recipient has zero-drain history) is safe.'
    ),
    dict(
        feature_name   = 'recipient_txn_count',
        source         = 'graph_historical',
        derivation     = 'For each txn: count of prior transactions where nameDest == current nameDest, step < current step',
        availability   = 'pre_settlement',
        leakage_risk   = 'none (if computed correctly)',
        leakage_reason = 'Only counts prior transactions. Current transaction excluded.',
        model_1        = 'NO — requires historical join',
        model_2        = 'YES',
        model_3        = 'graph feature / node embedding input',
        notes          = 'Compute with time-ordered groupby shift. First appearance = 0. Strong fraud signal: new recipients are high-risk.'
    ),
    dict(
        feature_name   = 'recipient_total_received',
        source         = 'graph_historical',
        derivation     = 'Cumulative sum of amount for prior transactions to same nameDest',
        availability   = 'pre_settlement',
        leakage_risk   = 'none (if computed correctly)',
        leakage_reason = 'Historical only',
        model_1        = 'NO',
        model_2        = 'YES',
        model_3        = 'graph feature',
        notes          = 'Apply log1p. Captures whether recipient is a high-volume account (likely merchant) vs new account.'
    ),
    dict(
        feature_name   = 'recipient_distinct_senders',
        source         = 'graph_historical',
        derivation     = 'Count of unique nameOrig accounts that sent to this nameDest before current step',
        availability   = 'pre_settlement',
        leakage_risk   = 'none (if computed correctly)',
        leakage_reason = 'Historical only',
        model_1        = 'NO',
        model_2        = 'YES',
        model_3        = 'graph feature',
        notes          = 'High value = merchant/mule. Low value (0-1) = new or rarely-used account.'
    ),
    dict(
        feature_name   = 'is_new_recipient',
        source         = 'graph_historical',
        derivation     = 'Binary: 1 if recipient_txn_count == 0 else 0',
        availability   = 'pre_settlement',
        leakage_risk   = 'none',
        leakage_reason = 'Derived from recipient_txn_count',
        model_1        = 'NO',
        model_2        = 'YES',
        model_3        = 'graph feature',
        notes          = 'Simple but powerful: throwaway mule accounts often appear only once. Fraud % much higher for new recipients.'
    ),
    dict(
        feature_name   = 'pair_prior_interactions',
        source         = 'graph_historical',
        derivation     = 'Count of prior transactions with this exact (nameOrig, nameDest) pair',
        availability   = 'pre_settlement',
        leakage_risk   = 'none (if computed correctly)',
        leakage_reason = 'Historical only',
        model_1        = 'NO',
        model_2        = 'YES (low priority)',
        model_3        = 'edge weight in graph',
        notes          = 'Low priority: senders almost always transact once, so this will be 0 for nearly every row.'
    ),
]

register_df = pd.DataFrame(register_rows)
register_path = REPORTS_DIR / 'feature_register.csv'
register_df.to_csv(register_path, index=False)
print(f'Feature register saved: {register_path}')

print('\n--- SUMMARY TABLE: Feature inclusion by model ---')
summary_cols = ['feature_name', 'leakage_risk', 'model_1', 'model_2', 'model_3']
print(register_df[summary_cols].to_string(index=False))

sep('PHASE 11 — CONFIRMED FEATURE LISTS')

print('\n=== MODEL 1 (XGBoost Baseline — raw features only) ===')
m1_features = register_df[register_df['model_1'] == 'yes']['feature_name'].tolist()
print('Features:')
for f in m1_features:
    print(f'  + {f}')

print('\n=== MODEL 2 (XGBoost + Graph Features) ===')
print('Raw features: same as Model 1, plus:')
m2_extra = register_df[
    (register_df['model_2'].str.startswith('YES')) &
    (register_df['model_1'] == 'NO')
]['feature_name'].tolist()
for f in m2_extra:
    print(f'  + {f}')

print('\n=== MODEL 3 (GNN — GraphSAGE) ===')
print('Node features: same as Model 1 raw features (one feature vector per transaction-node)')
print('Graph structure: edges from shared nameOrig or nameDest within 30-step window')

# ═════════════════════════════════════════════════════════════════════════════
# PHASE 12 — VALIDATION STRATEGY
# ═════════════════════════════════════════════════════════════════════════════
sep('PHASE 12 — WHY CHRONOLOGICAL SPLITS (NOT RANDOM K-FOLD)?')

note("""
  In most ML courses, you learn to use random train/test splits or k-fold
  cross-validation. For time-series or transaction data, this is WRONG.
  Here is why:

  Imagine transaction #1000 is in the test set and transaction #1001 is in
  the training set. With a random split, your model is trained on data that
  comes FROM THE FUTURE relative to the test point. In the real world, when
  you deploy a fraud model, you make decisions on today's transactions using
  only history from before today. You never have access to "future" history.

  CHRONOLOGICAL SPLITS enforce this discipline:
    - Training set:   step 1 to 534   (the past)
    - Validation set: step 535 to 638 (near future — used for tuning)
    - Test set:       step 639+        (far future — untouched until final eval)

  Any feature computed from historical aggregates (recipient_txn_count, etc.)
  must only look backwards — never at transactions in the same or future split.

  GRAPH FEATURES: When computing recipient_txn_count for a VALIDATION row,
  you use the full training history + prior validation transactions up to
  that step. When computing for a TEST row, you use training + all validation +
  prior test transactions up to that step. This mirrors real deployment.
""")

sep('PHASE 12 — CONFIRMED SPLIT STATISTICS')

print('Reproducing split statistics from Batch 2 analysis:')
split_stats = df.groupby('split', observed=True).agg(
    total=('isFraud','count'),
    fraud_count=('isFraud','sum'),
    fraud_rate=('isFraud','mean')
).assign(
    fraud_rate_pct = lambda x: (x['fraud_rate']*100).round(4),
    pct_of_data    = lambda x: (x['total'] / len(df) * 100).round(1)
)
print(split_stats.to_string())

note("""
  CRITICAL OBSERVATION — FRAUD RATE DRIFT:
    Train : 0.097% fraud rate
    Val   : 0.70%  fraud rate   (7x higher than train)
    Test  : 1.36%  fraud rate   (14x higher than train)

  This is NOT a problem with our split — it reflects reality in PaySim.
  Fraud activity ramps up over simulated time (later steps = more fraud).

  IMPLICATION for scale_pos_weight:
  XGBoost's scale_pos_weight tells the model "treat each positive (fraud)
  example as if it appeared this many times, to compensate for imbalance."
  It is used DURING TRAINING. So we calculate it from the TRAINING SPLIT only.

  Correct calculation:
    n_negatives / n_positives = (6,118,803 - 5,935) / 5,935 = 1030.7

  Using the full-dataset ratio (773.7) would UNDERCOUNT how rare fraud is in
  the training set, making the model slightly under-compensate for imbalance
  during training. The error is small (773 vs 1031) but it compounds when
  you're dealing with a 0.097% fraud rate — every bit of calibration matters.

  At EVALUATION TIME (val and test), scale_pos_weight is irrelevant —
  you just score each transaction and compare against the true label.
  The changing fraud rate in val/test does NOT affect the model's internal
  calibration, but it DOES affect the probability scores you compare against
  a threshold. This is why we choose the threshold on the VALIDATION set
  (not the training set).
""")

sep('PHASE 12 — SCALE_POS_WEIGHT CORRECTION')

n_train_neg = int(split_stats.loc['train','total'] - split_stats.loc['train','fraud_count'])
n_train_pos = int(split_stats.loc['train','fraud_count'])
spw_correct = round(n_train_neg / n_train_pos, 1)

print(f'  Training set negatives (legit)  : {n_train_neg:,}')
print(f'  Training set positives (fraud)  : {n_train_pos:,}')
print(f'  Correct scale_pos_weight        : {spw_correct}')
print(f'  Previously used (full dataset)  : 773.7')
print(f'\n  [ACTION REQUIRED] Update configs/baseline_xgboost.yaml:')
print(f'    scale_pos_weight: {spw_correct}')

sep('PHASE 12 — WHY PR-AUC, NOT ROC-AUC OR ACCURACY?')

note("""
  THREE METRICS EXPLAINED:

  1. ACCURACY = (TP + TN) / total
     BAD for fraud. If we predict "legit" for every single transaction,
     accuracy = 99.87% because 99.87% of rows ARE legit.
     That model catches zero fraud. Accuracy hides the failure completely.

  2. ROC-AUC = area under the Receiver Operating Characteristic curve.
     Better than accuracy. Measures how well the model ranks fraud above legit.
     Problem: ROC-AUC is optimistic on very imbalanced datasets because it
     considers ALL negatives in the denominator. With 6.3M negatives, the model
     gets "credit" for correctly ranking negatives even if it misses most fraud.
     ROC-AUC of 0.97 sounds great but the model might still miss 40% of fraud.

  3. PR-AUC = area under the Precision-Recall curve.
     BEST for fraud. Precision = "of transactions flagged as fraud, how many
     really are fraud?" Recall = "of all actual fraud, how many did we catch?"
     PR-AUC is sensitive to the performance on the POSITIVE CLASS only. With
     0.1% fraud rate, PR-AUC rewards models that both catch fraud (high recall)
     and don't spam the fraud team with false alarms (high precision).

  PRIMARY METRIC: PR-AUC
  SECONDARY METRICS: F1, Precision@K, Recall@threshold
  NEVER: Accuracy as a standalone metric
""")

sep('PHASE 12 — THRESHOLD SELECTION STRATEGY')

note("""
  XGBoost outputs a PROBABILITY (0 to 1) for each transaction. We must
  choose a THRESHOLD above which we call a transaction "fraud."

  Default threshold = 0.5. This is almost always wrong for imbalanced data.
  The model's output probabilities are calibrated to the training set.
  With 0.097% fraud in training, the model will almost never output a
  probability > 0.5 — so using 0.5 as a threshold will flag almost nothing.

  CORRECT APPROACH:
    1. Train the model on the training set.
    2. Score the VALIDATION set.
    3. Sweep thresholds from 0.01 to 0.99 in steps of 0.01.
    4. For each threshold, compute Precision, Recall, F1 on the validation set.
    5. Choose the threshold that maximises F1 (or the business-appropriate
       trade-off: a fraud team might accept lower precision to maximise recall).
    6. Apply that chosen threshold to the TEST set — ONCE — and report final metrics.

  The test set is UNTOUCHED until the final evaluation of each model.
  Choosing the threshold on the validation set ensures we do not overfit to
  the test set's specific fraud distribution.
""")

sep('PHASE 12 — PREPROCESSING RULES (no-leakage checklist)')

print("""
  Rule 1 — FIT ONLY ON TRAINING DATA
    Any transformation with learned parameters (StandardScaler, MinMaxScaler,
    LabelEncoder, OneHotEncoder, log transforms) must be FIT on training data
    only. Then TRANSFORM val and test using the training-fitted parameters.

    Wrong:   scaler.fit_transform(all_data)    <- leaks test stats into training
    Right:   scaler.fit(train); scaler.transform(val); scaler.transform(test)

  Rule 2 — LOG TRANSFORMS
    For XGBoost, we apply log1p(x) to skewed numeric features (amount,
    oldbalanceOrg, oldbalanceDest). log1p is a deterministic function
    (no learned parameters), so it can be applied to train, val, test identically.
    No fitting needed.

  Rule 3 — ONE-HOT ENCODING
    When OHE-ing 'type', fit the encoder on the training set so it learns the
    5 category values. The val/test sets are transformed using those 5 categories.
    If a category appears in val/test but not in train, the encoder returns 0 for
    that column. (In PaySim this won't happen — all 5 types appear in training.)

  Rule 4 — GRAPH FEATURES
    Historical aggregates (recipient_txn_count, etc.) are computed SEQUENTIALLY,
    sorted by step. For each transaction, only prior transactions are used.
    Never sort randomly and compute a rolling window — that uses future data.

  Rule 5 — SCALING (for XGBoost — actually OPTIONAL)
    Decision tree based models like XGBoost do NOT require feature scaling.
    They make decisions by splitting on thresholds, not distances. Log1p
    transforms help because they reduce the influence of extreme outliers on
    split thresholds, but StandardScaler is not needed for XGBoost.
    It IS needed for the GNN (Model 3) which uses neural network layers.
""")

# ═════════════════════════════════════════════════════════════════════════════
# PHASE 13 — EDA CLOSURE AND HANDOVER
# ═════════════════════════════════════════════════════════════════════════════
sep('PHASE 13 — EDA CLOSURE: FULL FINDINGS SUMMARY')

findings_summary = {
    'Dataset'                 : '6,362,620 rows x 11 columns. No missing values. No duplicates.',
    'Fraud rate (full)'       : '0.1291% (8,213 fraud rows). Severe class imbalance.',
    'Fraud by type'           : 'ONLY in TRANSFER (0.77% rate) and CASH_OUT (0.18% rate). Never in PAYMENT, CASH_IN, DEBIT.',
    'Time range'              : 'step 1 to 743 (simulated hours). No calendar dates. No missing steps.',
    'Split boundaries'        : 'train <= 534 | val 535-638 | test > 638',
    'Fraud per split'         : 'train=5,935 (0.097%) | val=1,106 (0.70%) | test=1,172 (1.36%)',
    'Feature drift'           : 'No significant drift in amount, oldbalanceOrg, oldbalanceDest across splits (all PSI < 0.10)',
    'Fraud rate drift'        : 'Fraud rate is 14x higher in test than train — expected in simulation, not a data problem',
    'scale_pos_weight'        : f'Corrected to {spw_correct} (training split). Was 773.7 (full dataset).',
    'Leakage exclusions'      : 'newbalanceOrig (post-settlement) | newbalanceDest (post-settlement) | isFlaggedFraud (sim artefact)',
    'Drain pattern'           : 'Fraud drains sender to exactly zero. pct_balance_used median=100% in fraud.',
    'balance_discrepancy'     : 'LEAKY (uses newbalanceOrig). EXCLUDED from Model 1 and Model 2 current-txn features.',
    'Entity profiling'        : 'Senders: 99%+ transact once. Recipients: up to 113 txns. Super-nodes: 1,033.',
    'Sender features dropped' : 'sender_txn_count and sender variants excluded — near-zero variance.',
    'Top feature candidates'  : 'pct_balance_used, oldbalanceOrg, amount, type, recipient_txn_count (Model 2+)',
    'Primary metric'          : 'PR-AUC. Secondary: F1, Precision, Recall. Never accuracy.',
    'Threshold strategy'      : 'Select threshold on val set; apply once to test set.',
}

print('Complete EDA findings:')
for k, v in findings_summary.items():
    print(f'  {k:30s}: {v}')

sep('PHASE 13 — CONFIG UPDATE: WRITE CORRECTED baseline_xgboost.yaml')

note("""
  We update baseline_xgboost.yaml with the corrected scale_pos_weight (1030.7)
  and the confirmed feature exclusion list. This is the single source of truth
  for Model 1 hyperparameters.
""")

yaml_path = PROJECT_ROOT / 'configs' / 'baseline_xgboost.yaml'
yaml_content = f"""# ---------------------------------------------------------------------------
# Model 1 — XGBoost Baseline (transaction features only)
# Updated by run_eda_batch3.py — confirmed from full EDA (Batches 1-3)
# ---------------------------------------------------------------------------

mlflow_experiment: "fraud_model_1_baseline"

model:
  n_estimators: 500
  max_depth: 6
  learning_rate: 0.05
  subsample: 0.8
  colsample_bytree: 0.8
  # CORRECTED: recalculated on training split only (step <= {TRAIN_END})
  # Formula: n_negatives / n_positives = {n_train_neg:,} / {n_train_pos:,} = {spw_correct}
  # Full-dataset ratio (773.7) was wrong — training set has lower fraud rate (0.097%)
  scale_pos_weight: {spw_correct}
  eval_metric: "aucpr"
  early_stopping_rounds: 30
  random_state: 42

features:
  # Apply log1p transform to these at preprocessing time (skewed, large range)
  log1p_transform:
    - "amount"
    - "oldbalanceOrg"
    - "oldbalanceDest"

  # One-hot encode (5 categories: CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER)
  one_hot_encode:
    - "type"

  # Engineered raw features (no leakage, derivable from raw columns)
  engineered:
    - "hour_of_day"          # step % 24
    - "day_of_sim"           # step // 24
    - "pct_balance_used"     # amount / oldbalanceOrg (cap div-by-zero at 0)

  # EXCLUDED: post-settlement leakage
  exclude_leakage:
    - "newbalanceOrig"       # post-settlement; 98% zero in fraud — LEAKAGE
    - "newbalanceDest"       # post-settlement — LEAKAGE

  # EXCLUDED: simulation artefact
  exclude_artefact:
    - "isFlaggedFraud"       # 16 flagged rows, 0.19% recall, simulation rule-based

  # EXCLUDED: identifiers (high-cardinality, not encodable directly)
  exclude_identifiers:
    - "nameOrig"             # 6.3M unique — graph construction only
    - "nameDest"             # 2.7M unique — graph construction only

  # EXCLUDED: balance_discrepancy (uses newbalanceOrig — post-settlement leakage)
  # pct_balance_used is the safe alternative (uses only amount + oldbalanceOrg)
  exclude_leakage_engineered:
    - "balance_discrepancy"

validation:
  train_end_step: {TRAIN_END}
  val_end_step: {VAL_END}
  primary_metric: "pr_auc"
  secondary_metrics:
    - "roc_auc"
    - "f1"
    - "precision"
    - "recall"
  # Threshold is selected by sweeping val set; applied once to test set
  threshold_sweep_range: [0.01, 0.99]
  threshold_sweep_steps: 99

artifacts_dir: "artifacts/model_1_baseline"
predictions_dir: "data/model_1_baseline/output"
"""

with open(yaml_path, 'w', encoding='utf-8') as fh:
    fh.write(yaml_content)
print(f'  Updated: {yaml_path}')

sep('PHASE 13 — MODEL 2 FEATURE CONSTRUCTION NOTES')

note("""
  For Model 2, graph features must be computed in a specific order.
  Here is the construction algorithm (to be implemented in Phase 3):

  STEP 1 — Sort ALL data by step (ascending).

  STEP 2 — Compute recipient-side expanding window features:
    For each row i (in step order):
      recipient_txn_count[i]      = number of rows j where j < i AND nameDest[j] == nameDest[i]
      recipient_total_received[i] = sum of amount[j] for those rows j
      recipient_distinct_senders[i] = nunique of nameOrig[j] for those rows j
      is_new_recipient[i]         = 1 if recipient_txn_count[i] == 0 else 0

    Implementation: use pandas groupby + cumcount/cumsum with shift(1).
    shift(1) ensures we exclude the current transaction from its own aggregate.

  STEP 3 — Compute pair interaction features:
    pair_prior_interactions[i] = count of rows j where j < i AND
                                  nameOrig[j] == nameOrig[i] AND nameDest[j] == nameDest[i]
    (Expected to be near-zero for all rows given sender one-shot behaviour)

  STEP 4 — These features are computed ONCE on the FULL sorted dataset.
    The train/val/test split is applied AFTER. This is correct because
    each row only uses data from prior rows — there is no look-ahead.
    (This is equivalent to computing expanding window features on a live stream.)

  WARNING: Do NOT compute these features separately per split and concatenate.
    A val-set transaction that comes after many train-set transactions would
    have incorrect counts if computed in isolation.
""")

sep('PHASE 13 — OPEN QUESTIONS AND KNOWN RISKS')

print("""
  1. FRAUD RATE RAMP-UP
     Fraud rate is 14x higher in test than train. The model will be trained
     on 0.097% fraud but evaluated on 1.36% fraud. This is EXPECTED in PaySim
     but means test set scores (PR-AUC, F1) will look better than they would
     on a random sample. When comparing models, compare only within the same
     test set, not across different splits.

  2. SIMULATION ARTEFACTS
     PaySim is synthetic. The balance drain pattern (100% account drain) is
     very clean — cleaner than real fraud data. Real fraud may show partial
     drains, splits across accounts, and more varied behaviour. Be cautious
     about claiming strong real-world performance from PaySim results.

  3. ZERO OLDBALANCEDEST
     65% of fraud rows have oldbalanceDest = 0 (throwaway mule accounts).
     This is a strong signal but may be simulation-specific. Flag this in
     model evaluation.

  4. GNN SUPER-NODES
     1,033 recipient accounts receive >50 transactions. In the transaction-node
     graph, these super-nodes will be connected to 50+ transaction-nodes.
     During GraphSAGE neighbourhood sampling, this creates very large
     neighbourhoods that can dominate aggregation. Cap neighbourhood size
     (fan-out) at 10-25 neighbours during GNN training. See Phase 4 notes.

  5. THRESHOLD CALIBRATION
     The model's output probabilities may not be well-calibrated given the
     extreme class imbalance. Consider Platt scaling or isotonic regression
     on the validation set if raw probability scores are needed for business logic.
""")

sep('PHASE 13 — FINAL CHART: EDA Summary Dashboard')

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Chart 1: Fraud count per split
split_stats['fraud_count'].plot(kind='bar', ax=axes[0,0],
    color=['steelblue','darkorange','seagreen'], edgecolor='white')
axes[0,0].set_title('Fraud count per split')
axes[0,0].set_xlabel('')
axes[0,0].tick_params(axis='x', rotation=0)
for bar, val in zip(axes[0,0].patches, split_stats['fraud_count'].values):
    axes[0,0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+10,
                   f'{int(val):,}', ha='center', va='bottom', fontsize=9)

# Chart 2: Fraud rate per split
(split_stats['fraud_rate_pct']).plot(kind='bar', ax=axes[0,1],
    color=['steelblue','darkorange','seagreen'], edgecolor='white')
axes[0,1].set_title('Fraud rate (%) per split\n(note 14x increase from train to test)')
axes[0,1].set_xlabel('')
axes[0,1].tick_params(axis='x', rotation=0)

# Chart 3: Model 1 feature list
feature_text = '\n'.join([f'+ {f}' for f in m1_features])
axes[0,2].text(0.05, 0.95, 'MODEL 1 FEATURES\n\n' + feature_text,
               transform=axes[0,2].transAxes, va='top', fontsize=9,
               fontfamily='monospace')
axes[0,2].set_title('Confirmed Model 1 feature list')
axes[0,2].axis('off')

# Chart 4: pct_balance_used by class
tf_co = df[df['type'].isin(['TRANSFER','CASH_OUT'])].copy()
tf_co['pct_balance_used'] = (tf_co['amount'] /
    tf_co['oldbalanceOrg'].replace(0, np.nan)).fillna(0).clip(0, 5)
for label, color, lname in [(0,'steelblue','Legit'), (1,'tomato','Fraud')]:
    sub = tf_co[tf_co['isFraud']==label]['pct_balance_used']
    samp = sub.sample(min(5000, len(sub)), random_state=42)
    axes[1,0].hist(samp, bins=50, alpha=0.55, color=color, label=lname, density=True)
axes[1,0].set_title('pct_balance_used distribution\n(TRANSFER+CASH_OUT, clipped at 5x)')
axes[1,0].set_xlabel('amount / oldbalanceOrg')
axes[1,0].legend()

# Chart 5: hour_of_day fraud rate
hourly_fraud = df.groupby('hour_of_day')['isFraud'].mean() * 100
axes[1,1].bar(hourly_fraud.index, hourly_fraud.values, color='tomato', edgecolor='white', alpha=0.8)
axes[1,1].set_title('Fraud rate (%) by hour of day\n(step % 24, simulated)')
axes[1,1].set_xlabel('Hour of simulated day')
axes[1,1].set_ylabel('Fraud rate (%)')

# Chart 6: scale_pos_weight comparison
spw_values  = [773.7, spw_correct]
spw_labels  = ['Full dataset\n(wrong)', f'Training split\n(correct={spw_correct})']
colors      = ['tomato', 'seagreen']
bars = axes[1,2].bar(spw_labels, spw_values, color=colors, edgecolor='white', width=0.4)
for bar, val in zip(bars, spw_values):
    axes[1,2].text(bar.get_x()+bar.get_width()/2, bar.get_height()+5,
                   f'{val}', ha='center', va='bottom', fontsize=10, fontweight='bold')
axes[1,2].set_title('scale_pos_weight correction\n(train split has rarer fraud than full dataset)')
axes[1,2].set_ylabel('scale_pos_weight value')

plt.suptitle('Phase 13 — EDA Closure Dashboard', fontsize=13, fontweight='bold')
plt.tight_layout()
save_chart(fig, 'eda_closure_dashboard')

sep('PHASE 13 — FINAL SUMMARY REPORT')

summary_path = REPORTS_DIR / 'eda_batch3_summary.md'
with open(summary_path, 'w', encoding='utf-8') as fh:
    fh.write('# EDA Batch 3 Summary — Phases 10-13\n\n')
    fh.write('## Feature Register\n\n')
    fh.write('See: `reports/data_quality/feature_register.csv`\n\n')
    fh.write('## Confirmed Model 1 Features\n\n')
    for f in m1_features:
        fh.write(f'- `{f}`\n')
    fh.write('\n## Confirmed Model 2 Additional Features\n\n')
    for f in m2_extra:
        fh.write(f'- `{f}` (historical aggregate — see construction notes)\n')
    fh.write('\n## Key Corrections from Batch 3\n\n')
    fh.write(f'- `scale_pos_weight` corrected from 773.7 to **{spw_correct}** (training split)\n')
    fh.write(f'- `balance_discrepancy` confirmed LEAKY — excluded from Models 1 and 2 (current txn)\n')
    fh.write(f'- Sender-side RFM features dropped (99%+ of senders transact exactly once)\n')
    fh.write(f'- Recipient-side features confirmed as primary graph feature set\n')
    fh.write('\n## Validation Strategy\n\n')
    fh.write(f'- Chronological splits: train ≤ {TRAIN_END} | val ≤ {VAL_END} | test > {VAL_END}\n')
    fh.write(f'- Primary metric: PR-AUC\n')
    fh.write(f'- Threshold selected on val set, applied once to test set\n')
    fh.write(f'- Preprocessing fit on train only\n')
    fh.write('\n## Next Step\n\nPhase 2 — Model 1: XGBoost Baseline pipeline.\n')
print(f'  Saved: {summary_path}')

sep('EDA BATCH 3 COMPLETE — FULL EDA DONE')
print("""
  All EDA phases (1-13) are now complete. Outputs:
    reports/data_quality/feature_register.csv
    reports/data_quality/eda_batch3_summary.md
    reports/data_quality/charts/eda_closure_dashboard.png
    configs/baseline_xgboost.yaml  (UPDATED with corrected scale_pos_weight)

  READY FOR: Phase 2 — Model 1 (XGBoost Baseline)
  Say "Start Model 1" to begin the XGBoost baseline pipeline.
""")
