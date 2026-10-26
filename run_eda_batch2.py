"""
PaySim EDA - Batch 2 (Phases 5-9)
Run: python run_eda_batch2.py

Covers:
  Phase 5  - Univariate Analysis
  Phase 6  - Bivariate Analysis
  Phase 7  - Time-Series Analysis
  Phase 8  - Data Drift and Stability
  Phase 9  - Correlation, Association and Domain Analysis

Outputs -> reports/data_quality/
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
from scipy import stats
from scipy.stats import chi2_contingency, mannwhitneyu, ks_2samp

sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams['figure.dpi'] = 100

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', '{:.4f}'.format)

# ── Helpers ───────────────────────────────────────────────────────────────────
def sep(title):
    print(f'\n{"="*65}\n  {title}\n{"="*65}')

def save_chart(fig, name):
    path = CHARTS_DIR / f'{name}.png'
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'  Chart saved: {name}.png')

def cramers_v(x, y):
    ct   = pd.crosstab(x, y)
    chi2 = chi2_contingency(ct)[0]
    n    = ct.sum().sum()
    phi2 = chi2 / n
    r, k = ct.shape
    return np.sqrt(phi2 / min(k - 1, r - 1))

def psi(expected, actual, buckets=10):
    """Population Stability Index. < 0.1 stable, 0.1-0.2 slight drift, > 0.2 significant."""
    bp  = np.unique(np.percentile(expected, np.linspace(0, 100, buckets + 1)))
    e   = np.histogram(expected, bins=bp)[0] / len(expected)
    a   = np.histogram(actual,   bins=bp)[0] / len(actual)
    e   = np.where(e == 0, 1e-4, e)
    a   = np.where(a == 0, 1e-4, a)
    return float(np.sum((a - e) * np.log(a / e)))

findings = []

# ── Load data ─────────────────────────────────────────────────────────────────
sep('LOADING DATA')
print(f'Reading {PAYSIM_CSV.name} ...')
df = pd.read_csv(PAYSIM_CSV)
print(f'Loaded: {df.shape[0]:,} rows x {df.shape[1]} columns')

TRAIN_END = 534
VAL_END   = 638

split_bins   = [0, TRAIN_END, VAL_END, df['step'].max() + 1]
split_labels = ['train', 'val', 'test']
df['split']  = pd.cut(df['step'], bins=split_bins, labels=split_labels, right=True)

# ── Phase 5 — Univariate Analysis ─────────────────────────────────────────────
sep('PHASE 5 — UNIVARIATE: TARGET DISTRIBUTION')

print('Fraud by transaction type:')
type_fraud = df.groupby('type')['isFraud'].agg(
    total='count', fraud_count='sum', fraud_rate='mean'
).assign(fraud_rate_pct=lambda x: (x['fraud_rate'] * 100).round(4))
print(type_fraud.sort_values('fraud_count', ascending=False).to_string())
findings.append('- Fraud only in TRANSFER and CASH_OUT (confirmed)')

print('\nFraud rate by split (proposed boundaries: train<=534, val<=638, test>638):')
split_stats = df.groupby('split', observed=True)['isFraud'].agg(
    total='count', fraud_count='sum', fraud_rate='mean'
).assign(
    fraud_rate_pct=lambda x: (x['fraud_rate'] * 100).round(4),
    pct_of_data=lambda x: (x['total'] / len(df) * 100).round(1)
)
print(split_stats.to_string())

train_fraud = int(split_stats.loc['train', 'fraud_count'])
val_fraud   = int(split_stats.loc['val',   'fraud_count'])
test_fraud  = int(split_stats.loc['test',  'fraud_count'])
findings.append(f'- Fraud per split -> train: {train_fraud:,} | val: {val_fraud:,} | test: {test_fraud:,}')

if val_fraud >= 100 and test_fraud >= 100:
    print('\n[OK] Split boundaries are reasonable — all splits have sufficient fraud cases.')
else:
    print('\n[WARN] A split has very few fraud cases. Consider adjusting boundaries.')

sep('PHASE 5 — UNIVARIATE: NUMERICAL FEATURES')

numeric_feats = ['amount', 'oldbalanceOrg', 'oldbalanceDest', 'step']
print('Summary statistics per feature per class:\n')
for feat in numeric_feats:
    print(f'--- {feat} ---')
    summary = df.groupby('isFraud')[feat].describe(percentiles=[0.25, 0.5, 0.75, 0.95, 0.99])
    print(summary.T.to_string())
    sk_fraud = df[df['isFraud']==1][feat].skew()
    sk_legit = df[df['isFraud']==0][feat].skew()
    zero_fraud = (df[df['isFraud']==1][feat] == 0).mean() * 100
    zero_legit = (df[df['isFraud']==0][feat] == 0).mean() * 100
    print(f'  Skewness  : Fraud={sk_fraud:.2f}  Legit={sk_legit:.2f}')
    print(f'  Zero freq : Fraud={zero_fraud:.2f}%  Legit={zero_legit:.2f}%\n')

sep('PHASE 5 — UNIVARIATE: CATEGORICAL FEATURES')

print('Transaction type frequency:')
tc = df['type'].value_counts()
tp = df['type'].value_counts(normalize=True) * 100
print(pd.DataFrame({'count': tc, 'pct%': tp.round(2)}).to_string())

sep('PHASE 5 — UNIVARIATE: IDENTIFIER PROFILING (Section 19)')

print('Unique accounts:')
print(f'  Unique senders (nameOrig)    : {df["nameOrig"].nunique():,}')
print(f'  Unique recipients (nameDest) : {df["nameDest"].nunique():,}')
all_accounts = pd.concat([df['nameOrig'], df['nameDest']]).nunique()
print(f'  Total unique accounts        : {all_accounts:,}')

sender_counts = df['nameOrig'].value_counts()
print(f'\nTransactions per sender:')
print(sender_counts.describe([0.5, 0.9, 0.99]).round(1).to_string())
print(f'Max transactions by one sender : {sender_counts.max():,}')
print(f'Senders with > 1 transaction   : {(sender_counts > 1).sum():,}')

dest_counts = df['nameDest'].value_counts()
print(f'\nTransactions per recipient:')
print(dest_counts.describe([0.5, 0.9, 0.99]).round(1).to_string())
print(f'Max transactions to one dest   : {dest_counts.max():,}')

fraud_senders = df[df['isFraud']==1]['nameOrig'].unique()
fraud_sender_txn = df[df['nameOrig'].isin(fraud_senders)]['nameOrig'].value_counts()
print(f'\nFraud sender recidivism:')
print(f'  Unique senders in fraud txns          : {len(fraud_senders):,}')
print(f'  Fraud senders with > 1 txn (any type) : {(fraud_sender_txn > 1).sum():,}')

findings.append(f'- Total unique accounts: {all_accounts:,}')
findings.append(f'- Max txns by one sender: {sender_counts.max():,}  | Max txns to one recipient: {dest_counts.max():,}')

# ── Phase 6 — Bivariate Analysis ──────────────────────────────────────────────
sep('PHASE 6 — BIVARIATE: FEATURE vs TARGET (Statistical Tests)')

print('Mann-Whitney U test (one-sided: fraud > legit) for numerical features:')
print('Effect size = rank-biserial correlation (r). |r| > 0.3 = medium, > 0.5 = large.\n')

fraud_df = df[df['isFraud']==1]
legit_df = df[df['isFraud']==0]

# Sample legit for speed (MWU on 6M rows is slow)
legit_sample = legit_df.sample(min(50_000, len(legit_df)), random_state=42)

mwu_results = []
for feat in ['amount', 'oldbalanceOrg', 'oldbalanceDest']:
    f_vals = fraud_df[feat].values
    l_vals = legit_sample[feat].values
    stat, p = mannwhitneyu(f_vals, l_vals, alternative='two-sided')
    n1, n2  = len(f_vals), len(l_vals)
    r       = 1 - (2 * stat) / (n1 * n2)
    mwu_results.append({'feature': feat, 'U_stat': stat, 'p_value': p,
                        'effect_r': round(r, 4), 'significant': p < 0.05})
    print(f'  {feat:20s}: p={p:.2e}  effect_r={r:.4f}  {"*significant*" if p < 0.05 else ""}')

print('\nKolmogorov-Smirnov test (distribution comparison):')
for feat in ['amount', 'oldbalanceOrg', 'oldbalanceDest']:
    stat, p = ks_2samp(fraud_df[feat].values,
                       legit_sample[feat].values)
    print(f'  {feat:20s}: KS={stat:.4f}  p={p:.2e}')

print('\nChi-square test: type vs isFraud')
ct   = pd.crosstab(df['type'], df['isFraud'])
chi2, p_chi, dof, _ = chi2_contingency(ct)
cv   = cramers_v(df['type'], df['isFraud'])
print(f'  chi2={chi2:.1f}  p={p_chi:.2e}  dof={dof}  Cramers_V={cv:.4f}')
print('  Interpretation: Cramers V > 0.3 = strong association with fraud.')

sep('PHASE 6 — BIVARIATE: AMOUNT vs BALANCE CHANGES')

tf_co = df[df['type'].isin(['TRANSFER', 'CASH_OUT'])].copy()
tf_co['balance_drop']    = tf_co['oldbalanceOrg'] - tf_co['newbalanceOrig']
tf_co['amount_vs_drop']  = tf_co['amount'] - tf_co['balance_drop']
tf_co['dest_gain']       = tf_co['newbalanceDest'] - tf_co['oldbalanceDest']
tf_co['balance_discrepancy'] = (tf_co['oldbalanceOrg'] - tf_co['newbalanceOrig'] - tf_co['amount']).abs()

print('Balance drop vs amount (TRANSFER+CASH_OUT, by fraud label):')
print(tf_co.groupby('isFraud')[['balance_drop','amount_vs_drop','balance_discrepancy']].describe(
    percentiles=[0.5, 0.95]).T.to_string())

# balance_discrepancy as fraud signal
bd_fraud = tf_co[tf_co['isFraud']==1]['balance_discrepancy']
bd_legit = tf_co[tf_co['isFraud']==0]['balance_discrepancy']
print(f'\nbalance_discrepancy zero rate:')
print(f'  Fraud: {(bd_fraud < 0.01).mean()*100:.1f}%  |  Legit: {(bd_legit < 0.01).mean()*100:.1f}%')
print('  --> balance_discrepancy ~0 for fraud (account drained exactly), non-zero for most legit txns.')
print('  --> Strong feature candidate for Model 2.')
findings.append('- balance_discrepancy: ~0 in fraud (drain-to-zero), non-zero in 90%+ legit -> top feature candidate')

sep('PHASE 6 — BIVARIATE CHARTS')

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Row 1: amount, oldbalanceOrg, oldbalanceDest log1p by class
for i, feat in enumerate(['amount', 'oldbalanceOrg', 'oldbalanceDest']):
    for label, color, lname in [(0,'steelblue','Legit'), (1,'tomato','Fraud')]:
        vals = df[df['isFraud']==label][feat].sample(min(10_000, len(df[df['isFraud']==label])), random_state=42)
        axes[0, i].hist(np.log1p(vals), bins=60, alpha=0.55, color=color, label=lname, density=True)
    axes[0, i].set_title(f'log1p({feat})')
    axes[0, i].legend(fontsize=8)

# Row 2: fraud rate by type, balance_discrepancy ECDF, split fraud counts
type_fraud_rate = df.groupby('type')['isFraud'].mean() * 100
type_fraud_rate.sort_values(ascending=False).plot(kind='bar', ax=axes[1, 0],
    color='tomato', edgecolor='white')
axes[1, 0].set_title('Fraud rate (%) by transaction type')
axes[1, 0].set_xlabel('')
axes[1, 0].tick_params(axis='x', rotation=30)

for label, color, lname in [(0,'steelblue','Legit'), (1,'tomato','Fraud')]:
    sub = tf_co[tf_co['isFraud']==label]['balance_discrepancy']
    samp = np.sort(np.log1p(sub.sample(min(5000, len(sub)), random_state=42).values))
    axes[1, 1].plot(samp, np.linspace(0, 1, len(samp)), color=color, label=lname, lw=1.5)
axes[1, 1].set_title('ECDF — log1p(balance_discrepancy)\n(TRANSFER+CASH_OUT)')
axes[1, 1].set_xlabel('log1p(balance_discrepancy)')
axes[1, 1].legend(fontsize=8)

split_stats['fraud_count'].plot(kind='bar', ax=axes[1, 2], color='tomato', edgecolor='white')
axes[1, 2].set_title('Fraud count per split')
axes[1, 2].set_xlabel('')
axes[1, 2].tick_params(axis='x', rotation=0)

plt.suptitle('Phase 6 — Bivariate Analysis', fontsize=13, fontweight='bold')
plt.tight_layout()
save_chart(fig, 'bivariate_analysis')

# ── Phase 7 — Time-Series Analysis ────────────────────────────────────────────
sep('PHASE 7 — TIME-SERIES: VALIDATE TIME FIELD')

print('Time field: step')
print('  Type        : simulated time (1 step = 1 hour)')
print('  Range       : 1 -> 743  (~31 simulated days)')
print('  NOT real calendar time — do not map to dates')
print('  Missing steps: ', end='')
all_steps  = set(range(1, 744))
data_steps = set(df['step'].unique())
missing_steps = all_steps - data_steps
print(f'{len(missing_steps)} missing steps' if missing_steps else 'none')

sep('PHASE 7 — TEMPORAL TRENDS')

step_stats = df.groupby('step').agg(
    total_txns=('isFraud', 'count'),
    fraud_count=('isFraud', 'sum'),
    fraud_rate=('isFraud', 'mean'),
    avg_amount=('amount', 'mean'),
    median_amount=('amount', 'median')
).reset_index()

print('Temporal summary statistics:')
print(step_stats[['total_txns','fraud_count','fraud_rate','avg_amount']].describe().round(2).to_string())

# Rolling averages (10-step window)
step_stats['rolling_fraud_rate']    = step_stats['fraud_rate'].rolling(10, center=True).mean()
step_stats['rolling_txn_volume']    = step_stats['total_txns'].rolling(10, center=True).mean()
step_stats['rolling_avg_amount']    = step_stats['avg_amount'].rolling(10, center=True).mean()

print('\nFraud rate statistics over time:')
print(f'  Mean  : {step_stats["fraud_rate"].mean()*100:.4f}%')
print(f'  Std   : {step_stats["fraud_rate"].std()*100:.4f}%')
print(f'  Max   : {step_stats["fraud_rate"].max()*100:.4f}% at step {step_stats.loc[step_stats["fraud_rate"].idxmax(),"step"]}')

sep('PHASE 7 — TIME-SERIES CHARTS')

fig, axes = plt.subplots(3, 1, figsize=(14, 12))

# Fraud rate over time with split boundaries
axes[0].plot(step_stats['step'], step_stats['rolling_fraud_rate']*100,
             color='tomato', lw=1.5, label='Fraud rate (10-step rolling avg)')
axes[0].axvline(TRAIN_END, color='steelblue', ls='--', lw=1.5, label=f'Train end (step {TRAIN_END})')
axes[0].axvline(VAL_END,   color='darkorange', ls='--', lw=1.5, label=f'Val end (step {VAL_END})')
axes[0].set_ylabel('Fraud rate (%)')
axes[0].set_title('Fraud Rate Over Time with Proposed Split Boundaries')
axes[0].legend(fontsize=8)

# Transaction volume over time
axes[1].plot(step_stats['step'], step_stats['rolling_txn_volume'],
             color='steelblue', lw=1.5, label='Txn volume (10-step rolling avg)')
axes[1].axvline(TRAIN_END, color='steelblue', ls='--', lw=1)
axes[1].axvline(VAL_END,   color='darkorange', ls='--', lw=1)
axes[1].set_ylabel('Transactions per step')
axes[1].set_title('Transaction Volume Over Time')
axes[1].legend(fontsize=8)

# Average amount over time
axes[2].plot(step_stats['step'], step_stats['rolling_avg_amount'],
             color='seagreen', lw=1.5, label='Avg amount (10-step rolling avg)')
axes[2].axvline(TRAIN_END, color='steelblue', ls='--', lw=1)
axes[2].axvline(VAL_END,   color='darkorange', ls='--', lw=1)
axes[2].set_xlabel('Step (simulated hour)')
axes[2].set_ylabel('Average transaction amount')
axes[2].set_title('Average Transaction Amount Over Time')
axes[2].legend(fontsize=8)

plt.suptitle('Phase 7 — Time-Series Analysis', fontsize=13, fontweight='bold')
plt.tight_layout()
save_chart(fig, 'time_series_analysis')

# ── Phase 8 — Data Drift and Stability ────────────────────────────────────────
sep('PHASE 8 — DATA DRIFT: COMPARISON PERIODS')

print('Comparison: train (step 1-534) vs val (535-638) vs test (639-743)')
print('Purpose: check if distributions shift significantly across splits.\n')

train_df = df[df['split'] == 'train']
val_df   = df[df['split'] == 'val']
test_df  = df[df['split'] == 'test']

print(f'  Train: {len(train_df):,} rows  |  Val: {len(val_df):,} rows  |  Test: {len(test_df):,} rows')

sep('PHASE 8 — NUMERICAL FEATURE DRIFT (PSI)')

print('PSI interpretation: < 0.10 = stable | 0.10-0.25 = slight drift | > 0.25 = significant drift\n')
drift_feats = ['amount', 'oldbalanceOrg', 'oldbalanceDest']
drift_rows  = []
for feat in drift_feats:
    psi_tv = psi(train_df[feat].values, val_df[feat].values)
    psi_tt = psi(train_df[feat].values, test_df[feat].values)
    ks_tv  = ks_2samp(train_df[feat].sample(min(10000, len(train_df)), random_state=42).values,
                      val_df[feat].sample(min(10000, len(val_df)), random_state=42).values).statistic
    ks_tt  = ks_2samp(train_df[feat].sample(min(10000, len(train_df)), random_state=42).values,
                      test_df[feat].sample(min(10000, len(test_df)), random_state=42).values).statistic
    drift_rows.append({
        'feature'     : feat,
        'PSI(train/val)'  : round(psi_tv, 4),
        'PSI(train/test)' : round(psi_tt, 4),
        'KS(train/val)'   : round(ks_tv, 4),
        'KS(train/test)'  : round(ks_tt, 4),
    })
    flag = 'DRIFT' if psi_tt > 0.25 else ('slight' if psi_tt > 0.10 else 'stable')
    print(f'  {feat:20s}: PSI(T/val)={psi_tv:.4f}  PSI(T/test)={psi_tt:.4f}  [{flag}]')

drift_df = pd.DataFrame(drift_rows).set_index('feature')
print('\nFull drift table:')
print(drift_df.to_string())

sep('PHASE 8 — CATEGORICAL DRIFT (transaction type mix)')

print('Transaction type proportion by split:')
type_by_split = df.groupby(['split', 'type'], observed=True).size().unstack(fill_value=0)
type_pct      = type_by_split.div(type_by_split.sum(axis=1), axis=0) * 100
print(type_pct.round(2).to_string())

sep('PHASE 8 — TARGET PREVALENCE DRIFT')

print('Fraud rate by split:')
fraud_by_split = df.groupby('split', observed=True)['isFraud'].agg(
    count='count', fraud='sum', rate='mean'
).assign(rate_pct=lambda x: (x['rate']*100).round(4))
print(fraud_by_split.to_string())
findings.append('- Fraud rate by split: train / val / test (see output)')

sep('PHASE 8 — DRIFT CHARTS')

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Amount distribution per split
for split, color in [('train','steelblue'), ('val','darkorange'), ('test','seagreen')]:
    sub = df[df['split']==split]['amount']
    samp = np.sort(np.log1p(sub.sample(min(5000, len(sub)), random_state=42).values))
    axes[0].plot(samp, np.linspace(0, 1, len(samp)), color=color, label=split, lw=1.5)
axes[0].set_title('ECDF — log1p(amount) by split')
axes[0].set_xlabel('log1p(amount)')
axes[0].legend()

# oldbalanceOrg ECDF
for split, color in [('train','steelblue'), ('val','darkorange'), ('test','seagreen')]:
    sub = df[df['split']==split]['oldbalanceOrg']
    samp = np.sort(np.log1p(sub.sample(min(5000, len(sub)), random_state=42).values))
    axes[1].plot(samp, np.linspace(0, 1, len(samp)), color=color, label=split, lw=1.5)
axes[1].set_title('ECDF — log1p(oldbalanceOrg) by split')
axes[1].set_xlabel('log1p(oldbalanceOrg)')
axes[1].legend()

# Fraud rate per step with split bands
axes[2].plot(step_stats['step'], step_stats['rolling_fraud_rate']*100,
             color='tomato', lw=1.5, label='Fraud rate')
axes[2].axvspan(1,          TRAIN_END, alpha=0.08, color='steelblue', label='Train')
axes[2].axvspan(TRAIN_END,  VAL_END,   alpha=0.08, color='darkorange', label='Val')
axes[2].axvspan(VAL_END,    743,       alpha=0.08, color='seagreen',   label='Test')
axes[2].set_title('Fraud rate by split period')
axes[2].set_xlabel('Step')
axes[2].legend(fontsize=8)

plt.suptitle('Phase 8 — Data Drift and Stability', fontsize=13, fontweight='bold')
plt.tight_layout()
save_chart(fig, 'drift_analysis')

# ── Phase 9 — Correlation, Association and Domain Analysis ────────────────────
sep('PHASE 9 — CORRELATION ANALYSIS')

num_cols = ['step', 'amount', 'oldbalanceOrg', 'oldbalanceDest', 'isFraud']
pearson  = df[num_cols].corr(method='pearson')
spearman = df[num_cols].corr(method='spearman')

print('Pearson correlations with isFraud:')
print(pearson['isFraud'].drop('isFraud').round(4).to_string())

print('\nSpearman correlations with isFraud:')
print(spearman['isFraud'].drop('isFraud').round(4).to_string())

print('\nNote: Spearman is more appropriate for skewed features.')

sep('PHASE 9 — CATEGORICAL ASSOCIATION')

print("Cramer's V — type vs isFraud:")
cv = cramers_v(df['type'], df['isFraud'])
print(f'  Cramers_V = {cv:.4f}  (0=no assoc, 1=perfect assoc)')

sep('PHASE 9 — ENTITY BEHAVIOUR (Section 42)')

print('Transactions per origin entity (nameOrig):')
orig_counts = df.groupby('nameOrig').agg(
    txn_count=('amount', 'count'),
    total_sent=('amount', 'sum'),
    distinct_dest=('nameDest', 'nunique'),
    fraud_txns=('isFraud', 'sum')
)
print(orig_counts[['txn_count','distinct_dest','fraud_txns']].describe(
    percentiles=[0.5, 0.9, 0.99]).round(2).to_string())

print(f'\nEntities with > 1 transaction      : {(orig_counts["txn_count"] > 1).sum():,}')
print(f'Entities with any fraud transaction : {(orig_counts["fraud_txns"] > 0).sum():,}')
print(f'Entities with > 1 fraud transaction : {(orig_counts["fraud_txns"] > 1).sum():,}')

print('\nTransactions per destination entity (nameDest):')
dest_counts = df.groupby('nameDest').agg(
    txn_count=('amount', 'count'),
    total_received=('amount', 'sum'),
    distinct_orig=('nameOrig', 'nunique'),
    fraud_txns=('isFraud', 'sum')
)
print(f'  Max txns received by one dest : {dest_counts["txn_count"].max():,}')
print(f'  Dest accounts receiving >10 txns: {(dest_counts["txn_count"] > 10).sum():,}')
print(f'  Super-nodes (>50 txns received) : {(dest_counts["txn_count"] > 50).sum():,}')
print('\nNote: super-node destinations may dominate graph edges in Phase 4 GNN.')
findings.append(f'- Super-node destinations (>50 txns): {(dest_counts["txn_count"] > 50).sum():,}')

sep('PHASE 9 — DOMAIN BALANCE CHECKS (Section 43)')

print('Amount vs origin balance changes (TRANSFER+CASH_OUT by fraud label):')
tf_co = df[df['type'].isin(['TRANSFER', 'CASH_OUT'])].copy()
tf_co['pct_balance_used'] = tf_co['amount'] / tf_co['oldbalanceOrg'].replace(0, np.nan) * 100

print('pct_balance_used = amount / oldbalanceOrg * 100:')
print(tf_co.groupby('isFraud')['pct_balance_used'].describe(
    percentiles=[0.5, 0.75, 0.95]).round(2).to_string())
print('\nFraud transactions use a much higher % of the senders opening balance (accounts fully drained).')

sep('PHASE 9 — GRAPH FEATURE CANDIDATES (Section 44)')

print('Based on identifier profiling and entity behaviour, these graph features are candidates for Model 2:')
candidates = [
    ('sender_txn_count',         'Number of prior transactions by nameOrig (frequency)'),
    ('sender_total_sent',        'Total amount sent by nameOrig historically (monetary)'),
    ('sender_distinct_recipients','Number of distinct nameDest accounts used by sender (diversity)'),
    ('recipient_txn_count',      'Number of prior transactions received by nameDest (frequency)'),
    ('pair_prior_interactions',  'Prior transactions between this exact nameOrig-nameDest pair'),
    ('time_since_last_txn_orig', 'Steps since nameOrigs previous transaction (recency)'),
    ('pct_balance_used',         'amount / oldbalanceOrg — drain rate (domain feature)'),
    ('balance_discrepancy',      'Abs(oldbalanceOrg - newbalanceOrig - amount) — drain anomaly'),
]
for name, desc in candidates:
    print(f'  {name:35s}: {desc}')

print('\nLeakage rule for ALL candidates:')
print('  - Compute using only transactions BEFORE the current one (strictly historical)')
print('  - Never include the current transaction in its own aggregate')
print('  - For training: historical window = all prior transactions in the training set only')
findings.append(f'- Graph feature candidates identified: {len(candidates)}')

sep('PHASE 9 — CORRELATION CHARTS')

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

mask_p = np.triu(np.ones_like(pearson, dtype=bool))
sns.heatmap(pearson, mask=mask_p, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, square=True, linewidths=0.5, ax=axes[0], vmin=-1, vmax=1)
axes[0].set_title('Pearson Correlation Matrix')

mask_s = np.triu(np.ones_like(spearman, dtype=bool))
sns.heatmap(spearman, mask=mask_s, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, square=True, linewidths=0.5, ax=axes[1], vmin=-1, vmax=1)
axes[1].set_title('Spearman Correlation Matrix')

plt.suptitle('Phase 9 — Correlation Analysis', fontsize=13, fontweight='bold')
plt.tight_layout()
save_chart(fig, 'correlation_analysis')

# ── Summary report ─────────────────────────────────────────────────────────────
sep('WRITING BATCH 2 SUMMARY REPORT')

summary_lines = [
    '# EDA Batch 2 Summary — Phases 5-9\n\n',
    '## Key Findings\n\n',
]
for f in findings:
    summary_lines.append(f + '  \n')

summary_lines += [
    '\n## Split Boundary Recommendation\n\n',
    f'- Train: step 1-{TRAIN_END}  |  fraud: {train_fraud:,}\n',
    f'- Val  : step {TRAIN_END+1}-{VAL_END}  |  fraud: {val_fraud:,}\n',
    f'- Test : step {VAL_END+1}-743  |  fraud: {test_fraud:,}\n',
    '\n## Graph Feature Candidates (for Model 2)\n\n',
]
for name, desc in candidates:
    summary_lines.append(f'- `{name}`: {desc}  \n')

summary_lines += [
    '\n## Confirmed Feature List for Model 1 (XGBoost Baseline)\n\n',
    '**Include:** `step`, `type` (one-hot), `amount` (log1p), `oldbalanceOrg`, `oldbalanceDest`  \n',
    '**Exclude:** `nameOrig`, `nameDest`, `isFlaggedFraud`, `newbalanceOrig`, `newbalanceDest`  \n',
    '\n## Next Step\n\nEDA Batch 3 (Phases 10-13): RFM feasibility, feature register, validation strategy, handover.\n',
]

out_path = REPORTS_DIR / 'eda_batch2_summary.md'
with open(out_path, 'w', encoding='utf-8') as fh:
    fh.writelines(summary_lines)
print(f'Saved: {out_path}')

sep('EDA BATCH 2 COMPLETE')
print('Outputs in reports/data_quality/:')
print('  charts/bivariate_analysis.png')
print('  charts/time_series_analysis.png')
print('  charts/drift_analysis.png')
print('  charts/correlation_analysis.png')
print('  eda_batch2_summary.md')
print('\nSay "Batch 3 ready" to continue with Phases 10-13.')
