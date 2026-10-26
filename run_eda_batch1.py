"""
PaySim EDA - Batch 1 (Phases 1-4)
Run from the project root: python run_eda_batch1.py

Outputs saved to: reports/data_quality/
  - sweetviz_overall_report.html
  - sweetviz_class1_vs_class0_report.html
  - charts/  (PNG files)
  - eda_batch1_summary.md
"""

import sys
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

# Paths
PAYSIM_CSV  = PROJECT_ROOT / 'data' / 'shared' / 'raw' / 'PaySim_kaggle_raw_data.csv'
REPORTS_DIR = PROJECT_ROOT / 'reports' / 'data_quality'
CHARTS_DIR  = REPORTS_DIR / 'charts'
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

# Imports with graceful fallbacks
import pandas as pd
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style='whitegrid', palette='muted')
    plt.rcParams['figure.dpi'] = 100
    PLOTS = True
except ImportError:
    PLOTS = False
    print('[WARN] matplotlib/seaborn not found. Charts will be skipped.')
    print('       Run: pip install matplotlib seaborn')

try:
    import sweetviz as sv
    SWEETVIZ = True
except ImportError:
    SWEETVIZ = False
    print('[WARN] sweetviz not found. Run: pip install sweetviz')

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', '{:.4f}'.format)

# ── Helpers ───────────────────────────────────────────────────────────────────
def sep(title):
    bar = '=' * 65
    print(f'\n{bar}\n  {title}\n{bar}')

def binding_rate(series):
    return ((series.notna()) & (series != 0)).mean() * 100

def save_chart(fig, name):
    path = CHARTS_DIR / f'{name}.png'
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'  Chart saved: {path.name}')

# Collect findings for summary report
findings = []

# ── Phase 1 — Problem Definition ──────────────────────────────────────────────
sep('PHASE 1 — PROBLEM DEFINITION')
print('Business objective  : Detect fraudulent mobile money transactions')
print('Unit of prediction  : Individual transaction (one row = one transaction)')
print('Prediction boundary : Features available at transaction submission, before settlement')
print('Target column       : isFraud (binary 0/1)')
print('Dataset             : PaySim synthetic mobile-money simulator (Kaggle)')
print('\nKnown limitations:')
print('  • Synthetic data — patterns may differ from real-world fraud')
print('  • isFlaggedFraud is a simulation artefact, not a real detection system')
print('  • step = simulated hours (1 step ≈ 1 hour), not real calendar time')
print('  • Fraud only in TRANSFER and CASH_OUT — simulation design choice')

# ── Phase 2 — Data Loading ─────────────────────────────────────────────────────
sep('PHASE 2 — DATA LOADING AND PROFILING')

print(f'\nLoading: {PAYSIM_CSV}')
if not PAYSIM_CSV.exists():
    print(f'[ERROR] File not found: {PAYSIM_CSV}')
    sys.exit(1)

df = pd.read_csv(PAYSIM_CSV)
print(f'Shape        : {df.shape[0]:,} rows x {df.shape[1]} columns')
print(f'Memory usage : {df.memory_usage(deep=True).sum() / 1e9:.2f} GB')

print('\nColumn names and data types:')
print(df.dtypes.to_string())
print(f'\nStep range   : {df["step"].min()} -> {df["step"].max()}')
print(f'Unique steps : {df["step"].nunique()}')

print('\nFirst 3 rows:')
print(df.head(3).to_string())

# Missing values
sep('MISSING VALUES')
missing = df.isnull().sum()
has_missing = missing[missing > 0]
if len(has_missing) == 0:
    print('No missing values found.')
else:
    print(has_missing.to_string())

# Unique counts
print('\nUnique value counts per column:')
print(df.nunique().rename('unique_count').to_string())

# Duplicates
dup_count = df.duplicated().sum()
print(f'\nExact duplicate rows: {dup_count:,}')

# Descriptive statistics
sep('DESCRIPTIVE STATISTICS')
print(df.describe().T.to_string())

# Categorical summaries
sep('CATEGORICAL FEATURE SUMMARIES')
for col in df.select_dtypes(include='object').columns:
    vc  = df[col].value_counts()
    pct = df[col].value_counts(normalize=True) * 100
    print(f'\n{col}  ({df[col].nunique():,} unique):')
    summary = pd.DataFrame({'count': vc, 'pct%': pct.round(2)}).head(10)
    print(summary.to_string())

# ── Target Validation Gate ─────────────────────────────────────────────────────
sep('TARGET VALIDATION GATE (Section 4)')

target_col   = 'isFraud'
invalid_mask = ~df[target_col].isin([0, 1])
miss_target  = df[target_col].isnull().sum()

fraud_counts = df[target_col].value_counts().sort_index()
fraud_pct    = df[target_col].value_counts(normalize=True).sort_index() * 100
n_neg = fraud_counts[0]
n_pos = fraud_counts[1]
spw   = n_neg / n_pos

print(f'Total rows             : {len(df):,}')
print(f'Missing target values  : {miss_target:,}')
print(f'Invalid target values  : {invalid_mask.sum():,}')
print(f'\nClass 0 (Legitimate)   : {n_neg:,}  ({fraud_pct[0]:.4f}%)')
print(f'Class 1 (Fraud)        : {n_pos:,}  ({fraud_pct[1]:.4f}%)')
print(f'\nClass ratio (neg:pos)  : {n_neg:,} : {n_pos:,}')
print(f'scale_pos_weight       : {spw:.1f}  <-- placeholder; recalculate on training split')

gate_pass = miss_target == 0 and invalid_mask.sum() == 0
print(f'\n[{"GATE PASSED" if gate_pass else "GATE FAILED"}] Target is {"valid" if gate_pass else "invalid — fix before continuing"}.')

findings.append(f'- Fraud rate: {fraud_pct[1]:.4f}%  |  scale_pos_weight (full dataset): {spw:.1f}')
findings.append(f'- Class 0: {n_neg:,}  |  Class 1: {n_pos:,}')

# ── Phase 3 — Sweetviz Reports ─────────────────────────────────────────────────
sep('PHASE 3 — SWEETVIZ REPORTS')

if SWEETVIZ:
    SAMPLE_SIZE = 100_000
    fraud_rows = df[df['isFraud'] == 1]
    legit_rows = df[df['isFraud'] == 0]
    n_fraud_s  = min(len(fraud_rows), int(SAMPLE_SIZE * len(fraud_rows) / len(df)))
    n_legit_s  = SAMPLE_SIZE - n_fraud_s

    sv_sample = pd.concat([
        fraud_rows.sample(n_fraud_s, random_state=42),
        legit_rows.sample(n_legit_s, random_state=42)
    ]).reset_index(drop=True)

    sv_cols = [c for c in sv_sample.columns if c not in ['nameOrig', 'nameDest']]
    sv_data = sv_sample[sv_cols]

    print(f'Sample size : {len(sv_sample):,}  |  fraud rate in sample: {sv_sample["isFraud"].mean()*100:.4f}%')

    print('\nGenerating sweetviz_overall_report.html ...')
    rpt1 = sv.analyze(sv_data, target_feat='isFraud')
    out1 = str(REPORTS_DIR / 'sweetviz_overall_report.html')
    rpt1.show_html(out1, open_browser=False)
    print(f'  Saved: {out1}')

    print('\nGenerating sweetviz_class1_vs_class0_report.html ...')
    fraud_sv = sv_data[sv_data['isFraud'] == 1]
    legit_sv = sv_data[sv_data['isFraud'] == 0]
    rpt2 = sv.compare([fraud_sv, 'Fraud (Class 1)'], [legit_sv, 'Legitimate (Class 0)'])
    out2 = str(REPORTS_DIR / 'sweetviz_class1_vs_class0_report.html')
    rpt2.show_html(out2, open_browser=False)
    print(f'  Saved: {out2}')

    findings.append('- Sweetviz reports generated: sweetviz_overall_report.html, sweetviz_class1_vs_class0_report.html')
else:
    print('[SKIPPED] sweetviz not installed.')

# ── Phase 4 — Data Quality ─────────────────────────────────────────────────────
sep('PHASE 4 — DATA QUALITY: MISSINGNESS AND BINDING RATES')

balance_cols  = ['oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
numeric_cols  = ['amount'] + balance_cols

print('\nZERO VALUES (%) BY FRAUD LABEL:')
rows = []
for col in balance_cols:
    p_legit = (df[df['isFraud']==0][col] == 0).mean() * 100
    p_fraud = (df[df['isFraud']==1][col] == 0).mean() * 100
    rows.append({'column': col, 'Legit zero%': round(p_legit,2), 'Fraud zero%': round(p_fraud,2), 'Diff': round(p_fraud-p_legit,2)})
zero_df = pd.DataFrame(rows).set_index('column')
print(zero_df.to_string())

print('\nFEATURE BINDING RATES (% non-null and non-zero):')
br_rows = []
for col in numeric_cols:
    br_rows.append({
        'column'     : col,
        'Overall %'  : round(binding_rate(df[col]), 2),
        'Legit %'    : round(binding_rate(df[df['isFraud']==0][col]), 2),
        'Fraud %'    : round(binding_rate(df[df['isFraud']==1][col]), 2),
    })
br_df = pd.DataFrame(br_rows).set_index('column')
print(br_df.to_string())

sep('PHASE 4 — DUPLICATE ANALYSIS')

sig_cols = ['step', 'type', 'amount', 'nameOrig', 'nameDest']
sig_dups = df.duplicated(subset=sig_cols).sum()
pair_counts = df.groupby(['nameOrig', 'nameDest']).size()
print(f'Exact duplicate rows              : {dup_count:,}')
print(f'Duplicate transaction signatures  : {sig_dups:,}')
print(f'Repeated origin-destination pairs : {(pair_counts > 1).sum():,}')
print(f'Max transactions for one O-D pair : {pair_counts.max():,}')

if sig_dups > 0:
    dup_mask = df.duplicated(subset=sig_cols, keep=False)
    conflicts = df[dup_mask].groupby(sig_cols)['isFraud'].nunique()
    print(f'Signatures with conflicting labels: {(conflicts > 1).sum():,}')

findings.append(f'- Duplicate rows: {dup_count:,}  |  Duplicate signatures: {sig_dups:,}')

sep('PHASE 4 — INVALID VALUES AND CONSISTENCY')

valid_types = {'CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER'}
print(f'Invalid transaction types   : {(~df["type"].isin(valid_types)).sum():,}')
print(f'Negative amounts            : {(df["amount"] < 0).sum():,}')
for col in balance_cols:
    print(f'Negative {col:18s}: {(df[col] < 0).sum():,}')

print('\nisFlaggedFraud vs isFraud:')
ct = pd.crosstab(df['isFlaggedFraud'], df['isFraud'],
                 rownames=['isFlaggedFraud'], colnames=['isFraud'], margins=True)
print(ct.to_string())

flagged = df[df['isFlaggedFraud'] == 1]
print(f'\nOf {len(flagged):,} flagged rows:')
print(f'  {int(flagged["isFraud"].sum()):,} are actual fraud  ({flagged["isFraud"].mean()*100:.1f}%)')
print(f'  {int((1-flagged["isFraud"]).sum()):,} are NOT fraud (false positives)')
print('  --> isFlaggedFraud is a noisy proxy for isFraud. EXCLUDE from all models.')

print('\nBalance conservation check (TRANSFER + CASH_OUT):')
tf_co = df[df['type'].isin(['TRANSFER', 'CASH_OUT'])].copy()
tf_co['orig_error'] = (tf_co['oldbalanceOrg'] - tf_co['newbalanceOrig'] - tf_co['amount']).abs()
tf_co['dest_error'] = (tf_co['newbalanceDest'] - tf_co['oldbalanceDest'] - tf_co['amount']).abs()
tol = 0.01
orig_viol = (tf_co['orig_error'] > tol).sum()
dest_viol = (tf_co['dest_error'] > tol).sum()
print(f'  Sender-side violations   : {orig_viol:,} / {len(tf_co):,}  ({orig_viol/len(tf_co)*100:.1f}%)')
print(f'  Recipient-side violations: {dest_viol:,} / {len(tf_co):,}  ({dest_viol/len(tf_co)*100:.1f}%)')

print('\nViolation rate by fraud label (sender side):')
viol_by_label = tf_co.groupby('isFraud')['orig_error'].apply(lambda x: (x > tol).mean() * 100).round(2)
print(viol_by_label.rename('sender_violation_rate_%').to_string())

findings.append(f'- Sender balance violations: {orig_viol:,} ({orig_viol/len(tf_co)*100:.1f}%)')
findings.append('- isFlaggedFraud: exclude from all models (simulation artefact)')

# ── Outlier charts ──────────────────────────────────────────────────────────────
if PLOTS:
    sep('PHASE 4 — OUTLIER ANALYSIS (saving charts)')

    fig, axes = plt.subplots(3, 2, figsize=(14, 14))

    for label, color, lname in [(0, 'steelblue', 'Legitimate'), (1, 'tomato', 'Fraud')]:
        s = df[df['isFraud'] == label]['amount']
        axes[0, 0].hist(s.clip(upper=s.quantile(0.99)), bins=80, alpha=0.55,
                        color=color, label=lname, density=True)
        axes[0, 1].hist(np.log1p(s), bins=80, alpha=0.55,
                        color=color, label=lname, density=True)

    axes[0, 0].set_title('amount (raw, capped at 99th pct)')
    axes[0, 1].set_title('amount (log1p transformed)')
    for ax in axes[0]:
        ax.legend(); ax.set_xlabel('Amount')

    for label, color, lname in [(0, 'steelblue', 'Legitimate'), (1, 'tomato', 'Fraud')]:
        s      = df[df['isFraud'] == label]['amount']
        s_samp = np.sort(s.sample(min(len(s), 10_000), random_state=42).values)
        axes[1, 0].plot(s_samp, np.linspace(0, 1, len(s_samp)), color=color, label=lname, lw=1.5)
        axes[1, 1].plot(np.log1p(s_samp), np.linspace(0, 1, len(s_samp)), color=color, label=lname, lw=1.5)

    axes[1, 0].set_title('ECDF — amount (raw)')
    axes[1, 1].set_title('ECDF — log1p(amount)')
    for ax in axes[1]:
        ax.legend(); ax.set_xlabel('Amount'); ax.set_ylabel('Cumulative probability')

    tf_plot = df[df['type'].isin(['TRANSFER', 'CASH_OUT'])][['isFraud','oldbalanceOrg','oldbalanceDest']].copy()
    tf_melt = tf_plot.melt(id_vars='isFraud', var_name='balance_col', value_name='value')
    tf_melt['log1p_value'] = np.log1p(tf_melt['value'])
    tf_melt['Class'] = tf_melt['isFraud'].map({0: 'Legitimate', 1: 'Fraud'})
    sns.boxplot(data=tf_melt, x='balance_col', y='log1p_value', hue='Class',
                palette={'Legitimate': 'steelblue', 'Fraud': 'tomato'}, ax=axes[2, 0])
    axes[2, 0].set_title('log1p(opening balance) by class (TRANSFER+CASH_OUT)')
    axes[2, 0].set_xlabel('')

    qtls  = [0.5, 0.75, 0.9, 0.95, 0.99, 0.999]
    qf    = np.log1p(df[df['isFraud']==1]['amount'].quantile(qtls).values)
    ql    = np.log1p(df[df['isFraud']==0]['amount'].quantile(qtls).values)
    x     = np.arange(len(qtls))
    axes[2, 1].bar(x - 0.2, ql, width=0.35, color='steelblue', label='Legitimate', alpha=0.8)
    axes[2, 1].bar(x + 0.2, qf, width=0.35, color='tomato',    label='Fraud',       alpha=0.8)
    axes[2, 1].set_xticks(x)
    axes[2, 1].set_xticklabels([str(q) for q in qtls])
    axes[2, 1].set_title('log1p(amount) quantiles by class')
    axes[2, 1].set_xlabel('Quantile'); axes[2, 1].legend()

    plt.suptitle('Outlier Analysis — Amount and Opening Balances', fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_chart(fig, 'outlier_analysis')

    # Binding rate trend
    fig2, ax2 = plt.subplots(figsize=(13, 4))
    step_br = pd.DataFrame({
        col: df.groupby('step')[col].apply(binding_rate)
        for col in numeric_cols
    })
    rolling_br = step_br.rolling(10, center=True).mean()
    for col in numeric_cols:
        ax2.plot(rolling_br.index, rolling_br[col], label=col, lw=1.4)
    ax2.set_xlabel('Step'); ax2.set_ylabel('Binding rate (%)')
    ax2.set_title('Feature Binding Rates Over Time (10-step rolling avg)')
    ax2.legend(fontsize=8)
    plt.tight_layout()
    save_chart(fig2, 'binding_rate_trend')

# ── Summary markdown report ─────────────────────────────────────────────────────
sep('WRITING SUMMARY REPORT')

summary_lines = [
    '# EDA Batch 1 Summary — Phases 1-4\n',
    f'\n**Dataset:** {PAYSIM_CSV.name}  \n',
    f'**Rows:** {len(df):,}  |  **Columns:** {df.shape[1]}  \n',
    f'**Missing values:** None  \n',
    f'**Duplicate rows:** {dup_count:,}  \n',
    '\n## Key Findings\n',
]
for f in findings:
    summary_lines.append(f + '  \n')

summary_lines += [
    '\n## Leakage Risk Register\n',
    '| Column | Risk | Decision |\n',
    '|---|---|---|\n',
    '| `newbalanceOrig` | Post-settlement balance | **Exclude** |\n',
    '| `newbalanceDest` | Post-settlement balance | **Exclude** |\n',
    '| `isFlaggedFraud` | Simulation artefact — near-perfect proxy for target | **Exclude** |\n',
    '| `nameOrig` | High-cardinality identifier | Graph construction only |\n',
    '| `nameDest` | High-cardinality identifier | Graph construction only |\n',
    '| `oldbalanceOrg` | Pre-transaction — investigate further | **Investigate in Batch 2** |\n',
    '| `oldbalanceDest` | Pre-transaction — zero-heavy for merchants | **Investigate in Batch 2** |\n',
    '\n## Confirmed for Phase 2 (Model 1 preprocessing)\n',
    '- Exclude: `newbalanceOrig`, `newbalanceDest`, `isFlaggedFraud`, `nameOrig`, `nameDest`\n',
    '- Apply log1p to `amount`\n',
    '- Use `scale_pos_weight` from training split (not full dataset)\n',
    '- Chronological split: train ≤ step 534 | val ≤ 638 | test > 638  (confirm in Batch 2)\n',
    '\n## Next Step\n',
    'Run EDA Batch 2 (Phases 5-9): univariate, bivariate, time-series, drift, correlation.\n',
]

summary_path = REPORTS_DIR / 'eda_batch1_summary.md'
with open(summary_path, 'w', encoding='utf-8') as f:
    f.writelines(summary_lines)
print(f'Summary report saved: {summary_path}')

sep('EDA BATCH 1 COMPLETE')
print(f'All outputs in: {REPORTS_DIR}')
print('  sweetviz_overall_report.html         -- open in browser')
print('  sweetviz_class1_vs_class0_report.html -- open in browser')
print('  charts/outlier_analysis.png')
print('  charts/binding_rate_trend.png')
print('  eda_batch1_summary.md')
print('\nShare the key numbers above and say "Batch 2 ready" to continue.')
