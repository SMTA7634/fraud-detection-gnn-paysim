"""
create_subset.py
Creates a 50,000-row development subset from the full PaySim dataset.

Strategy:
  - Keep ALL 8,213 fraud rows (we never want to lose fraud signal)
  - Randomly sample (50,000 - 8,213) = 41,787 non-fraud rows
  - Save to data/shared/raw/paysim_subset_50k.csv

Run: python create_subset.py
"""

from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DIR      = PROJECT_ROOT / 'data' / 'shared' / 'raw'
FULL_CSV     = RAW_DIR / 'PaySim_kaggle_raw_data.csv'
SUBSET_CSV   = RAW_DIR / 'paysim_subset_250k.csv'

SUBSET_SIZE  = 250_000
RANDOM_SEED  = 42
TRAIN_END    = 534
VAL_END      = 638

print('Loading full dataset...')
df = pd.read_csv(FULL_CSV)
print(f'  Full dataset: {len(df):,} rows  |  fraud: {df["isFraud"].sum():,}')

fraud_df  = df[df['isFraud'] == 1]
legit_df  = df[df['isFraud'] == 0]

n_fraud   = len(fraud_df)
n_legit_sample = SUBSET_SIZE - n_fraud

print(f'\nSubset composition:')
print(f'  All fraud rows kept : {n_fraud:,}')
print(f'  Legit rows sampled  : {n_legit_sample:,}')
print(f'  Total               : {SUBSET_SIZE:,}')

legit_sample = legit_df.sample(n=n_legit_sample, random_state=RANDOM_SEED)

subset = pd.concat([fraud_df, legit_sample], ignore_index=True)
# Sort by step to preserve temporal order
subset = subset.sort_values('step').reset_index(drop=True)

subset.to_csv(SUBSET_CSV, index=False)
print(f'\nSaved: {SUBSET_CSV}')

# ── Report statistics ─────────────────────────────────────────────────────────
print('\n' + '='*55)
print('  SUBSET STATISTICS')
print('='*55)

fraud_rate = subset['isFraud'].mean() * 100
print(f'\n  Total rows       : {len(subset):,}')
print(f'  Fraud rows       : {subset["isFraud"].sum():,}')
print(f'  Legit rows       : {(subset["isFraud"]==0).sum():,}')
print(f'  Fraud rate       : {fraud_rate:.2f}%  (vs 0.13% in full dataset)')

# Split breakdown
subset['split'] = pd.cut(
    subset['step'],
    bins=[0, TRAIN_END, VAL_END, subset['step'].max() + 1],
    labels=['train', 'val', 'test'],
    right=True
)

split_stats = subset.groupby('split', observed=True)['isFraud'].agg(
    total='count', fraud='sum', fraud_rate='mean'
).assign(
    fraud_rate_pct = lambda x: (x['fraud_rate'] * 100).round(2),
    pct_of_data    = lambda x: (x['total'] / len(subset) * 100).round(1)
)
print(f'\n  Split breakdown:')
print(split_stats.to_string())

# scale_pos_weight for training split
n_train_neg = int(split_stats.loc['train', 'total'] - split_stats.loc['train', 'fraud'])
n_train_pos = int(split_stats.loc['train', 'fraud'])
spw = round(n_train_neg / n_train_pos, 1)

print(f'\n  scale_pos_weight (training split) : {spw}')
print(f'  (was 1030.0 on full dataset — much lower now because fraud rate is higher)')

print('\n  Transaction types in subset:')
print(subset['type'].value_counts().to_string())

print('\n  Fraud by type:')
print(subset.groupby('type')['isFraud'].agg(
    count='count', fraud='sum', rate='mean'
).assign(rate_pct=lambda x: (x['rate']*100).round(2)).sort_values('fraud', ascending=False).to_string())

print(f'\n  Step range: {subset["step"].min()} -> {subset["step"].max()}')
print(f'\nDone. Use paysim_subset_50k.csv for all model development.')
