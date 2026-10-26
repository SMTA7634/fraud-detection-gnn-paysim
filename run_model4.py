"""
run_model4.py  —  Model 4: Account-Node GraphSAGE
Run: python run_model4.py

═══════════════════════════════════════════════════════════════════
  MODEL 3 vs MODEL 4 — THE KEY DIFFERENCE
═══════════════════════════════════════════════════════════════════

  MODEL 3 (transaction-node graph):
  ┌─────────────────────────────────────────────────────────────┐
  │  Question: "Is this TRANSACTION fraudulent?"                │
  │  Node     = one transaction (50k / 250k nodes)              │
  │  Edge     = two transactions share a sender or recipient    │
  │  Label    = isFraud (0/1 per transaction)                   │
  │  Use case = real-time scoring before settlement             │
  └─────────────────────────────────────────────────────────────┘

  MODEL 4 (account-node graph):
  ┌─────────────────────────────────────────────────────────────┐
  │  Question: "Is this ACCOUNT a fraudster?"                   │
  │  Node     = one account (each unique nameOrig/nameDest)     │
  │  Edge     = a transaction between two accounts              │
  │  Label    = is this account the sender in a fraud txn?      │
  │  Use case = account-level risk scoring, fraud ring detect.  │
  └─────────────────────────────────────────────────────────────┘

  WHY THIS MATTERS:
    Transaction scoring answers: "Block this payment?"
    Account scoring answers:     "Should we flag / suspend this account?"
                                 "Are these accounts part of a fraud ring?"

  In PaySim, fraudsters DRAIN their own account in one TRANSFER or
  CASH_OUT. The account-node view lets the GNN see: this account
  is connected (via prior transactions) to accounts that later become
  fraud victims — a pattern a transaction-level model can't see.

═══════════════════════════════════════════════════════════════════
  NODE SETUP DIAGRAM
═══════════════════════════════════════════════════════════════════

  Transaction data (raw):
    step  nameOrig   nameDest   amount   type      isFraud
    100   C111       M456       9800     TRANSFER  1        ← T1
    105   C222       M456       3200     TRANSFER  1        ← T2
    200   C333       M456       410      PAYMENT   0        ← T3
    215   C444       Q789       75       CASH_OUT  0        ← T4

  Account-node graph built from this:
    Nodes (accounts):    C111  C222  C333  C444  M456  Q789
    Node indices:          0     1     2     3     4     5

    Edges (one per transaction, undirected):
      T1: C111 → M456   adds edges (0,4) and (4,0)
      T2: C222 → M456   adds edges (1,4) and (4,1)
      T3: C333 → M456   adds edges (2,4) and (4,2)
      T4: C444 → Q789   adds edges (3,5) and (5,3)

    Graph visualised:
      C111 ─────────┐
      C222 ──────── M456      C444 ── Q789
      C333 ─────────┘

    Labels (fraud sender = nameOrig in any fraud transaction):
      C111: FRAUD (sent T1, which is fraud)
      C222: FRAUD (sent T2, which is fraud)
      C333: LEGIT
      C444: LEGIT
      M456: LEGIT  ← recipient account, not the fraudster
      Q789: LEGIT

  Node feature vector (aggregated from ALL that account's transactions):
    [total_txn_count, out_degree, in_degree,
     total_sent_log, total_received_log,
     avg_sent_log, drain_fraction, transfer_ratio,
     out_in_ratio, max_single_sent_log]

  KEY INSIGHT about M456 in this graph:
    M456 received from C111 (fraud) and C222 (fraud) AND C333 (legit).
    After GraphSAGE aggregation, M456's embedding will be influenced by
    its fraud neighbours. When C111 asks "what do my neighbours look like?"
    it sees M456, which also received from another fraud account.
    This neighbourhood pattern is invisible to a flat XGBoost model.
"""

import sys, warnings, os
warnings.filterwarnings('ignore')
os.environ['MLFLOW_DISABLE_AGENT_HINT'] = '1'
os.environ['GIT_PYTHON_REFRESH']        = 'quiet'

from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT  = Path(__file__).resolve().parent
SUBSET_CSV    = PROJECT_ROOT / 'data' / 'shared' / 'raw' / 'paysim_subset_250k.csv'
ARTIFACTS_DIR = PROJECT_ROOT / 'artifacts' / 'model_4_account_gnn'
CHARTS_DIR    = ARTIFACTS_DIR / 'charts'
PREDS_DIR     = PROJECT_ROOT / 'data' / 'model_4_account_gnn' / 'output'

for d in [ARTIFACTS_DIR, CHARTS_DIR, PREDS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn   import SAGEConv

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    average_precision_score, roc_auc_score,
    confusion_matrix, precision_recall_curve
)
import mlflow

sns.set_theme(style='whitegrid')
plt.rcParams['figure.dpi'] = 100

OOT_STEP_BOUNDARY = 594
RANDOM_SEED       = 42
TEST_SIZE         = 0.20
HIDDEN_DIM        = 64
N_EPOCHS          = 150
LR                = 0.001
PATIENCE          = 20

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    print(f'  Saved: {name}.png')

print(f'  Device: {device}')
if device.type == 'cuda':
    print(f'  GPU   : {torch.cuda.get_device_name(0)}')

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 — LOAD DATA AND BUILD ACCOUNT INDEX
# ═════════════════════════════════════════════════════════════════════════════
sep('SECTION 1 — LOAD DATA AND BUILD ACCOUNT INDEX')

note("""
  STEP 1: Every unique account name (nameOrig or nameDest) becomes a NODE.

  Unlike the transaction graph where each ROW was a node, here each
  ACCOUNT is a node. A single account may be involved in many transactions
  — those transactions become its EDGES and its features are aggregated
  across all of them.

  Account types in PaySim:
    nameOrig format:  'C' + digits  (customer — can send and receive)
    nameDest format:  'C' + digits  (customers) or 'M' + digits (merchants)

  We collect ALL unique account names from both columns into a single
  node set, then assign each a sequential integer index (0, 1, 2, ...).
  This index is what the graph uses — account names are just the lookup key.
""")

df = pd.read_csv(SUBSET_CSV)
df = df.sort_values('step').reset_index(drop=True)
print(f'  Loaded: {len(df):,} transactions')

# Build account → integer index mapping
all_accounts = pd.Index(
    pd.concat([df['nameOrig'], df['nameDest']]).unique()
)
account_to_idx = {acc: i for i, acc in enumerate(all_accounts)}
n_accounts     = len(all_accounts)
print(f'  Unique accounts (nodes): {n_accounts:,}')
print(f'    nameDest accounts      : {df["nameDest"].nunique():,}')
print(f'    nameOrig accounts      : {df["nameOrig"].nunique():,}')

# Map account names to integer node indices in the transaction dataframe
df['orig_idx'] = df['nameOrig'].map(account_to_idx)
df['dest_idx'] = df['nameDest'].map(account_to_idx)

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 — BUILD NODE FEATURES
# ═════════════════════════════════════════════════════════════════════════════
sep('SECTION 2 — BUILD NODE FEATURES (aggregated per account)')

note("""
  WHAT FEATURES CAN AN ACCOUNT NODE HAVE?

  Unlike a transaction node (which has direct features: amount, type, etc.),
  an account node's features must be AGGREGATED across all its transactions.

  We compute two groups:

  OUTGOING (account acted as SENDER in these transactions):
    out_degree          : number of transactions sent
    out_total_log       : log1p(total amount sent)
    out_avg_log         : log1p(average amount sent per transaction)
    out_drain_frac      : fraction of outgoing where pct_balance_used ≈ 1.0
                          (account was completely drained → strong fraud signal)
    out_transfer_frac   : fraction of outgoing that are TRANSFER or CASH_OUT
                          (only these transaction types contain fraud)

  INCOMING (account acted as RECIPIENT in these transactions):
    in_degree           : number of transactions received
    in_total_log        : log1p(total amount received)
    in_distinct_senders : how many unique senders paid into this account

  COMBINED:
    total_degree        : out_degree + in_degree
    out_in_ratio        : out_degree / total_degree  (1.0 = only sends, never receives)

  LEAKAGE NOTE (important):
    Here we aggregate ALL transactions for each account, including those
    after the OOT boundary. This means an account's node features include
    future information — technically leaky.
    In a production system, features would be computed up to the prediction
    timestamp only (event-time feature store).
    For this learning exercise, we accept this simplification and note it.
""")

df['pct_balance_used'] = np.where(
    df['oldbalanceOrg'] > 0, df['amount'] / df['oldbalanceOrg'], 0.0
).clip(0, 10)
df['is_drain']        = (df['pct_balance_used'] >= 0.99).astype(int)
df['is_xfer_cashout'] = df['type'].isin(['TRANSFER', 'CASH_OUT']).astype(int)

# --- Outgoing aggregation ---
out_agg = df.groupby('orig_idx').agg(
    out_degree          = ('amount', 'count'),
    out_total           = ('amount', 'sum'),
    out_avg             = ('amount', 'mean'),
    out_drain_count     = ('is_drain', 'sum'),
    out_xfer_count      = ('is_xfer_cashout', 'sum'),
    orig_first_step     = ('step', 'min'),
).rename_axis('node_idx')

out_agg['out_drain_frac']    = out_agg['out_drain_count'] / out_agg['out_degree']
out_agg['out_transfer_frac'] = out_agg['out_xfer_count']  / out_agg['out_degree']
out_agg['out_total_log']     = np.log1p(out_agg['out_total'])
out_agg['out_avg_log']       = np.log1p(out_agg['out_avg'])

# --- Incoming aggregation ---
in_agg = df.groupby('dest_idx').agg(
    in_degree           = ('amount', 'count'),
    in_total            = ('amount', 'sum'),
    in_distinct_senders = ('orig_idx', 'nunique'),
    dest_first_step     = ('step', 'min'),
).rename_axis('node_idx')

in_agg['in_total_log'] = np.log1p(in_agg['in_total'])

# --- Combine into one node feature DataFrame ---
node_df = pd.DataFrame(index=range(n_accounts))
node_df.index.name = 'node_idx'
node_df = node_df.join(out_agg[['out_degree','out_total_log','out_avg_log',
                                 'out_drain_frac','out_transfer_frac',
                                 'orig_first_step']], how='left')
node_df = node_df.join(in_agg[['in_degree','in_total_log',
                                'in_distinct_senders',
                                'dest_first_step']], how='left')

node_df = node_df.fillna(0)
node_df['total_degree']  = node_df['out_degree'] + node_df['in_degree']
node_df['out_in_ratio']  = node_df['out_degree'] / node_df['total_degree'].replace(0, 1)
# first transaction step for this account (used for splitting)
node_df['first_step']    = np.where(
    (node_df['orig_first_step'] > 0) & (node_df['dest_first_step'] > 0),
    node_df[['orig_first_step','dest_first_step']].min(axis=1),
    node_df['orig_first_step'] + node_df['dest_first_step']
)
node_df['first_step'] = node_df['first_step'].replace(0, 999)

FEATURE_COLS = ['out_degree','out_total_log','out_avg_log',
                'out_drain_frac','out_transfer_frac',
                'in_degree','in_total_log','in_distinct_senders',
                'total_degree','out_in_ratio']

print(f'  Node feature dimensions: {len(FEATURE_COLS)}')
print(f'  Features: {FEATURE_COLS}')
print(f'\n  Sample node stats:')
print(node_df[FEATURE_COLS].describe().round(3).to_string())

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 — BUILD NODE LABELS (fraud accounts)
# ═════════════════════════════════════════════════════════════════════════════
sep('SECTION 3 — BUILD NODE LABELS')

note("""
  WHAT MAKES AN ACCOUNT "FRAUDULENT"?

  In PaySim, fraud is committed by SENDERS:
    - A compromised customer account (nameOrig) initiates a fraudulent
      TRANSFER or CASH_OUT, draining their own balance.
    - The recipient (nameDest) is often a mule account — it receives the
      stolen funds but is not itself the attacker.

  So we label an account as FRAUD if it appears as nameOrig in at least
  one fraud transaction.

  NOTE: This is a coarser label than transaction-level.
  An account that made 10 legitimate transactions and 1 fraud transaction
  is still labelled FRAUD (class 1).

  WHY NOT LABEL RECIPIENTS TOO?
  We could — but PaySim's simulation means the recipient is a randomly
  assigned mule, not a structural member of the fraud. Including them
  would flood the fraud class with borderline cases and make evaluation
  less clean for this learning exercise.
""")

fraud_orig_idxs = set(df.loc[df['isFraud'] == 1, 'orig_idx'].unique())
node_df['is_fraud_account'] = node_df.index.map(
    lambda i: 1 if i in fraud_orig_idxs else 0
)

fraud_accounts = node_df['is_fraud_account'].sum()
total_accounts = len(node_df)
print(f'  Total accounts (nodes)  : {total_accounts:,}')
print(f'  Fraud accounts (label=1): {fraud_accounts:,}  ({fraud_accounts/total_accounts*100:.2f}%)')
print(f'  Legit accounts (label=0): {total_accounts - fraud_accounts:,}')

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4 — TRAIN / TEST / OOT SPLIT (account-level)
# ═════════════════════════════════════════════════════════════════════════════
sep('SECTION 4 — ACCOUNT-LEVEL TRAIN/TEST/OOT SPLIT')

note("""
  HOW TO SPLIT ACCOUNTS (NOT TRANSACTIONS)?

  We cannot use the same row-level split as before — here our rows are
  ACCOUNTS, not transactions.

  Strategy: split accounts by when they FIRST appeared in the simulation.
    OOT accounts   : first_step > 594  (brand new accounts, never seen in dev)
    Dev accounts   : first_step ≤ 594  (appeared during training window)
    Train/test     : 80/20 stratified split of dev accounts on fraud label

  This tests whether the GNN can generalise to genuinely unseen accounts
  that join the system after the training window closes.
""")

oot_account_mask  = node_df['first_step'] > OOT_STEP_BOUNDARY
dev_account_mask  = ~oot_account_mask

dev_indices  = node_df[dev_account_mask].index.tolist()
oot_indices  = node_df[oot_account_mask].index.tolist()

y_dev = node_df.loc[dev_indices, 'is_fraud_account'].values

if y_dev.sum() == 0:
    print('  WARNING: No fraud accounts in dev set. Using all dev as train, no test.')
    train_indices, test_indices = dev_indices, []
else:
    train_indices, test_indices = train_test_split(
        dev_indices,
        test_size    = TEST_SIZE,
        random_state = RANDOM_SEED,
        stratify     = y_dev
    )

y_all = node_df['is_fraud_account'].values.astype(np.float32)

print(f'  Dev accounts  : {len(dev_indices):,}   '
      f'| fraud: {y_all[dev_indices].sum():.0f} ({y_all[dev_indices].mean()*100:.3f}%)')
print(f'  Train accounts: {len(train_indices):,}   '
      f'| fraud: {y_all[train_indices].sum():.0f} ({y_all[train_indices].mean()*100:.3f}%)')
print(f'  Test accounts : {len(test_indices):,}    '
      f'| fraud: {y_all[test_indices].sum():.0f} ({y_all[test_indices].mean()*100:.3f}%)')
print(f'  OOT accounts  : {len(oot_indices):,}   '
      f'| fraud: {y_all[oot_indices].sum():.0f} ({y_all[oot_indices].mean()*100:.3f}%)')

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5 — SCALE NODE FEATURES
# ═════════════════════════════════════════════════════════════════════════════
sep('SECTION 5 — SCALE NODE FEATURES')

X_raw = node_df[FEATURE_COLS].values.astype(np.float32)
scaler = StandardScaler()
scaler.fit(X_raw[train_indices])         # fit only on train accounts
X_all  = scaler.transform(X_raw)

print(f'  Scaler fit on {len(train_indices):,} train accounts only.')
print(f'  Feature matrix shape: {X_all.shape}')

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6 — BUILD EDGE INDEX (transactions as edges)
# ═════════════════════════════════════════════════════════════════════════════
sep('SECTION 6 — BUILD EDGE INDEX (transactions become edges)')

note("""
  EDGE CONSTRUCTION FOR ACCOUNT-NODE GRAPH:

  Each transaction is an EDGE between two account nodes:
    nameOrig → nameDest

  This is much simpler (and faster) than the transaction-node graph:
    - No time-window check needed
    - Each row of the dataframe becomes exactly one edge
    - We make it undirected (add both directions)

  EDGE COUNT:
    250k transactions → 250k directed edges
    Undirected = 500k directed edge entries in edge_index

  COMPARISON TO MODEL 3:
    Model 3 (transaction-node): 13,400 undirected edges from shared recipients
    Model 4 (account-node):     250,000 undirected edges — one per transaction

  The account-node graph is MUCH denser. Every account is connected to
  every other account it ever transacted with. This is why the account-node
  graph is better for detecting fraud rings — the connectivity mirrors the
  real financial network.
""")

src = df['orig_idx'].values.tolist()
dst = df['dest_idx'].values.tolist()

# Undirected: add both directions
edge_src = src + dst
edge_dst = dst + src

edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
print(f'  Transactions (edges)      : {len(df):,}')
print(f'  Undirected edge entries   : {edge_index.shape[1]:,}')
print(f'  Avg edges per account node: {edge_index.shape[1] / n_accounts:.2f}')

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 7 — ASSEMBLE PyG DATA OBJECT
# ═════════════════════════════════════════════════════════════════════════════
sep('SECTION 7 — ASSEMBLE PyG DATA OBJECT')

x_tensor = torch.tensor(X_all,  dtype=torch.float)
y_tensor = torch.tensor(y_all,  dtype=torch.float)

n = n_accounts
train_mask = torch.zeros(n, dtype=torch.bool)
test_mask  = torch.zeros(n, dtype=torch.bool)
oot_mask   = torch.zeros(n, dtype=torch.bool)
train_mask[train_indices] = True
test_mask[test_indices]   = True
oot_mask[oot_indices]     = True

data = Data(x=x_tensor, edge_index=edge_index, y=y_tensor)
data.train_mask = train_mask
data.test_mask  = test_mask
data.oot_mask   = oot_mask
data = data.to(device)

print(f'  Account nodes      : {data.num_nodes:,}')
print(f'  Edges (directed)   : {data.num_edges:,}')
print(f'  Node features      : {data.num_features}')
print(f'  Train / Test / OOT : {train_mask.sum()} / {test_mask.sum()} / {oot_mask.sum()}')
print(f'  Data on            : {data.x.device}')

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 8 — MODEL DEFINITION
# ═════════════════════════════════════════════════════════════════════════════
sep('SECTION 8 — GRAPHSAGE MODEL (same architecture as Model 3)')

note("""
  Same SAGEConv architecture — but now operating on ACCOUNT nodes.

  What changes at each layer:
    Layer 1: each account aggregates features from directly connected accounts
             (the accounts it transacted with)
    Layer 2: each account aggregates its connected accounts' Layer-1 embeddings
             (i.e. summaries of those accounts' own neighbours — 2-hop view)

  For a fraud account C111 that sent to mule M456:
    1-hop neighbours of C111: M456 and anyone else C111 transacted with
    2-hop neighbours of C111: everyone M456 ever transacted with
        → including other fraud senders C222 who also sent to M456

  The GNN learns that C111 shares a 2-hop neighbourhood with another fraud
  account (C222), which is a signal invisible to flat feature models.
""")

class FraudAccountGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels=64):
        super().__init__()
        self.conv1      = SAGEConv(in_channels, hidden_channels)
        self.conv2      = SAGEConv(hidden_channels, hidden_channels)
        self.classifier = nn.Linear(hidden_channels, 1)
        self.dropout    = nn.Dropout(p=0.3)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index);  x = F.relu(x);  x = self.dropout(x)
        x = self.conv2(x, edge_index);  x = F.relu(x);  x = self.dropout(x)
        return self.classifier(x).squeeze(-1)

model_gnn = FraudAccountGNN(
    in_channels     = data.num_features,
    hidden_channels = HIDDEN_DIM
).to(device)

total_params = sum(p.numel() for p in model_gnn.parameters())
print(f'  Parameters: {total_params:,}')

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 9 — TRAIN
# ═════════════════════════════════════════════════════════════════════════════
sep('SECTION 9 — TRAINING')

n_fraud_train = int(y_all[train_indices].sum())
n_legit_train = len(train_indices) - n_fraud_train
pos_weight_val = n_legit_train / max(n_fraud_train, 1)
print(f'  pos_weight = {pos_weight_val:.1f}  '
      f'({n_legit_train:,} legit / {n_fraud_train:,} fraud accounts in train)')

pos_weight_t = torch.tensor([pos_weight_val], dtype=torch.float).to(device)
criterion    = nn.BCEWithLogitsLoss(pos_weight=pos_weight_t)
optimizer    = torch.optim.Adam(model_gnn.parameters(), lr=LR, weight_decay=1e-4)

best_prauc    = 0.0
best_state    = None
patience_ctr  = 0
history       = {'loss': [], 'test_prauc': []}

print(f'  Training up to {N_EPOCHS} epochs (patience={PATIENCE}) ...\n')

for epoch in range(1, N_EPOCHS + 1):
    model_gnn.train()
    optimizer.zero_grad()
    logits = model_gnn(data.x, data.edge_index)
    loss   = criterion(logits[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

    model_gnn.eval()
    with torch.no_grad():
        all_logits = model_gnn(data.x, data.edge_index)
        if test_mask.sum() > 0:
            probs_test = torch.sigmoid(all_logits[data.test_mask]).cpu().numpy()
            y_test_np  = data.y[data.test_mask].cpu().numpy()
            if y_test_np.sum() > 0:
                val_prauc = average_precision_score(y_test_np, probs_test)
            else:
                val_prauc = 0.0
        else:
            val_prauc = 0.0

    history['loss'].append(loss.item())
    history['test_prauc'].append(val_prauc)

    if epoch % 15 == 0 or epoch == 1:
        print(f'  Epoch {epoch:>4}  |  loss={loss.item():.4f}  |  test PR-AUC={val_prauc:.4f}')

    if val_prauc > best_prauc + 1e-4:
        best_prauc   = val_prauc
        best_state   = {k: v.clone() for k, v in model_gnn.state_dict().items()}
        patience_ctr = 0
    else:
        patience_ctr += 1
        if patience_ctr >= PATIENCE:
            print(f'\n  Early stopping at epoch {epoch}.')
            break

if best_state:
    model_gnn.load_state_dict(best_state)
print(f'\n  Best test PR-AUC: {best_prauc:.4f}')

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 10 — THRESHOLD + OOT EVALUATION
# ═════════════════════════════════════════════════════════════════════════════
sep('SECTION 10 — THRESHOLD SELECTION AND OOT EVALUATION')

model_gnn.eval()
with torch.no_grad():
    prob_all = torch.sigmoid(model_gnn(data.x, data.edge_index)).cpu().numpy()

y_train_np = y_all[train_indices]
y_test_np  = y_all[test_indices]  if test_indices  else np.array([])
y_oot_np   = y_all[oot_indices]

prob_test_np = prob_all[test_indices]  if test_indices  else np.array([])
prob_oot_np  = prob_all[oot_indices]

if len(y_test_np) > 0 and y_test_np.sum() > 0:
    pr_auc_test  = average_precision_score(y_test_np, prob_test_np)
    roc_auc_test = roc_auc_score(y_test_np, prob_test_np)
    thresholds   = np.arange(0.01, 1.00, 0.01)
    f1_scores    = [f1_score(y_test_np, (prob_test_np >= t).astype(int), zero_division=0)
                    for t in thresholds]
    best_threshold = thresholds[int(np.argmax(f1_scores))]
    print(f'  Test PR-AUC  : {pr_auc_test:.4f}')
    print(f'  Test ROC-AUC : {roc_auc_test:.4f}')
    print(f'  Best threshold (test F1): {best_threshold:.2f}')
else:
    best_threshold = 0.50
    pr_auc_test    = float('nan')
    roc_auc_test   = float('nan')
    print(f'  Skipping threshold sweep (no fraud in test set). Using default 0.50.')

if y_oot_np.sum() > 0:
    pr_auc_oot   = average_precision_score(y_oot_np, prob_oot_np)
    roc_auc_oot  = roc_auc_score(y_oot_np, prob_oot_np)
    oot_preds    = (prob_oot_np >= best_threshold).astype(int)
    oot_f1       = f1_score(y_oot_np, oot_preds, zero_division=0)
    oot_prec     = precision_score(y_oot_np, oot_preds, zero_division=0)
    oot_rec      = recall_score(y_oot_np, oot_preds, zero_division=0)
    oot_cm       = confusion_matrix(y_oot_np, oot_preds)
    tn, fp, fn, tp = oot_cm.ravel() if oot_cm.shape == (2,2) else (0,0,0,0)

    print(f'\n  OOT set evaluation (threshold={best_threshold:.2f}):')
    print(f'    PR-AUC    : {pr_auc_oot:.4f}')
    print(f'    ROC-AUC   : {roc_auc_oot:.4f}')
    print(f'    F1        : {oot_f1:.4f}')
    print(f'    Precision : {oot_prec:.4f}')
    print(f'    Recall    : {oot_rec:.4f}')
    print(f'    TN={tn:,}  FP={fp:,}  FN={fn:,}  TP={tp:,}')
else:
    pr_auc_oot = roc_auc_oot = oot_f1 = oot_prec = oot_rec = float('nan')
    tn = fp = fn = tp = 0
    print('  No fraud accounts in OOT set — cannot evaluate.')

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 11 — CHARTS
# ═════════════════════════════════════════════════════════════════════════════
sep('SECTION 11 — CHARTS')

epochs_done = len(history['loss'])
fig, axes   = plt.subplots(1, 3, figsize=(18, 5))

ax = axes[0]; ax2 = ax.twinx()
ax.plot(range(1, epochs_done+1), history['loss'],     color='tomato',    lw=1.5, label='Loss')
ax2.plot(range(1, epochs_done+1), history['test_prauc'], color='steelblue', lw=1.5, label='Test PR-AUC')
ax.set_xlabel('Epoch'); ax.set_ylabel('Loss', color='tomato')
ax2.set_ylabel('PR-AUC', color='steelblue')
ax.set_title('Model 4 Training Curve')
lines1,l1 = ax.get_legend_handles_labels()
lines2,l2 = ax2.get_legend_handles_labels()
ax.legend(lines1+lines2, l1+l2, fontsize=9)

if y_oot_np.sum() > 0:
    prec_c, rec_c, _ = precision_recall_curve(y_oot_np, prob_oot_np)
    axes[1].plot(rec_c, prec_c, color='darkorchid', lw=2,
                 label=f'Account GNN OOT PR-AUC={pr_auc_oot:.4f}')
    axes[1].scatter([oot_rec], [oot_prec], color='purple', s=120, zorder=5)
    axes[1].axhline(y_oot_np.mean(), color='grey', ls='--', lw=1, label='Random')
else:
    axes[1].text(0.5, 0.5, 'No OOT fraud accounts', ha='center', va='center',
                 transform=axes[1].transAxes, color='grey')
axes[1].set_title('PR Curve — Model 4 (OOT)')
axes[1].set_xlabel('Recall'); axes[1].set_ylabel('Precision')
axes[1].legend(fontsize=9); axes[1].set_xlim([0,1]); axes[1].set_ylim([0,1.02])

# Graph density comparison bar chart
models   = ['M3\n(txn-node\n50k)', 'M3\n(txn-node\n250k)', 'M4\n(acc-node\n250k)']
edges    = [496, 13400, edge_index.shape[1] // 2]
colors   = ['#4e79a7', '#59a14f', '#b07aa1']
axes[2].bar(models, edges, color=colors, edgecolor='white', width=0.55)
axes[2].set_title('Graph Density: Undirected Edges')
axes[2].set_ylabel('Edges')
for i, (bar, val) in enumerate(zip(axes[2].patches, edges)):
    axes[2].text(bar.get_x()+bar.get_width()/2, bar.get_height()+500,
                 f'{val:,}', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.suptitle('Model 4 — Account-Node GraphSAGE', fontsize=13, fontweight='bold')
plt.tight_layout()
save_chart(fig, 'model4_summary')

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 12 — SAVE
# ═════════════════════════════════════════════════════════════════════════════
sep('SECTION 12 — SAVE OUTPUTS')

torch.save(best_state or model_gnn.state_dict(), ARTIFACTS_DIR / 'gnn_model.pt')
print(f'  Model: {ARTIFACTS_DIR / "gnn_model.pt"}')

oot_out = pd.DataFrame({
    'account'    : list(all_accounts[oot_indices]) if oot_indices else [],
    'prob_fraud' : prob_oot_np,
    'pred_label' : (prob_oot_np >= best_threshold).astype(int),
    'true_label' : y_oot_np.astype(int)
})
oot_out.to_csv(PREDS_DIR / 'predictions_oot.csv', index=False)
print(f'  Preds: {PREDS_DIR / "predictions_oot.csv"}')

mlflow.set_tracking_uri(f'sqlite:///{PROJECT_ROOT}/mlflow.db')
mlflow.set_experiment('fraud_model_4_account_gnn')
with mlflow.start_run(run_name='account_gnn_v1'):
    mlflow.log_params({
        'n_accounts': n_accounts, 'n_edges': edge_index.shape[1]//2,
        'hidden_dim': HIDDEN_DIM, 'epochs_run': epochs_done,
        'lr': LR, 'patience': PATIENCE, 'best_threshold': best_threshold
    })
    mlflow.log_metrics({
        'pr_auc_oot': float(pr_auc_oot)   if not np.isnan(pr_auc_oot)  else -1,
        'roc_auc_oot': float(roc_auc_oot) if not np.isnan(roc_auc_oot) else -1,
        'f1_oot': float(oot_f1)           if not np.isnan(oot_f1)      else -1,
    })

sep('MODEL 4 COMPLETE')
print(f"""
  Node setup recap:
    Nodes  : {n_accounts:,} unique accounts
    Edges  : {edge_index.shape[1]//2:,} undirected (one per transaction)
    Labels : {int(y_all.sum()):,} fraud accounts ({y_all.mean()*100:.3f}%)

  Outputs:
    artifacts/model_4_account_gnn/gnn_model.pt
    artifacts/model_4_account_gnn/charts/model4_summary.png
    data/model_4_account_gnn/output/predictions_oot.csv
""")
