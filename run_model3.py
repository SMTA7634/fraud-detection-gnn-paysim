"""
run_model3.py  —  Phase 4: GraphSAGE GNN (Model 3)
Run: python run_model3.py

WHY A GRAPH NEURAL NETWORK?
  Models 1 and 2 treat each transaction in isolation (or with a simple
  historical count). A GNN goes further: it lets each transaction NODE
  aggregate information from its NEIGHBOURING transactions in the graph.

  Neighbour = another transaction that shares the same sender or recipient
  account within 30 simulated time steps.

  After aggregation, a transaction's representation encodes not just its
  own features, but a summary of what its neighbours look like. A fraud
  transaction surrounded by other fraud transactions will look very different
  from an isolated fraud transaction — the GNN can learn this.

WHAT IS GraphSAGE?
  GraphSAGE (Hamilton et al., 2017) is an INDUCTIVE graph learning method.
  'Inductive' means it can handle new nodes it has never seen during training
  — important for deployment where new transactions arrive continuously.

  At each layer, GraphSAGE:
    1. SAMPLES a fixed number of neighbours for each node
    2. AGGREGATES their feature vectors (we use mean aggregation)
    3. CONCATENATES the aggregated neighbour vector with the node's own vector
    4. TRANSFORMS through a learned weight matrix + activation

  Two layers of this give each node a view of its 2-hop neighbourhood
  (neighbours of neighbours), capturing slightly wider fraud patterns.

GRAPH DESIGN (transaction-node graph):
  - Each of the 50,000 transactions = one NODE
  - Node features = the 12 features from Model 1 (standardised for neural net)
  - EDGES: connect two transactions if they share nameOrig OR nameDest
            AND their steps are within 30 of each other
  - Undirected graph (edges go both ways)

SPLIT STRATEGY:
  Same as Models 1 and 2:
    OOT   = step > 594 (time-based holdout)
    Train = 80% of step ≤ 594, stratified
    Test  = 20% of step ≤ 594, stratified

  For graph training: train/test/OOT are NODE MASKS on the same full graph.
  The graph structure is shared — OOT nodes can be connected to train nodes.
  This is realistic: when a new transaction arrives it connects to historical
  transactions. We never use OOT LABELS during training.

Dataset  : paysim_subset_50k.csv
Hardware : GTX 1650 Ti (4GB VRAM) — confirmed CUDA available
Outputs  : artifacts/model_3_gnn/ and data/model_3_gnn/output/
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
ARTIFACTS_DIR = PROJECT_ROOT / 'artifacts' / 'model_3_gnn'
CHARTS_DIR    = ARTIFACTS_DIR / 'charts'
PREDS_DIR     = PROJECT_ROOT / 'data' / 'model_3_gnn' / 'output'

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

from sklearn.model_selection  import train_test_split
from sklearn.preprocessing    import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    confusion_matrix, roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve
)
import mlflow

sns.set_theme(style='whitegrid')
plt.rcParams['figure.dpi'] = 100

OOT_STEP_BOUNDARY = 594
RANDOM_SEED       = 42
TEST_SIZE         = 0.20
EDGE_TIME_WINDOW  = 30     # connect transactions within 30 steps of each other
HIDDEN_DIM        = 64     # neurons in each GNN layer
N_EPOCHS          = 100
LR                = 0.001
PATIENCE          = 15     # early stopping patience (epochs without val improvement)

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
    print(f'  Saved chart: {name}.png')

print(f'  Device: {device}')
if device.type == 'cuda':
    print(f'  GPU   : {torch.cuda.get_device_name(0)}')

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 — LOAD AND PREPROCESS NODE FEATURES
# ═════════════════════════════════════════════════════════════════════════════
sep('SECTION 1 — LOAD DATA AND ENGINEER NODE FEATURES')

note("""
  Each transaction becomes a NODE in the graph. The node's feature vector
  is the same 12-feature set used in Model 1, with one addition:

  STANDARDISATION (StandardScaler):
    XGBoost does not need feature scaling — it splits on thresholds.
    Neural networks DO need it. If one feature has values 0-10,000,000
    and another has values 0-1, the gradient steps will be dominated by
    the large-scale feature. StandardScaler transforms each feature to
    have mean=0 and standard deviation=1, putting all features on the
    same scale.

    As always: fit the scaler on TRAINING nodes only, then transform
    all nodes (including test and OOT) using training statistics.
""")

df = pd.read_csv(SUBSET_CSV)
df = df.sort_values('step').reset_index(drop=True)
print(f'  Loaded: {len(df):,} rows')

# Engineer features
df['hour_of_day']      = df['step'] % 24
df['day_of_sim']       = df['step'] // 24
df['pct_balance_used'] = np.where(df['oldbalanceOrg'] > 0,
                                  df['amount'] / df['oldbalanceOrg'], 0.0).clip(0, 10)

LOG1P_COLS = ['amount', 'oldbalanceOrg', 'oldbalanceDest']
for col in LOG1P_COLS:
    df[col] = np.log1p(df[col])

# OHE for type
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore', dtype=np.float32)

# Splits — define masks first so we fit scaler on train only
oot_mask_bool  = df['step'] >  OOT_STEP_BOUNDARY
dev_mask_bool  = df['step'] <= OOT_STEP_BOUNDARY
dev_indices    = df[dev_mask_bool].index.tolist()

train_idx, test_idx = train_test_split(
    dev_indices,
    test_size    = TEST_SIZE,
    random_state = RANDOM_SEED,
    stratify     = df.loc[dev_indices, 'isFraud']
)
oot_idx   = df[oot_mask_bool].index.tolist()

# Fit OHE and scaler on TRAIN NODES ONLY
ohe.fit(df.loc[train_idx, ['type']])
type_cols  = [f'type_{c}' for c in ohe.categories_[0]]
type_enc   = pd.DataFrame(ohe.transform(df[['type']]),
                          columns=type_cols, index=df.index)

FEATURE_COLS = ['step', 'amount', 'oldbalanceOrg', 'oldbalanceDest',
                'hour_of_day', 'day_of_sim', 'pct_balance_used'] + type_cols

df_feat = pd.concat([df[['step','amount','oldbalanceOrg','oldbalanceDest',
                          'hour_of_day','day_of_sim','pct_balance_used']],
                     type_enc], axis=1)

scaler = StandardScaler()
scaler.fit(df_feat.loc[train_idx])              # fit on train nodes only
X_all  = scaler.transform(df_feat).astype(np.float32)   # transform all nodes

y_all  = df['isFraud'].values.astype(np.float32)

print(f'  Node features: {X_all.shape[1]} dimensions')
print(f'  Train nodes  : {len(train_idx):,}  '
      f'| fraud: {y_all[train_idx].sum():.0f} ({y_all[train_idx].mean()*100:.2f}%)')
print(f'  Test nodes   : {len(test_idx):,}   '
      f'| fraud: {y_all[test_idx].sum():.0f} ({y_all[test_idx].mean()*100:.2f}%)')
print(f'  OOT nodes    : {len(oot_idx):,}   '
      f'| fraud: {y_all[oot_idx].sum():.0f} ({y_all[oot_idx].mean()*100:.2f}%)')

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 — BUILD THE TRANSACTION GRAPH
# ═════════════════════════════════════════════════════════════════════════════
sep('SECTION 2 — BUILD TRANSACTION GRAPH (edges)')

note("""
  GRAPH CONSTRUCTION RULES:
    Add an undirected edge between transaction i and transaction j if:
      (a) They share the same SENDER (nameOrig)  OR
          they share the same RECIPIENT (nameDest)
      AND
      (b) |step_i - step_j| <= 30  (within 30 simulated hours)

  WHY THE 30-STEP WINDOW?
    Without a time window, a recipient with 100 transactions spread over
    743 hours would create 100*99/2 = 4,950 edges. Most of those pairs
    are far apart in time and share no meaningful context. A 30-step
    window keeps edges local in time — neighbouring transactions are
    genuinely related events.

  WHY UNDIRECTED?
    Fraud can propagate in both directions in a network. A fraudulent
    TRANSFER at step 100 is relevant context for both earlier and later
    transactions involving the same account. Using undirected edges lets
    information flow both ways.

  IMPLEMENTATION:
    GroupBy nameOrig / nameDest, then within each group find all pairs
    where |step_i - step_j| <= 30. This is O(k²) per group where k is
    the group size — efficient because groups are small (max k=3 in 50k subset).
""")

print('  Building edges from shared nameOrig ...')
edges_src, edges_dst = [], []

for _, grp in df.groupby('nameOrig'):
    if len(grp) < 2:
        continue
    idxs  = grp.index.tolist()
    steps = grp['step'].values
    for i in range(len(idxs)):
        for j in range(i + 1, len(idxs)):
            if abs(int(steps[i]) - int(steps[j])) <= EDGE_TIME_WINDOW:
                edges_src += [idxs[i], idxs[j]]
                edges_dst += [idxs[j], idxs[i]]

orig_edges = len(edges_src) // 2
print(f'  Sender-based edges      : {orig_edges:,}')

print('  Building edges from shared nameDest ...')
for _, grp in df.groupby('nameDest'):
    if len(grp) < 2:
        continue
    idxs  = grp.index.tolist()
    steps = grp['step'].values
    for i in range(len(idxs)):
        for j in range(i + 1, len(idxs)):
            if abs(int(steps[i]) - int(steps[j])) <= EDGE_TIME_WINDOW:
                edges_src += [idxs[i], idxs[j]]
                edges_dst += [idxs[j], idxs[i]]

dest_edges = len(edges_src) // 2 - orig_edges
total_undirected = len(edges_src) // 2
print(f'  Recipient-based edges   : {dest_edges:,}')
print(f'  Total undirected edges  : {total_undirected:,}')
print(f'  Total directed edges    : {len(edges_src):,}  (each undirected edge stored twice)')
print(f'  Avg edges per node      : {len(edges_src) / len(df):.2f}')

if len(edges_src) == 0:
    print('\n  [NOTE] No edges found within 30-step window.')
    print('  This is expected for the 50k subset — most accounts appear only once.')
    print('  Adding self-loops so every node has at least one edge to aggregate over.')
    edges_src = list(range(len(df)))
    edges_dst = list(range(len(df)))

edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
print(f'\n  edge_index shape: {edge_index.shape}')

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 — BUILD PyG DATA OBJECT
# ═════════════════════════════════════════════════════════════════════════════
sep('SECTION 3 — ASSEMBLE PyTorch Geometric DATA OBJECT')

note("""
  PyTorch Geometric (PyG) expects a Data object that bundles:
    x           : node feature matrix  [num_nodes, num_features]
    edge_index  : edge list            [2, num_edges]  (COO format)
    y           : node labels          [num_nodes]

  COO (Coordinate) format stores edges as two lists:
    edge_index[0] = source node indices
    edge_index[1] = destination node indices
  So edge_index[:, k] = (source_k, destination_k) for the k-th edge.

  We also create boolean masks for train/test/OOT nodes so the training
  loop knows which nodes to include in the loss calculation.
""")

x_tensor = torch.tensor(X_all, dtype=torch.float)
y_tensor = torch.tensor(y_all, dtype=torch.float)

n = len(df)
train_mask = torch.zeros(n, dtype=torch.bool)
test_mask  = torch.zeros(n, dtype=torch.bool)
oot_mask   = torch.zeros(n, dtype=torch.bool)
train_mask[train_idx] = True
test_mask[test_idx]   = True
oot_mask[oot_idx]     = True

data = Data(x=x_tensor, edge_index=edge_index, y=y_tensor)
data.train_mask = train_mask
data.test_mask  = test_mask
data.oot_mask   = oot_mask
data = data.to(device)

print(f'  Nodes          : {data.num_nodes:,}')
print(f'  Edges (directed): {data.num_edges:,}')
print(f'  Node features  : {data.num_features}')
print(f'  Train nodes    : {data.train_mask.sum().item():,}')
print(f'  Test nodes     : {data.test_mask.sum().item():,}')
print(f'  OOT nodes      : {data.oot_mask.sum().item():,}')
print(f'  Data on device : {data.x.device}')

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4 — DEFINE THE GRAPHSAGE MODEL
# ═════════════════════════════════════════════════════════════════════════════
sep('SECTION 4 — GRAPHSAGE MODEL ARCHITECTURE')

note("""
  GRAPHSAGE LAYER (SAGEConv):
    For each node v:
      1. Sample neighbours N(v)
      2. Aggregate: h_neigh = MEAN( h_u  for u in N(v) )
      3. Concatenate: [h_v || h_neigh]
      4. Transform: h_v_new = ReLU( W * [h_v || h_neigh] + b )

  Our architecture:
    Input     : 12 features (standardised)
    Layer 1   : SAGEConv(12 → 64)  + ReLU + Dropout(0.3)
    Layer 2   : SAGEConv(64 → 64)  + ReLU + Dropout(0.3)
    Classifier: Linear(64 → 1)     + Sigmoid (fraud probability)

  DROPOUT (p=0.3):
    Randomly zeros 30% of neurons during each forward pass.
    Forces the network to not rely on any single neuron — improves
    generalisation. Only applied during training, not inference.

  TWO LAYERS = 2-HOP NEIGHBOURHOOD:
    After 2 GraphSAGE layers, each node's representation has incorporated
    information from its neighbours AND their neighbours.
    For a fraud transaction, this means the model sees:
      - The fraud transaction's own features
      - Features of transactions sharing the same account (1-hop)
      - Features of transactions sharing accounts with those (2-hop)
""")

class FraudGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels=64):
        super().__init__()
        self.conv1      = SAGEConv(in_channels, hidden_channels)
        self.conv2      = SAGEConv(hidden_channels, hidden_channels)
        self.classifier = nn.Linear(hidden_channels, 1)
        self.dropout    = nn.Dropout(p=0.3)

    def forward(self, x, edge_index):
        # Layer 1 — aggregate 1-hop neighbourhood
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        # Layer 2 — aggregate 2-hop neighbourhood
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        # Classification head — output raw logit (no sigmoid here;
        # BCEWithLogitsLoss applies sigmoid internally for numerical stability)
        x = self.classifier(x)
        return x.squeeze(-1)   # shape: [num_nodes]

model_gnn = FraudGNN(
    in_channels     = data.num_features,
    hidden_channels = HIDDEN_DIM
).to(device)

total_params = sum(p.numel() for p in model_gnn.parameters())
print(f'  Model architecture:')
print(f'    SAGEConv({data.num_features} → {HIDDEN_DIM})  →  ReLU  →  Dropout(0.3)')
print(f'    SAGEConv({HIDDEN_DIM} → {HIDDEN_DIM})  →  ReLU  →  Dropout(0.3)')
print(f'    Linear({HIDDEN_DIM} → 1)')
print(f'  Total parameters : {total_params:,}')

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5 — TRAIN THE GNN
# ═════════════════════════════════════════════════════════════════════════════
sep('SECTION 5 — TRAINING LOOP')

note("""
  LOSS FUNCTION: BCEWithLogitsLoss
    Binary Cross-Entropy combines a sigmoid activation and cross-entropy
    loss in a numerically stable way. For each node:
      loss = -[y * log(sigmoid(logit)) + (1-y) * log(1 - sigmoid(logit))]

    Cross-entropy penalises confident wrong predictions heavily.
    If the model outputs logit=5 (very confident = fraud) for a legit
    transaction, the loss is enormous. This drives it to be accurate.

  OPTIMISER: Adam (Adaptive Moment Estimation)
    Adam adjusts the learning rate per parameter. It keeps a running
    average of gradients (momentum) and their squared values (adaptive
    rate). This makes training more stable than vanilla SGD, especially
    in early epochs when gradients are large and noisy.

  EARLY STOPPING:
    After each epoch, evaluate PR-AUC on the TEST nodes.
    If PR-AUC does not improve for PATIENCE=15 consecutive epochs,
    stop training and restore the best weights.
    This prevents overfitting — the model memorising the training nodes
    at the expense of generalisation to test/OOT.

  FULL-GRAPH TRAINING:
    Because our graph is small (50k nodes, sparse edges), we do full-graph
    forward passes — all 50k nodes are processed each epoch.
    The LOSS is only computed on TRAIN MASK nodes (labels never used for
    test/OOT nodes during training).
""")

# pos_weight = n_legit_train / n_fraud_train
# When the dataset is imbalanced (e.g. 250k has 3.3% fraud vs 50k's 13.8%),
# the loss function would otherwise be dominated by legit transactions and the
# model learns to predict "legit" for everything. pos_weight tells the loss:
# "treat each fraud sample as if it were pos_weight times as important."
# This is the GNN equivalent of XGBoost's scale_pos_weight.
n_fraud_train = int(y_all[train_mask.numpy()].sum())
n_legit_train = int(train_mask.sum().item()) - n_fraud_train
pos_weight_val = n_legit_train / n_fraud_train
print(f'  Class balance  : {n_fraud_train:,} fraud / {n_legit_train:,} legit '
      f'(pos_weight = {pos_weight_val:.1f})')
pos_weight_tensor = torch.tensor([pos_weight_val], dtype=torch.float).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
optimizer = torch.optim.Adam(model_gnn.parameters(), lr=LR, weight_decay=1e-4)

best_val_prauc  = 0.0
best_state      = None
patience_counter = 0
history         = {'train_loss': [], 'test_prauc': []}

print(f'  Training for up to {N_EPOCHS} epochs (early stop patience={PATIENCE}) ...\n')

for epoch in range(1, N_EPOCHS + 1):

    # ── Training step ─────────────────────────────────────────────────────
    model_gnn.train()
    optimizer.zero_grad()
    logits = model_gnn(data.x, data.edge_index)
    loss   = criterion(logits[data.train_mask],
                       data.y[data.train_mask])
    loss.backward()
    optimizer.step()

    # ── Validation step (test nodes) ──────────────────────────────────────
    model_gnn.eval()
    with torch.no_grad():
        logits_all  = model_gnn(data.x, data.edge_index)
        probs_test  = torch.sigmoid(logits_all[data.test_mask]).cpu().numpy()
        y_test_np   = data.y[data.test_mask].cpu().numpy()
        val_prauc   = average_precision_score(y_test_np, probs_test)

    history['train_loss'].append(loss.item())
    history['test_prauc'].append(val_prauc)

    if epoch % 10 == 0 or epoch == 1:
        print(f'  Epoch {epoch:>4}  |  loss={loss.item():.4f}  |  test PR-AUC={val_prauc:.4f}')

    # ── Early stopping ────────────────────────────────────────────────────
    if val_prauc > best_val_prauc + 1e-4:
        best_val_prauc   = val_prauc
        best_state       = {k: v.clone() for k, v in model_gnn.state_dict().items()}
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f'\n  Early stopping at epoch {epoch} '
                  f'(no improvement for {PATIENCE} epochs).')
            break

# Restore best weights
model_gnn.load_state_dict(best_state)
print(f'\n  Best test PR-AUC: {best_val_prauc:.4f}')

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6 — THRESHOLD SELECTION AND OOT EVALUATION
# ═════════════════════════════════════════════════════════════════════════════
sep('SECTION 6 — THRESHOLD SELECTION AND OOT EVALUATION')

model_gnn.eval()
with torch.no_grad():
    logits_all = model_gnn(data.x, data.edge_index)
    prob_all   = torch.sigmoid(logits_all).cpu().numpy()

prob_train_np = prob_all[train_mask.numpy()]
prob_test_np  = prob_all[test_mask.numpy()]
prob_oot_np   = prob_all[oot_mask.numpy()]

y_train_np = y_all[train_mask.numpy()]
y_test_np  = y_all[test_mask.numpy()]
y_oot_np   = y_all[oot_mask.numpy()]

pr_auc_test  = average_precision_score(y_test_np, prob_test_np)
pr_auc_oot   = average_precision_score(y_oot_np,  prob_oot_np)
roc_auc_test = roc_auc_score(y_test_np, prob_test_np)
roc_auc_oot  = roc_auc_score(y_oot_np,  prob_oot_np)

print(f'\n  {"Set":<8} {"PR-AUC":>10} {"ROC-AUC":>10}')
print(f'  {"Test":<8} {pr_auc_test:>10.4f} {roc_auc_test:>10.4f}')
print(f'  {"OOT":<8} {pr_auc_oot:>10.4f} {roc_auc_oot:>10.4f}')

# Threshold sweep on test set
thresholds  = np.arange(0.01, 1.00, 0.01)
f1_scores   = [f1_score(y_test_np, (prob_test_np >= t).astype(int), zero_division=0)
               for t in thresholds]
best_idx       = int(np.argmax(f1_scores))
best_threshold = thresholds[best_idx]
prec_scores = [precision_score(y_test_np, (prob_test_np >= t).astype(int), zero_division=0)
               for t in thresholds]
rec_scores  = [recall_score(y_test_np, (prob_test_np >= t).astype(int), zero_division=0)
               for t in thresholds]

print(f'\n  Best threshold (test set F1): {best_threshold:.2f}')

# Final OOT evaluation
oot_preds = (prob_oot_np >= best_threshold).astype(int)
oot_f1    = f1_score(y_oot_np,    oot_preds, zero_division=0)
oot_prec  = precision_score(y_oot_np, oot_preds, zero_division=0)
oot_rec   = recall_score(y_oot_np,   oot_preds, zero_division=0)
oot_cm    = confusion_matrix(y_oot_np, oot_preds)
tn, fp, fn, tp = oot_cm.ravel()

print(f'\n  OOT set evaluation (threshold = {best_threshold:.2f}):')
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

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 7 — ALL THREE MODELS COMPARED
# ═════════════════════════════════════════════════════════════════════════════
sep('SECTION 7 — MODEL 1 vs MODEL 2 vs MODEL 3 COMPARISON')

note("""
  We now have three models trained on the same 50k dataset with the same
  OOT split. The differences are:

  Model 1 (XGBoost baseline)    : 12 raw features, no graph structure
  Model 2 (XGBoost + graph)     : 16 features, recipient history as flat features
  Model 3 (GraphSAGE GNN)       : 12 raw features, but graph structure via message passing

  The comparison tells us:
    Model 1 vs Model 2 : does adding graph features as flat inputs help?
    Model 1 vs Model 3 : does processing graph STRUCTURE (not just summaries) help?
    Model 2 vs Model 3 : is message-passing GNN better than hand-crafted graph features?
""")

# Reference metrics from Models 1 and 2
m1 = dict(pr_auc=1.0000, f1=0.9997, prec=1.0000, rec=0.9994, fn=1,  fp=0)
m2 = dict(pr_auc=1.0000, f1=0.9997, prec=0.9994, rec=1.0000, fn=0,  fp=1)
m3 = dict(pr_auc=pr_auc_oot, f1=oot_f1, prec=oot_prec, rec=oot_rec, fn=int(fn), fp=int(fp))

print(f'\n  {"Metric":<25} {"Model 1":>12} {"Model 2":>12} {"Model 3 (GNN)":>14}')
print(f'  {"-"*68}')
for metric, label in [('pr_auc','PR-AUC (OOT)'),('f1','F1 (OOT)'),
                       ('prec','Precision (OOT)'),('rec','Recall (OOT)'),
                       ('fn','Missed fraud (FN)'),('fp','False alarms (FP)')]:
    fmt = '.4f' if metric not in ('fn','fp') else ','
    print(f'  {label:<25} {m1[metric]:>12{fmt}} {m2[metric]:>12{fmt}} {m3[metric]:>14{fmt}}')

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 8 — CHARTS
# ═════════════════════════════════════════════════════════════════════════════
sep('SECTION 8 — CHARTS')

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Training curve
epochs_done = len(history['train_loss'])
ax = axes[0]
ax2 = ax.twinx()
ax.plot(range(1, epochs_done+1), history['train_loss'],
        color='tomato', lw=1.5, label='Train loss')
ax2.plot(range(1, epochs_done+1), history['test_prauc'],
         color='steelblue', lw=1.5, label='Test PR-AUC')
ax.set_xlabel('Epoch'); ax.set_ylabel('Loss', color='tomato')
ax2.set_ylabel('PR-AUC', color='steelblue')
ax.set_title('GNN Training Curve')
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1+lines2, labels1+labels2, fontsize=9)

# PR curve (OOT)
prec_c, rec_c, _ = precision_recall_curve(y_oot_np, prob_oot_np)
axes[1].plot(rec_c, prec_c, color='seagreen', lw=2,
             label=f'GNN OOT PR-AUC = {pr_auc_oot:.4f}')
axes[1].scatter([oot_rec], [oot_prec], color='darkgreen', s=120, zorder=5)
axes[1].axhline(y_oot_np.mean(), color='grey', ls='--', lw=1, label='Random')
axes[1].set_xlabel('Recall'); axes[1].set_ylabel('Precision')
axes[1].set_title('PR Curve — Model 3 GNN (OOT)')
axes[1].legend(fontsize=9); axes[1].set_xlim([0,1]); axes[1].set_ylim([0,1.02])

# Model comparison bar chart (F1)
models = ['Model 1\n(XGBoost)', 'Model 2\n(XGB+Graph)', 'Model 3\n(GNN)']
f1s    = [m1['f1'], m2['f1'], m3['f1']]
colors = ['steelblue', 'darkorange', 'seagreen']
bars   = axes[2].bar(models, f1s, color=colors, edgecolor='white', width=0.5)
axes[2].set_ylim([min(f1s) - 0.01, 1.002])
axes[2].set_title('F1 Score Comparison (OOT set)')
axes[2].set_ylabel('F1')
for bar, val in zip(bars, f1s):
    axes[2].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.0002,
                 f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.suptitle('Model 3 — GraphSAGE GNN', fontsize=13, fontweight='bold')
plt.tight_layout()
save_chart(fig, 'model3_summary')

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 9 — SAVE OUTPUTS
# ═════════════════════════════════════════════════════════════════════════════
sep('SECTION 9 — SAVE OUTPUTS')

torch.save(best_state, ARTIFACTS_DIR / 'gnn_model.pt')
print(f'  Model saved: {ARTIFACTS_DIR / "gnn_model.pt"}')

oot_out = pd.DataFrame({
    'step'       : df.loc[oot_idx, 'step'].values,
    'prob_fraud' : prob_oot_np,
    'pred_label' : oot_preds,
    'true_label' : y_oot_np.astype(int)
})
oot_out.to_csv(PREDS_DIR / 'predictions_oot.csv', index=False)
print(f'  Predictions: {PREDS_DIR / "predictions_oot.csv"}')

mlflow.set_tracking_uri(f'sqlite:///{PROJECT_ROOT}/mlflow.db')
mlflow.set_experiment('fraud_model_3_gnn')
with mlflow.start_run(run_name='graphsage_v1'):
    mlflow.log_params({'hidden_dim': HIDDEN_DIM, 'epochs_run': epochs_done,
                       'lr': LR, 'patience': PATIENCE,
                       'edge_time_window': EDGE_TIME_WINDOW,
                       'n_edges': len(edges_src),
                       'best_threshold': best_threshold})
    mlflow.log_metrics({'pr_auc_oot': pr_auc_oot, 'roc_auc_oot': roc_auc_oot,
                        'f1_oot': oot_f1, 'precision_oot': oot_prec,
                        'recall_oot': oot_rec, 'fn_oot': int(fn), 'fp_oot': int(fp)})

sep('MODEL 3 COMPLETE')
print("""
  Outputs:
    artifacts/model_3_gnn/gnn_model.pt
    artifacts/model_3_gnn/charts/model3_summary.png
    data/model_3_gnn/output/predictions_oot.csv

  All three models are now complete.
  Final comparison is in Section 7 above.
""")
