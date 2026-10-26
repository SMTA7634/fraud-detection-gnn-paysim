"""
run_final_report.py  —  Phase 6: Final Model Comparison Report
Run: python run_final_report.py

Generates a clean summary comparing all 4 models and saves
a final comparison chart + metrics CSV to artifacts/final_report/.
"""

import warnings, os
warnings.filterwarnings('ignore')
os.environ['GIT_PYTHON_REFRESH'] = 'quiet'

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

PROJECT_ROOT = Path(__file__).resolve().parent
OUT_DIR      = PROJECT_ROOT / 'artifacts' / 'final_report'
OUT_DIR.mkdir(parents=True, exist_ok=True)

def sep(t): print(f'\n{"="*65}\n  {t}\n{"="*65}')

sep('FINAL MODEL COMPARISON REPORT')

# ── Metrics (from actual run outputs) ─────────────────────────────────────
results = pd.DataFrame([
    dict(model='Model 1',  label='XGBoost\nBaseline',
         dataset='50k',  features=12,  graph='None',
         pr_auc=1.0000, roc_auc=1.0000, f1=0.9997,
         precision=1.0000, recall=0.9994, fn=1,   fp=0,
         prediction_unit='transaction'),
    dict(model='Model 2',  label='XGBoost\n+Graph Feats',
         dataset='50k',  features=16,  graph='Hand-crafted recipient features',
         pr_auc=1.0000, roc_auc=1.0000, f1=0.9997,
         precision=0.9994, recall=1.0000, fn=0,   fp=1,
         prediction_unit='transaction'),
    dict(model='Model 3',  label='GraphSAGE\nTxn-Node',
         dataset='250k', features=12,  graph='13,400 edges (shared nameDest)',
         pr_auc=0.9706, roc_auc=0.9872, f1=0.8868,
         precision=0.9410, recall=0.8386, fn=267, fp=87,
         prediction_unit='transaction'),
    dict(model='Model 4',  label='GraphSAGE\nAccount-Node',
         dataset='250k', features=10,  graph='250,000 edges (transactions)',
         pr_auc=0.8051, roc_auc=0.9638, f1=0.7457,
         precision=0.6478, recall=0.8785, fn=201, fp=790,
         prediction_unit='account'),
])

print('\n  All model results (OOT set):')
print(results[['model','dataset','features','pr_auc','roc_auc',
               'f1','precision','recall','fn','fp']].to_string(index=False))

results.to_csv(OUT_DIR / 'model_comparison.csv', index=False)
print(f'\n  Saved: artifacts/final_report/model_comparison.csv')

# ── Chart ──────────────────────────────────────────────────────────────────
colors = ['#4e79a7', '#59a14f', '#b07aa1', '#e15759']
labels = results['label'].tolist()
x      = np.arange(4)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('PaySim Fraud Detection — All 4 Models Comparison (OOT set)',
             fontsize=14, fontweight='bold')

metrics = [
    ('pr_auc',    'PR-AUC',    True),
    ('roc_auc',   'ROC-AUC',   True),
    ('f1',        'F1 Score',  True),
    ('precision', 'Precision', True),
    ('recall',    'Recall',    True),
]

for i, (col, title, higher_better) in enumerate(metrics):
    ax   = axes[i // 3][i % 3]
    vals = results[col].values
    bars = ax.bar(x, vals, color=colors, edgecolor='white', width=0.6, alpha=0.9)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_ylim([max(0, min(vals) - 0.08), 1.02])
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.004,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Bottom right: FN / FP counts
ax = axes[1][2]
w  = 0.3
fn_vals = results['fn'].values.astype(float)
fp_vals = results['fp'].values.astype(float)
bars1 = ax.bar(x - w/2, fn_vals, w, label='FN (missed fraud)', color='#e15759', alpha=0.85, edgecolor='white')
bars2 = ax.bar(x + w/2, fp_vals, w, label='FP (false alarm)',  color='#4e79a7', alpha=0.85, edgecolor='white')
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
ax.set_title('FN vs FP Count', fontsize=11, fontweight='bold')
ax.legend(fontsize=9)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
for bar, val in zip(list(bars1)+list(bars2), list(fn_vals)+list(fp_vals)):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
            f'{int(val):,}', ha='center', fontsize=8)

plt.tight_layout()
fig.savefig(OUT_DIR / 'model_comparison_chart.png', bbox_inches='tight', facecolor='white')
plt.close(fig)
print('  Saved: artifacts/final_report/model_comparison_chart.png')

sep('FINAL REPORT COMPLETE')
print("""
  Key findings across all 4 models:

  1. XGBoost (Models 1 & 2) achieves near-perfect scores on PaySim.
     The fraud signal is concentrated in individual transaction features
     (pct_balance_used, type_TRANSFER) — not in network structure.

  2. Model 2 (XGBoost + graph features) matched Model 1 with 0 missed
     fraud (FN=0), suggesting recipient history adds a small edge.

  3. Model 3 (GNN, transaction-node) reached PR-AUC=0.97 but is
     bottlenecked by sparse graph structure (0.11 edges/node).
     On the full 6.3M dataset, this gap would likely narrow or close.

  4. Model 4 (GNN, account-node) answers a DIFFERENT question —
     account risk vs. transaction risk. ROC-AUC=0.96 shows the model
     reliably RANKS fraud accounts, even if absolute precision is lower.
     High recall (87.9%) is valuable for account investigation workflows.

  5. For a production fraud system, the recommended stack is:
     Model 1/2 for real-time transaction blocking (milliseconds)
     Model 4   for overnight account-level risk scoring (batch)
""")
