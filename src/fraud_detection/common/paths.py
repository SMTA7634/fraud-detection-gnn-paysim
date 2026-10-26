"""Central path registry. Import from here — never hardcode paths in other modules."""

from pathlib import Path

# Project root is the parent of src/
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# ── Raw data ──────────────────────────────────────────────────────────────────
RAW_DIR = PROJECT_ROOT / "data" / "shared" / "raw"
PAYSIM_CSV = RAW_DIR / "PaySim_kaggle_raw_data.csv"

# ── Per-model data dirs ───────────────────────────────────────────────────────
def model_data_dir(model_key: str, stage: str) -> Path:
    """Return data/<model_key>/<stage>/ and create it if missing.

    Args:
        model_key: one of 'model_1_baseline', 'model_2_graph_features', 'model_3_gnn'
        stage: one of 'raw', 'intermediate', 'output'
    """
    p = PROJECT_ROOT / "data" / model_key / stage
    p.mkdir(parents=True, exist_ok=True)
    return p


# ── Artifacts ─────────────────────────────────────────────────────────────────
def artifacts_dir(model_key: str) -> Path:
    p = PROJECT_ROOT / "artifacts" / model_key
    p.mkdir(parents=True, exist_ok=True)
    return p


# ── Reports ───────────────────────────────────────────────────────────────────
REPORTS_DIR = PROJECT_ROOT / "reports"

def reports_dir(category: str) -> Path:
    """category: 'data_quality' | 'model_evaluation' | 'explainability' | 'experiment_logs'"""
    p = REPORTS_DIR / category
    p.mkdir(parents=True, exist_ok=True)
    return p


# ── Configs ───────────────────────────────────────────────────────────────────
CONFIGS_DIR = PROJECT_ROOT / "configs"
