# Hardware & Software Requirements

## Your Current Hardware (Detected 2026-10-17)

| Component | Spec | Status |
|---|---|---|
| CPU | Intel Core i7-10750H @ 2.60GHz (6 cores / 12 threads) | ✅ Sufficient |
| RAM | 15.9 GB | ⚠️ Tight but workable — see notes |
| GPU | NVIDIA GeForce GTX 1650 Ti (4 GB VRAM) | ✅ CUDA-capable |
| NVIDIA Driver | 617.42 | ✅ Confirmed |
| CUDA Version | 12.6 (cu126) | ✅ Confirmed working |
| Disk (free) | ~118 GB | ✅ Sufficient |

---

## Minimum Requirements

| Component | Minimum | Recommended |
|---|---|---|
| RAM | 8 GB | 16 GB |
| CPU | Any modern 4-core | i5/i7 with 8+ threads |
| GPU | None (CPU-only works) | NVIDIA with 4+ GB VRAM |
| Disk | 10 GB free | 20 GB free |
| Python | 3.11 | 3.11 |

---

## GPU / CUDA Setup (Important)

Your GTX 1650 Ti is CUDA-capable and will significantly speed up GNN training (Phase 4). NVIDIA driver was updated 2026-10-18.

### Setup Steps

1. **Verify driver** — run `nvidia-smi` in PowerShell. CUDA Version (top-right) must show 11.8+.

2. **Install CUDA Toolkit 11.8** — download from `https://developer.nvidia.com/cuda-11-8-0-download-archive` (Windows → x86_64 → 11 → exe local). Custom install: keep CUDA Runtime, CUDA Development, cuDNN only — skip the driver. Restart after.

3. **Create venv and install PyTorch with CUDA 11.8:**
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   pip install -e .
   pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu118
   pip install torch_geometric
   ```

4. **Verify CUDA visible to PyTorch:**
   ```powershell
   python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
   ```
   Expected: `True` then `NVIDIA GeForce GTX 1650 Ti`

### CPU-only fallback (if CUDA setup fails)

Works fine for Phases 1–3. Phase 4 (GNN) will be slow (~hours per epoch vs ~minutes with GPU).
```powershell
pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu
pip install torch_geometric
```

---

## RAM Notes

PaySim has 6.3 million rows. With 15.9 GB RAM:

| Operation | Est. RAM | Status |
|---|---|---|
| Loading raw CSV | ~1.5 GB | ✅ Fine |
| Full DataFrame in memory | ~1.5–2 GB | ✅ Fine |
| Graph construction (all edges) | ~4–8 GB | ⚠️ Use chunked processing |
| GNN mini-batch training | ~2–4 GB | ✅ Fine with neighbor sampling |
| MLflow + Jupyter running | ~1 GB overhead | Factor in |

**The key:** Graph construction in Phase 4 must use the mini-batch neighbor sampling already configured in `configs/gnn.yaml`. Never try to load the entire graph into GPU VRAM (only 4 GB).

---

## Software Requirements

### Operating System
- Windows 11 ✅ (PowerShell 5.1+)

### Python Environment
- System Python 3.13 — no venv, no conda

### Key Package Versions (see requirements.txt for full list)

| Package | Version | Purpose |
|---|---|---|
| pandas | 2.1+ | Data manipulation |
| numpy | 1.26+ | Numerical ops |
| scikit-learn | 1.4+ | Preprocessing, metrics |
| xgboost | 2.0+ | Models 1 & 2 |
| torch | 2.7.1 (cu126) | Deep learning backend |
| torch_geometric | 2.5+ | GNN framework (Model 3) |
| networkx | 3.2+ | Graph analysis |
| mlflow | 2.11+ | Experiment tracking |
| shap | 0.45+ | Explainability (Phase 5) |
| matplotlib | 3.8+ | Visualisation |
| seaborn | 0.13+ | Statistical plots |

---

## GNN Feasibility on Your Laptop

**Short answer: Yes, fully feasible with mini-batch training.**

With the GTX 1650 Ti (4 GB VRAM) and mini-batch neighbor sampling:
- Training per epoch: ~5–15 minutes (CPU) or ~1–3 minutes (GPU with CUDA)
- Total Phase 4 training: ~2–4 hours (CPU) or ~30 min (GPU)
- The configuration in `configs/gnn.yaml` already accounts for your hardware:
  - `batch_size: 512` — safe for 4 GB VRAM
  - `num_neighbors: [10, 5]` — samples a small neighborhood, not the full graph
  - `max_degree: 50` — caps super-node connections

The GNN does **not** require loading all 6.3M transaction edges into memory at once.
