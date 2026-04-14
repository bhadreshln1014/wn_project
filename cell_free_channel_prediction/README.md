# GNN-CNN Hybrid Channel Prediction for Cell-Free Massive MIMO

**Research area:** AI-Native Physical Layer Design for 6G  
**Problem:** Multi-step forward channel state prediction in distributed cell-free massive MIMO under high user mobility

---

## What This Project Does

In cell-free massive MIMO, the channel estimate obtained at pilot time is stale by the time the precoder uses it. This *channel aging* gap worsens as users move faster. Rather than re-estimating the channel after every pilot, this project predicts it forward in time.

The core contribution is a **GNN-CNN hybrid** that does two things conventional approaches cannot:

1. **Shared CNN per AP** — extracts temporal features from each AP's channel history independently, with weights shared across all APs (parameter-efficient, topology-agnostic)
2. **Single GCN layer** — aggregates information across spatially correlated neighbouring APs using a learned message-passing step, exploiting the Gudmundson inter-AP shadow fading correlation that makes neighbouring APs genuinely see correlated channels

No prior work applies this specific architecture to multi-step channel prediction in the cell-free setting under high mobility. The Kim et al. (2025) survey explicitly lists this as an open direction.

---

## Architecture

```
Input: (B, L, 2NK, T_history)
         ↓
 ┌─────────────────────────────┐
 │  Stage 1 — LocalCNN         │  Conv1D×3 → BN → ReLU → AvgPool → Linear
 │  Shared weights across APs  │  Output: h_loc ∈ R^64 per AP
 └─────────────────────────────┘
         ↓
 ┌─────────────────────────────┐
 │  Stage 2 — GCNLayer         │  Â h_loc W^T   (Â = D^{-½}(A+I)D^{-½})
 │  Single layer, no smoothing │  Output: h_agg ∈ R^64 per AP
 └─────────────────────────────┘
         ↓
 ┌─────────────────────────────┐
 │  Stage 3 — PredictionHead   │  Linear(64→128→2NK·T_predict)
 └─────────────────────────────┘
         ↓
Output: (B, L, 2NK, T_predict)
```

**Parameter target:** < 100K total  
**Graph construction:** K-NN (K=3) from fixed AP positions  
**Channel model:** Jake's AR(1) small-scale fading + Gudmundson spatially correlated large-scale fading (the latter is what gives the GCN real signal to work with)

---

## System Parameters

| Parameter | Value |
|-----------|-------|
| APs (L) | 16 |
| UEs (K) | 4 |
| Antennas per AP (N) | 4 |
| Carrier frequency | 3.5 GHz |
| User velocities | 30 / 60 / 120 km/h |
| History window (T) | 10 time steps |
| Prediction horizon | 3 steps ahead |
| Area | 500 × 500 m |
| Sampling interval | 1 ms |
| Shadow decorrelation distance | 9 m (Gudmundson / Ngo 2017) |

---

## File Structure

```
.
├── dataset.py       # Channel data generation (Jake's + Gudmundson model)
├── model.py         # GNNCNNHybrid, CNNOnly, loss functions, latency utils
├── baselines.py     # AR(4) predictor, Kalman filter, evaluation helpers
├── train.py         # Training loop, early stopping, checkpointing
├── evaluate.py      # All experiments, plots, summary tables
├── data/            # Generated datasets (created on first run)
├── checkpoints/     # Saved model weights (created during training)
├── results/         # Plots and JSON result files (created during evaluation)
├── requirements.txt
└── README.md
```

---

## Setup

```bash
# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

`requirements.txt`:
```
torch>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
matplotlib>=3.7.0
```

> **Note:** PyTorch Geometric is *not* required. The GCN layer is implemented manually using a precomputed normalised adjacency matrix, which handles batched graphs cleanly without DataLoader overhead.

---

## Running the Project

### Step 1 — Generate datasets

Generates all velocity × layout combinations (~10 min on first run, saved to `data/`):

```bash
python dataset.py
```

### Step 2 — Train all models

Train GNN-CNN and CNN-Only for each layout and velocity (6 training runs per model type, 12 total):

```bash
# GNN-CNN
for layout in random grid; do
  for vel in 30 60 120; do
    python train.py --layout $layout --velocity $vel --model gnn_cnn
  done
done

# CNN-Only (ablation baseline)
for layout in random grid; do
  for vel in 30 60 120; do
    python train.py --layout $layout --velocity $vel --model cnn_only
  done
done
```

Or train a single configuration directly:

```bash
python train.py --layout random --velocity 120 --model gnn_cnn
```

Training flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--layout` | `random` | AP topology: `random` or `grid` |
| `--velocity` | `60` | User velocity in km/h |
| `--model` | `gnn_cnn` | `gnn_cnn` or `cnn_only` |
| `--epochs` | `100` | Max training epochs |
| `--batch` | `32` | Batch size |
| `--lr` | `1e-3` | Initial learning rate |

### Step 3 — Evaluate

Run all experiments at once:

```bash
python evaluate.py
```

Or run individual experiments:

```bash
python evaluate.py --exp velocity      # Exp 1: NMSE vs velocity (PRIMARY)
python evaluate.py --exp step          # Exp 2: NMSE vs prediction step k
python evaluate.py --exp ablation_gcn  # Exp 3: CNN-Only vs GNN-CNN
python evaluate.py --exp ablation_topo # Exp 4: Grid vs random AP layout
python evaluate.py --exp efficiency    # Exp 5: Parameter count + latency
```

---

## What the Results Prove

| Claim | Experiment | How |
|-------|-----------|-----|
| Beats AR and Kalman at high mobility | Exp 1 | NMSE vs velocity at 120 km/h |
| GCN adds value over local CNN alone | Exp 3 | Ablation: CNN-Only vs GNN-CNN |
| Works on irregular AP topology | Exp 4 | Ablation: random vs grid layout |
| Lightweight enough for edge deployment | Exp 5 | Parameter count + inference latency |

The NMSE comparison across methods at 120 km/h (Experiment 1, random layout) is the single most important result. Everything else supports or contextualises it.

---

## Output Files

After evaluation, `results/` contains:

```
results/
├── nmse_vs_velocity_random.json
├── nmse_vs_velocity_grid.json
├── nmse_vs_step_random_60.json
├── nmse_vs_step_random_120.json
├── ablation_gcn_random.json
├── ablation_gcn_grid.json
├── ablation_topology_60.json
├── ablation_topology_120.json
├── efficiency.json
├── fig1_nmse_vs_velocity_random.png   ← PRIMARY result figure
├── fig1_nmse_vs_velocity_grid.png
├── fig2_nmse_vs_step_random_60.png
├── fig2_nmse_vs_step_random_120.png
├── fig3_ablation_gcn_random.png
├── fig3_ablation_gcn_grid.png
├── fig4_ablation_topology_60.png
└── fig4_ablation_topology_120.png
```

---

## Key Design Decisions

**Why a single GCN layer?**  
Multiple GCN layers cause over-smoothing — all AP embeddings converge toward the same value, removing the spatial differentiation that makes the GCN useful. One layer is sufficient to aggregate one hop of neighbourhood information.

**Why shared CNN weights across APs?**  
The temporal channel dynamics at each AP follow the same Jake's model regardless of AP location. Sharing weights reduces the parameter count by a factor of L (16×) and forces the model to learn a generalised temporal feature extractor rather than memorising per-AP statistics.

**Why Gudmundson shadow fading?**  
Without spatially correlated large-scale fading, neighbouring APs see statistically independent channels. The GCN would have nothing meaningful to aggregate, and the CNN-vs-GNN ablation would show negligible difference — which would invalidate the central argument. The Gudmundson model (Ngo et al. 2017, the canonical cell-free reference) gives the GCN real cross-AP mutual information.

**Why K-NN graph construction?**  
The graph must capture which APs genuinely share spatial correlation. Distance-based K-NN (K=3) is a principled proxy for this: nearby APs have lower d_ll' and thus higher Gudmundson correlation C_ll' = 2^(-d_ll'/d_decorr). Using K=3 keeps the graph sparse, preventing over-aggregation.

---

## References

The following works from the literature survey are directly relevant to implementation choices:

- Ngo et al. (2017) — cell-free massive MIMO baseline; Gudmundson shadow fading model
- Kim et al. (2021) — AR/Kalman vs ML benchmark; NMSE evaluation protocol
- Zheng et al. (2021) — channel aging analysis in cell-free systems
- Jiang et al. (2022) — Transformer channel predictor; NMSE target reference
- Kim et al. (2025) survey — confirms the open research gap this project addresses
