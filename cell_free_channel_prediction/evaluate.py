"""
evaluate.py — Full evaluation suite

Experiments
-----------
  1. NMSE vs Velocity     — all methods at 30, 60, 120 km/h  [PRIMARY]
  2. NMSE vs Prediction step k=1,2,3  [SECONDARY]
  3. Ablation: CNN-Only vs GNN-CNN    [proves GCN value]
  4. Ablation: Grid vs Random AP layout [proves topology handling]
  5. Efficiency: parameter count + inference latency

Usage
-----
  # Train first (all combinations), then run all experiments:
  python evaluate.py

  # Run a specific experiment only:
  python evaluate.py --exp velocity
  python evaluate.py --exp step
  python evaluate.py --exp ablation_gcn
  python evaluate.py --exp ablation_topo
  python evaluate.py --exp efficiency
"""

import argparse
import os
import json
import time
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader, random_split

from dataset import generate_dataset, compute_norm_adj
from model import (GNNCNNHybrid, CNNOnly, nmse_db,
                   count_parameters, measure_latency)
from baselines import (ARPredictor, KalmanPredictor,
                       flat_to_complex, evaluate_baseline)
from train import make_loaders, load_or_generate, train_model


# ──────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────

VELOCITIES  = [30, 60, 120]
LAYOUTS     = ['random', 'grid']
DATA_DIR    = 'data'
CKPT_DIR    = 'checkpoints'
RESULTS_DIR = 'results'
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL_PARAMS = dict(L=16, N=4, K=4, T_history=10, T_predict=3, hidden_dim=64)
DS_PARAMS    = dict(L=16, K=4, N=4, fc=3.5e9, T_history=10, T_predict=3,
                    num_samples=5000)

# Consistent plot style across all figures
PLOT_STYLES = {
    'AR':       dict(marker='s', linestyle='--', color='#e74c3c'),
    'Kalman':   dict(marker='^', linestyle='--', color='#e67e22'),
    'CNN-Only': dict(marker='o', linestyle='-',  color='#3498db'),
    'GNN-CNN':  dict(marker='D', linestyle='-',  color='#2ecc71', linewidth=2),
}


# ──────────────────────────────────────────────────────────────
# Load helpers
# ──────────────────────────────────────────────────────────────

def load_model(model_type: str, layout: str, velocity: float,
               device: torch.device):
    """Load a trained model checkpoint. Returns None if checkpoint missing."""
    ckpt = os.path.join(CKPT_DIR,
                        f'{model_type}_{layout}_{int(velocity)}kmh.pt')
    if not os.path.exists(ckpt):
        return None
    mp = MODEL_PARAMS.copy()
    cls = GNNCNNHybrid if model_type == 'gnn_cnn' else CNNOnly
    model = cls(**mp).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False))
    model.eval()
    return model


def get_test_loader(layout: str, velocity: float, batch_size: int = 64):
    ds = load_or_generate(layout, velocity, data_dir=DATA_DIR, **DS_PARAMS)
    _, _, test_loader = make_loaders(ds, batch_size=batch_size)
    return test_loader, ds


def eval_nn_model(model, test_loader, adj_norm, device,
                  k_step: int = None) -> float:
    """Return NMSE in dB on the test set (optionally for a single k-step)."""
    model.eval()
    adj_norm = adj_norm.to(device)
    preds, targets = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            p  = model(xb, adj_norm).cpu()
            preds.append(p)
            targets.append(yb)
    preds   = torch.cat(preds)    # (S, L, C, T_predict)
    targets = torch.cat(targets)

    if k_step is not None:
        return nmse_db(preds[..., k_step], targets[..., k_step])
    return nmse_db(preds, targets)


def eval_baselines_on_dataset(layout: str, velocity: float,
                               k_step: int = None) -> dict:
    """Evaluate AR and Kalman on a dataset. Returns dict of NMSE dB values."""
    ds = load_or_generate(layout, velocity, data_dir=DATA_DIR, **DS_PARAMS)
    X, Y = ds['X'], ds['Y']

    # Reproduce the same 10% test split used in train.py
    S = len(X)
    n_train = int(S * 0.8)
    n_val   = int(S * 0.1)
    n_test  = S - n_train - n_val
    gen = torch.Generator().manual_seed(0)
    _, _, test_idx = random_split(
        list(range(S)), [n_train, n_val, n_test], generator=gen)
    idx = list(test_idx)

    N, K       = ds['N'], ds['K']
    T_predict  = ds['T_predict']
    fc, Ts     = ds['fc'], ds['Ts']

    X_cplx = flat_to_complex(X[idx], N, K)   # (S, L, N, K, T_h)
    Y_cplx = flat_to_complex(Y[idx], N, K)   # (S, L, N, K, T_p)

    ar = ARPredictor(order=4)
    kf = KalmanPredictor(fc=fc, velocity_kmh=velocity, Ts=Ts)

    results = {}
    for name, pred_obj in [('AR', ar), ('Kalman', kf)]:
        results[name] = evaluate_baseline(
            pred_obj, X_cplx, Y_cplx, T_predict, k_step=k_step)
    return results


# ──────────────────────────────────────────────────────────────
# Experiment 1: NMSE vs Velocity  [PRIMARY]
# ──────────────────────────────────────────────────────────────

def exp_nmse_vs_velocity(layout: str = 'random') -> dict:
    """Primary result: aggregate NMSE vs velocity for all methods."""
    print(f"\n{'═'*55}")
    print(f" Experiment 1: NMSE vs Velocity  [layout={layout}]")
    print(f"{'═'*55}")

    results = {m: [] for m in ['AR', 'Kalman', 'CNN-Only', 'GNN-CNN']}

    for vel in VELOCITIES:
        print(f"\n  Velocity = {vel} km/h")

        # Baselines
        bl = eval_baselines_on_dataset(layout, vel)
        results['AR'].append(bl['AR'])
        results['Kalman'].append(bl['Kalman'])
        print(f"    AR       : {bl['AR']:+.2f} dB")
        print(f"    Kalman   : {bl['Kalman']:+.2f} dB")

        # Neural models
        test_loader, ds = get_test_loader(layout, vel)
        adj_norm = ds['adj_norm']

        for mtype, mkey in [('cnn_only', 'CNN-Only'), ('gnn_cnn', 'GNN-CNN')]:
            model = load_model(mtype, layout, vel, DEVICE)
            if model is None:
                print(f"    [{mkey}] checkpoint missing — run train.py first")
                results[mkey].append(None)
            else:
                val = eval_nn_model(model, test_loader, adj_norm, DEVICE)
                results[mkey].append(val)
                print(f"    {mkey:10s}: {val:+.2f} dB")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    json_path = os.path.join(RESULTS_DIR, f'nmse_vs_velocity_{layout}.json')
    with open(json_path, 'w') as f:
        json.dump({'velocities': VELOCITIES, 'results': results}, f, indent=2)
    print(f"\n  Results saved → {json_path}")

    _plot_nmse_vs_velocity(VELOCITIES, results, layout)
    return results


def _plot_nmse_vs_velocity(velocities, results, layout):
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, vals in results.items():
        clean = [v if v is not None else np.nan for v in vals]
        ax.plot(velocities, clean, label=name,
                markersize=8, **PLOT_STYLES.get(name, {}))

    ax.set_xlabel('User Velocity (km/h)', fontsize=13)
    ax.set_ylabel('NMSE (dB)', fontsize=13)
    ax.set_title(f'NMSE vs User Velocity  [AP layout: {layout}]', fontsize=13)
    ax.set_xticks(velocities)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()   # lower NMSE = better, so better results appear higher

    path = os.path.join(RESULTS_DIR, f'fig1_nmse_vs_velocity_{layout}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Figure saved  → {path}")


# ──────────────────────────────────────────────────────────────
# Experiment 2: NMSE vs Prediction Step  [SECONDARY]
# ──────────────────────────────────────────────────────────────

def exp_nmse_vs_step(layout: str = 'random', velocity: float = 60.0) -> dict:
    """Show NMSE degradation from k=1 to k=3 for all methods."""
    print(f"\n{'═'*55}")
    print(f" Experiment 2: NMSE vs Prediction Step")
    print(f" layout={layout}  |  velocity={int(velocity)} km/h")
    print(f"{'═'*55}")

    T_predict = MODEL_PARAMS['T_predict']
    results   = {m: [] for m in ['AR', 'Kalman', 'CNN-Only', 'GNN-CNN']}

    test_loader, ds = get_test_loader(layout, velocity)
    adj_norm = ds['adj_norm']

    for k in range(T_predict):
        print(f"\n  k = {k+1}")
        bl = eval_baselines_on_dataset(layout, velocity, k_step=k)
        results['AR'].append(bl['AR'])
        results['Kalman'].append(bl['Kalman'])
        print(f"    AR       : {bl['AR']:+.2f} dB")
        print(f"    Kalman   : {bl['Kalman']:+.2f} dB")

        for mtype, mkey in [('cnn_only', 'CNN-Only'), ('gnn_cnn', 'GNN-CNN')]:
            model = load_model(mtype, layout, velocity, DEVICE)
            if model is None:
                results[mkey].append(None)
                print(f"    [{mkey}] checkpoint missing")
            else:
                val = eval_nn_model(model, test_loader, adj_norm, DEVICE, k_step=k)
                results[mkey].append(val)
                print(f"    {mkey:10s}: {val:+.2f} dB")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    json_path = os.path.join(
        RESULTS_DIR, f'nmse_vs_step_{layout}_{int(velocity)}.json')
    with open(json_path, 'w') as f:
        json.dump({'steps': list(range(1, T_predict + 1)),
                   'results': results}, f, indent=2)
    print(f"\n  Results saved → {json_path}")

    _plot_nmse_vs_step(T_predict, results, layout, velocity)
    return results


def _plot_nmse_vs_step(T_predict, results, layout, velocity):
    fig, ax = plt.subplots(figsize=(7, 5))
    x = list(range(1, T_predict + 1))
    for name, vals in results.items():
        clean = [v if v is not None else np.nan for v in vals]
        ax.plot(x, clean, label=name,
                markersize=8, **PLOT_STYLES.get(name, {}))

    ax.set_xlabel('Prediction Step k', fontsize=13)
    ax.set_ylabel('NMSE (dB)', fontsize=13)
    ax.set_title(f'NMSE vs Prediction Step  '
                 f'[{layout}, {int(velocity)} km/h]', fontsize=13)
    ax.set_xticks(x)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()

    path = os.path.join(RESULTS_DIR,
                        f'fig2_nmse_vs_step_{layout}_{int(velocity)}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Figure saved  → {path}")


# ──────────────────────────────────────────────────────────────
# Experiment 3: Ablation — CNN-Only vs GNN-CNN
# ──────────────────────────────────────────────────────────────

def exp_ablation_gcn(layout: str = 'random') -> dict:
    """Compare CNN-Only vs GNN-CNN to isolate the GCN contribution."""
    print(f"\n{'═'*55}")
    print(f" Experiment 3: Ablation — GCN Contribution  [layout={layout}]")
    print(f"{'═'*55}")

    cnn_vals, gnn_vals = [], []

    for vel in VELOCITIES:
        print(f"\n  Velocity = {vel} km/h")
        test_loader, ds = get_test_loader(layout, vel)
        adj_norm = ds['adj_norm']

        cnn_model = load_model('cnn_only', layout, vel, DEVICE)
        gnn_model = load_model('gnn_cnn',  layout, vel, DEVICE)

        cnn_v = (eval_nn_model(cnn_model, test_loader, adj_norm, DEVICE)
                 if cnn_model else None)
        gnn_v = (eval_nn_model(gnn_model, test_loader, adj_norm, DEVICE)
                 if gnn_model else None)

        cnn_vals.append(cnn_v)
        gnn_vals.append(gnn_v)

        if cnn_v is not None and gnn_v is not None:
            # Negative gain = GNN-CNN is lower (better) NMSE
            gain = gnn_v - cnn_v
            print(f"    CNN-Only : {cnn_v:+.2f} dB")
            print(f"    GNN-CNN  : {gnn_v:+.2f} dB  (GCN gain: {gain:+.2f} dB)")
        else:
            print("    One or both checkpoints missing.")

    results = {'CNN-Only': cnn_vals, 'GNN-CNN': gnn_vals}

    os.makedirs(RESULTS_DIR, exist_ok=True)
    json_path = os.path.join(RESULTS_DIR, f'ablation_gcn_{layout}.json')
    with open(json_path, 'w') as f:
        json.dump({'velocities': VELOCITIES, 'results': results}, f, indent=2)
    print(f"\n  Results saved → {json_path}")

    _plot_ablation_gcn(cnn_vals, gnn_vals, layout)
    return results


def _plot_ablation_gcn(cnn_vals, gnn_vals, layout):
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(VELOCITIES))
    w = 0.35

    safe = lambda v: v if v is not None else 0.0
    ax.bar(x - w / 2, [safe(v) for v in cnn_vals],
           w, label='CNN-Only', color='#3498db', alpha=0.85)
    ax.bar(x + w / 2, [safe(v) for v in gnn_vals],
           w, label='GNN-CNN',  color='#2ecc71', alpha=0.85)

    # Annotate bars with dB values
    for i, (c, g) in enumerate(zip(cnn_vals, gnn_vals)):
        if c is not None:
            ax.text(x[i] - w / 2, c - 0.3, f'{c:.1f}',
                    ha='center', va='top', fontsize=9)
        if g is not None:
            ax.text(x[i] + w / 2, g - 0.3, f'{g:.1f}',
                    ha='center', va='top', fontsize=9)

    ax.set_xlabel('User Velocity (km/h)', fontsize=13)
    ax.set_ylabel('NMSE (dB)', fontsize=13)
    ax.set_title(f'Ablation: GCN Contribution  [layout={layout}]', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{v} km/h' for v in VELOCITIES])
    ax.legend(fontsize=11)
    ax.grid(True, axis='y', alpha=0.3)
    ax.invert_yaxis()

    path = os.path.join(RESULTS_DIR, f'fig3_ablation_gcn_{layout}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Figure saved  → {path}")


# ──────────────────────────────────────────────────────────────
# Experiment 4: Ablation — Grid vs Random AP Layout
# ──────────────────────────────────────────────────────────────

def exp_ablation_topology(velocity: float = 60.0) -> dict:
    """Compare GNN-CNN on grid vs random AP layout to prove topology handling."""
    print(f"\n{'═'*55}")
    print(f" Experiment 4: Topology Ablation  [v={int(velocity)} km/h]")
    print(f"{'═'*55}")

    results = {}
    for lay in LAYOUTS:
        print(f"\n  Layout = {lay}")
        test_loader, ds = get_test_loader(lay, velocity)
        adj_norm = ds['adj_norm']
        model = load_model('gnn_cnn', lay, velocity, DEVICE)
        if model is None:
            results[lay] = None
            print(f"    Checkpoint missing — run: python train.py "
                  f"--layout {lay} --velocity {int(velocity)}")
        else:
            val = eval_nn_model(model, test_loader, adj_norm, DEVICE)
            results[lay] = val
            print(f"    GNN-CNN NMSE : {val:+.2f} dB")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    json_path = os.path.join(RESULTS_DIR, f'ablation_topology_{int(velocity)}.json')
    with open(json_path, 'w') as f:
        json.dump({'layouts': LAYOUTS, 'velocity': velocity,
                   'results': results}, f, indent=2)
    print(f"\n  Results saved → {json_path}")

    _plot_ablation_topology(results, velocity)
    return results


def _plot_ablation_topology(results, velocity):
    fig, ax = plt.subplots(figsize=(5, 5))
    vals   = [results.get(l) or 0.0 for l in LAYOUTS]
    colors = ['#9b59b6', '#f39c12']
    bars   = ax.bar(LAYOUTS, vals, color=colors, alpha=0.85, width=0.4)

    # Annotate bars
    for bar, v in zip(bars, vals):
        if v != 0.0:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    v - 0.2,
                    f'{v:.2f} dB',
                    ha='center', va='top', fontsize=11, fontweight='bold')

    ax.set_ylabel('NMSE (dB)', fontsize=13)
    ax.set_title(f'Topology Ablation  [GNN-CNN, {int(velocity)} km/h]', fontsize=13)
    ax.grid(True, axis='y', alpha=0.3)
    ax.invert_yaxis()
    ax.set_xticklabels(['Random AP Layout', 'Grid AP Layout'], fontsize=11)

    path = os.path.join(RESULTS_DIR,
                        f'fig4_ablation_topology_{int(velocity)}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Figure saved  → {path}")


# ──────────────────────────────────────────────────────────────
# Experiment 5: Efficiency Metrics
# ──────────────────────────────────────────────────────────────

def exp_efficiency(layout: str = 'random', velocity: float = 60.0) -> dict:
    """
    Report parameter count and inference latency for all neural models.
    Baselines use a timing loop; their 'params' reflect the AR order / KF state size.
    """
    print(f"\n{'═'*55}")
    print(f" Experiment 5: Efficiency Metrics")
    print(f" layout={layout}  |  velocity={int(velocity)} km/h")
    print(f"{'═'*55}")

    mp = MODEL_PARAMS.copy()
    ds = load_or_generate(layout, velocity, data_dir=DATA_DIR, **DS_PARAMS)
    adj_norm = ds['adj_norm'].to(DEVICE)

    # Dummy batch for timing (batch size = 1 → realistic edge-deployment scenario)
    x_dummy = torch.randn(1, mp['L'], 2 * mp['N'] * mp['K'],
                          mp['T_history']).to(DEVICE)

    results = {}

    # ── Neural models ──
    for mtype, label in [('gnn_cnn', 'GNN-CNN'), ('cnn_only', 'CNN-Only')]:
        model = load_model(mtype, layout, velocity, DEVICE)
        if model is None:
            # Still count parameters from a fresh (untrained) instance
            cls = GNNCNNHybrid if mtype == 'gnn_cnn' else CNNOnly
            model = cls(**mp).to(DEVICE)
            trained = False
        else:
            trained = True

        n_params = count_parameters(model)
        mean_ms, std_ms = measure_latency(model, x_dummy, adj_norm, n_runs=200)

        results[label] = {
            'params': n_params,
            'latency_mean_ms': round(mean_ms, 3),
            'latency_std_ms':  round(std_ms, 3),
            'trained': trained,
        }
        print(f"\n  {label}")
        print(f"    Parameters   : {n_params:,}")
        print(f"    Latency      : {mean_ms:.3f} ± {std_ms:.3f} ms"
              f"{'  (untrained model)' if not trained else ''}")

    # ── Baselines — latency via direct timing ──
    N, K, T_h, T_p = mp['N'], mp['K'], mp['T_history'], mp['T_predict']
    L = mp['L']
    fc, Ts = ds['fc'], ds['Ts']

    x_np = np.random.randn(L, N, K, T_h) + 1j * np.random.randn(L, N, K, T_h)

    for bl_cls, label, extra_kw in [
        (ARPredictor,     'AR',     dict(order=4)),
        (KalmanPredictor, 'Kalman', dict(fc=fc, velocity_kmh=velocity, Ts=Ts)),
    ]:
        predictor = bl_cls(**extra_kw)
        # Warm-up
        for _ in range(3):
            predictor.predict(x_np, T_p)

        times = []
        for _ in range(50):   # fewer runs — baselines are much slower
            t0 = time.perf_counter()
            predictor.predict(x_np, T_p)
            times.append((time.perf_counter() - t0) * 1000)

        mean_ms = float(np.mean(times))
        std_ms  = float(np.std(times))

        # AR(4): 4 coefficients per scalar channel coefficient → L*N*K*4
        # Kalman: state size = 1 per scalar channel coefficient
        param_equiv = (L * N * K * 4) if label == 'AR' else (L * N * K)

        results[label] = {
            'params': param_equiv,
            'latency_mean_ms': round(mean_ms, 3),
            'latency_std_ms':  round(std_ms, 3),
            'note': 'AR coefficients (not trainable)'
                    if label == 'AR' else 'KF state variables (not trainable)',
        }
        print(f"\n  {label}")
        print(f"    Param-equiv  : {param_equiv:,}  ({results[label]['note']})")
        print(f"    Latency      : {mean_ms:.3f} ± {std_ms:.3f} ms")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    json_path = os.path.join(RESULTS_DIR, 'efficiency.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved → {json_path}")

    _print_efficiency_table(results)
    return results


def _print_efficiency_table(results: dict):
    """Print a formatted efficiency summary table to stdout."""
    print(f"\n{'─'*62}")
    print(f"  {'Method':<12} {'Parameters':>12}  {'Latency (ms)':>16}  Edge?")
    print(f"{'─'*62}")
    edge_threshold_ms = 5.0   # sub-5ms = deployable at the AP
    for method, info in results.items():
        p      = info['params']
        lat    = info['latency_mean_ms']
        std    = info['latency_std_ms']
        edge   = '✓' if lat < edge_threshold_ms else '✗'
        p_str  = f"{p:,}" if p < 1_000_000 else f"{p/1e6:.2f}M"
        print(f"  {method:<12} {p_str:>12}  {lat:>8.3f} ± {std:<5.3f}   {edge}")
    print(f"{'─'*62}")
    print(f"  Edge threshold: < {edge_threshold_ms} ms inference latency")


# ──────────────────────────────────────────────────────────────
# Summary table (fills in Table 4 from literature survey)
# ──────────────────────────────────────────────────────────────

def print_summary_table(layout: str = 'random'):
    """
    Print a summary NMSE table across all methods and velocities —
    corresponds to the filled-in version of Table 4 in the survey.
    """
    print(f"\n{'═'*65}")
    print(f" NMSE Summary Table  [layout={layout}]  (dB, lower is better)")
    print(f"{'═'*65}")
    header = f"  {'Method':<12}" + "".join(f"  {v:>6} km/h" for v in VELOCITIES)
    print(header)
    print(f"{'─'*65}")

    # Load pre-computed velocity results if available
    json_path = os.path.join(RESULTS_DIR, f'nmse_vs_velocity_{layout}.json')
    if not os.path.exists(json_path):
        print("  Run exp_nmse_vs_velocity first to generate this table.")
        return

    with open(json_path) as f:
        data = json.load(f)

    for method, vals in data['results'].items():
        row = f"  {method:<12}"
        for v in vals:
            row += f"  {v:>10.2f}" if v is not None else f"  {'N/A':>10}"
        print(row)

    print(f"{'─'*65}")


# ──────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────

EXP_MAP = {
    'velocity':     lambda: (
        exp_nmse_vs_velocity('random'),
        exp_nmse_vs_velocity('grid'),
        print_summary_table('random'),
        print_summary_table('grid'),
    ),
    'step':         lambda: (
        exp_nmse_vs_step('random', 60.0),
        exp_nmse_vs_step('random', 120.0),
    ),
    'ablation_gcn': lambda: (
        exp_ablation_gcn('random'),
        exp_ablation_gcn('grid'),
    ),
    'ablation_topo': lambda: (
        exp_ablation_topology(60.0),
        exp_ablation_topology(120.0),
    ),
    'efficiency':   lambda: exp_efficiency('random', 60.0),
}


def run_all():
    """Run every experiment in sequence."""
    exp_nmse_vs_velocity('random')
    exp_nmse_vs_velocity('grid')
    print_summary_table('random')
    print_summary_table('grid')

    exp_nmse_vs_step('random', 60.0)
    exp_nmse_vs_step('random', 120.0)

    exp_ablation_gcn('random')
    exp_ablation_gcn('grid')

    exp_ablation_topology(60.0)
    exp_ablation_topology(120.0)

    exp_efficiency('random', 60.0)

    print(f"\n{'═'*55}")
    print(f" All experiments complete. Results in → ./{RESULTS_DIR}/")
    print(f"{'═'*55}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate GNN-CNN channel predictor')
    parser.add_argument(
        '--exp',
        default='all',
        choices=['all', 'velocity', 'step', 'ablation_gcn',
                 'ablation_topo', 'efficiency'],
        help='Which experiment to run (default: all)')
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"Device : {DEVICE}")
    print(f"Results: ./{RESULTS_DIR}/")

    if args.exp == 'all':
        run_all()
    else:
        EXP_MAP[args.exp]()


if __name__ == '__main__':
    main()
