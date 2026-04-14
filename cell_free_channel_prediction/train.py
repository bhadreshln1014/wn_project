"""
train.py — Training loop for GNNCNNHybrid and CNNOnly

Setup
-----
  • 80 / 10 / 10 train / val / test split
  • Loss: NMSE (linear)
  • Optimizer: Adam, lr=1e-3
  • Scheduler: ReduceLROnPlateau, patience=5
  • Early stopping: patience=10
  • Batch size: 32, max epochs: 100

The same training function handles both GNNCNNHybrid and CNNOnly,
allowing a fair comparison for the ablation study.

Usage
-----
  python train.py --layout random --velocity 60
  python train.py --layout random --velocity 60 --model cnn_only
"""

import argparse
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split

from dataset import generate_dataset, compute_norm_adj
from model import GNNCNNHybrid, CNNOnly, nmse_loss, nmse_db, count_parameters


# ──────────────────────────────────────────────────────────────
# Dataset helper
# ──────────────────────────────────────────────────────────────

def load_or_generate(layout: str, velocity: float,
                     data_dir: str = 'data', **kwargs) -> dict:
    path = os.path.join(data_dir, f'dataset_{layout}_{int(velocity)}kmh.pt')
    if os.path.exists(path):
        print(f"Loading dataset from {path}")
        return torch.load(path, weights_only=False)
    print(f"Dataset not found at {path}, generating...")
    os.makedirs(data_dir, exist_ok=True)
    return generate_dataset(layout=layout, velocity_kmh=velocity,
                             save_path=path, **kwargs)


def make_loaders(dataset: dict, batch_size: int = 32,
                 train_frac: float = 0.8, val_frac: float = 0.1,
                 seed: int = 0):
    X, Y = dataset['X'], dataset['Y']
    full = TensorDataset(X, Y)
    S = len(full)
    n_train = int(S * train_frac)
    n_val = int(S * val_frac)
    n_test = S - n_train - n_val

    gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(
        full, [n_train, n_val, n_test], generator=gen)

    loader_kw = dict(batch_size=batch_size, num_workers=0, pin_memory=False)
    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kw)
    test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kw)

    return train_loader, val_loader, test_loader


# ──────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    adj_norm: torch.Tensor,
    device: torch.device,
    save_path: str,
    lr: float = 1e-3,
    max_epochs: int = 100,
    patience_sched: int = 5,
    patience_stop: int = 10,
    verbose: bool = True,
) -> dict:
    """
    Train a model (GNNCNNHybrid or CNNOnly) and return training history.

    Returns dict with keys: train_loss, val_loss, best_val_nmse_db
    """
    model = model.to(device)
    adj_norm = adj_norm.to(device)

    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode='min', factor=0.5, patience=patience_sched)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_loss': [], 'val_nmse_db_per_step': []}

    for epoch in range(1, max_epochs + 1):
        t0 = time.time()

        # ── Train ──
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimiser.zero_grad()
            pred = model(xb, adj_norm)
            loss = nmse_loss(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimiser.step()
            train_losses.append(loss.item())

        # ── Validate ──
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb, adj_norm)
                val_losses.append(nmse_loss(pred, yb).item())

            # Per-step NMSE in dB (k=1,2,3)
            all_preds, all_targets = [], []
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                all_preds.append(model(xb, adj_norm))
                all_targets.append(yb)
            all_preds   = torch.cat(all_preds)
            all_targets = torch.cat(all_targets)

            T_predict = all_preds.shape[-1]
            per_step_nmse = [
                nmse_db(all_preds[..., k], all_targets[..., k])
                for k in range(T_predict)
            ]

        mean_train = float(torch.tensor(train_losses).mean())
        mean_val   = float(torch.tensor(val_losses).mean())
        history['train_loss'].append(mean_train)
        history['val_loss'].append(mean_val)
        history['val_nmse_db_per_step'].append(per_step_nmse)

        scheduler.step(mean_val)

        if mean_val < best_val_loss:
            best_val_loss = mean_val
            epochs_no_improve = 0
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            torch.save(model.state_dict(), save_path)
        else:
            epochs_no_improve += 1

        if verbose:
            step_str = '  '.join(
                [f"k{k+1}={ps:.2f}dB" for k, ps in enumerate(per_step_nmse)])
            print(f"Ep {epoch:3d}/{max_epochs} | "
                  f"train={mean_train:.4f}  val={mean_val:.4f} | "
                  f"{step_str} | "
                  f"lr={optimiser.param_groups[0]['lr']:.5f} | "
                  f"{time.time()-t0:.1f}s")

        if epochs_no_improve >= patience_stop:
            if verbose:
                print(f"Early stopping at epoch {epoch} "
                      f"(no val improvement for {patience_stop} epochs)")
            break

    history['best_val_nmse_linear'] = best_val_loss
    return history


# ──────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Train GNN-CNN or CNN-Only channel predictor')
    parser.add_argument('--layout',   default='random',
                        choices=['random', 'grid'])
    parser.add_argument('--velocity', type=float, default=60.0,
                        help='User velocity in km/h (30 | 60 | 120)')
    parser.add_argument('--model',    default='gnn_cnn',
                        choices=['gnn_cnn', 'cnn_only'])
    parser.add_argument('--epochs',   type=int, default=100)
    parser.add_argument('--batch',    type=int, default=32)
    parser.add_argument('--lr',       type=float, default=1e-3)
    parser.add_argument('--data_dir', default='data')
    parser.add_argument('--ckpt_dir', default='checkpoints')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load / generate dataset
    ds = load_or_generate(args.layout, args.velocity,
                           data_dir=args.data_dir,
                           L=16, K=4, N=4, fc=3.5e9,
                           T_history=10, T_predict=3,
                           num_samples=5000)

    L, K, N = ds['L'], ds['K'], ds['N']
    T_history, T_predict = ds['T_history'], ds['T_predict']
    adj_norm = ds['adj_norm']

    train_loader, val_loader, _ = make_loaders(ds, batch_size=args.batch)

    # Build model
    model_kwargs = dict(L=L, N=N, K=K, T_history=T_history,
                        T_predict=T_predict, hidden_dim=64)
    if args.model == 'gnn_cnn':
        model = GNNCNNHybrid(**model_kwargs)
    else:
        model = CNNOnly(**model_kwargs)

    print(f"\nModel : {args.model.upper()}")
    print(f"Params: {count_parameters(model):,}")
    print(f"Layout: {args.layout}  |  Velocity: {args.velocity} km/h\n")

    ckpt = os.path.join(args.ckpt_dir,
                         f'{args.model}_{args.layout}_{int(args.velocity)}kmh.pt')

    history = train_model(
        model, train_loader, val_loader, adj_norm,
        device=device, save_path=ckpt,
        lr=args.lr, max_epochs=args.epochs,
    )

    print(f"\nBest val NMSE (linear): {history['best_val_nmse_linear']:.5f}")
    print(f"Checkpoint saved: {ckpt}")


if __name__ == '__main__':
    main()
