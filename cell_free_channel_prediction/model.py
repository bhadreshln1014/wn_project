"""
model.py — GNN-CNN Hybrid and CNN-Only models

Architecture (GNNCNNHybrid):
  Stage 1 — LocalCNN:  Conv1D × 3 per AP (shared weights) → h_loc ∈ R^64
  Stage 2 — GCNLayer:  single normalised GCN → h_agg ∈ R^64
  Stage 3 — PredHead:  Linear(64→128→out) per AP → (2*N*K, T_predict)

Design rationale:
  • Shared CNN weights: parameter-efficient and topology-agnostic
  • Single GCN layer: prevents over-smoothing, keeps model lightweight
  • Manual normalised-adjacency GCN: handles batched graphs cleanly
    without requiring PyG DataLoader gymnastics
"""

import numpy as np
import torch
import torch.nn as nn
import time


# ──────────────────────────────────────────────────────────────
# Building blocks
# ──────────────────────────────────────────────────────────────

class LocalCNN(nn.Module):
    """
    Per-AP temporal feature extractor (shared weights across all APs).
    Input  : (B*L, in_channels, T_history)
    Output : (B*L, hidden_dim)
    """

    def __init__(self, in_channels: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),   # global average pooling over time
        )
        self.proj = nn.Sequential(
            nn.Linear(64, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B*L, C, T)
        h = self.net(x).squeeze(-1)   # (B*L, 64)
        return self.proj(h)            # (B*L, hidden_dim)


class GCNLayer(nn.Module):
    """
    Single GCN message-passing layer operating on batched node features.

    Update rule:  h_agg = ReLU( Â h_loc W^T + b )
    where Â = D^{-1/2}(A+I)D^{-1/2}  (precomputed, passed at runtime).

    Input  : h  (B, L, in_features),  adj_norm (L, L)
    Output : (B, L, out_features)
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=True)
        self.relu = nn.ReLU()

    def forward(self, h: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        # h       : (B, L, in_features)
        # adj_norm: (L, L)
        support = self.linear(h)                           # (B, L, out_features)
        out = torch.einsum('lm,bmd->bld', adj_norm, support)  # graph aggregation
        return self.relu(out)


class PredictionHead(nn.Module):
    """
    Per-AP prediction MLP.
    Input  : (B*L, hidden_dim)
    Output : (B*L, out_channels * T_predict)
    """

    def __init__(self, hidden_dim: int, out_channels: int, T_predict: int):
        super().__init__()
        self.T_predict = T_predict
        self.out_channels = out_channels
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, out_channels * T_predict),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)   # (B*L, out_channels * T_predict)


# ──────────────────────────────────────────────────────────────
# Full GNN-CNN Hybrid
# ──────────────────────────────────────────────────────────────

class GNNCNNHybrid(nn.Module):
    """
    Three-stage GNN-CNN hybrid for multi-step channel prediction
    in cell-free massive MIMO.

    Parameters
    ----------
    L         : number of APs
    N         : antennas per AP
    K         : number of UEs
    T_history : input time steps
    T_predict : output (future) time steps
    hidden_dim: internal feature size (default 64, keeps params < 100K)
    """

    def __init__(self, L: int, N: int, K: int,
                 T_history: int, T_predict: int,
                 hidden_dim: int = 64):
        super().__init__()
        self.L = L
        self.T_predict = T_predict
        self.out_channels = 2 * N * K  # real + imag

        # Stage 1
        self.local_cnn = LocalCNN(in_channels=self.out_channels,
                                   hidden_dim=hidden_dim)
        # Stage 2
        self.gcn = GCNLayer(hidden_dim, hidden_dim)

        # Stage 3
        self.pred_head = PredictionHead(hidden_dim, self.out_channels, T_predict)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        """
        x        : (B, L, C, T_history)
        adj_norm : (L, L)  — precomputed normalised adjacency
        returns  : (B, L, C, T_predict)
        """
        B, L, C, T = x.shape
        
        # 1. Normalize per AP
        scale = torch.sqrt((x**2).mean(dim=(-1, -2), keepdim=True) + 1e-8)
        x_norm = x / scale

        # Stage 1 — shared CNN
        x_flat = x_norm.reshape(B * L, C, T)
        h_loc = self.local_cnn(x_flat)           # (B*L, hidden_dim)
        h_loc = h_loc.reshape(B, L, -1)          # (B, L, hidden_dim)

        # Stage 2 — GCN aggregation
        h_agg = self.gcn(h_loc, adj_norm)         # (B, L, hidden_dim)

        # Stage 3 — prediction
        h_flat = h_agg.reshape(B * L, -1)
        pred_flat = self.pred_head(h_flat)        # (B*L, C*T_predict)
        pred_norm = pred_flat.reshape(B, L, self.out_channels, self.T_predict)
        
        # 2. Denormalize
        pred = pred_norm * scale
        return pred


# ──────────────────────────────────────────────────────────────
# CNN-Only ablation (identical to GNN-CNN but GCN stage removed)
# ──────────────────────────────────────────────────────────────

class CNNOnly(nn.Module):
    """
    Ablation model: CNN per AP with NO graph aggregation.
    Used to isolate the contribution of the GCN stage.
    Architecture is otherwise identical to GNNCNNHybrid.
    """

    def __init__(self, L: int, N: int, K: int,
                 T_history: int, T_predict: int,
                 hidden_dim: int = 64):
        super().__init__()
        self.L = L
        self.T_predict = T_predict
        self.out_channels = 2 * N * K

        self.local_cnn = LocalCNN(in_channels=self.out_channels,
                                   hidden_dim=hidden_dim)
        self.pred_head = PredictionHead(hidden_dim, self.out_channels, T_predict)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor = None) -> torch.Tensor:
        """adj_norm is accepted but ignored (API compatibility)."""
        B, L, C, T = x.shape
        
        # 1. Normalize per AP
        scale = torch.sqrt((x**2).mean(dim=(-1, -2), keepdim=True) + 1e-8)
        x_norm = x / scale
        
        x_flat = x_norm.reshape(B * L, C, T)
        h_loc = self.local_cnn(x_flat)              # (B*L, hidden_dim)
        pred_flat = self.pred_head(h_loc)            # (B*L, C*T_predict)
        pred_norm = pred_flat.reshape(B, L, self.out_channels, self.T_predict)
        
        # 2. Denormalize
        pred = pred_norm * scale
        return pred


# ──────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────

def nmse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    NMSE loss (linear, not dB) — used during training.
    Averaged over batch; summed over L, C, T.
    """
    num = ((pred - target) ** 2).sum(dim=(1, 2, 3))
    den = (target ** 2).sum(dim=(1, 2, 3)) + 1e-8
    return (num / den).mean()


def nmse_db(pred: torch.Tensor, target: torch.Tensor) -> float:
    """NMSE in dB — used during evaluation."""
    with torch.no_grad():
        num = ((pred - target) ** 2).sum()
        den = (target ** 2).sum() + 1e-8
        return 10.0 * torch.log10(num / den).item()


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def measure_latency(model: nn.Module, x: torch.Tensor,
                    adj_norm: torch.Tensor,
                    n_runs: int = 100) -> tuple:
    """Measure mean inference latency in ms over n_runs forward passes."""
    model.eval()
    device = next(model.parameters()).device
    x = x.to(device)
    adj_norm = adj_norm.to(device)

    # Warm-up
    with torch.no_grad():
        for _ in range(10):
            _ = model(x, adj_norm)

    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = model(x, adj_norm)
            times.append((time.perf_counter() - t0) * 1000)

    return float(np.mean(times)), float(np.std(times))


# ──────────────────────────────────────────────────────────────
# Quick sanity check
# ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    from dataset import generate_ap_positions, build_knn_graph, compute_norm_adj

    L, N, K, T_h, T_p = 16, 4, 4, 10, 3

    ap_pos = generate_ap_positions(L)
    _, adj_np = build_knn_graph(ap_pos)
    adj_norm = compute_norm_adj(adj_np)

    B = 8
    x = torch.randn(B, L, 2 * N * K, T_h)

    for ModelClass, name in [(GNNCNNHybrid, 'GNNCNNHybrid'), (CNNOnly, 'CNNOnly')]:
        model = ModelClass(L, N, K, T_h, T_p)
        pred = model(x, adj_norm)
        n = count_parameters(model)
        print(f"{name:20s} | params={n:,} | output={tuple(pred.shape)}")
        assert pred.shape == (B, L, 2 * N * K, T_p), "Shape mismatch!"

    print("All shape checks passed.")
