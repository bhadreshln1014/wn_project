"""
dataset.py — Channel data generation for GNN-CNN Cell-Free Massive MIMO

Channel model:
  - Large-scale fading: 3GPP UMa path loss + spatially correlated shadow fading
  - Small-scale fading: Jake's isotropic scattering (AR(1) process)
  - Spatial AP correlation via Gudmundson model: C_ll' = 2^(-d_ll' / d_decorr)

This spatial correlation is the key addition over plain Jake's model —
it gives the GCN real inter-AP information to aggregate.
"""

import numpy as np
import torch
import os
from scipy.special import j0  # zeroth-order Bessel function


# ──────────────────────────────────────────────────────────────
# AP / UE geometry
# ──────────────────────────────────────────────────────────────

def generate_ap_positions(L: int, area_size: float = 500.0,
                          layout: str = 'random', seed: int = 42) -> np.ndarray:
    """Return AP positions (L, 2) in metres."""
    rng = np.random.RandomState(seed)
    if layout == 'random':
        return rng.uniform(0, area_size, (L, 2))
    elif layout == 'grid':
        g = int(np.sqrt(L))
        assert g * g == L, "L must be a perfect square for grid layout"
        xs = np.linspace(area_size / (2 * g), area_size - area_size / (2 * g), g)
        ys = np.linspace(area_size / (2 * g), area_size - area_size / (2 * g), g)
        xx, yy = np.meshgrid(xs, ys)
        return np.column_stack([xx.flatten(), yy.flatten()])
    else:
        raise ValueError(f"Unknown layout: {layout}")


def generate_ue_positions(K: int, area_size: float = 500.0,
                          seed: int = 0) -> np.ndarray:
    """Return UE positions (K, 2) in metres, kept away from edges."""
    rng = np.random.RandomState(seed)
    margin = 0.1 * area_size
    return rng.uniform(margin, area_size - margin, (K, 2))


# ──────────────────────────────────────────────────────────────
# Large-scale fading with spatial correlation
# ──────────────────────────────────────────────────────────────

def _path_loss_uma(d_m: np.ndarray, fc_ghz: float = 3.5) -> np.ndarray:
    """3GPP UMa path loss (linear scale). d_m in metres."""
    d_m = np.maximum(d_m, 10.0)          # clip min distance
    PL_dB = 128.1 + 37.6 * np.log10(d_m / 1000.0)
    return 10.0 ** (-PL_dB / 10.0)


def _shadow_corr_matrix(ap_pos: np.ndarray, d_decorr: float = 9.0) -> np.ndarray:
    """
    Gudmundson inter-AP shadow fading correlation matrix (L, L).
    C_ll' = 2^(-||p_l - p_l'|| / d_decorr)
    Standard model used in cell-free MIMO literature (Ngo et al. 2017).
    """
    L = len(ap_pos)
    diff = ap_pos[:, None, :] - ap_pos[None, :, :]   # (L, L, 2)
    dist = np.linalg.norm(diff, axis=-1)              # (L, L)
    C = np.power(2.0, -dist / d_decorr)
    C += 1e-6 * np.eye(L)                            # numerical stability
    return C


def generate_large_scale_fading(ap_pos: np.ndarray, ue_pos: np.ndarray,
                                 fc_ghz: float = 3.5,
                                 shadow_std_db: float = 8.0,
                                 seed: int = 0) -> np.ndarray:
    """
    Generate beta_lk = PL_lk * 10^(z_lk / 10)  shape (L, K).

    Shadow fading z_lk is spatially correlated across APs l for each UE k,
    using the Gudmundson model.  This is what gives the GCN meaningful
    cross-AP information to aggregate.
    """
    L, K = len(ap_pos), len(ue_pos)
    rng = np.random.RandomState(seed)

    # Path loss
    beta = np.zeros((L, K))
    for k in range(K):
        d = np.linalg.norm(ap_pos - ue_pos[k], axis=1)  # (L,)
        beta[:, k] = _path_loss_uma(d, fc_ghz)

    # Spatially correlated shadow fading via Cholesky
    C = _shadow_corr_matrix(ap_pos)
    try:
        L_chol = np.linalg.cholesky(C)
    except np.linalg.LinAlgError:
        C += 1e-4 * np.eye(L)
        L_chol = np.linalg.cholesky(C)

    for k in range(K):
        z_raw = rng.randn(L)
        z_db = shadow_std_db * (L_chol @ z_raw)   # correlated in dB
        beta[:, k] *= 10.0 ** (z_db / 10.0)

    return beta   # (L, K)


# ──────────────────────────────────────────────────────────────
# Temporal channel (Jake's AR(1) model)
# ──────────────────────────────────────────────────────────────

def generate_temporal_channel(beta: np.ndarray, ap_pos: np.ndarray, ue_pos: np.ndarray,
                               N: int, T_total: int, fc: float, velocity_kmh: float,
                               Ts: float = 1e-3, num_clusters: int = 20, seed: int = 0) -> np.ndarray:
    """
    Geometry-Based Stochastic Model (GBSM) approximating 3GPP/WINNER II channel models.
    Generates realistic spatial correlation across both distributed APs and local AP antennas.
    """
    rng = np.random.RandomState(seed)
    L, K = beta.shape
    c_speed = 3e8
    lam = c_speed / fc
    
    # 1. 2D Environment setup - Place scattering clusters randomly
    area_min = min(ap_pos.min(), ue_pos.min()) - 100
    area_max = max(ap_pos.max(), ue_pos.max()) + 100
    cluster_pos = rng.uniform(area_min, area_max, (num_clusters, 2))
    
    # 2. UE velocities (random directions for each user)
    v_ms = velocity_kmh / 3.6
    angles = rng.uniform(0, 2 * np.pi, K)
    v_vec = np.stack([np.cos(angles), np.sin(angles)], axis=1) * v_ms  # (K, 2)
    
    # 3. Base complex scattering amplitude for each cluster -> UE link
    A = (rng.randn(num_clusters, K) + 1j * rng.randn(num_clusters, K)) / np.sqrt(2)
    
    H = np.zeros((T_total, L, N, K), dtype=complex)
    
    # Distance: Cluster to APs (Fixed over time)
    diff_c_ap = cluster_pos[:, None, :] - ap_pos[None, :, :]
    d_c_ap = np.linalg.norm(diff_c_ap, axis=-1)  # (C, L)
    
    # Angle of Arrival (AoA) at the AP from the cluster.
    aoa_c_ap = np.arctan2(diff_c_ap[:, :, 1], diff_c_ap[:, :, 0]) # (C, L)
    
    # ULA array steering vectors for N antennas (spacing = lambda/2)
    antenna_spacing = lam / 2.0
    n_idx = np.arange(N)
    
    for t in range(T_total):
        # Move UEs geometrically
        ue_pos_t = ue_pos + v_vec * (t * Ts)  # (K, 2)
        
        # Distance: UE to Cluster (Time-varying) -> generates Doppler physically!
        diff_ue_c = ue_pos_t[None, :, :] - cluster_pos[:, None, :]  # (C, K, 2)
        d_ue_c = np.linalg.norm(diff_ue_c, axis=-1)  # (C, K)
        
        # Total propagation distance: D_clk(t) = d(UE, Cluster) + d(Cluster, AP)
        D = d_c_ap[:, :, None] + d_ue_c[:, None, :]  # (C, L, K)
        
        # Base phase from path length
        phase_base = -2j * np.pi * D / lam  # (C, L, K)
        
        # Array Response Phase Offset
        array_phase = -2j * np.pi * (antenna_spacing / lam) * n_idx[None, None, :] * np.cos(aoa_c_ap)[:, :, None]
        
        # Total complex phase (C, L, N, K)
        total_phase = phase_base[:, :, None, :] + array_phase[:, :, :, None]
        
        # Superimpose the arriving waves from all C clusters!
        h_t = np.sum(A[:, None, None, :] * np.exp(total_phase), axis=0)  # (L, N, K)
        
        # Normalize so variance is 1
        h_t /= np.sqrt(num_clusters)
        
        # Scale to match the large-scale envelope (pathloss + shadow fading)
        H[t] = h_t * np.sqrt(beta[:, None, :])
        
    return H   # (T_total, L, N, K)


# ──────────────────────────────────────────────────────────────
# Graph construction
# ──────────────────────────────────────────────────────────────

def build_knn_graph(ap_pos: np.ndarray, k_neighbors: int = 3):
    """
    Build symmetric K-NN adjacency from AP positions.
    Returns:
        edge_index : (2, E) torch.long
        adj        : (L, L) numpy float32
    """
    L = len(ap_pos)
    diff = ap_pos[:, None, :] - ap_pos[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dist, np.inf)

    adj = np.zeros((L, L), dtype=np.float32)
    for i in range(L):
        nbrs = np.argsort(dist[i])[:k_neighbors]
        adj[i, nbrs] = 1.0
        adj[nbrs, i] = 1.0   # symmetric

    rows, cols = np.nonzero(adj)
    edge_index = torch.tensor(np.stack([rows, cols]), dtype=torch.long)
    return edge_index, adj


def compute_norm_adj(adj: np.ndarray, self_loops: bool = True) -> torch.Tensor:
    """
    Compute D^{-1/2} (A + I) D^{-1/2}  for GCN propagation.
    Returns (L, L) float32 tensor.
    """
    A = adj.copy().astype(np.float32)
    if self_loops:
        A += np.eye(len(A), dtype=np.float32)
    D = A.sum(axis=1)
    D_inv_sqrt = np.diag(D ** -0.5)
    A_norm = D_inv_sqrt @ A @ D_inv_sqrt
    return torch.tensor(A_norm, dtype=torch.float32)


# ──────────────────────────────────────────────────────────────
# Full dataset generation
# ──────────────────────────────────────────────────────────────

def generate_dataset(
    L: int = 16,
    K: int = 4,
    N: int = 4,
    fc: float = 3.5e9,
    velocity_kmh: float = 60.0,
    T_history: int = 10,
    T_predict: int = 3,
    num_samples: int = 5000,
    layout: str = 'random',
    area_size: float = 500.0,
    ap_seed: int = 42,
    Ts: float = 1e-3,
    save_path: str = None,
    verbose: bool = True,
) -> dict:
    """
    Generate full dataset.

    X shape : (num_samples, L, 2*N*K, T_history)   — real/imag flattened input
    Y shape : (num_samples, L, 2*N*K, T_predict)   — real/imag flattened target

    AP layout is FIXED per dataset (same positions for all samples).
    UE positions are randomised per sample to improve generalisation.
    """
    ap_pos = generate_ap_positions(L, area_size, layout, seed=ap_seed)
    edge_index, adj_np = build_knn_graph(ap_pos, k_neighbors=3)
    adj_norm = compute_norm_adj(adj_np)

    T_total = T_history + T_predict
    C = 2 * N * K   # channels per AP (real + imag)

    X_list, Y_list = [], []

    for s in range(num_samples):
        if verbose and s % 500 == 0:
            print(f"  Generating sample {s}/{num_samples}...")

        ue_pos = generate_ue_positions(K, area_size, seed=s)
        beta = generate_large_scale_fading(ap_pos, ue_pos,
                                            fc_ghz=fc / 1e9, seed=s)
        H = generate_temporal_channel(beta, ap_pos, ue_pos, N, T_total, fc,
                                       velocity_kmh, Ts, seed=s)
        # H: (T_total, L, N, K) complex

        # Flatten N×K → N*K and split real/imag → 2*N*K
        H_r = H.real.reshape(T_total, L, N * K)
        H_i = H.imag.reshape(T_total, L, N * K)
        H_flat = np.concatenate([H_r, H_i], axis=-1).astype(np.float32)
        # H_flat: (T_total, L, C)  where C = 2*N*K

        # Transpose to (L, C, T) for Conv1D compatibility
        H_t = H_flat.transpose(1, 2, 0)   # (L, C, T_total)

        X_list.append(H_t[:, :, :T_history])          # (L, C, T_history)
        Y_list.append(H_t[:, :, T_history:])          # (L, C, T_predict)

    X = torch.tensor(np.stack(X_list))   # (S, L, C, T_history)
    Y = torch.tensor(np.stack(Y_list))   # (S, L, C, T_predict)

    dataset = {
        'X': X,
        'Y': Y,
        'edge_index': edge_index,
        'adj_norm': adj_norm,
        'adj_np': adj_np,
        'ap_pos': torch.tensor(ap_pos, dtype=torch.float32),
        'layout': layout,
        'velocity_kmh': velocity_kmh,
        'L': L, 'K': K, 'N': N,
        'T_history': T_history,
        'T_predict': T_predict,
        'fc': fc,
        'Ts': Ts,
    }

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        torch.save(dataset, save_path)
        if verbose:
            print(f"Saved → {save_path}  |  X:{X.shape}  Y:{Y.shape}")

    return dataset


# ──────────────────────────────────────────────────────────────
# Convenience: generate all velocity × layout combinations
# ──────────────────────────────────────────────────────────────

def generate_all_datasets(data_dir: str = 'data', **kwargs):
    """Pre-generate all required datasets and save to disk."""
    os.makedirs(data_dir, exist_ok=True)
    velocities = [30, 60, 120]
    layouts = ['random', 'grid']

    for vel in velocities:
        for lay in layouts:
            path = os.path.join(data_dir, f'dataset_{lay}_{vel}kmh.pt')
            if os.path.exists(path):
                print(f"Already exists, skipping: {path}")
                continue
            print(f"\n── Generating: layout={lay}, velocity={vel} km/h ──")
            generate_dataset(velocity_kmh=vel, layout=lay,
                             save_path=path, **kwargs)


if __name__ == '__main__':
    print("Generating all datasets (this takes ~10 min on first run)...")
    generate_all_datasets(
        data_dir='data',
        L=16, K=4, N=4,
        fc=3.5e9,
        T_history=10,
        T_predict=3,
        num_samples=5000,
        Ts=1e-3,
    )
    print("Done.")
