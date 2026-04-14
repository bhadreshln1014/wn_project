"""
baselines.py — AR and Kalman Filter channel predictors

Both baselines operate on the raw complex channel H, not the flattened
real/imag format used by the neural models. They are evaluated on the
same test samples to ensure a fair NMSE comparison.

AR Model
--------
  Fits an AR(P) model per scalar channel coefficient using Yule-Walker
  equations, then predicts T_predict steps ahead autoregressively.
  Order P=4 as per the plan.

Kalman Filter
-------------
  Optimal linear predictor for the Jake's AR(1) channel model.
  State:  h_t (scalar complex)
  Model:  h_t = a * h_{t-1} + w_t,  w_t ~ CN(0, Q)
          y_t = h_t + v_t,           v_t ~ CN(0, R)
  The AR coefficient a = J0(2*pi*fD*Ts) is computed from velocity.
  Q and R are estimated from the observed history window.
"""

import numpy as np
from scipy.special import j0
from typing import Tuple


# ──────────────────────────────────────────────────────────────
# AR Model
# ──────────────────────────────────────────────────────────────

def _yule_walker(x: np.ndarray, order: int) -> np.ndarray:
    """
    Estimate AR coefficients via Yule-Walker equations.
    x    : 1-D complex array of observations
    order: AR order P
    Returns: coefficients array of shape (order,)
    """
    n = len(x)
    # Autocorrelation
    r = np.array([np.dot(np.conj(x[:n - k]), x[k:]) / (n - k)
                  for k in range(order + 1)])
    # Toeplitz system R * a = r[1:]
    R = np.array([[r[abs(i - j)] for j in range(order)]
                  for i in range(order)])
    try:
        a = np.linalg.solve(R, r[1:])
    except np.linalg.LinAlgError:
        a = np.zeros(order, dtype=complex)
    return a


def ar_predict(h_history: np.ndarray, T_predict: int,
               order: int = 4) -> np.ndarray:
    """
    Predict T_predict future values of a 1-D complex time series using AR(P).

    h_history : (T_history,) complex array
    Returns   : (T_predict,) complex array
    """
    P = min(order, len(h_history) - 1)
    if P < 1:
        # Not enough history — repeat last value
        return np.full(T_predict, h_history[-1], dtype=complex)

    coeffs = _yule_walker(h_history, P)

    buf = list(h_history[-P:])    # rolling buffer of length P
    preds = []
    for _ in range(T_predict):
        pred = np.dot(coeffs, buf[-P:][::-1])
        preds.append(pred)
        buf.append(pred)

    return np.array(preds)


class ARPredictor:
    """
    AR(P) predictor applied independently to each scalar channel coefficient.

    Usage
    -----
    pred = ARPredictor(order=4).predict(X_np, T_predict)
    """

    def __init__(self, order: int = 4):
        self.order = order

    def predict(self, X: np.ndarray, T_predict: int) -> np.ndarray:
        """
        X       : (L, N, K, T_history) complex — one sample
        Returns : (L, N, K, T_predict) complex
        """
        L, N, K, T_h = X.shape
        Y = np.zeros((L, N, K, T_predict), dtype=complex)
        for l in range(L):
            for n in range(N):
                for k in range(K):
                    Y[l, n, k] = ar_predict(X[l, n, k], T_predict, self.order)
        return Y


# ──────────────────────────────────────────────────────────────
# Vector Kalman Filter
# ──────────────────────────────────────────────────────────────

def _estimate_ar_scalar_joint(X_seq: np.ndarray, p: int = 1) -> np.ndarray:
    """Estimate a single scalar AR(p) model by averaging ACF across N antennas."""
    N, T = X_seq.shape
    if T <= p:
        return np.zeros(p, dtype=complex)
    
    r = np.zeros(p + 1, dtype=complex)
    for tau in range(p + 1):
        for n in range(N):
            r[tau] += np.dot(np.conj(X_seq[n, :T - tau]), X_seq[n, tau:]) / (T - tau)
    r /= N
    
    R = np.zeros((p, p), dtype=complex)
    for i in range(p):
        for j in range(p):
            R[i, j] = r[abs(i - j)] if i >= j else np.conj(r[abs(i - j)])
            
    try:
        a = np.linalg.solve(R, r[1:])
    except np.linalg.LinAlgError:
        a = np.zeros(p, dtype=complex)
    return a

class VectorKalmanPredictor:
    """
    Joint Vector Kalman filter predictor applied over the entire array of an AP.
    Tracks state vectors with cross-antenna covariance for optimal prediction in GBSM.
    """
    
    def __init__(self, order: int = 1, fc: float = 3.5e9,
                 velocity_kmh: float = 60.0, Ts: float = 1e-3):
        self.p = order

    def predict(self, X: np.ndarray, T_predict: int) -> np.ndarray:
        L, N, K, T_h = X.shape
        Y = np.zeros((L, N, K, T_predict), dtype=complex)
        
        for l in range(L):
            for k in range(K):
                h_seq = X[l, :, k, :]  # (N, T_h)
                p = min(self.p, T_h - 1)
                
                if p < 1:
                    Y[l, :, k, :] = h_seq[:, -1:]
                    continue
                    
                a = _estimate_ar_scalar_joint(h_seq, p)
                
                if T_h > p:
                    W = np.zeros((N, T_h - p), dtype=complex)
                    for t in range(p, T_h):
                        h_t = h_seq[:, t]
                        h_pred = np.zeros(N, dtype=complex)
                        for i in range(p):
                            h_pred += a[i] * h_seq[:, t - 1 - i]
                        W[:, t - p] = h_t - h_pred
                    Q = (W @ np.conj(W).T) / (T_h - p)
                else:
                    Q = np.eye(N, dtype=complex) * 1e-6
                
                Q += 1e-6 * np.eye(N)
                R_obs = Q * 0.1
                
                S = np.zeros(p * N, dtype=complex)
                for i in range(p):
                    S[i*N : (i+1)*N] = h_seq[:, p - 1 - i]
                
                P = np.eye(p * N, dtype=complex) * np.mean(np.diag(np.abs(Q)))
                
                F = np.zeros((p * N, p * N), dtype=complex)
                for i in range(p):
                    F[0:N, i*N : (i+1)*N] = a[i] * np.eye(N)
                for i in range(1, p):
                    F[i*N : (i+1)*N, (i-1)*N : i*N] = np.eye(N)
                    
                Q_ss = np.zeros((p * N, p * N), dtype=complex)
                Q_ss[0:N, 0:N] = Q
                
                H_obs = np.zeros((N, p * N), dtype=complex)
                H_obs[0:N, 0:N] = np.eye(N)
                
                for t in range(p, T_h):
                    y_t = h_seq[:, t]
                    
                    S_pred = F @ S
                    P_pred = F @ P @ np.conj(F).T + Q_ss
                    
                    Inn = y_t - (H_obs @ S_pred)
                    S_cov = H_obs @ P_pred @ np.conj(H_obs).T + R_obs
                    try:
                        K_gain = P_pred @ np.conj(H_obs).T @ np.linalg.inv(S_cov)
                    except np.linalg.LinAlgError:
                        K_gain = np.zeros((p*N, N), dtype=complex)
                        
                    S = S_pred + K_gain @ Inn
                    P = (np.eye(p * N) - K_gain @ H_obs) @ P_pred
                    
                for step in range(T_predict):
                    S = F @ S
                    Y[l, :, k, step] = S[0:N]
                    
        return Y


# ──────────────────────────────────────────────────────────────
# Shared evaluation helper
# ──────────────────────────────────────────────────────────────

def nmse_complex(pred: np.ndarray, target: np.ndarray) -> float:
    """NMSE in dB for complex-valued arrays."""
    num = np.sum(np.abs(pred - target) ** 2)
    den = np.sum(np.abs(target) ** 2) + 1e-12
    return 10.0 * np.log10(num / den)


def evaluate_baseline(predictor, X_complex: np.ndarray,
                       Y_complex: np.ndarray,
                       T_predict: int,
                       k_step: int = None) -> float:
    """
    Evaluate a baseline predictor on a batch of test samples.

    X_complex : (S, L, N, K, T_history) complex
    Y_complex : (S, L, N, K, T_predict) complex
    k_step    : if given, evaluate only the k-th prediction step (0-indexed)

    Returns NMSE in dB.
    """
    S = X_complex.shape[0]
    preds = []
    for s in range(S):
        pred_s = predictor.predict(X_complex[s], T_predict)  # (L,N,K,T_predict)
        preds.append(pred_s)
    preds = np.stack(preds)   # (S, L, N, K, T_predict)

    if k_step is not None:
        return nmse_complex(preds[..., k_step], Y_complex[..., k_step])
    return nmse_complex(preds, Y_complex)


# ──────────────────────────────────────────────────────────────
# Convert flat real/imag tensor back to complex for baselines
# ──────────────────────────────────────────────────────────────

def flat_to_complex(X_flat: 'torch.Tensor', N: int, K: int) -> np.ndarray:
    """
    Convert dataset format (S, L, 2*N*K, T) → (S, L, N, K, T) complex numpy.
    """
    X_np = X_flat.numpy()                    # (S, L, 2*N*K, T)
    S, L, C, T = X_np.shape
    NK = N * K
    X_r = X_np[:, :, :NK, :].reshape(S, L, N, K, T)
    X_i = X_np[:, :, NK:, :].reshape(S, L, N, K, T)
    return X_r + 1j * X_i                   # (S, L, N, K, T) complex


# ──────────────────────────────────────────────────────────────
# Quick self-test
# ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    L, N, K, T_h, T_p = 4, 4, 4, 10, 3
    rng = np.random.default_rng(0)
    X = (rng.standard_normal((L, N, K, T_h))
         + 1j * rng.standard_normal((L, N, K, T_h)))
    Y = (rng.standard_normal((L, N, K, T_p))
         + 1j * rng.standard_normal((L, N, K, T_p)))

    ar = ARPredictor(order=4)
    kf = VectorKalmanPredictor(order=1)

    for pred_obj, name in [(ar, 'AR'), (kf, 'VKF')]:
        pred = pred_obj.predict(X, T_p)
        err = nmse_complex(pred, Y)
        print(f"{name:10s} NMSE = {err:.2f} dB  (random baseline, expect ~0 dB)")

    print("Baseline self-test complete.")
