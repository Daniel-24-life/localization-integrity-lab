"""Research-grade local linear-algebra audits.

The routines here are deliberately separate from the production filters.  They
make the finite-horizon observability and continuous-to-discrete noise claims
explicit, dimensionless, and reproducible without changing estimator behavior.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm


def _finite_matrix(value: np.ndarray, name: str) -> np.ndarray:
    value = np.asarray(value, dtype=float)
    if not np.isfinite(value).all():
        raise ValueError(f"{name} must contain finite values")
    return value


def transition_matrix(A: np.ndarray, elapsed: float) -> np.ndarray:
    """Return the exact frozen-linear transition ``exp(A * elapsed)``."""
    A = _finite_matrix(A, "A")
    elapsed = float(elapsed)
    if A.ndim != 2 or A.shape[0] != A.shape[1] or not np.isfinite(elapsed) or elapsed < 0:
        raise ValueError("A must be square and elapsed must be nonnegative")
    return expm(A * elapsed)


def scaled_whitened_observability(
    A: np.ndarray,
    H: np.ndarray,
    R: np.ndarray,
    elapsed: np.ndarray,
    scales: np.ndarray,
    valid: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, int, float | None]:
    """Build a dimensionless finite-horizon observation matrix.

    Each valid row block is ``R^(-1/2) H exp(A*t) S`` where ``S`` maps
    dimensionless initial coordinates to physical state coordinates.  The
    resulting singular values are therefore meaningful only for the supplied
    scales and frozen-linear interval; they are not an estimator covariance or
    a global nonlinear observability proof.
    """
    A = _finite_matrix(A, "A")
    H = _finite_matrix(H, "H")
    R = _finite_matrix(R, "R")
    elapsed = _finite_matrix(elapsed, "elapsed").reshape(-1)
    scales = _finite_matrix(scales, "scales").reshape(-1)
    if A.ndim != 2 or A.shape[0] != A.shape[1] or H.ndim != 2:
        raise ValueError("A and H dimensions do not match")
    n = A.shape[0]
    if H.shape[1] != n:
        raise ValueError("A and H dimensions do not match")
    if R.shape != (H.shape[0], H.shape[0]) or not np.allclose(R, R.T, atol=1e-12):
        raise ValueError("R must be symmetric with one row/column per observation")
    if np.any(np.linalg.eigvalsh(R) <= 0) or scales.shape != (n,) or np.any(scales <= 0):
        raise ValueError("R must be positive definite and scales positive")
    if np.any(elapsed < 0):
        raise ValueError("elapsed must be nonnegative")
    if valid is None:
        valid = np.ones(elapsed.shape, dtype=bool)
    valid = np.asarray(valid)
    if valid.shape != elapsed.shape or valid.dtype != np.bool_:
        raise ValueError("valid must be a boolean mask matching elapsed")
    chol = np.linalg.cholesky(R)
    S = np.diag(scales)
    blocks = []
    for t, keep in zip(elapsed, valid):
        if keep:
            blocks.append(np.linalg.solve(chol, H @ transition_matrix(A, float(t)) @ S))
    O = np.vstack(blocks) if blocks else np.zeros((0, n), dtype=float)
    singular_values = np.linalg.svd(O, compute_uv=False) if O.size else np.zeros(n)
    rank = int(np.linalg.matrix_rank(O)) if O.size else 0
    positive = singular_values[singular_values > 0]
    condition = float(positive[0] / positive[-1]) if len(positive) and rank == n else None
    return O, singular_values, rank, condition


def van_loan_discrete_covariance(
    A: np.ndarray, noise_map: np.ndarray, spectral_covariance: np.ndarray, dt: float
) -> tuple[np.ndarray, np.ndarray]:
    """Compute exact frozen-linear ``Phi, Qd`` using the Van Loan block exponential.

    ``spectral_covariance`` is a continuous-time covariance density.  It is
    distinct from the production filter's per-sample input standard deviation.
    """
    A = _finite_matrix(A, "A")
    noise_map = _finite_matrix(noise_map, "noise_map")
    spectral_covariance = _finite_matrix(spectral_covariance, "spectral_covariance")
    dt = float(dt)
    if A.ndim != 2 or A.shape[0] != A.shape[1] or noise_map.ndim != 2:
        raise ValueError("A and noise_map dimensions do not match")
    n = A.shape[0]
    if noise_map.shape[0] != n:
        raise ValueError("A and noise_map dimensions do not match")
    qdim = noise_map.shape[1]
    if spectral_covariance.shape != (qdim, qdim) or not np.allclose(spectral_covariance, spectral_covariance.T, atol=1e-12):
        raise ValueError("spectral covariance shape/symmetry mismatch")
    if np.any(np.linalg.eigvalsh(spectral_covariance) < -1e-12) or not np.isfinite(dt) or dt <= 0:
        raise ValueError("spectral covariance must be PSD and dt positive")
    Qc = noise_map @ spectral_covariance @ noise_map.T
    block = np.block([[A, Qc], [np.zeros_like(A), -A.T]])
    exp_block = expm(block * dt)
    Phi = exp_block[:n, :n]
    Qd = exp_block[:n, n:] @ Phi.T
    Qd = (Qd + Qd.T) / 2
    return Phi, Qd
