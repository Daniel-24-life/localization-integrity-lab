"""Run a bounded finite-horizon observability and noise-discretisation audit."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import quad

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.run_artifacts import ROOT, RunSession, write_json
from src.models.research_linear_audit import (
    scaled_whitened_observability,
    transition_matrix,
    van_loan_discrete_covariance,
)
from src.models.velocity_bias import bias_jacobian


def numerical_covariance_integral(A, Qc, dt):
    """Independent scalar quadrature of integral Phi Q Phi^T for an audit check."""
    n = A.shape[0]
    out = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            out[i, j] = quad(
                lambda tau: (transition_matrix(A, tau) @ Qc @ transition_matrix(A, tau).T)[i, j],
                0.0,
                dt,
                epsabs=1e-11,
                epsrel=1e-11,
                limit=100,
            )[0]
    return (out + out.T) / 2


def main():
    state = np.zeros(4)
    H = np.hstack([np.eye(2), np.zeros((2, 2))])
    R = np.eye(2) * 0.15**2
    scales = np.array([0.5, 0.5, 0.2, 0.1])
    elapsed = np.linspace(0.0, 5.0, 51)
    scenarios = {
        "moving_position_updates": (np.array([1.0, 0.0]), np.ones(len(elapsed), dtype=bool)),
        "stationary_position_updates": (np.array([0.0, 0.0]), np.ones(len(elapsed), dtype=bool)),
        "moving_no_position_updates": (np.array([1.0, 0.0]), np.zeros(len(elapsed), dtype=bool)),
        "moving_first_and_last_only": (np.array([1.0, 0.0]), np.array([True] + [False] * (len(elapsed) - 2) + [True])),
    }
    audit = {"state_order": ["x", "y", "theta", "b_v"], "elapsed_seconds": elapsed.tolist(), "scales": scales.tolist(), "scenarios": {}}
    for name, (control, valid) in scenarios.items():
        A = bias_jacobian(state, control)
        O, singular_values, rank, condition = scaled_whitened_observability(A, H, R, elapsed, scales, valid)
        audit["scenarios"][name] = {
            "control_v_omega": control.tolist(),
            "valid_measurements": int(valid.sum()),
            "matrix_shape": list(O.shape),
            "rank": rank,
            "singular_values": singular_values.tolist(),
            "condition_number_if_full_rank": condition,
        }

    # Continuous spectral density used only for this numerical comparison.
    noise_map = np.eye(4)
    spectral = np.diag([0.01, 0.01, 0.0025, 0.0009])
    A = bias_jacobian(state, np.array([1.0, 0.0]))
    dt = 0.2
    phi, qd = van_loan_discrete_covariance(A, noise_map, spectral, dt)
    direct = numerical_covariance_integral(A, spectral, dt)
    phi_half, qd_half = van_loan_discrete_covariance(A, noise_map, spectral, dt / 2)
    composed = qd_half + phi_half @ qd_half @ phi_half.T
    audit["noise_discretisation"] = {
        "dt_seconds": dt,
        "van_loan_vs_quadrature_max_abs": float(np.max(np.abs(qd - direct))),
        "semigroup_composition_max_abs": float(np.max(np.abs(qd - composed))),
        "phi_max_abs": float(np.max(np.abs(phi))),
        "qd_min_eigenvalue": float(np.linalg.eigvalsh(qd).min()),
        "spectral_covariance_units": "continuous covariance density; state-unit dependent",
    }

    config = {
        "purpose": "finite_horizon_scaled_whitened_observability_and_van_loan_audit",
        "model": "frozen local augmented [x,y,theta,b_v]",
        "full_nonlinear_filter_rerun": False,
        "measurement_noise_standard_deviation_m": 0.15,
        "observability_window_seconds": 5.0,
        "notes": "Position is the only observation; GT is not used.",
    }
    inputs = [ROOT / "src/models/research_linear_audit.py", ROOT / "src/models/velocity_bias.py", ROOT / "configs/default_config.yaml"]
    with RunSession("linear_research_audit", config=config, inputs=inputs) as run:
        write_json(run.path / "audit.json", audit)
        (run.path / "README.md").write_text(
            "This is a bounded frozen-linear audit. It does not modify or rerun production filters.\n",
            encoding="utf-8",
        )
        print(run.path)


if __name__ == "__main__":
    main()
