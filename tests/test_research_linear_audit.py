import numpy as np
import pytest

from src.models.research_linear_audit import (
    scaled_whitened_observability,
    van_loan_discrete_covariance,
)
from src.models.velocity_bias import bias_jacobian


def _inputs():
    H = np.hstack([np.eye(2), np.zeros((2, 2))])
    R = np.eye(2) * 0.15**2
    times = np.linspace(0.0, 3.0, 31)
    scales = np.array([0.5, 0.5, 0.2, 0.1])
    return H, R, times, scales


def test_dynamic_rank_differs_from_single_frame_rank():
    H, R, times, scales = _inputs()
    valid = np.ones(len(times), dtype=bool)
    moving = bias_jacobian(np.zeros(4), np.array([1.0, 0.0]))
    stationary = bias_jacobian(np.zeros(4), np.array([0.0, 0.0]))
    _, moving_sv, moving_rank, moving_cond = scaled_whitened_observability(moving, H, R, times, scales, valid)
    _, stationary_sv, stationary_rank, stationary_cond = scaled_whitened_observability(stationary, H, R, times, scales, valid)
    assert np.linalg.matrix_rank(H) == 2
    assert moving_rank == 4
    assert stationary_rank == 3
    assert moving_cond is not None and moving_cond > 1
    assert stationary_cond is None
    assert moving_sv[0] > moving_sv[1] > 0
    assert stationary_sv[-1] == pytest.approx(0.0)


def test_no_measurements_have_zero_information_and_sparse_motion_can_be_full_rank():
    H, R, times, scales = _inputs()
    moving = bias_jacobian(np.zeros(4), np.array([1.0, 0.0]))
    _, sv0, rank0, condition0 = scaled_whitened_observability(
        moving, H, R, times, scales, np.zeros(len(times), dtype=bool)
    )
    valid = np.zeros(len(times), dtype=bool)
    valid[[0, -1]] = True
    _, sv_sparse, rank_sparse, condition_sparse = scaled_whitened_observability(
        moving, H, R, times, scales, valid
    )
    assert rank0 == 0 and condition0 is None and np.all(sv0 == 0)
    assert rank_sparse == 4 and condition_sparse is not None and sv_sparse[-1] > 0


def test_van_loan_matches_scalar_analytic_and_is_psd():
    A = np.array([[0.4]])
    L = np.array([[1.0]])
    q = np.array([[0.7]])
    dt = 0.3
    phi, qd = van_loan_discrete_covariance(A, L, q, dt)
    expected_phi = np.exp(0.4 * dt)
    expected_q = 0.7 * (np.exp(2 * 0.4 * dt) - 1.0) / (2 * 0.4)
    assert phi[0, 0] == pytest.approx(expected_phi)
    assert qd[0, 0] == pytest.approx(expected_q)
    assert np.linalg.eigvalsh(qd).min() >= -1e-12


def test_invalid_masks_and_scales_rejected():
    H, R, times, scales = _inputs()
    A = bias_jacobian(np.zeros(4), np.array([1.0, 0.0]))
    with pytest.raises(ValueError):
        scaled_whitened_observability(A, H, R, times, np.zeros(4))
    with pytest.raises(ValueError):
        scaled_whitened_observability(A, H, R, times, scales, np.ones(len(times)))


def test_correlated_noise_semigroup_and_psd():
    A=np.array([[0.,1.],[-.3,-.2]])
    density=np.array([[.2,.06],[.06,.1]])
    phi,qd=van_loan_discrete_covariance(A,np.eye(2),density,.4)
    half,qhalf=van_loan_discrete_covariance(A,np.eye(2),density,.2)
    np.testing.assert_allclose(phi,half@half,atol=1e-14)
    np.testing.assert_allclose(qd,qhalf+half@qhalf@half.T,atol=1e-14)
    assert np.linalg.eigvalsh(qd).min()>0


def test_coordinate_rotation_preserves_whitened_singular_values():
    H,R,times,scales=_inputs()
    A=bias_jacobian(np.zeros(4),np.array([1.,0.]))
    angle=.7;c,s=np.cos(angle),np.sin(angle)
    rotation=np.array([[c,-s],[s,c]])
    _,sv,rank,_=scaled_whitened_observability(A,H,R,times,scales)
    _,sv_rot,rank_rot,_=scaled_whitened_observability(A,rotation@H,rotation@R@rotation.T,times,scales)
    np.testing.assert_allclose(sv,sv_rot,rtol=1e-12,atol=1e-12)
    assert rank==rank_rot==4
