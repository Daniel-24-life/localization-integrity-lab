"""Unit tests for bounded antenna lever-arm uncertainty model."""
import numpy as np
import pytest
from src.models.lever_arm_uncertainty import LeverArmUncertainty


def test_lever_arm_initialization():
    model = LeverArmUncertainty(r_max=0.35, k_integrity=3.0)
    assert model.r_max == 0.35
    assert model.k_integrity == 3.0
    assert model.position_bias_bound() == 0.35

    with pytest.raises(ValueError):
        LeverArmUncertainty(r_max=-0.1)
    with pytest.raises(ValueError):
        LeverArmUncertainty(k_integrity=0)


def test_antenna_position_geometry():
    model = LeverArmUncertainty(r_max=0.30)
    p_body = np.array([10.0, 5.0])
    r_ant = np.array([0.2, 0.0])  # 20cm ahead of body center

    # Heading 0: pointing East (+x)
    p_ant_0 = model.antenna_position(p_body, 0.0, r_ant)
    np.testing.assert_allclose(p_ant_0, [10.2, 5.0])

    # Heading pi/2: pointing North (+y)
    p_ant_pi2 = model.antenna_position(p_body, np.pi / 2, r_ant)
    np.testing.assert_allclose(p_ant_pi2, [10.0, 5.2], atol=1e-10)

    # Invariance of distance
    dist = np.linalg.norm(p_ant_pi2 - p_body)
    np.testing.assert_allclose(dist, 0.2)


def test_conservative_protection_level():
    model = LeverArmUncertainty(r_max=0.25)
    nom_pl = 0.60
    cons_pl = model.conservative_protection_level(nom_pl)
    assert cons_pl == pytest.approx(0.85)

    # Vectorized
    pl_vec = np.array([0.4, 0.5, 0.6])
    cons_vec = model.conservative_protection_level(pl_vec)
    np.testing.assert_allclose(cons_vec, [0.65, 0.75, 0.85])

    with pytest.raises(ValueError):
        model.conservative_protection_level(-0.1)


def test_inflated_covariance():
    model = LeverArmUncertainty(r_max=0.20)
    P_pos = np.array([[0.04, 0.0], [0.0, 0.04]])
    inflated = model.inflated_covariance(P_pos)

    # Inflation is r_max^2 = 0.04
    expected = np.array([[0.08, 0.0], [0.0, 0.08]])
    np.testing.assert_allclose(inflated, expected)

    # 3D tensor shape (N, 2, 2)
    P_seq = np.zeros((5, 2, 2))
    inflated_seq = model.inflated_covariance(P_seq)
    assert inflated_seq.shape == (5, 2, 2)
    np.testing.assert_allclose(inflated_seq[0], 0.04 * np.eye(2))


def test_bounds_summary():
    model = LeverArmUncertainty(r_max=0.30)
    summary = model.bounds_summary(nominal_pl=0.50, nominal_ate_rmse=0.15)
    assert summary["r_max_m"] == 0.30
    assert summary["conservative_pl_m"] == pytest.approx(0.80)
    assert summary["conservative_ate_rmse_bound_m"] == pytest.approx(0.45)
    assert summary["lever_arm_variance_m2"] == pytest.approx(0.09)
