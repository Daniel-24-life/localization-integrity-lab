import copy
import numpy as np
import pytest
from src.data.synthetic_generator import SyntheticTrajectoryGenerator
from src.evaluation.run_artifacts import load_config
from src.filters.fixed_ekf import FixedEKF
from src.filters.gated_ekf import GatedEKF
from src.filters.adaptive_ekf import AdaptiveEKF
from src.filters.velocity_bias_ekf import VelocityBiasEKF
from src.filters.bias_model_bank import BiasModelBank
from src.filters.antenna_ekf import (
    AntennaFixedEKF,
    AntennaGatedEKF,
    AntennaAdaptiveEKF,
    AntennaVelocityBiasEKF,
    AntennaBiasModelBank,
)
from src.models.antenna_observation import antenna_observation


def fixture():
    cfg = load_config(); cfg['simulation']['duration'] = 3.
    cfg['filter']['position_lever_flu_m'] = [0., 0., 0.]
    return cfg, SyntheticTrajectoryGenerator(cfg).generate('circle')


@pytest.mark.parametrize('old,new', [
    (FixedEKF, AntennaFixedEKF),
    (GatedEKF, AntennaGatedEKF),
    (AdaptiveEKF, AntennaAdaptiveEKF),
    (VelocityBiasEKF, AntennaVelocityBiasEKF),
    (BiasModelBank, AntennaBiasModelBank),
])
def test_zero_lever_reproduces_existing_baselines(old, new):
    cfg, data = fixture(); initial = np.array([data.gt_x[0], data.gt_y[0], data.gt_theta[0]])
    a, b = old(cfg, initial).run_sequence(data), new(cfg, initial).run_sequence(data)
    for key in a: np.testing.assert_allclose(a[key], b[key], atol=1e-11, rtol=1e-11, equal_nan=True)


def test_unavailable_wheel_rate_is_explicit_not_silently_replaced():
    cfg, data = fixture(); data.wheel_omega[:] = np.nan
    with pytest.raises(ValueError): data.validate()
    data.metadata.update(wheel_angular_velocity_status='unavailable', imu_frame='wheel', imu_rotation_verified=True)
    data.validate()
    with pytest.raises(ValueError): FixedEKF(cfg).run_sequence(data)
    cfg['filter']['angular_input'] = 'imu'
    assert np.isfinite(AntennaFixedEKF(cfg).run_sequence(data)['state']).all()
    data.wheel_omega[0] = 0.
    with pytest.raises(ValueError): data.validate()


@pytest.mark.parametrize('model', [AntennaFixedEKF, AntennaGatedEKF, AntennaAdaptiveEKF])
def test_future_and_reference_changes_do_not_enter_filter(model):
    cfg, data = fixture(); cfg['filter']['position_lever_flu_m'] = [.2, -.1, .5]
    initial = np.array([data.gt_x[0], data.gt_y[0], data.gt_theta[0]])
    baseline = model(cfg, initial).run_sequence(data)
    altered = copy.deepcopy(data); cutoff = 25
    altered.gt_x += 1000.; altered.gt_y -= 1000.; altered.gt_theta += .7
    altered.gnss_x[cutoff:] += 50.; altered.wheel_v[cutoff:] *= 10.
    other = model(cfg, initial).run_sequence(altered)
    np.testing.assert_array_equal(baseline['state'][:cutoff], other['state'][:cutoff])


def test_perfect_antenna_observation_has_zero_innovation_and_positive_covariance():
    cfg, _ = fixture(); cfg['filter']['position_lever_flu_m'] = [.3, .1, 1.]
    state = np.array([2., 3., 1.2]); model = AntennaFixedEKF(cfg, state)
    z, _ = antenna_observation(state, model.lever)
    _, P, nis, residual = model.update(z)
    np.testing.assert_allclose(residual, 0.)
    assert nis == 0. and np.linalg.eigvalsh(P).min() > 0.


@pytest.mark.parametrize('model', [AntennaFixedEKF, AntennaGatedEKF, AntennaAdaptiveEKF])
def test_entire_reference_can_be_removed_after_initialization(model):
    cfg, data = fixture()
    initial = np.array([data.gt_x[0], data.gt_y[0], data.gt_theta[0]])
    expected = model(cfg, initial).run_sequence(data)
    for key in ('gt_x', 'gt_y', 'gt_theta'): getattr(data, key)[:] = np.nan
    actual = model(cfg, initial).run_sequence(data)
    for key in expected: np.testing.assert_array_equal(actual[key], expected[key])


def test_gate_equality_accepts_and_missing_sample_does_not_hold_alarm():
    cfg, data = fixture(); model = AntennaGatedEKF(cfg)
    z = np.array([.5, 0.]); y = z - model.state[:2]
    model.gate_threshold = float(y @ np.linalg.solve(model.P[:2, :2] + model.R, y))
    model.update(z); assert not model._gated[-1]
    data.gnss_valid[:] = False; data.gnss_valid[1] = True; data.gnss_x[1] = 100.; data.gnss_y[1] = 0.
    out = AntennaGatedEKF(cfg).run_sequence(data)
    assert out['gated'][1] and not out['gated'][2] and np.isnan(out['nis'][2])


def test_antenna_adaptive_ekf_lever_arm_adaptation_and_bounds():
    cfg, data = fixture()
    cfg['filter']['position_lever_flu_m'] = [0.35, -0.12, 0.58]
    cfg['filter']['adaptation_window'] = 5
    cfg['filter']['adaptation_rate'] = 0.2
    model = AntennaAdaptiveEKF(cfg)
    out = model.run_sequence(data)
    assert 'measurement_covariance_used' in out
    assert 'measurement_covariance_next' in out
    assert 'R_projection_clipped' in out
    # Verify eigenvalues stay within bounds
    covs = out['measurement_covariance_next']
    for cov in covs:
        eigs = np.linalg.eigvalsh(cov)
        assert eigs.min() >= model.floor - 1e-9
        assert eigs.max() <= model.ceiling + 1e-9


def test_antenna_bias_model_bank_with_nonzero_lever_arm():
    cfg, data = fixture()
    cfg['filter']['position_lever_flu_m'] = [0.25, -0.1, 0.4]
    initial = np.array([data.gt_x[0], data.gt_y[0], data.gt_theta[0]])
    bank = AntennaBiasModelBank(cfg, initial)
    out = bank.run_sequence(data)
    assert np.isfinite(out['state']).all()
    assert np.isfinite(out['covariance']).all()
    assert (np.linalg.eigvalsh(out['covariance']) > 0).all()
    np.testing.assert_allclose(out['model_weights'].sum(axis=1), 1.0, atol=1e-12)
