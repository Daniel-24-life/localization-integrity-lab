import copy
import numpy as np
import pytest
from src.evaluation.run_artifacts import load_config
from src.filters.adaptive_ekf import AdaptiveEKF
from src.filters.fixed_ekf import FixedEKF
from src.data.synthetic_generator import SyntheticTrajectoryGenerator
from src.faults.injector import FaultInjector


def test_zero_adaptation_is_fixed_baseline():
    cfg = load_config(); cfg['simulation']['duration'] = 3
    cfg['adaptive'] = {'update_rate': 0.}
    data = SyntheticTrajectoryGenerator(cfg).generate('circle')
    fixed, adaptive = FixedEKF(cfg).run_sequence(data), AdaptiveEKF(cfg).run_sequence(data)
    for key in fixed:
        np.testing.assert_array_equal(fixed[key], adaptive[key])


def test_formula_and_current_nis_use_old_R():
    cfg = load_config(); cfg['adaptive'] = dict(window_observations=2, minimum_observations=2, update_rate=1.)
    model = AdaptiveEKF(cfg); innovations = []
    for z in [np.array([1., .2]), np.array([-.5, 2.])]:
        projected = model.H @ model.P @ model.H.T
        old = model.R.copy(); y = z-model.H @ model.state
        expected_nis = float(y @ np.linalg.solve(projected+old, y))
        result = model.update(z); innovations.append(y)
        assert result[2] == pytest.approx(expected_nis)
    samples = np.asarray(innovations)
    value = samples.T @ samples / 2 - projected
    eig, vec = np.linalg.eigh(value)
    np.testing.assert_allclose(model.R, (vec*np.clip(eig, .05**2, 3.**2)) @ vec.T)


def test_no_adaptation_during_dropout_and_missing_diagnostics():
    cfg = load_config(); cfg['simulation']['duration'] = 6
    cfg['adaptive'] = dict(window_observations=2, minimum_observations=2)
    data = FaultInjector.inject_gnss_dropout(SyntheticTrajectoryGenerator(cfg).generate('straight'), 2, 2)
    result = AdaptiveEKF(cfg).run_sequence(data)
    mask = (data.timestamp >= 2) & (data.timestamp < 4)
    assert np.isnan(result['nis'][mask]).all()
    assert np.isnan(result['measurement_covariance_used'][mask]).all()
    R = result['measurement_covariance_next'][mask]
    np.testing.assert_array_equal(R, np.repeat(R[:1], len(R), axis=0))
    assert np.linalg.eigvalsh(result['covariance']).min() >= -1e-12


def test_prefix_causality_and_no_GT_dependency():
    cfg = load_config(); cfg['simulation']['duration'] = 3
    cfg['adaptive'] = dict(window_observations=2, minimum_observations=2)
    data = SyntheticTrajectoryGenerator(cfg).generate('figure8')
    changed = copy.deepcopy(data); changed.gnss_x[-4:] += 100
    changed.gt_x[:] += 500; changed.gt_y[:] -= 500
    a, b = AdaptiveEKF(cfg).run_sequence(data), AdaptiveEKF(cfg).run_sequence(changed)
    for key in a:
        np.testing.assert_array_equal(a[key][:-4], b[key][:-4])


@pytest.mark.parametrize('options', [dict(window_observations=1), dict(minimum_observations=21),
    dict(update_rate=np.nan), dict(minimum_sigma_m=-.1), dict(maximum_sigma_m=.01)])
def test_invalid_options(options):
    cfg = load_config(); cfg['adaptive'] = options
    with pytest.raises(ValueError): AdaptiveEKF(cfg)
