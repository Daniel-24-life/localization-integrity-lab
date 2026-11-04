import numpy as np
import pytest
from src.faults.temporal import wheel_bias_profile, inject_temporal_wheel_bias
from src.data.synthetic_generator import SyntheticTrajectoryGenerator
from src.evaluation.run_artifacts import load_config


@pytest.mark.parametrize('mode,expected', [('constant', 4.), ('intermittent', 2.), ('ramp', 2.)])
def test_exact_exposure_on_nonuniform_grid(mode, expected):
    t = np.array([0., 19.3, 20.1, 21.7, 25.2, 32.6, 39.8, 40.3, 60.])
    s = wheel_bias_profile(t, mode, .2, 20., 20.)
    assert s[:-1] @ np.diff(t) == pytest.approx(expected, abs=1e-12)
    assert s[-1] == 0
    assert np.all((s >= 0) & (s <= .2))


def test_dose_matched_family_and_fractional_support():
    t = np.arange(0., 60.05, .05)
    doses = [wheel_bias_profile(t, m, p, 20., 20.)[:-1] @ np.diff(t)
             for m, p in [('constant', .1), ('intermittent', .2), ('ramp', .2)]]
    np.testing.assert_allclose(doses, 2., atol=1e-12)
    assert wheel_bias_profile([0., 1., 3.], 'constant', .2, .5, 1.)[:-1] @ [1., 2.] == pytest.approx(.2)


def test_identity_no_mutation_and_GNSS_unchanged():
    cfg = load_config(); cfg['simulation']['duration'] = 3
    data = SyntheticTrajectoryGenerator(cfg).generate('straight')
    original = data.wheel_v.copy()
    result = inject_temporal_wheel_bias(data, np.zeros(len(data)))
    np.testing.assert_array_equal(result.wheel_v, original)
    profile = wheel_bias_profile(data.timestamp, 'ramp', .2, 1., 1.)
    result = inject_temporal_wheel_bias(data, profile)
    np.testing.assert_array_equal(data.wheel_v, original)
    np.testing.assert_array_equal(result.gnss_valid, data.gnss_valid)
    np.testing.assert_allclose(result.wheel_v, original*(1-profile))
    assert result.metadata['temporal_wheel_bias']['exposure_sec'] == pytest.approx(.1)


@pytest.mark.parametrize('kwargs', [dict(mode='unknown'), dict(peak=1.), dict(duration=-1.), dict(period=0.), dict(duty=0.)])
def test_invalid_parameters(kwargs):
    args = dict(mode='constant', peak=.2, start=0., duration=1.); args.update(kwargs)
    with pytest.raises(ValueError): wheel_bias_profile([0., 1.], **args)
