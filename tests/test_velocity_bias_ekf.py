import copy
import numpy as np
import pytest
from src.evaluation.run_artifacts import load_config
from src.filters.fixed_ekf import FixedEKF
from src.filters.velocity_bias_ekf import VelocityBiasEKF
from src.data.synthetic_generator import SyntheticTrajectoryGenerator
from src.faults.injector import FaultInjector


def test_disabled_bias_reproduces_fixed_EKF():
    cfg=load_config();cfg['simulation']['duration']=3
    cfg['velocity_bias']=dict(initial_sigma_m_s=0.,diffusion_std_m_s_sqrt_s=0.)
    d=SyntheticTrajectoryGenerator(cfg).generate('circle')
    fixed=FixedEKF(cfg).run_sequence(d);candidate=VelocityBiasEKF(cfg).run_sequence(d)
    for k in fixed:np.testing.assert_allclose(fixed[k],candidate[k],atol=1e-12,equal_nan=True)


def test_no_online_GT_or_future_dependency():
    cfg=load_config();cfg['simulation']['duration']=3
    d=SyntheticTrajectoryGenerator(cfg).generate('figure8');changed=copy.deepcopy(d)
    changed.gt_x[:]=100;changed.gt_y[:]=-100;changed.gt_theta[:]=2.
    changed.gnss_x[-5:]+=10
    a=VelocityBiasEKF(cfg).run_sequence(d);b=VelocityBiasEKF(cfg).run_sequence(changed)
    for k in a:np.testing.assert_array_equal(a[k][:-5],b[k][:-5])


def test_diffusion_cross_covariance_and_missing_update():
    cfg=load_config();cfg['simulation']['duration']=4
    d=FaultInjector.inject_gnss_dropout(SyntheticTrajectoryGenerator(cfg).generate('straight'),1.,2.)
    model=VelocityBiasEKF(cfg);R=model.R.copy();r=model.run_sequence(d)
    missing=(d.timestamp>=1)&(d.timestamp<3)
    assert np.isnan(r['nis'][missing]).all()
    np.testing.assert_array_equal(model.R,R)
    assert np.linalg.eigvalsh(r['augmented_covariance']).min()>0
    np.testing.assert_allclose(r['augmented_covariance'],r['augmented_covariance'].transpose(0,2,1),atol=1e-14)
    assert r['augmented_covariance'][np.flatnonzero(missing)[-1],3,3]>r['augmented_covariance'][np.flatnonzero(missing)[0],3,3]


@pytest.mark.parametrize('options',[dict(initial_sigma_m_s=-1),dict(diffusion_std_m_s_sqrt_s=np.nan),dict(initial_mean_m_s=np.inf)])
def test_invalid_prior(options):
    cfg=load_config();cfg['velocity_bias']=options
    with pytest.raises(ValueError):VelocityBiasEKF(cfg)
