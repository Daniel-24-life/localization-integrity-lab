import numpy as np
import pytest
from src.models.absorbing_pde import (generator, mean_exit_time,
    solve_absorbing_drift_diffusion as solve, brownian_survival_centered,
    unabsorbed_outside_probability)
from src.data.frame_transforms import camera_imu_vectors_to_wheel, load_wheel_to_camera_imu
from src.evaluation.run_artifacts import ROOT


def test_generator_substochastic_and_no_clipped_mass():
    _,K,_,_=generator(.2,.003,n_intervals=100)
    off=K.toarray(); np.fill_diagonal(off,0)
    assert off.min()>=0
    assert np.max(np.asarray(K.sum(axis=0)))<1e-10
    r=solve(.2,.003,n_intervals=200,t_max=4,dt=.01)
    np.testing.assert_allclose(r['survival']+r['left_absorbed']+r['right_absorbed'],1,atol=1e-10)
    assert r['min_mass']>=-1e-12 and np.all(np.diff(r['survival'])<=1e-10)
    assert r['survival'][-1]<.1  # No per-step renormalization of surviving mass.


def test_brownian_absorption_matches_independent_spectral_solution():
    r=solve(0,.01,n_intervals=400,t_max=10,dt=.005)
    t=np.array([1.,3.,5.,10.]); computed=np.interp(t,r['times'],r['survival'])
    np.testing.assert_allclose(computed,brownian_survival_centered(t,.01),atol=.001)
    assert r['mean_exit_time_discrete']==pytest.approx(25.,rel=1e-10)


def test_reflection_and_exact_mean_exit():
    pos=solve(.2,.003,n_intervals=400,t_max=3,dt=.01)
    neg=solve(-.2,.003,n_intervals=400,t_max=3,dt=.01)
    np.testing.assert_allclose(pos['first_exit_cdf'],neg['first_exit_cdf'],atol=1e-12)
    np.testing.assert_allclose(pos['left_absorbed'],neg['right_absorbed'],atol=1e-12)
    for x in [-.2,0,.2]:
        assert mean_exit_time(.2,.003,initial_error=x)==pytest.approx(mean_exit_time(-.2,.003,initial_error=-x))
    assert mean_exit_time(.2,.003)==pytest.approx(2.5*np.tanh(.5*.2/.003))


def test_zero_noise_stationary_and_deterministic_limits():
    r=solve(0,0,n_intervals=50,t_max=1,dt=.1)
    np.testing.assert_allclose(r['survival'],1)
    assert np.isinf(r['mean_exit_time_exact'])
    assert mean_exit_time(.2,0)==pytest.approx(2.5)
    assert mean_exit_time(1e-12,.003,initial_error=.1)==pytest.approx((.25-.01)/.003,rel=1e-8)


def test_first_exit_is_not_occupied_outside_and_near_boundary_initialization():
    r=solve(0,.01,n_intervals=200,t_max=10,dt=.01)
    outside=unabsorbed_outside_probability(np.array([10.]),0,.01)[0]
    assert r['first_exit_cdf'][-1]>outside+.05
    near=solve(0,.01,initial_error=.499,n_intervals=100,t_max=.1,dt=.01)
    assert near['first_exit_cdf'][0]==0
    assert abs(near['initial_grid_mean_error'])<=near['spatial_step']


@pytest.mark.parametrize('kwargs',[{'variance_rate':-.1},{'alert_limit':0},{'initial_error':.5},{'dt':0},{'n_intervals':3},{'drift':np.nan}])
def test_invalid_parameters_fail(kwargs):
    p={'drift':.2,'variance_rate':.003,'t_max':1,'n_intervals':40};p.update(kwargs)
    with pytest.raises(ValueError):solve(**p)


def test_frame_rotation_roundtrip_and_proper_rotation_validation():
    T=load_wheel_to_camera_imu(ROOT/'materials/references/m3dgr_20261027/calibration.md')
    R=T[:3,:3]; wheel=np.array([[0,0,1.],[1.,2.,3.]])
    camera=wheel@R.T
    np.testing.assert_allclose(camera_imu_vectors_to_wheel(camera,R),wheel,atol=1e-12)
    with pytest.raises(ValueError):camera_imu_vectors_to_wheel(camera,2*R)
