import numpy as np
import pytest
from scipy.linalg import expm
from src.models.linear_uncertainty import finite_observability_gramian as gramian, straight_line_error_moments, bounded_position_error
from src.evaluation.metrics import PerformanceEvaluator as E


def test_finite_gramian_moving_static_and_absent_measurements():
    t=np.linspace(0,3,31);H=np.array([[1.,0,0],[0,1.,0]]);R=np.eye(2)*.15**2
    A=np.array([[0.,0,0],[0,0,1.],[0,0,0]])
    transitions=np.array([expm(A*s) for s in t]);valid=np.ones(len(t),bool)
    W=gramian(transitions,H,R,valid,[.5,.5,.1])
    assert np.linalg.eigvalsh(W).min()>0
    static=gramian(np.repeat(np.eye(3)[None,:,:],len(t),axis=0),H,R,valid,[.5,.5,.1])
    assert np.linalg.matrix_rank(static)==2
    np.testing.assert_allclose(gramian(transitions,H,R,np.zeros(len(t),bool),[.5,.5,.1]),0)
    with pytest.raises(ValueError):gramian(transitions,H,R,valid,[.5,0,.1])


def test_moments_distinguish_mean_bias_from_covariance():
    m,C,M=straight_line_error_moments(np.array([0.,1.,2.]),1.,.2,.04,.003,.0004)
    assert m[-1,0]==pytest.approx(-.4)
    assert m[-1,1]==pytest.approx(-.08)
    assert C[-1,1,1]/C[1,1,1]==pytest.approx(8.)
    np.testing.assert_allclose(M-C,m[:,:,None]*m[:,None,:])
    assert np.linalg.eigvalsh(C[-1]).min()>0
    bound=bounded_position_error(np.array([0.,1.,2.]),.1,.05,1.,.2,.04)
    np.testing.assert_allclose(bound,[.1,.37,.68])


def test_se2_rpe_unchanged_under_global_rigid_transform():
    t=np.linspace(0,5,11);ref=np.column_stack([t,np.sin(t),.1*t])
    angle=.7;c,s=np.cos(angle),np.sin(angle);rot=np.array([[c,-s],[s,c]])
    est=ref.copy();est[:,:2]=ref[:,:2]@rot.T+np.array([3.,-2.]);est[:,2]+=angle
    r=E.compute_rpe(est,ref,t)
    assert r['rpe_pair_count']==9
    assert r['rpe_translation_rmse']<1e-12 and r['rpe_rotation_rmse_rad']<1e-12


def test_rpe_lag_translation_bias_and_no_pairs():
    t=np.linspace(0,5,11);ref=np.column_stack([t,np.zeros(len(t)),np.zeros(len(t))]);est=ref.copy();est[:,0]*=1.1
    assert E.compute_rpe(est,ref,t)['rpe_translation_rmse']==pytest.approx(.1)
    r=E.compute_rpe(est[:2],ref[:2],t[:2],delta_sec=1)
    assert r['rpe_pair_count']==0 and r['rpe_translation_rmse'] is None
    with pytest.raises(ValueError):E.compute_rpe(est,ref,t[::-1])
