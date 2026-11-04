import numpy as np
import pytest
from src.filters.colored_position_ekf import ColoredPositionEKF
from src.evaluation.run_artifacts import load_config

def make(tau=(100.,200.)):
    c=load_config();c['filter']['position_lever_flu_m']=[0,0,0]
    return ColoredPositionEKF(c,np.zeros(3),tau,np.eye(2)*.001)

def test_bias_state_responds_to_innovation_and_covariance_stays_psd():
    f=make()
    for _ in range(20):f.predict(1.,.1,.05);assert f.update([2.,1.])>=0
    assert np.linalg.norm(f.x[3:])>0
    assert np.linalg.eigvalsh(f.P).min()>0

def test_random_walk_and_ou_exact_bias_variance():
    f=make((np.inf,np.inf));old=f.P[3:,3:].copy();f.predict(0,0,.2)
    np.testing.assert_allclose(f.P[3:,3:],old+np.eye(2)*.0002)
    g=make((100.,100.));g.x[3:]=1.;g.predict(0,0,.2)
    np.testing.assert_allclose(g.x[3:],np.exp(-.2/100))

def test_reference_not_read_in_filter_loop():
    class Inputs:
        timestamp=np.arange(5)*.1;wheel_v=np.ones(5);gnss_valid=np.zeros(5,dtype=bool)
        def __len__(self):return 5
        def validate(self):pass
        def angular_velocity(self,source):return np.zeros(5)
        def __getattr__(self,name):raise AssertionError('Unexpected input access: '+name)
    out=make().run_sequence(Inputs())
    assert np.isnan(out['nis']).all()
    assert out['state'][-1,0]==pytest.approx(.4)
