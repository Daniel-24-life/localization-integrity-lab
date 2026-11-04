import numpy as np
import pytest
from src.models.integrity_geometry import fault_sensitivity


def test_restricted_single_measurement_fault_exact_bound():
    r=fault_sensitivity([[1.],[1.]],np.eye(2),[[1.],[0.]],[1.],residual_budget=9.)
    assert not r['unbounded']
    assert r['bound']==pytest.approx(3/np.sqrt(2))
    assert r['witness_residual_norm']**2==pytest.approx(9.)
    assert abs(r['witness_state_component'])==pytest.approx(r['bound'])
    Q=r['residual_projector'];np.testing.assert_allclose(Q@Q,Q,atol=1e-14)


def test_coordinated_fault_detects_unbounded_nullspace():
    r=fault_sensitivity([[1.],[1.]],np.eye(2),np.eye(2),[1.])
    assert r['unbounded'] and np.isinf(r['bound'])
    assert r['witness_residual_norm']<1e-14
    assert abs(r['witness_state_component'])>.5
    # Reparameterizing the same unrestricted fault space cannot make it bounded.
    tiny=fault_sensitivity([[1.],[1.]],np.eye(2),np.eye(2)*1e-12,[1.])
    assert tiny['unbounded'] and np.isinf(tiny['bound'])


def test_unobservable_single_frame_and_nonpositive_covariance_rejected():
    with pytest.raises(ValueError,match='not identifiable'):
        fault_sensitivity([[1,0,0],[0,1,0]],np.eye(2),np.eye(2),[1,0,0])
    with pytest.raises(np.linalg.LinAlgError):
        fault_sensitivity([[1.],[1.]],np.zeros((2,2)),np.eye(2),[1.])


def test_bound_invariant_to_measurement_coordinates_and_fault_parameter_scale():
    H=np.array([[1.,0],[0,1],[1,1],[1,-1]])
    F=np.eye(4)[:,:1];R=np.diag([1.,2,3,4]);a=[1.,0]
    r=fault_sensitivity(H,R,F,a,2.)
    T=np.diag([2.,.5,3.,4.])
    r2=fault_sensitivity(T@H,T@R@T.T,T@F*7,a,2.)
    assert r2['bound']==pytest.approx(r['bound'])
