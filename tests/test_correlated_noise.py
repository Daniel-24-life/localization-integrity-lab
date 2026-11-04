import numpy as np
import pytest
from src.evaluation.correlated_noise import fit_models,forecast,evaluate_segment

def test_frozen_parameters_and_positive_covariance():
    rng=np.random.default_rng(3);x=np.cumsum(rng.normal(size=(200,2)),axis=0)
    p=fit_models(x[:100]);before=p['phi'].copy()
    rows,_=evaluate_segment(x,150,200,p)
    np.testing.assert_array_equal(p['phi'],before)
    assert len(rows)==12
    for h in [1,5,20]:
        _,c=forecast('diagonal_ou',x[150],h,p)
        assert np.linalg.eigvalsh(c).min()>0

def test_ou_closed_form_mean_and_covariance_composition():
    p={'mean':np.array([1.,2.]),'phi':np.array([.8,.9]),'innovation_covariance':np.array([[.2,.01],[.01,.3]])}
    m,c=forecast('diagonal_ou',np.array([3.,4.]),2,p)
    np.testing.assert_allclose(m,p['mean']+p['phi']**2*2)
    d=np.diag(p['phi']);np.testing.assert_allclose(c,p['innovation_covariance']+d@p['innovation_covariance']@d)

def test_unidentifiable_and_unknown_rejected():
    with pytest.raises(ValueError):fit_models(np.ones((30,2)))
    with pytest.raises(ValueError):forecast('missing',np.zeros(2),1,{})


def test_calibration_does_not_read_test_tail():
    from src.evaluation.correlated_noise import calibrate_covariance
    rng=np.random.default_rng(2);x=np.cumsum(rng.normal(size=(300,2)),axis=0)
    p=fit_models(x[:150]);a=calibrate_covariance(x,150,225,p)
    x[225:]+=1000
    assert calibrate_covariance(x,150,225,p)==a
    p['validation_covariance_scale']=a
    mean,cov=forecast('diagonal_ou',x[150],5,p)
    adjusted,large=forecast('ou_calibrated',x[150],5,p)
    np.testing.assert_allclose(mean,adjusted)
    np.testing.assert_allclose(large,cov*a)
