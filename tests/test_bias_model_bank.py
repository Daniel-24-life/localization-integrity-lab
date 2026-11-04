import copy
import numpy as np
import pytest
from src.filters.bias_model_bank import BiasModelBank,mixture_moments
from src.data.synthetic_generator import SyntheticTrajectoryGenerator
from src.faults.injector import FaultInjector
from src.evaluation.run_artifacts import load_config


def test_covariance_contains_between_model_spread():
    x=np.zeros((2,4));x[1,0]=2
    mean,P=mixture_moments(x,np.zeros((2,4,4)),[.5,.5])
    assert mean[0]==1 and P[0,0]==1


def test_heading_wrap_and_ambiguous_case():
    x=np.zeros((2,4));x[:,2]=np.deg2rad([179.,-179.])
    mean,P=mixture_moments(x,np.zeros((2,4,4)),[.5,.5])
    assert abs(abs(mean[2])-np.pi)<1e-10
    assert P[2,2]<.001
    x[:,2]=[0.,np.pi]
    with pytest.raises(ValueError):mixture_moments(x,np.zeros((2,4,4)),[.5,.5])


def test_probabilities_missing_data_and_covariance():
    cfg=load_config();cfg['simulation']['duration']=4
    d=FaultInjector.inject_gnss_dropout(SyntheticTrajectoryGenerator(cfg).generate('straight'),1.,2.)
    r=BiasModelBank(cfg).run_sequence(d);mask=(d.timestamp>=1)&(d.timestamp<3)
    p=r['model_weights'][mask]
    np.testing.assert_array_equal(p,np.repeat(p[:1],len(p),axis=0))
    np.testing.assert_allclose(r['model_weights'].sum(axis=1),1.)
    assert np.isnan(r['nis'][mask]).all()
    assert np.linalg.eigvalsh(r['covariance']).min()>0


@pytest.mark.parametrize('learn',[False,True])
def test_fixed_weight_ablation_and_online_causality(learn):
    cfg=load_config();cfg['simulation']['duration']=3;cfg['model_bank']={'learn_weights':learn}
    d=SyntheticTrajectoryGenerator(cfg).generate('circle');changed=copy.deepcopy(d)
    changed.gt_x[:]=999;changed.gnss_x[-5:]+=10
    a=BiasModelBank(cfg).run_sequence(d);b=BiasModelBank(cfg).run_sequence(changed)
    if not learn:np.testing.assert_allclose(a['model_weights'],np.tile([.95,.05],(len(d),1)))
    for k in a:np.testing.assert_array_equal(a[k][:-5],b[k][:-5])


@pytest.mark.parametrize('options',[dict(initial_bias_probability=0),dict(posterior_uniform_floor=.5),dict(learn_weights=1)])
def test_invalid_settings(options):
    cfg=load_config();cfg['model_bank']=options
    with pytest.raises(ValueError):BiasModelBank(cfg)
