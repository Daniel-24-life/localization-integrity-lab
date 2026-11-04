"""Unit tests for CandidateBankV2 and ablation mechanisms."""
import copy
import numpy as np
import pytest

from src.data.synthetic_generator import SyntheticTrajectoryGenerator
from src.faults.temporal import wheel_bias_profile, inject_temporal_wheel_bias
from src.filters.bias_model_bank import BiasModelBank
from src.filters.candidate_bank_v2 import CandidateBankV2
from src.evaluation.run_artifacts import load_config


def test_candidate_v2_backward_compatibility():
    """When all switches are disabled, V2 must be array-identical to BiasModelBank (V1)."""
    cfg = load_config()
    cfg['project']['random_seed'] = 42
    cfg['simulation']['random_seed'] = 42
    
    gen = SyntheticTrajectoryGenerator(cfg)
    clean = gen.generate('circle')
    
    bank_v1 = BiasModelBank(cfg)
    res_v1 = bank_v1.run_sequence(clean)
    
    bank_v2_off = CandidateBankV2(
        cfg,
        enable_cusum=False,
        enable_degradation=False,
        enable_se2=False,
    )
    res_v2_off = bank_v2_off.run_sequence(clean)
    
    # Check states, covariances, weights, pl are array-identical
    np.testing.assert_allclose(res_v1['state'], res_v2_off['state'], atol=1e-12)
    np.testing.assert_allclose(res_v1['covariance'], res_v2_off['covariance'], atol=1e-12)
    np.testing.assert_allclose(res_v1['model_weights'], res_v2_off['model_weights'], atol=1e-12)
    np.testing.assert_allclose(res_v1['pl'], res_v2_off['pl'], atol=1e-12)


def test_candidate_v2_degradation_alert():
    """Prolonged outage exceeding tau_crit must trigger degradation alert."""
    cfg = load_config()
    cfg['project']['random_seed'] = 42
    cfg['simulation']['random_seed'] = 42
    
    gen = SyntheticTrajectoryGenerator(cfg)
    clean = gen.generate('straight')
    
    # Introduce a 5-second GNSS gap between t=10 and t=15
    data_gap = copy.deepcopy(clean)
    mask = (data_gap.timestamp >= 10.0) & (data_gap.timestamp <= 15.0)
    data_gap.gnss_valid[mask] = False
    
    # V2 with degradation enabled
    bank_v2 = CandidateBankV2(cfg, enable_degradation=True, tau_crit=1.5, alert_limit=0.5)
    res = bank_v2.run_sequence(data_gap)
    
    # Check that after t=11.6s, alert is triggered
    alert_mask = (data_gap.timestamp >= 11.6) & (data_gap.timestamp <= 15.0)
    assert np.all(res['system_alert'][alert_mask]), "Degradation alert must fire during prolonged outage"
    assert np.all(res['pl'][alert_mask] >= 0.5), "PL must exceed AL during degradation alert"


def test_candidate_v2_cusum_recovery():
    """CUSUM reset must accelerate post-fault recovery."""
    cfg = load_config()
    cfg['project']['random_seed'] = 42
    cfg['simulation']['random_seed'] = 42
    
    gen = SyntheticTrajectoryGenerator(cfg)
    clean = gen.generate('straight')
    
    # Inject slip between 10s and 20s
    prof = wheel_bias_profile(clean.timestamp, mode='constant', peak=0.20, start=10.0, duration=10.0)
    faulty = inject_temporal_wheel_bias(clean, prof)
    
    # Run V1
    bank_v1 = BiasModelBank(cfg)
    res_v1 = bank_v1.run_sequence(faulty)
    
    # Run V2 with CUSUM
    bank_v2 = CandidateBankV2(cfg, enable_cusum=True, enable_degradation=False, enable_se2=False)
    res_v2 = bank_v2.run_sequence(faulty)
    
    # Check that after recovery onset (e.g. t=24.0s), V2 healthy weight recovers much faster than V1
    idx_24 = np.searchsorted(faulty.timestamp, 24.0)
    assert res_v2['model_weights'][idx_24, 0] > 0.90, "V2 must rapidly recover healthy weight to > 0.90"
    assert res_v2['model_weights'][idx_24, 0] > res_v1['model_weights'][idx_24, 0] + 0.50, \
        "V2 healthy weight must significantly exceed V1 during recovery phase"
