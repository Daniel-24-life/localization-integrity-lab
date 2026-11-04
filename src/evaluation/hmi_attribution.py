"""Symmetric accounting of fixed error/alarm masks; not causal identification."""
import numpy as np


def decompose_hmi_change(exceeded_baseline,alarm_baseline,exceeded_candidate,alarm_candidate,weights):
    masks=[np.asarray(x) for x in [exceeded_baseline,alarm_baseline,exceeded_candidate,alarm_candidate]]
    w=np.asarray(weights,dtype=float)
    if w.ndim!=1 or not len(w) or not np.isfinite(w).all() or np.any(w<0) or w.sum()<=0:
        raise ValueError('Finite nonnegative weights with positive total required')
    if any(x.shape!=w.shape or x.dtype!=np.bool_ for x in masks):
        raise ValueError('Aligned boolean masks required')
    e0,a0,e1,a1=masks
    rate=lambda x:float(w@x/w.sum())
    h00=rate(e0&~a0);h10=rate(e1&~a0);h01=rate(e0&~a1);h11=rate(e1&~a1)
    error_effect=.5*((h00-h10)+(h01-h11))
    alarm_effect=.5*((h00-h01)+(h10-h11))
    return dict(baseline_hmi=h00,candidate_hmi=h11,mixed_candidate_error_baseline_alarm=h10,
        mixed_baseline_error_candidate_alarm=h01,reduction=h00-h11,
        error_accounting_contribution=error_effect,alarm_accounting_contribution=alarm_effect,
        decomposition_residual=(h00-h11)-error_effect-alarm_effect)
