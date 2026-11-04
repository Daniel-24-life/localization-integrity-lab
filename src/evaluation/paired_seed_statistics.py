"""Paired method comparisons with seed-level resampling, not row pseudoreplication."""
import numpy as np
import pandas as pd

KEYS = ['seed', 'trajectory', 'severity', 'dropout_sec']
METRICS = ['hmi', 'safe_availability', 'false_alarm', 'ate_rmse']


def paired_rows(frame, baseline='C_bank_v1', candidate='C_bank_v2'):
    """Reject ambiguous or incomplete method pairs instead of silently dropping rows."""
    required = KEYS + ['model'] + METRICS
    if not set(required).issubset(frame.columns):
        raise ValueError('Missing comparison columns')
    df = frame[frame.model.isin([baseline, candidate])].copy()
    if df.empty or df[KEYS + ['model']].isna().any().any():
        raise ValueError('Empty or missing keys')
    if df.duplicated(KEYS + ['model']).any():
        raise ValueError('Duplicate method-condition rows')
    if not np.isfinite(df[METRICS].to_numpy(float)).all():
        raise ValueError('Nonfinite metric; unknown is not zero')
    if not ((df[['hmi','safe_availability','false_alarm']] >= 0).all().all() and
            (df[['hmi','safe_availability','false_alarm']] <= 1).all().all()):
        raise ValueError('Duration ratios must be in [0,1]')
    a, b = (df[df.model == m].set_index(KEYS)[METRICS].sort_index() for m in [baseline,candidate])
    if not a.index.equals(b.index):
        raise ValueError('Unmatched method-condition keys')
    # Every seed must have the same grid; otherwise seed means target different experiments.
    grids = [set(g.index.droplevel('seed')) for _,g in a.groupby(level='seed')]
    if any(g != grids[0] for g in grids[1:]):
        raise ValueError('Different condition grid across seeds')
    return a.add_suffix('_baseline').join(b.add_suffix('_candidate')).reset_index()


def seed_summary(pairs, metric, resamples=10000, random_seed=20261102, minimum_seeds=10):
    """Conditional percentile intervals; minimum is a reporting policy, not a theorem."""
    if metric not in METRICS or pairs.empty:
        raise ValueError('Unknown metric or empty subgroup')
    if resamples < 100 or minimum_seeds < 2:
        raise ValueError('Invalid resampling configuration')
    means = pairs.groupby('seed')[[metric+'_baseline',metric+'_candidate']].mean()
    a,b = means.to_numpy().T
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Nonfinite seed aggregate')
    delta = b-a
    result = {'seed_count':len(a),'paired_conditions':len(pairs),'baseline_mean':float(a.mean()),
              'candidate_mean':float(b.mean()),'delta_candidate_minus_baseline':float(delta.mean()),
              'ci95':None,'ci_status':'insufficient_seed_clusters',
              'seed_delta_range':[float(delta.min()),float(delta.max())],
              'relative_reduction':float(1-b.mean()/a.mean()) if a.mean()>0 else None,
              'relative_reduction_ci95':None,'bootstrap_valid_relative_fraction':None}
    if len(a)>=minimum_seeds:
        idx=np.random.default_rng(random_seed).integers(0,len(a),(resamples,len(a)))
        da=a[idx].mean(axis=1); db=b[idx].mean(axis=1)
        result['ci95']=np.quantile(db-da,[.025,.975]).tolist()
        result['ci_status']='conditional_percentile_seed_bootstrap' if np.ptp(delta)>0 else 'degenerate_empirical_seed_distribution'
        valid=da>0
        result['bootstrap_valid_relative_fraction']=float(valid.mean())
        if valid.all(): result['relative_reduction_ci95']=np.quantile(1-db/da,[.025,.975]).tolist()
    return result
