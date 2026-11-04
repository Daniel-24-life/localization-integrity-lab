import numpy as np
import pandas as pd
import pytest
from src.evaluation.paired_seed_statistics import paired_rows, seed_summary

def fixture_frame(n=12):
    return pd.DataFrame([dict(seed=s,trajectory='straight',severity=c,dropout_sec=0,model=m,
        hmi=.1+.001*s-(.02 if m=='C_bank_v2' else 0),safe_availability=.7,false_alarm=.1,ate_rmse=.2)
        for s in range(n) for c in [0,.1] for m in ['C_bank_v1','C_bank_v2']])

def test_paired_constant_effect_and_shuffle():
    p=paired_rows(fixture_frame().sample(frac=1,random_state=4))
    r=seed_summary(p,'hmi',resamples=1000)
    assert r['seed_count']==12 and r['paired_conditions']==24
    np.testing.assert_allclose(r['ci95'],[-.02,-.02],atol=1e-15)

@pytest.mark.parametrize('failure',['missing','duplicate','nan','different_grid'])
def test_fail_closed(failure):
    df=fixture_frame()
    if failure=='missing': df=df.iloc[1:]
    if failure=='duplicate': df=pd.concat([df,df.iloc[:1]])
    if failure=='nan': df.loc[0,'hmi']=np.nan
    if failure=='different_grid': df=df.drop(df.index[:2])
    with pytest.raises(ValueError): paired_rows(df)

def test_two_seeds_not_narrow_confidence():
    r=seed_summary(paired_rows(fixture_frame(2)),'hmi')
    assert r['ci95'] is None and r['ci_status']=='insufficient_seed_clusters'

def test_zero_baseline_relative_is_unknown():
    df=fixture_frame(); df['hmi']=0
    r=seed_summary(paired_rows(df),'hmi',resamples=1000)
    assert r['relative_reduction'] is None and r['relative_reduction_ci95'] is None
    assert r['ci_status']=='degenerate_empirical_seed_distribution'
