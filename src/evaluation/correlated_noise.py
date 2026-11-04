"""Frozen residual forecast models; diagnostic only, not a navigation filter.

Past reference residuals initialise forecasts. This oracle information prevents
interpreting prediction scores as deployable localisation performance.
"""
import numpy as np


def fit_models(train):
    x=np.asarray(train,dtype=float)
    if x.ndim!=2 or x.shape[1]!=2 or len(x)<20 or not np.isfinite(x).all():
        raise ValueError('Need at least 20 finite 2D training events')
    mean=x.mean(axis=0); z=x-mean
    denominator=(z[:-1]**2).sum(axis=0)
    if np.any(denominator<=1e-12):
        raise ValueError('Constant training component cannot identify correlation')
    phi_raw=(z[:-1]*z[1:]).sum(axis=0)/denominator
    # Positive stable OU-compatible candidate, declared before evaluation.
    phi=np.clip(phi_raw,1e-4,.9999)
    innovations=z[1:]-phi*z[:-1]
    covariance=np.cov(innovations,rowvar=False)
    covariance=.9*covariance+.1*np.diag(np.diag(covariance))+np.eye(2)*1e-8
    return {'mean':mean,'phi':phi,'phi_raw':phi_raw,'innovation_covariance':covariance,
            'iid_covariance':np.cov(x,rowvar=False)+np.eye(2)*1e-8,
            'random_walk_covariance':np.cov(np.diff(x,axis=0),rowvar=False)+np.eye(2)*1e-8}


def forecast(model, initial, horizon, parameters):
    if horizon<1 or int(horizon)!=horizon:
        raise ValueError('Positive integer horizon required')
    initial=np.asarray(initial,float)
    if initial.shape!=(2,) or not np.isfinite(initial).all():
        raise ValueError('Finite initial 2D residual required')
    p=parameters
    if model=='zero_fixed':return np.zeros(2),np.eye(2)*.732**2
    if model=='constant_bias':return p['mean'].copy(),p['iid_covariance'].copy()
    if model=='persistence':return initial.copy(),horizon*p['random_walk_covariance']
    if model=='ou_calibrated':
        pred,cov=forecast('diagonal_ou',initial,horizon,parameters)
        return pred,cov*parameters['validation_covariance_scale']
    if model!='diagonal_ou':raise ValueError('Unknown model')
    power=p['phi']**horizon
    prediction=p['mean']+power*(initial-p['mean'])
    covariance=np.zeros((2,2))
    for k in range(horizon):
        d=np.diag(p['phi']**k)
        covariance+=d@p['innovation_covariance']@d
    return prediction,covariance


def evaluate_segment(values, start, stop, parameters, horizons=(1,5,20)):
    x=np.asarray(values,float)
    if x.ndim!=2 or x.shape[1]!=2 or not np.isfinite(x).all() or not 1<=start<stop<=len(x):
        raise ValueError('Invalid finite sequence or chronological segment')
    rows=[]; series={}
    for h in horizons:
        # Origins also inside the segment: no fit/eval overlap within scores.
        origins=np.arange(start,stop-h)
        if len(origins)<2:raise ValueError('Segment too short for horizon')
        names=['zero_fixed','constant_bias','persistence','diagonal_ou']
        if 'validation_covariance_scale' in parameters:names.append('ou_calibrated')
        for name in names:
            prediction=[];nll=[];inside=[];mahal=[]
            for i in origins:
                pred,cov=forecast(name,x[i],h,parameters)
                delta=x[i+h]-pred
                chol=np.linalg.cholesky(cov); white=np.linalg.solve(chol,delta)
                m=float(white@white); logdet=2*np.log(np.diag(chol)).sum()
                nll.append(.5*(2*np.log(2*np.pi)+logdet+m));inside.append(m<=5.991464547)
                mahal.append(m);prediction.append(pred)
            prediction=np.asarray(prediction);error=x[origins+h]-prediction
            rows.append({'model':name,'horizon_events':h,'forecast_count':len(origins),
                         'rmse_norm_m':float(np.sqrt(np.mean(np.sum(error**2,axis=1)))),
                         'mean_nll':float(np.mean(nll)),'ellipse95_empirical_coverage':float(np.mean(inside)),
                         'mean_mahalanobis2':float(np.mean(mahal)),
                         'status':'oracle_past_reference_residual_forecast_not_navigation'})
            series[f'{name}_h{h}']=prediction
        series[f'target_indices_h{h}']=origins+h
    return rows,series


def calibrate_covariance(values,start,stop,parameters,horizons=(1,5,20)):
    """One common inflation scale fitted on validation only, preserving Q shape."""
    x=np.asarray(values,float);scales=[]
    if not 1<=start<stop<=len(x) or not np.isfinite(x).all():raise ValueError('Invalid calibration segment')
    for h in horizons:
        scores=[]
        for i in range(start,stop-h):
            pred,cov=forecast('diagonal_ou',x[i],h,parameters)
            err=x[i+h]-pred;scores.append(float(err@np.linalg.solve(cov,err)))
        if len(scores)<20:raise ValueError('Too few calibration forecasts')
        scales.append(float(np.quantile(scores,.95,method='higher')/5.991464547))
    return max(1.,*scales)
