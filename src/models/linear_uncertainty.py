"""Finite-horizon local information and straight-motion error-moment models."""
import numpy as np


def finite_observability_gramian(transitions, H, R, valid, scales):
    """Whitened discrete-observation Gramian in dimensionless coordinates.

    transitions[j] maps the initial state error to measurement time j.
    No duration multiplier: these are discrete measurements, not continuous ones.
    """
    transitions,H,R=np.asarray(transitions),np.asarray(H),np.asarray(R)
    valid=np.asarray(valid);scales=np.asarray(scales,dtype=float)
    if transitions.ndim!=3 or transitions.shape[1]!=transitions.shape[2]:
        raise ValueError("Expected (N,n,n) transitions")
    N,n,_=transitions.shape
    if H.ndim!=2 or H.shape[1]!=n or R.shape!=(H.shape[0],H.shape[0]) or valid.shape!=(N,) or valid.dtype!=np.bool_ or scales.shape!=(n,):
        raise ValueError("Gramian input shapes/valid dtype do not match")
    if not all(np.isfinite(a).all() for a in [transitions,H,R,scales]) or np.any(scales<=0) or not np.allclose(R,R.T):
        raise ValueError("Finite values, positive scales and symmetric R required")
    chol=np.linalg.cholesky(R)
    W=np.zeros((n,n));S=np.diag(scales)
    for transition in transitions[valid]:
        whitened=np.linalg.solve(chol,H@transition@S)
        W+=whitened.T@whitened
    return (W+W.T)/2


def straight_line_error_moments(times,speed,bias_v,bias_omega,qc_v,qc_omega,initial_heading_error=0.):
    """Local small-heading-error model, zero initial random covariance.

    e=true-estimate; input measurement=true+bias. qc are continuous intensities.
    This does not silently change the existing filter's per-sample noise model.
    """
    t=np.asarray(times,dtype=float)
    if t.ndim!=1 or not np.isfinite(t).all() or np.any(t<0) or not np.isfinite([speed,bias_v,bias_omega,qc_v,qc_omega,initial_heading_error]).all() or min(qc_v,qc_omega)<0:
        raise ValueError("Invalid moment-model parameters")
    mean=np.column_stack([-bias_v*t,speed*initial_heading_error*t-.5*speed*bias_omega*t*t,initial_heading_error-bias_omega*t])
    C=np.zeros((len(t),3,3));C[:,0,0]=qc_v*t;C[:,2,2]=qc_omega*t
    C[:,1,1]=speed*speed*qc_omega*t**3/3
    C[:,1,2]=C[:,2,1]=.5*speed*qc_omega*t*t
    second=C+mean[:,:,None]*mean[:,None,:]
    return mean,C,second


def bounded_position_error(times,initial_position_error,initial_heading_error,speed_bound,velocity_bias_bound,angular_bias_bound):
    t=np.asarray(times,dtype=float)
    vals=[initial_position_error,initial_heading_error,speed_bound,velocity_bias_bound,angular_bias_bound]
    if not np.isfinite(t).all() or np.any(t<0) or not np.isfinite(vals).all() or min(initial_position_error,speed_bound,velocity_bias_bound,angular_bias_bound)<0:
        raise ValueError("Invalid finite-time bound parameters")
    return initial_position_error+(velocity_bias_bound+speed_bound*abs(initial_heading_error))*t+.5*speed_bound*angular_bias_bound*t*t
