"""Validated-subproblem building blocks for dE=b dt+sqrt(q_c)dW.

Two absorbing endpoints at +/-AL. Interior nodal probability masses evolve
under a conservative substochastic generator. This is not an HMI solver.
"""
import numpy as np
from scipy.sparse import diags, eye
from scipy.sparse.linalg import factorized, spsolve
from scipy.special import ndtr


def _parameters(drift, variance_rate, alert_limit, initial_error):
    if not np.isfinite([drift, variance_rate, alert_limit, initial_error]).all():
        raise ValueError("Parameters must be finite")
    if variance_rate < 0 or alert_limit <= 0 or abs(initial_error) >= alert_limit:
        raise ValueError("Require q_c>=0, AL>0 and strictly interior initial error")


def mean_exit_time(drift, variance_rate, alert_limit=.5, initial_error=0.):
    """Exact finite-interval mean exit time; variance_rate has units m^2/s."""
    _parameters(drift, variance_rate, alert_limit, initial_error)
    a, x, b, q = alert_limit, initial_error, drift, variance_rate
    if q == 0:
        if b == 0:
            return float("inf")
        return float((a-np.sign(b)*x)/abs(b))
    if b == 0:
        return float((a*a-x*x)/q)
    if b < 0:  # reflect the problem to avoid exponential overflow
        return mean_exit_time(-b, q, a, -x)
    kappa = q/2
    pe = a*b/kappa
    if abs(pe) < 1e-5:
        # Small-drift expansion avoids cancellation in the exit-probability formula.
        return float((a*a-x*x)/(2*kappa) + b*x*(x*x-a*a)/(6*kappa*kappa))
    right_exit_probability = np.expm1(-b*(x+a)/kappa)/np.expm1(-2*b*a/kappa)
    return float((2*a*right_exit_probability-x-a)/b)


def generator(drift, variance_rate, alert_limit=.5, n_intervals=400):
    """Column generator K; missing boundary transitions lose interior mass."""
    _parameters(drift, variance_rate, alert_limit, 0.)
    if not isinstance(n_intervals, (int, np.integer)) or n_intervals < 4:
        raise ValueError("n_intervals must be an integer >=4")
    h = 2*alert_limit/n_intervals
    x = np.linspace(-alert_limit, alert_limit, n_intervals+1)[1:-1]
    r_plus = variance_rate/(2*h*h)+max(drift,0)/h
    r_minus = variance_rate/(2*h*h)+max(-drift,0)/h
    n = len(x)
    K = diags([np.full(n-1,r_plus),np.full(n,-r_plus-r_minus),np.full(n-1,r_minus)],[-1,0,1],format="csc")
    return x, K, r_minus, r_plus


def initial_mass(x, initial_error, alert_limit):
    """Unit point mass approximation; near-boundary grid displacement is reported."""
    mass=np.zeros(len(x))
    if initial_error<=x[0]:
        mass[0]=1.
    elif initial_error>=x[-1]:
        mass[-1]=1.
    else:
        index=np.searchsorted(x,initial_error,side="right")-1
        frac=(initial_error-x[index])/(x[index+1]-x[index])
        mass[index],mass[index+1]=1-frac,frac
    return mass,0.,0.


def solve_absorbing_drift_diffusion(drift, variance_rate, alert_limit=.5,
                                   initial_error=0., t_max=6., dt=.005,
                                   n_intervals=400, snapshot_count=9):
    _parameters(drift, variance_rate, alert_limit, initial_error)
    if not np.isfinite([t_max,dt]).all() or t_max <= 0 or dt <= 0:
        raise ValueError("t_max and dt must be positive and finite")
    if not isinstance(snapshot_count,int) or snapshot_count<2:
        raise ValueError("snapshot_count must be an integer >=2")
    x,K,left_rate,right_rate=generator(drift,variance_rate,alert_limit,n_intervals)
    mass,left0,right0=initial_mass(x,initial_error,alert_limit)
    initial_grid_mean=float(mass@x)
    n_steps=int(np.ceil(t_max/dt)); actual_dt=t_max/n_steps
    times=np.linspace(0,t_max,n_steps+1)
    solve=factorized(eye(len(x),format="csc")-actual_dt*K)
    S=np.empty(n_steps+1); L=np.empty_like(S); R=np.empty_like(S)
    S[0],L[0],R[0]=mass.sum(),left0,right0
    selected=set(np.linspace(0,n_steps,min(snapshot_count,n_steps+1),dtype=int))
    snapshots=[mass.copy()]; snap_times=[0.]
    minimum=float(mass.min()); max_balance=abs(S[0]+L[0]+R[0]-1)
    for i in range(1,n_steps+1):
        mass=solve(mass)
        S[i]=mass.sum()
        L[i]=L[i-1]+actual_dt*left_rate*mass[0]
        R[i]=R[i-1]+actual_dt*right_rate*mass[-1]
        minimum=min(minimum,float(mass.min()))
        max_balance=max(max_balance,abs(S[i]+L[i]+R[i]-1))
        if minimum < -1e-12 or S[i]>S[i-1]+1e-10 or max_balance>1e-8:
            raise RuntimeError("Nonpositivity/monotonicity/mass-balance failure; no clipping applied")
        if i in selected:
            snapshots.append(mass.copy());snap_times.append(times[i])
    exact_mean=mean_exit_time(drift,variance_rate,alert_limit,initial_error)
    discrete_mean=float("inf") if drift==0 and variance_rate==0 else float(initial_mass(x,initial_error,alert_limit)[0] @ spsolve(-K.T,np.ones(len(x))))
    return {"x":x,"times":times,"survival":S,"first_exit_cdf":1-S,
            "left_absorbed":L,"right_absorbed":R,"snapshot_times":np.array(snap_times),
            "mass_snapshots":np.array(snapshots),"density_snapshots":np.array(snapshots)/(2*alert_limit/n_intervals),
            "min_mass":minimum,"max_mass_balance_error":max_balance,
            "initial_boundary_interpolation_mass":left0+right0,
            "initial_grid_mean":initial_grid_mean,"initial_grid_mean_error":initial_grid_mean-initial_error,
            "mean_exit_time_exact":exact_mean,"mean_exit_time_discrete":discrete_mean,
            "actual_dt":actual_dt,"spatial_step":2*alert_limit/n_intervals,
            "numerical_diffusion_leading_term":abs(drift)*alert_limit/n_intervals}


def brownian_survival_centered(times, variance_rate, alert_limit=.5, terms=2000):
    """Absorbing spectral series for b=0,e0=0. t>0 must be resolved by terms."""
    times=np.asarray(times,dtype=float)
    if not np.isfinite(times).all() or np.any(times<0) or variance_rate<=0 or alert_limit<=0 or not isinstance(terms,int) or terms<1:
        raise ValueError("Require t>=0,q_c>0,AL>0")
    n=np.arange(terms); odd=2*n+1
    values=np.ones_like(times)
    for index in np.ndindex(times.shape):
        t=times[index]
        if t>0:
            if np.exp(-odd[-1]**2*np.pi**2*variance_rate*t/(8*alert_limit**2))>1e-10:
                raise ValueError("Insufficient spectral terms at this small time")
            values[index]=4/np.pi*np.sum((-1.)**n/odd*np.exp(-odd**2*np.pi**2*variance_rate*t/(8*alert_limit**2)))
    return values


def unabsorbed_outside_probability(times, drift, variance_rate, alert_limit=.5, initial_error=0.):
    _parameters(drift,variance_rate,alert_limit,initial_error)
    times=np.asarray(times,dtype=float)
    if not np.isfinite(times).all() or np.any(times<0):
        raise ValueError("Require finite nonnegative times")
    result=np.zeros_like(times)
    if variance_rate==0:
        return (np.abs(initial_error+drift*times)>alert_limit).astype(float)
    valid=times>0;sd=np.sqrt(variance_rate*times[valid]);mean=initial_error+drift*times[valid]
    result[valid]=ndtr((-alert_limit-mean)/sd)+ndtr((mean-alert_limit)/sd)
    return result
