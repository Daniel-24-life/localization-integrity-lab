"""Candidate additive wheel-measurement bias model; no estimator gains claimed."""
import numpy as np


def _inputs(state, control):
    x, u = np.asarray(state, dtype=float), np.asarray(control, dtype=float)
    if x.shape != (4,) or u.shape != (2,) or not np.isfinite(x).all() or not np.isfinite(u).all():
        raise ValueError('Expected finite [x,y,theta,b_v] and [v_measured,omega]')
    return x, u


def bias_ode(state, control):
    x, u = _inputs(state, control)
    c = u[0]-x[3]
    return np.array([c*np.cos(x[2]), c*np.sin(x[2]), u[1], 0.])


def bias_jacobian(state, control):
    x, u = _inputs(state, control)
    c = u[0]-x[3]; sn, cs = np.sin(x[2]), np.cos(x[2])
    A = np.zeros((4,4))
    A[0,2], A[1,2] = -c*sn, c*cs
    A[0,3], A[1,3] = -cs, -sn
    return A


def frozen_bias_diffusion_covariance(state, control, dt, density):
    """Exact Q for additive bias Brownian motion under frozen local Jacobian.

    density is variance rate of b_v, (m/s)^2/s. This is distinct from the
    existing per-input-sample q_v/q_omega standard deviations.
    """
    A = bias_jacobian(state, control)
    if not np.isfinite([dt,density]).all() or dt<=0 or density<0:
        raise ValueError('Positive dt and nonnegative bias variance density required')
    g = np.array([0.,0.,0.,1.]); a = A@g
    return density*(dt*np.outer(g,g)+dt**2/2*(np.outer(a,g)+np.outer(g,a))+dt**3/3*np.outer(a,a))


def frozen_position_observability(state, control):
    """[H; H A] local frozen model; not a global nonlinear proof."""
    A = bias_jacobian(state, control)
    H = np.eye(4)[:2]
    return np.vstack([H,H@A])
