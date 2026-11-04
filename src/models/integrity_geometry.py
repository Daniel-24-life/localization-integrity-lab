"""Research-only linear measurement geometry; not an EKF protection level.

Fault-only sensitivity under an explicit fault subspace. Noise, priors,
linearization validity and time evolution require separate treatment.
"""
import numpy as np


def fault_sensitivity(H, R, fault_basis, state_direction, residual_budget=1., rtol=1e-10):
    """max |a' delta_x| subject to ||Q L^-1 F u||^2 <= budget.

    SVD exposes an unbounded direction rather than regularizing away a nullspace.
    The returned witness uses the supplied fault basis coefficients.
    """
    H,R,F,a=map(lambda x:np.asarray(x,dtype=float),(H,R,fault_basis,state_direction))
    if H.ndim!=2 or F.ndim!=2 or F.shape[0]!=H.shape[0] or R.shape!=(H.shape[0],H.shape[0]) or a.shape!=(H.shape[1],):
        raise ValueError('Incompatible measurement/state/fault dimensions')
    if min(H.shape)==0 or F.shape[1]==0 or not all(np.isfinite(x).all() for x in [H,R,F,a]) or not np.allclose(R,R.T):
        raise ValueError('Finite nonempty matrices and symmetric covariance required')
    if not np.isfinite([residual_budget,rtol]).all() or residual_budget<0 or not 0<rtol<1:
        raise ValueError('Invalid budget/tolerance')
    L=np.linalg.cholesky(R)
    A=np.linalg.solve(L,H);Fw=np.linalg.solve(L,F)
    U,s,Vt=np.linalg.svd(A,full_matrices=False)
    if len(s)<H.shape[1] or s[-1]<=rtol*s[0]:
        raise ValueError('State not identifiable from these measurements alone')
    estimator=(Vt.T/s)@U.T
    Q=np.eye(H.shape[0])-U@U.T
    G=Q@Fw;d=a@estimator@Fw
    _,sg,Vg=np.linalg.svd(G,full_matrices=True)
    # Absolute scale from whitened fault basis prevents numerical projection
    # residuals near 1e-16 being mistaken for actual detectable information.
    tolerance=rtol*max(float(np.linalg.norm(Fw,ord=2)),np.finfo(float).tiny)
    rank=int(np.sum(sg>tolerance));V=Vg.T
    null_component=V[:,rank:].T@d
    unbounded=np.linalg.norm(null_component)>rtol*max(float(np.linalg.norm(d)),np.finfo(float).tiny)
    result=dict(unbounded=bool(unbounded),fault_residual_rank=rank,
        state_rank=len(s),residual_degrees_of_freedom=H.shape[0]-H.shape[1],
        residual_projector=Q,state_fault_map=estimator@Fw,
        singular_values=sg,scope='fault-only linear WLS; no probabilistic integrity guarantee')
    if unbounded:
        direction=V[:,rank:]@null_component/np.linalg.norm(null_component)
        result.update(bound=np.inf,witness_coefficients=direction,
            witness_residual_norm=float(np.linalg.norm(G@direction)),
            witness_state_component=float(d@direction))
    else:
        projected=V[:,:rank].T@d
        slope=float(np.linalg.norm(projected/sg[:rank])) if rank else 0.
        witness=np.zeros(F.shape[1]) if slope==0 else np.sqrt(residual_budget)/slope*(V[:,:rank]@(projected/sg[:rank]**2))
        result.update(bound=np.sqrt(residual_budget)*slope,witness_coefficients=witness,
            witness_residual_norm=float(np.linalg.norm(G@witness)),
            witness_state_component=float(d@witness))
    return result
