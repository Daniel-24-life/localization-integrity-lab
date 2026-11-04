"""Two-dimensional isotropic absorbing boundary Fokker-Planck PDE and Bessel spectral solutions.

This module provides analytical Bessel eigenfunction expansions and coupled Monte Carlo
verifications for 2D isotropic Brownian diffusion inside an absorbing disk of radius AL:
    ∂p/∂t = (qc / 2) ∇² p,  with p(AL, phi, t) = 0 and p(r, phi, 0) = δ(r) / (2π r).

Key analytical results:
1. Exact survival probability:
       S(t) = ∑_{n=1}^∞ [2 / (j_{0,n} J_1(j_{0,n}))] exp(- j_{0,n}^2 qc t / (2 AL^2))
2. Exact mean first exit time (Poisson equation solution):
       T_{2D}(0) = AL^2 / (2 qc)  [precisely 1/2 of the 1D interval exit time T_{1D} = AL^2 / qc]
"""
import numpy as np
from scipy.special import jn_zeros, jv
from scipy.stats import norm


def mean_exit_time_2d(qc: float, r0: float = 0.0, AL: float = 0.5) -> float:
    """Exact analytical mean first exit time from initial radius r0 for 2D disk of radius AL."""
    if qc <= 0 or AL <= 0:
        raise ValueError("qc and AL must be strictly positive")
    if not 0.0 <= r0 < AL:
        raise ValueError("Initial radius r0 must satisfy 0 <= r0 < AL")
    return (AL ** 2 - r0 ** 2) / (2.0 * qc)


def bessel_survival_2d(t_array: np.ndarray, qc: float, r0: float = 0.0, AL: float = 0.5, n_terms: int = 300) -> np.ndarray:
    """Calculate survival probability S(t | r0) inside disk of radius AL via Bessel spectral series.
    
    For initial position at radius r0 in [0, AL):
        S(t | r0) = sum_{n=1}^inf [2 J_0(j_{0,n} r0 / AL) / (j_{0,n} J_1(j_{0,n}))] exp(- j_{0,n}^2 qc t / (2 AL^2))
    When r0 = 0, J_0(0) = 1, which strictly reduces to the centered source expansion.
    """
    t = np.asarray(t_array, dtype=float)
    if np.any(t < 0):
        raise ValueError("Time array must be non-negative")
    if qc <= 0 or AL <= 0:
        raise ValueError("qc and AL must be strictly positive")
    if not 0.0 <= r0 < AL:
        raise ValueError("Initial radius r0 must satisfy 0 <= r0 < AL")

    zeros = jn_zeros(0, n_terms)
    j1_vals = jv(1, zeros)
    
    if r0 == 0.0:
        j0_r0 = 1.0
    else:
        j0_r0 = jv(0, zeros * r0 / AL)
        
    coeffs = 2.0 * j0_r0 / (zeros * j1_vals)

    # Compute outer product over t and zeros
    exponent = np.outer(t, -(zeros ** 2) * qc / (2.0 * AL ** 2))
    terms = coeffs * np.exp(exponent)
    s_t = np.sum(terms, axis=1)

    # Numerical safeguard: S(t) in [0, 1]
    return np.clip(s_t, 0.0, 1.0)


def bessel_first_exit_cdf_2d(t_array: np.ndarray, qc: float, r0: float = 0.0, AL: float = 0.5, n_terms: int = 300) -> np.ndarray:
    """Calculate cumulative distribution function F(t | r0) = 1 - S(t | r0) of first exit time."""
    return 1.0 - bessel_survival_2d(t_array, qc, r0=r0, AL=AL, n_terms=n_terms)


def solve_absorbing_pde_2d_fvm(
    qc: float,
    AL: float = 0.5,
    n_cells: int = 400,
    t_max: float = 6.0,
    dt: float = 0.001
) -> dict:
    """Solve 2D isotropic radial absorbing Fokker-Planck PDE via Conservative Finite Volume Method.
    
    Domain: r in [0, AL], with absorbing Dirichlet condition p(AL, t) = 0,
    and conservative reflective condition at r = 0 (zero flux).
    Initial condition: Dirac delta at origin, represented as unit mass in cell 0.
    
    Uses Crank-Nicolson implicit time-stepping with tridiagonal solver.
    """
    if qc <= 0 or AL <= 0 or t_max <= 0 or dt <= 0:
        raise ValueError("Physical and grid parameters must be positive")
        
    dr = AL / n_cells
    r_faces = np.linspace(0.0, AL, n_cells + 1)
    r_centers = 0.5 * (r_faces[:-1] + r_faces[1:])
    # Cell areas (2D polar volumes)
    cell_areas = np.pi * (r_faces[1:]**2 - r_faces[:-1]**2)
    
    # Initial state: unit point mass at origin (first cell)
    p = np.zeros(n_cells)
    p[0] = 1.0 / cell_areas[0]
    
    # Construct spatial tridiagonal operator A where V dp/dt = (qc / 2) A p
    # Interface conductances: c_{i+1/2} = 2 pi r_{i+1/2} / dr
    conductances = 2.0 * np.pi * r_faces / dr  # length n_cells + 1
    # For the outer boundary r_N = AL, distance from center to face is dr/2:
    conductances[-1] = 2.0 * np.pi * AL / (0.5 * dr)
    
    # Matrix diagonals for L: d p_i / dt = sum_j L_ij p_j
    # V_i dp_i/dt = (qc/2) * [ c_{i+1/2}(p_{i+1} - p_i) - c_{i-1/2}(p_i - p_{i-1}) ]
    diff_coeff = 0.5 * qc
    
    diag_main = np.zeros(n_cells)
    diag_upper = np.zeros(n_cells - 1)
    diag_lower = np.zeros(n_cells - 1)
    
    for i in range(n_cells):
        c_left = conductances[i]      # interface between i-1 and i
        c_right = conductances[i + 1]  # interface between i and i+1
        diag_main[i] = - diff_coeff * (c_left + c_right) / cell_areas[i]
        if i < n_cells - 1:
            diag_upper[i] = diff_coeff * c_right / cell_areas[i]
            diag_lower[i] = diff_coeff * c_right / cell_areas[i + 1]
            
    # Crank-Nicolson matrices: (I - 0.5 * dt * L) p^{n+1} = (I + 0.5 * dt * L) p^n
    # Form tridiagonal system A_lhs p^{n+1} = b_rhs
    a_sub = - 0.5 * dt * diag_lower
    b_main = 1.0 - 0.5 * dt * diag_main
    c_sup = - 0.5 * dt * diag_upper
    
    # Pre-factor Thomas algorithm coefficients
    n = n_cells
    cp = np.zeros(n - 1)
    dp = np.zeros(n)
    
    # Time stepping loop
    n_steps = int(round(t_max / dt))
    times = np.linspace(0.0, t_max, n_steps + 1)
    survival = np.zeros(n_steps + 1)
    survival[0] = 1.0
    
    # Snapshots at ~0.5s intervals
    snapshot_interval = int(round(0.5 / dt))
    snapshots = [p.copy()]
    snapshot_times = [0.0]
    
    curr_p = p.copy()
    
    for step in range(1, n_steps + 1):
        # Compute RHS: (I + 0.5 * dt * L) p^n
        rhs = curr_p + 0.5 * dt * (diag_main * curr_p)
        rhs[:-1] += 0.5 * dt * (diag_upper * curr_p[1:])
        rhs[1:] += 0.5 * dt * (diag_lower * curr_p[:-1])
        
        # Thomas algorithm solve
        # Forward sweep
        cp[0] = c_sup[0] / b_main[0]
        dp[0] = rhs[0] / b_main[0]
        for i in range(1, n - 1):
            denom = b_main[i] - a_sub[i - 1] * cp[i - 1]
            cp[i] = c_sup[i] / denom
            dp[i] = (rhs[i] - a_sub[i - 1] * dp[i - 1]) / denom
        denom_last = b_main[n - 1] - a_sub[n - 2] * cp[n - 2]
        dp[n - 1] = (rhs[n - 1] - a_sub[n - 2] * dp[n - 2]) / denom_last
        
        # Back substitution
        curr_p[n - 1] = dp[n - 1]
        for i in range(n - 2, -1, -1):
            curr_p[i] = dp[i] - cp[i] * curr_p[i + 1]
            
        # Mass in domain
        mass = np.sum(curr_p * cell_areas)
        survival[step] = max(0.0, min(1.0, mass))
        
        if step % snapshot_interval == 0:
            snapshots.append(curr_p.copy())
            snapshot_times.append(step * dt)
            
    exit_cdf = 1.0 - survival
    return {
        'times': times,
        'survival': survival,
        'first_exit_cdf': exit_cdf,
        'r_centers': r_centers,
        'r_faces': r_faces,
        'cell_areas': cell_areas,
        'snapshots': snapshots,
        'snapshot_times': snapshot_times,
        'qc': qc,
        'AL': AL,
        'dt': dt,
        'dr': dr,
    }


def coupled_monte_carlo_2d(
    qc: float,
    checkpoints: np.ndarray,
    AL: float = 0.5,
    n_paths: int = 20000,
    dt: float = 0.001,
    seed: int = 314159
) -> dict:
    """Coupled 2D Brownian Monte Carlo with Wilson confidence intervals."""
    rng = np.random.default_rng(seed)
    n_steps = int(round(checkpoints[-1] / dt))
    sigma = np.sqrt(qc * dt)

    pos_x = np.zeros(n_paths)
    pos_y = np.zeros(n_paths)
    first_exit = np.full(n_paths, np.inf)

    for step in range(1, n_steps + 1):
        pos_x += sigma * rng.standard_normal(n_paths)
        pos_y += sigma * rng.standard_normal(n_paths)
        r2 = pos_x ** 2 + pos_y ** 2
        crossed = (r2 >= AL ** 2) & np.isinf(first_exit)
        first_exit[crossed] = step * dt

    cdf_mc = np.array([(first_exit <= t).mean() for t in checkpoints])

    # Wilson 95% confidence intervals
    z = norm.ppf(1.0 - 0.05 / (2.0 * len(checkpoints)))
    denom = 1.0 + (z ** 2) / n_paths
    center = (cdf_mc + (z ** 2) / (2.0 * n_paths)) / denom
    half = z * np.sqrt(cdf_mc * (1.0 - cdf_mc) / n_paths + (z ** 2) / (4.0 * n_paths ** 2)) / denom

    return {
        'checkpoints': checkpoints,
        'mc_cdf': cdf_mc,
        'wilson_low': center - half,
        'wilson_high': center + half,
        'n_paths': n_paths,
        'dt': dt,
        'seed': seed,
        'first_exit_times': first_exit,
    }
