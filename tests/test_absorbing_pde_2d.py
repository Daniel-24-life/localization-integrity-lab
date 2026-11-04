"""Unit tests for 2D isotropic absorbing boundary Fokker-Planck PDE and Bessel spectral solutions."""
import numpy as np
import pytest

from src.models.absorbing_pde_2d import (
    bessel_survival_2d,
    bessel_first_exit_cdf_2d,
    mean_exit_time_2d,
    coupled_monte_carlo_2d,
    solve_absorbing_pde_2d_fvm,
)


def test_bessel_survival_properties():
    qc = 0.01
    AL = 0.5
    times = np.array([0.1, 1.0, 5.0, 10.0, 20.0, 40.0])
    s_t = bessel_survival_2d(times, qc=qc, AL=AL, n_terms=300)

    # Monotonically non-increasing
    assert np.all(np.diff(s_t) <= 1e-6), "Survival probability must be non-increasing"
    # Bound between 0 and 1
    assert np.all((s_t >= 0.0) & (s_t <= 1.0))
    # Near t=0, survival should be very close to 1
    s_0p1 = bessel_survival_2d(np.array([0.1]), qc=qc, AL=AL)
    assert s_0p1[0] > 0.999
    # Long time decay
    assert s_t[-1] < 0.05


def test_mean_exit_time_halving():
    # 2D mean exit time is precisely AL^2 / (2 qc)
    AL = 0.5
    qc = 0.01
    t_2d = mean_exit_time_2d(qc=qc, AL=AL)
    assert np.isclose(t_2d, 12.5)

    # Compared with 1D exit time AL^2 / qc = 25.0s
    t_1d = (AL ** 2) / qc
    assert np.isclose(t_2d, 0.5 * t_1d)


def test_cdf_vs_monte_carlo_wilson():
    qc = 0.01
    AL = 0.5
    checkpoints = np.array([2.0, 5.0, 10.0, 15.0])
    mc_res = coupled_monte_carlo_2d(qc=qc, checkpoints=checkpoints, AL=AL, n_paths=20000, dt=0.002, seed=42)

    bessel_cdf = bessel_first_exit_cdf_2d(checkpoints, qc=qc, AL=AL, n_terms=300)

    # Check that Bessel CDF falls within (or extremely close to due to discretization) Wilson CI
    diff = np.abs(bessel_cdf - mc_res['mc_cdf'])
    # Monte Carlo discrete-time Euler path introduces a small O(dt^0.5) boundary overestimation
    assert np.all(diff < 0.015), f"Max difference between Bessel series and MC is {np.max(diff):.5f}"


def test_fvm_vs_bessel_spectral():
    qc = 0.01
    AL = 0.5
    fvm_res = solve_absorbing_pde_2d_fvm(qc=qc, AL=AL, n_cells=200, t_max=10.0, dt=0.005)
    checkpoints = np.array([1.0, 2.5, 5.0, 7.5, 10.0])
    fvm_cdf = np.interp(checkpoints, fvm_res['times'], fvm_res['first_exit_cdf'])
    bessel_cdf = bessel_first_exit_cdf_2d(checkpoints, qc=qc, AL=AL, n_terms=300)
    
    max_err = np.max(np.abs(fvm_cdf - bessel_cdf))
    assert max_err < 0.015, f"FVM vs Bessel spectral max discrepancy {max_err:.5f} exceeds 0.015"


def test_bessel_off_center_r0():
    qc = 0.01
    AL = 0.5
    # When r0 = 0.25 (halfway to boundary), mean exit time must be (0.5^2 - 0.25^2)/(2*0.01) = 0.1875 / 0.02 = 9.375s
    t_mean_off = mean_exit_time_2d(qc=qc, r0=0.25, AL=AL)
    assert np.isclose(t_mean_off, 9.375)
    
    # Survival probability at r0 > 0 must be strictly less than at r0 = 0 (closer to absorbing boundary)
    times = np.array([1.0, 5.0, 10.0])
    s_center = bessel_survival_2d(times, qc=qc, r0=0.0, AL=AL)
    s_off = bessel_survival_2d(times, qc=qc, r0=0.25, AL=AL)
    assert np.all(s_off < s_center), "Off-center initial position must exit faster than origin"
