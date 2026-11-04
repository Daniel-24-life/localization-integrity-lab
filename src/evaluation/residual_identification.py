"""Auditable residual and diffusion diagnostics for a versioned trajectory.

This module deliberately does not alter a filter score.  The position residual is
an observation/reference diagnostic whose reference point, receiver correlation,
and antenna lever arm remain part of the evidence boundary.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from src.models.antenna_observation import antenna_observation


@dataclass(frozen=True)
class ResidualSeries:
    timestamp_s: np.ndarray
    residual_m: np.ndarray
    source_indices: np.ndarray
    metadata: dict
    wheel_point_offset_m: np.ndarray | None = None


def load_residual_series(npz_path: str | Path) -> ResidualSeries:
    """Load only valid position events and form raw observation-reference residuals.

    The arrays are taken from the versioned preparation artifact.  No centering,
    smoothing, interpolation, or outlier deletion is performed here.
    """
    path = Path(npz_path)
    sidecar = path.with_suffix(".metadata.json")
    if not sidecar.exists():
        raise ValueError("formal residual identification requires the versioned metadata sidecar")
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    required_metadata = ("reference_status", "reference_independent", "G1_G2_certified", "reference_heading_status", "position_lever_flu_m")
    missing_metadata = [name for name in required_metadata if name not in metadata]
    if missing_metadata:
        raise ValueError(f"metadata missing for formal residual identification: {missing_metadata}")
    lever = np.asarray(metadata["position_lever_flu_m"], dtype=float)
    if lever.shape != (3,) or not np.isfinite(lever).all():
        raise ValueError("position_lever_flu_m must be a finite length-3 vector")
    if metadata.get("reference_status") != "available":
        raise ValueError("an available reference is required for residual identification")
    if metadata.get("reference_heading_status") != "available":
        raise ValueError("nominal antenna projection requires an available reference heading")
    with np.load(path, allow_pickle=False) as data:
        required = ("timestamp", "gnss_x", "gnss_y", "gnss_valid", "gt_x", "gt_y", "gt_theta")
        missing = [name for name in required if name not in data]
        if missing:
            raise ValueError(f"missing arrays: {missing}")
        arrays = {name: np.asarray(data[name]) for name in required}
        n = len(arrays["timestamp"])
        if n < 4 or any(array.shape != (n,) for array in arrays.values()):
            raise ValueError("all residual input arrays must be one-dimensional with one common length")
        if data["gnss_valid"].dtype != np.bool_:
            raise ValueError("gnss_valid must be a boolean array; refusing coercion")
        timestamp = np.asarray(arrays["timestamp"], dtype=float)
        valid = np.asarray(arrays["gnss_valid"], dtype=bool)
        if not np.isfinite(timestamp).all() or not np.all(np.diff(timestamp) > 0):
            raise ValueError("timestamp must be finite and strictly increasing")
        indices = np.flatnonzero(valid)
        if len(indices) < 4:
            raise ValueError("at least four valid observation events are required")
        for name in ("gnss_x", "gnss_y", "gt_x", "gt_y", "gt_theta"):
            if not np.isfinite(np.asarray(data[name], dtype=float)[indices]).all():
                raise ValueError(f"valid event values for {name} must be finite")
        state = np.column_stack((arrays["gt_x"][indices], arrays["gt_y"][indices], arrays["gt_theta"][indices])).astype(float)
        # The stored reference is at the wheel point. Project it to the nominal
        # antenna point using the same planar observation convention as the
        # estimator. Roll/pitch and point identity remain explicit limitations.
        antenna_reference = np.vstack([antenna_observation(row, lever)[0] for row in state])
        observation = np.column_stack((arrays["gnss_x"][indices], arrays["gnss_y"][indices])).astype(float)
        residual = observation - antenna_reference
        wheel_offset = observation - state[:, :2]
        times = timestamp[indices]
    finite = np.isfinite(times) & np.isfinite(residual).all(axis=1) & np.isfinite(wheel_offset).all(axis=1)
    if not finite.all():
        raise ValueError("non-finite valid observation/reference residual")
    if len(times) < 4 or not np.all(np.diff(times) > 0):
        raise ValueError("residual event times must be finite and strictly increasing")
    return ResidualSeries(times, residual, indices, metadata, wheel_point_offset_m=wheel_offset)


def split_series(series: ResidualSeries, fraction: float = 0.5) -> tuple[ResidualSeries, ResidualSeries]:
    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction must be between zero and one")
    cut = int(np.floor(len(series.timestamp_s) * fraction))
    if cut < 2 or len(series.timestamp_s) - cut < 2:
        raise ValueError("both fit and evaluation sections need at least two events")
    def part(start: int, stop: int) -> ResidualSeries:
        wheel = None if series.wheel_point_offset_m is None else series.wheel_point_offset_m[start:stop]
        return ResidualSeries(series.timestamp_s[start:stop], series.residual_m[start:stop], series.source_indices[start:stop], series.metadata, wheel)
    return part(0, cut), part(cut, len(series.timestamp_s))


def residual_summary(residual_m: np.ndarray) -> dict:
    r = np.asarray(residual_m, dtype=float)
    if r.ndim != 2 or r.shape[1] != 2 or len(r) < 2 or not np.isfinite(r).all():
        raise ValueError("residual must be a finite (n, 2) array with n >= 2")
    mean = r.mean(axis=0)
    centered = r - mean
    cov = np.cov(r, rowvar=False, ddof=1)
    norms = np.linalg.norm(r, axis=1)
    centered_norms = np.linalg.norm(centered, axis=1)
    return {
        "count": int(len(r)),
        "mean_m": mean.tolist(),
        "raw_rmse_norm_m": float(np.sqrt(np.mean(np.sum(r * r, axis=1)))),
        "raw_component_rmse_m": np.sqrt(np.mean(r * r, axis=0)).tolist(),
        "centered_covariance_m2": cov.tolist(),
        "centered_component_std_m": np.sqrt(np.diag(cov)).tolist(),
        "raw_norm_median_m": float(np.median(norms)),
        "raw_norm_p95_m": float(np.quantile(norms, 0.95)),
        "centered_norm_rmse_m": float(np.sqrt(np.mean(centered_norms ** 2))),
        "centered_norm_p95_m": float(np.quantile(centered_norms, 0.95)),
    }


def autocorrelation(residual_m: np.ndarray, max_lag: int = 20) -> dict:
    r = np.asarray(residual_m, dtype=float)
    if r.ndim != 2 or r.shape[1] != 2 or len(r) < 4 or not np.isfinite(r).all():
        raise ValueError("at least four two-dimensional residuals are required")
    max_lag = min(int(max_lag), len(r) - 1)
    if max_lag < 1:
        raise ValueError("max_lag must be positive")
    acf = np.empty((max_lag + 1, 2), dtype=float)
    acf[0] = 1.0
    for axis in range(2):
        x = r[:, axis] - np.mean(r[:, axis])
        denom = float(np.dot(x, x))
        acf[1:, axis] = [float(np.dot(x[:-lag], x[lag:]) / denom) if denom > 0 else np.nan for lag in range(1, max_lag + 1)]
    # Initial-positive-sequence estimate.  It is a diagnostic, not an iid CI.
    tau = []
    first_nonpositive = []
    for axis in range(2):
        if not np.isfinite(acf[1, axis]):
            tau.append(float('nan'))
            first_nonpositive.append(None)
            continue
        positive = []
        first = None
        for lag, value in enumerate(acf[1:, axis], start=1):
            if not np.isfinite(value) or value <= 0:
                first = lag
                break
            positive.append(value)
        tau.append(float(max(1.0, 1.0 + 2.0 * sum(positive))))
        first_nonpositive.append(first)
    return {"max_lag_events": max_lag, "acf": acf.tolist(), "acf_at_max_lag": acf[-1].tolist(), "first_nonpositive_lag_events": first_nonpositive, "integrated_autocorrelation_time_events": tau, "effective_sample_size": (len(r) / np.asarray(tau)).tolist(), "ess_status": "truncated initial-positive diagnostic; not a certified independent sample count"}


def autocorrelation_sensitivity(residual_m: np.ndarray, max_lags: Iterable[int] = (20, 50, 100)) -> list[dict]:
    """Show how the truncated ESS changes with the declared maximum lag."""
    return [autocorrelation(residual_m, int(max_lag)) for max_lag in max_lags]


def bootstrap_sensitivity(values: np.ndarray, block_lengths: Iterable[int] = (20, 50, 100), draws: int = 2000) -> list[dict]:
    """Use predeclared block lengths; no block length is selected from the data."""
    return [block_bootstrap_mean_ci(values, block_length=int(length), draws=draws) for length in block_lengths]


def increment_diagnostics(series: ResidualSeries, lags_events: Iterable[int] = (1, 2, 5, 10, 20, 50)) -> list[dict]:
    """Estimate increment drift and variance-rate diagnostics at event lags.

    ``variance_rate_m2_per_s`` is a random-walk diagnostic for the residual
    process. It is not asserted to be the robot's PDE ``q_c``: measurement
    noise, shared RF errors, lever-arm mismatch, and reference error can all
    contribute to it.
    """
    t, r = series.timestamp_s, series.residual_m
    rows = []
    for lag in lags_events:
        lag = int(lag)
        if lag < 1 or lag >= len(r) - 1:
            continue
        dt = t[lag:] - t[:-lag]
        dr = r[lag:] - r[:-lag]
        mean_dt = float(np.mean(dt))
        centered = dr - dr.mean(axis=0)
        cov = np.cov(dr, rowvar=False, ddof=1) if len(dr) > 1 else np.full((2, 2), np.nan)
        rows.append({
            "lag_events": lag,
            "pairs": int(len(dr)),
            "mean_dt_s": mean_dt,
            "dt_cv": float(np.std(dt) / mean_dt) if mean_dt > 0 else None,
            "drift_m_per_s": np.mean(dr / dt[:, None], axis=0).tolist(),
            "variance_rate_m2_per_s": (np.diag(cov) / mean_dt).tolist(),
            "increment_covariance_m2": cov.tolist(),
            "interpretation": "diagnostic residual increment rate; not calibrated robot q_c",
        })
    return rows


def block_bootstrap_mean_ci(values: np.ndarray, block_length: int = 10, draws: int = 2000, seed: int = 20261102) -> dict:
    """Moving-block bootstrap CI for the component mean; retains local dependence."""
    x = np.asarray(values, dtype=float)
    if x.ndim != 2 or x.shape[1] != 2 or len(x) < 4:
        raise ValueError("values must be finite (n, 2), n >= 4")
    if not np.isfinite(x).all() or block_length < 1:
        raise ValueError("invalid bootstrap values or block length")
    rng = np.random.default_rng(seed)
    block_length = min(int(block_length), len(x))
    starts = np.arange(len(x) - block_length + 1)
    out = np.empty((draws, 2), dtype=float)
    blocks = int(np.ceil(len(x) / block_length))
    for i in range(draws):
        chosen = rng.choice(starts, size=blocks, replace=True)
        sample = np.concatenate([x[start:start + block_length] for start in chosen], axis=0)[:len(x)]
        out[i] = sample.mean(axis=0)
    return {"block_length_events": block_length, "draws": int(draws), "seed": int(seed), "mean_ci95_m": np.quantile(out, [0.025, 0.975], axis=0).tolist()}


def fit_then_evaluate(series: ResidualSeries) -> dict:
    fit, evaluation = split_series(series)
    fit_stats = residual_summary(fit.residual_m)
    eval_raw = residual_summary(evaluation.residual_m)
    train_bias = np.asarray(fit_stats["mean_m"])
    eval_after_fit_bias = residual_summary(evaluation.residual_m - train_bias)
    return {
        "fit_events": [int(fit.source_indices[0]), int(fit.source_indices[-1])],
        "evaluation_events": [int(evaluation.source_indices[0]), int(evaluation.source_indices[-1])],
        "fit_summary": fit_stats,
        "evaluation_raw_summary": eval_raw,
        "evaluation_after_fit_bias_diagnostic": eval_after_fit_bias,
        "fit_mean_block_bootstrap_sensitivity": bootstrap_sensitivity(fit.residual_m),
        "fit_autocorrelation": autocorrelation(fit.residual_m),
        "fit_autocorrelation_sensitivity": autocorrelation_sensitivity(fit.residual_m),
        "evaluation_autocorrelation": autocorrelation(evaluation.residual_m),
        "evaluation_autocorrelation_sensitivity": autocorrelation_sensitivity(evaluation.residual_m),
        "evaluation_increment_diagnostics": increment_diagnostics(evaluation),
        "score_policy": "raw evaluation summary is the only score-like value; bias diagnostic does not replace the project's original score",
    }
