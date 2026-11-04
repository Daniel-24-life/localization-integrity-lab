"""Conditional bounds for a fixed estimate/alarm trace, not reference calibration.

The caller must supply a uniform pointwise bound on the change in the error
vector. A receiver standard deviation or an RMS-only budget is insufficient.
"""
import numpy as np


def _nonnegative(value, name):
    value = float(value)
    if not np.isfinite(value) or value < 0:
        raise ValueError(f'{name} must be finite and nonnegative')
    return value


def rmse_comparison_bounds(dr_rmse, filter_rmse, budget):
    """Minkowski bounds, assuming the same budget applies to both fixed traces."""
    dr = _nonnegative(dr_rmse, 'dr_rmse')
    filtered = _nonnegative(filter_rmse, 'filter_rmse')
    bound = _nonnegative(budget, 'budget')
    dr_lower = max(0., dr - bound)
    filtered_upper = filtered + bound
    return dict(dr_lower=dr_lower, dr_upper=dr + bound,
                filter_lower=max(0., filtered - bound), filter_upper=filtered_upper,
                improvement_lower_percent=100 * (1 - filtered_upper / dr_lower) if dr_lower > 0 else None)


def hmi_classification_bounds(errors, alarms, timestamps, alert_limit, budget):
    """Definite/possible HMI conditional on fixed alarms and uniform error bound.

    Strict exceedance and left endpoint duration weights match project metrics.
    These masks bound classification only; they are not risk probabilities.
    """
    errors = np.asarray(errors, dtype=float)
    alarms = np.asarray(alarms)
    timestamps = np.asarray(timestamps, dtype=float)
    limit = _nonnegative(alert_limit, 'alert_limit')
    bound = _nonnegative(budget, 'budget')
    if errors.ndim != 1 or len(errors) < 2 or timestamps.shape != errors.shape or alarms.shape != errors.shape:
        raise ValueError('Aligned 1D arrays with at least two samples required')
    if alarms.dtype != np.bool_:
        raise ValueError('alarms must be boolean')
    if not np.all(np.isfinite(errors)) or np.any(errors < 0):
        raise ValueError('errors must be finite and nonnegative')
    if not np.all(np.isfinite(timestamps)) or np.any(np.diff(timestamps) <= 0):
        raise ValueError('timestamps must be finite and strictly increasing')
    weights = np.r_[np.diff(timestamps), 0.]
    definite = (np.maximum(0., errors - bound) > limit) & ~alarms
    possible = (errors + bound > limit) & ~alarms
    return dict(definite_mask=definite, possible_mask=possible,
                definite_frame_ratio=float(np.mean(definite)), possible_frame_ratio=float(np.mean(possible)),
                definite_duration_ratio=float(weights @ definite / weights.sum()),
                possible_duration_ratio=float(weights @ possible / weights.sum()))
