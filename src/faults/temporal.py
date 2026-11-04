"""Interval-averaged wheel bias profiles with explicit cumulative exposure.

These are measured-speed scaling faults, not universal wheel-slip physics.
"""
import copy
import numpy as np


def wheel_bias_profile(timestamp, mode, peak, start, duration, period=4., duty=.5):
    """Average s(t) on each input interval; last sample has zero duration/value.

    ramp rises continuously from zero to peak; intermittent starts with ON.
    Integration over fractional endpoints avoids sample-rate-dependent dose.
    """
    t = np.asarray(timestamp, dtype=float)
    if t.ndim != 1 or len(t) < 2 or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError('Finite strictly increasing timestamps required')
    if mode not in ['constant', 'intermittent', 'ramp']:
        raise ValueError('Unknown temporal mode')
    if not np.isfinite([peak, start, duration, period, duty]).all() or not 0 <= peak < 1 or duration < 0 or period <= 0 or not 0 < duty <= 1:
        raise ValueError('Invalid temporal parameters')
    if duration == 0 or peak == 0:
        return np.zeros(len(t))
    u = np.clip(t-start, 0., duration)
    if mode == 'constant':
        integral = peak*u
    elif mode == 'ramp':
        integral = peak*u*u/(2*duration)
    else:
        cycles = np.floor(u/period)
        remainder = u-cycles*period
        integral = peak*(cycles*period*duty+np.minimum(remainder, period*duty))
    profile = np.r_[np.diff(integral)/np.diff(t), 0.]
    # Roundoff can produce tiny negatives at exactly flat integral segments.
    return np.clip(profile, 0., peak)


def inject_temporal_wheel_bias(data, profile):
    data.validate()
    profile = np.asarray(profile, dtype=float)
    if profile.shape != data.timestamp.shape or not np.isfinite(profile).all() or np.any(profile < 0) or np.any(profile >= 1):
        raise ValueError('Aligned measured-speed scale bias in [0,1) required')
    result = copy.deepcopy(data)
    result.wheel_v = result.wheel_v*(1-profile)
    result.metadata['temporal_wheel_bias'] = dict(formula='measured_v_faulty=(1-s_interval)*measured_v_original',
        exposure_sec=float(profile[:-1] @ np.diff(data.timestamp)), last_sample_duration_zero=True)
    return result
