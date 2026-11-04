"""Explicit i2Nav conventions; no fitted offsets or reference in online inputs."""
import numpy as np
from scipy.spatial.transform import Rotation


NED_TO_ENU = np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., -1.]])
FRD_TO_FLU = np.diag([1., -1., -1.])


def frd_to_ned_rotation(rpy_deg):
    """Active body-to-navigation Rz(yaw) Ry(pitch) Rx(roll), degree inputs."""
    angles = np.asarray(rpy_deg, dtype=float)
    if angles.shape[-1:] != (3,) or not np.isfinite(angles).all():
        raise ValueError('Finite roll/pitch/yaw triples required')
    return Rotation.from_euler('ZYX', angles[..., ::-1], degrees=True).as_matrix()


def heading_enu_from_yaw_ned(yaw_deg):
    yaw = np.asarray(yaw_deg, dtype=float)
    if not np.isfinite(yaw).all():
        raise ValueError('Finite NED yaw required')
    return (np.pi / 2 - np.deg2rad(yaw) + np.pi) % (2 * np.pi) - np.pi


def incremental_body_rates(timestamp, increments):
    """Return interval-end times and average body rates, discarding first delta.

    Row i>0 is interpreted as integral over (t[i-1],t[i]], matching incremental
    IMU convention. The first row has no recorded left endpoint and is unused.
    Returns FRD rates. Apply FRD_TO_FLU explicitly for the estimator's frame.
    No interpolation, padding, reference data, or hidden first-interval guess.
    """
    t, delta = np.asarray(timestamp, dtype=float), np.asarray(increments, dtype=float)
    if t.ndim != 1 or len(t) < 2 or delta.shape != (len(t), 3):
        raise ValueError('Matching time and (N,3) increments required')
    if not np.isfinite(t).all() or not np.isfinite(delta).all() or np.any(np.diff(t) <= 0):
        raise ValueError('Finite data and strictly increasing time required')
    return t[1:].copy(), delta[1:] / np.diff(t)[:, None]


def four_wheel_body_velocity(speed, steering):
    """Signed forward/lateral velocity from the published symmetric wheel mean.

    All four wheels are used with the same weight; this does not infer yaw rate
    without wheel geometry. Magnitude is not substituted for signed forward v.
    """
    v, delta = np.asarray(speed, dtype=float), np.asarray(steering, dtype=float)
    if v.shape != delta.shape or v.shape[-1:] != (4,) or not np.isfinite(v).all() or not np.isfinite(delta).all():
        raise ValueError('Matching finite four-wheel speeds and angles required')
    return np.stack([np.mean(v * np.cos(delta), axis=-1), np.mean(v * np.sin(delta), axis=-1)], axis=-1)


def aligned_wheel_imu_intervals(imu_time, angle_increments_frd, wheel_time, forward_speed,
                                wheel_time_shift=-.01):
    """Measurement-time causal intervals on IMU endpoints, with signed wheel v.

    Snap each corrected wheel timestamp to the first IMU endpoint at/after it.
    Over the next state interval, hold that wheel sample and aggregate exactly
    the IMU increments ending within the interval. The resulting yaw rate at
    index k is consumed when predicting output k+1, not output k. No offline
    interpolation of partial/future increments; last rate is unused zero.

    This proves corrected-measurement-time causality, not wall-clock latency.
    Raw receipt/driver scheduling remains an explicitly separate validation.
    """
    t = np.asarray(imu_time, dtype=float)
    delta = np.asarray(angle_increments_frd, dtype=float)
    incremental_body_rates(t, delta)  # Shared strict shape/time validation.
    tw, v = np.asarray(wheel_time, dtype=float), np.asarray(forward_speed, dtype=float)
    if tw.ndim != 1 or len(tw) < 2 or v.shape != tw.shape or not np.isfinite(tw).all() or not np.isfinite(v).all():
        raise ValueError('Matching finite wheel times and forward speeds required')
    if np.any(np.diff(tw) <= 0) or not np.isfinite(wheel_time_shift):
        raise ValueError('Increasing wheel times and finite time shift required')
    corrected = tw + wheel_time_shift
    indices = np.searchsorted(t, corrected, side='left')
    keep = (corrected >= t[0]) & (indices >= 1) & (indices < len(t))
    indices = indices[keep]
    if len(indices) < 2 or np.any(np.diff(indices) <= 0):
        raise ValueError('Insufficient overlap or multiple wheel events on one IMU endpoint')
    grid = t[indices]
    cumulative = np.vstack([np.zeros(3), np.cumsum(delta[1:], axis=0)])
    integral = cumulative[indices[1:]] - cumulative[indices[:-1]]
    rates_flu = integral @ FRD_TO_FLU.T / np.diff(grid)[:, None]
    rate = np.r_[rates_flu[:, 2], 0.]
    return dict(timestamp=grid.copy(), forward_speed=v[keep].copy(), imu_yaw_rate_for_next_interval=rate,
        corrected_wheel_source_time=corrected[keep].copy(), raw_wheel_timestamp=tw[keep].copy(),
        imu_interval_end_time=grid[1:].copy(), imu_indices=indices.copy(),
        snap_delay_sec=grid - corrected[keep], dropped_wheel_samples=int((~keep).sum()),
        last_yaw_rate_unused=True, wall_clock_latency_certified=False)
