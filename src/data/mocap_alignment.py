"""Planar marker-to-wheel calibration, kept separate from online estimation.

p_W = p_M + R(theta_M) lever_M; theta_W = theta_M + phi.
The fit uses a designated calibration interval, never estimator errors.
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def transform_marker_pose(xy, yaw, phi, lever):
    xy, yaw, lever = np.asarray(xy), np.asarray(yaw), np.asarray(lever)
    if xy.shape != (len(yaw), 2) or lever.shape != (2,):
        raise ValueError("Expected planar poses and two-component lever")
    c, s = np.cos(yaw), np.sin(yaw)
    offset = np.column_stack([c*lever[0]-s*lever[1], s*lever[0]+c*lever[1]])
    return xy + offset, (yaw + phi + np.pi) % (2*np.pi) - np.pi


def synchronize_audited_mocap(raw):
    """Interpolate on common header support only; raw observations stay intact."""
    tw = raw['odom_header']; tg = raw['vrpn_client_node_UGV_pose_header']
    ti = raw['camera_imu_header']; pose = raw['vrpn_client_node_UGV_pose_values']
    for t in [tw, tg, ti]:
        if not np.isfinite(t).all() or not np.all(np.diff(t) > 0):
            raise ValueError("Header times must increase strictly before interpolation")
    valid = (tw >= max(tg[0], ti[0])) & (tw <= min(tg[-1], ti[-1]))
    t = tw[valid]
    rotation = Rotation.from_quat(pose[:, 3:]).as_matrix()
    yaw = np.unwrap(np.arctan2(rotation[:, 1, 0], rotation[:, 0, 0]))
    return dict(timestamp=t, wheel_v=raw['odom_values'][valid, 0],
                wheel_omega=raw['odom_values'][valid, 1],
                imu_gz=np.interp(t, ti, raw['rotated_imu_gyro'][:, 2]),
                marker_xy=np.column_stack([np.interp(t, tg, pose[:, k]) for k in [0, 1]]),
                marker_yaw=np.interp(t, tg, yaw),
                cropped_wheel_samples=int((~valid).sum()))


def velocity_equations(stream):
    t = stream['timestamp']; mid = (t[1:]+t[:-1])/2
    velocity = np.diff(stream['marker_xy'], axis=0)/np.diff(t)[:, None]
    yaw = (stream['marker_yaw'][1:]+stream['marker_yaw'][:-1])/2
    c, s = np.cos(yaw), np.sin(yaw)
    local = np.column_stack([c*velocity[:, 0]+s*velocity[:, 1], -s*velocity[:, 0]+c*velocity[:, 1]])
    return mid-t[0], np.interp(mid, t, stream['wheel_v']), np.interp(mid, t, stream['imu_gz']), local


def calibrate_marker(stream, duration_sec=30., min_speed=.25, max_reference_speed=2.):
    time, v, w, local = velocity_equations(stream)
    quality = (np.abs(v) > min_speed) & (np.linalg.norm(local, axis=1) < max_reference_speed)
    train = quality & (time < duration_sec)
    if train.sum() < 30:
        raise ValueError("Insufficient calibration observations")

    def residual(parameters, selection):
        phi, lx, ly = parameters
        pred = v[selection, None]*np.array([np.cos(phi), np.sin(phi)]) + w[selection, None]*np.array([ly, -lx])
        return pred-local[selection]

    fit = least_squares(lambda p: residual(p, train).ravel(), [-np.pi, .2, .03], loss='soft_l1', f_scale=.1)
    if not fit.success or np.linalg.matrix_rank(fit.jac) < 3:
        raise ValueError("Calibration failed or is not locally identifiable")
    result = dict(phi_rad=float(fit.x[0]), phi_deg=float(np.rad2deg(fit.x[0])), lever_m=fit.x[1:].tolist(),
                  calibration_duration_sec=duration_sec, min_speed=min_speed, max_reference_speed=max_reference_speed,
                  objective='soft_l1 velocity residual, f_scale=0.1 m/s', velocity_gain_fixed=1.,
                  jacobian_singular_values=np.linalg.svd(fit.jac, compute_uv=False).tolist(),
                  time_offset_fitted=False, official_marker_extrinsic=False)
    for name, mask in [('calibration', train), ('post_calibration', quality & (time >= duration_sec))]:
        r = residual(fit.x, mask)
        result[name] = dict(samples=int(mask.sum()), velocity_rmse_mps=float(np.sqrt(np.mean(np.sum(r*r, axis=1)))) if len(r) else None)
    return result
