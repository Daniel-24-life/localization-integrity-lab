"""
Evaluation & Integrity Metrics Module.
ATE, SE(2) relative-pose errors and project-specific integrity quadrants.
NEES and event-level TTA are not yet implemented.
"""
import numpy as np
from typing import Dict, Any
from src.data.dataset import TrajectoryData


class PerformanceEvaluator:
    @staticmethod
    def compute_rpe(est_state, reference_state, timestamps, delta_sec=1., max_time_error_sec=.1):
        """Unaligned SE(2) relative transform error at approximately delta seconds.

        Pair each sample with the first sample at/after t+delta. Discard pairs
        outside the stated timing tolerance; report an undefined metric if empty.
        """
        est,ref,t=np.asarray(est_state),np.asarray(reference_state),np.asarray(timestamps)
        if est.shape!=ref.shape or est.shape!=(len(t),3) or len(t)<2 or not all(np.isfinite(a).all() for a in [est,ref,t]) or not np.all(np.diff(t)>0):
            raise ValueError("RPE requires finite matching (N,3) states and increasing times")
        if not np.isfinite([delta_sec,max_time_error_sec]).all() or delta_sec<=0 or max_time_error_sec<0:
            raise ValueError("Invalid RPE lag/tolerance")
        i=np.arange(len(t));j=np.searchsorted(t,t+delta_sec)
        mask=j<len(t);i,j=i[mask],j[mask]
        mask=np.abs(t[j]-t[i]-delta_sec)<=max_time_error_sec+1e-12;i,j=i[mask],j[mask]
        if not len(i):
            return {"rpe_pair_count":0,"rpe_translation_rmse":None,"rpe_rotation_rmse_rad":None,"rpe_lag_sec":delta_sec}
        def local_displacement(state):
            d=state[j,:2]-state[i,:2];c=np.cos(state[i,2]);s=np.sin(state[i,2])
            return np.column_stack([c*d[:,0]+s*d[:,1],-s*d[:,0]+c*d[:,1]])
        relative_translation=local_displacement(est)-local_displacement(ref)
        angle=(est[j,2]-est[i,2]-(ref[j,2]-ref[i,2])+np.pi)%(2*np.pi)-np.pi
        return {"rpe_pair_count":len(i),"rpe_translation_rmse":float(np.sqrt(np.mean(np.sum(relative_translation**2,axis=1)))),
                "rpe_rotation_rmse_rad":float(np.sqrt(np.mean(angle**2))),"rpe_lag_sec":delta_sec,
                "rpe_mean_actual_lag_sec":float(np.mean(t[j]-t[i]))}

    @staticmethod
    def compute_ate(est_xy: np.ndarray, gt_xy: np.ndarray) -> Dict[str, float]:
        """计算绝对轨迹误差 (Absolute Trajectory Error, ATE)"""
        if est_xy.shape != gt_xy.shape or est_xy.ndim != 2 or est_xy.shape[1] != 2 or len(est_xy) == 0:
            raise ValueError("ATE requires matching nonempty (N, 2) arrays")
        if not (np.isfinite(est_xy).all() and np.isfinite(gt_xy).all()):
            raise ValueError("ATE requires finite estimates and reference")
        error_vectors = est_xy - gt_xy
        errors_norm = np.linalg.norm(error_vectors, axis=1)

        rmse = float(np.sqrt(np.mean(errors_norm**2)))
        mean_err = float(np.mean(errors_norm))
        max_err = float(np.max(errors_norm))

        return {
            "ate_rmse": rmse,
            "ate_mean": mean_err,
            "ate_max": max_err
        }

    @staticmethod
    def evaluate_integrity(
        est_traj: np.ndarray,
        data: TrajectoryData,
        pl_hist: np.ndarray,
        nis_hist: np.ndarray,
        alert_limit: float = 0.5,
        nis_threshold: float = 9.21,
        allow_dependent_reference: bool = False
    ) -> Dict[str, Any]:
        """
        完整性评估与静默失效 (HMI) 判别
        - e(t) = ||est_xy - gt_xy||
        - alarm(t) = (PL(t) > AL) 或 (NIS(t) > nis_threshold)
        - HMI: e(t) > AL 且 alarm(t) == False (错了但没报警)
        """
        data.require_reference(allow_dependent=allow_dependent_reference, require_heading=False)
        N = len(data)
        if not np.isfinite(alert_limit) or alert_limit <= 0 or not np.isfinite(nis_threshold) or nis_threshold <= 0:
            raise ValueError("Thresholds must be finite and positive")
        if np.shape(est_traj) != (N, 3) or np.shape(pl_hist) != (N,) or np.shape(nis_hist) != (N,):
            raise ValueError("Integrity input shapes do not match data")
        if not np.isfinite(est_traj).all() or not np.isfinite(pl_hist).all() or np.any(pl_hist < 0):
            raise ValueError("Estimates and PL must be finite; PL must be nonnegative")
        if np.isinf(nis_hist).any() or np.any(nis_hist[np.isfinite(nis_hist)] < 0):
            raise ValueError("NIS must be nonnegative or NaN (missing)")
        est_xy = est_traj[:, :2]
        gt_xy = np.column_stack([data.gt_x, data.gt_y])

        # 真实误差
        true_errors = np.linalg.norm(est_xy - gt_xy, axis=1)

        # 在线报警判断
        # Instantaneous alarms only: no sample-and-hold. NaN is unavailable, not zero.
        nis_available = np.isfinite(nis_hist)
        alarm = (pl_hist > alert_limit) | (nis_available & (nis_hist > nis_threshold))

        # 完整性四象限判别
        hmi_mask = (true_errors > alert_limit) & (~alarm)
        safe_alert_mask = (true_errors > alert_limit) & alarm
        false_alarm_mask = (true_errors <= alert_limit) & alarm
        normal_mask = (true_errors <= alert_limit) & (~alarm)

        hmi_count = int(np.sum(hmi_mask))
        # Left-endpoint hold over [t_i, t_{i+1}); final sample has zero duration.
        weights = np.append(np.diff(data.timestamp), 0.0)
        total_duration = float(weights.sum())
        duration_ratio = lambda mask: float(weights[mask].sum() / total_duration)
        hmi_duration_ratio = duration_ratio(hmi_mask)
        safe_availability = float(np.sum(normal_mask) / N)
        false_alarm_ratio = float(np.sum(false_alarm_mask) / N)

        return {
            "hmi_count": hmi_count,
            "hmi_frame_ratio": float(hmi_count / N),
            "hmi_duration_ratio": hmi_duration_ratio,
            "hmi_duration_sec": float(weights[hmi_mask].sum()),
            "total_duration_sec": total_duration,
            "safe_availability": safe_availability,
            "safe_availability_duration_ratio": duration_ratio(normal_mask),
            "false_alarm_ratio": false_alarm_ratio,
            "false_alarm_duration_ratio": duration_ratio(false_alarm_mask),
            "safe_alert_frame_ratio": float(np.mean(safe_alert_mask)),
            "safe_alert_duration_ratio": duration_ratio(safe_alert_mask),
            "max_true_error": float(np.max(true_errors)),
            "mean_true_error": float(np.mean(true_errors)),
            "true_errors": true_errors,
            "alarm_mask": alarm,
            "hmi_mask": hmi_mask,
            "normal_mask": normal_mask,
            "safe_alert_mask": safe_alert_mask,
            "false_alarm_mask": false_alarm_mask,
            "nis_available_mask": nis_available,
            "duration_weights": weights
        }
