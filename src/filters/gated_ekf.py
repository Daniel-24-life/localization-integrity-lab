"""
Baseline B2: NIS-Gated EKF.
Rejects anomalous GNSS updates when NIS exceeds the chi-square threshold.
"""
import numpy as np
from typing import Dict, Any
from .fixed_ekf import FixedEKF
from src.data.dataset import TrajectoryData


class GatedEKF(FixedEKF):
    def __init__(self, config: Dict[str, Any], initial_state: np.ndarray = None):
        super().__init__(config, initial_state)
        # 自由度为 2 的卡方分布，p=0.99 对应门限约为 9.21
        self.nis_threshold = config.get("integrity", {}).get("nis_threshold", 9.21)

    def run_sequence(self, data: TrajectoryData) -> Dict[str, np.ndarray]:
        data.validate()
        angular_velocity = data.angular_velocity(self.angular_input)
        N = len(data)
        est_traj = np.zeros((N, 3))
        cov_hist = np.zeros((N, 3, 3))
        nis_hist = np.full(N, np.nan)
        pl_hist = np.zeros(N)
        gated_flags = np.zeros(N, dtype=bool)

        est_traj[0] = self.state
        cov_hist[0] = self.P
        pl_hist[0] = self.compute_protection_level()

        for i in range(1, N):
            v = data.wheel_v[i-1]
            omega = angular_velocity[i-1]

            # 预测
            self.predict(v, omega, data.timestamp[i] - data.timestamp[i-1])

            # 观测门控检查
            if data.gnss_valid[i]:
                z = np.array([data.gnss_x[i], data.gnss_y[i]])
                z_pred = self.H @ self.state
                y = z - z_pred
                S = self.H @ self.P @ self.H.T + self.R
                S_inv = np.linalg.inv(S)
                nis = float(y.T @ S_inv @ y)
                nis_hist[i] = nis

                if nis <= self.nis_threshold:
                    # 门控通过，允许更新
                    K = self.P @ self.H.T @ S_inv
                    self.state += K @ y
                    self.state[2] = (self.state[2] + np.pi) % (2 * np.pi) - np.pi
                    I_KH = np.eye(3) - K @ self.H
                    self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T
                else:
                    # 门控拒绝，记录标志
                    gated_flags[i] = True

            est_traj[i] = self.state.copy()
            cov_hist[i] = self.P.copy()
            pl_hist[i] = self.compute_protection_level()

        return {
            "state": est_traj,
            "covariance": cov_hist,
            "nis": nis_hist,
            "pl": pl_hist,
            "gated": gated_flags
        }
