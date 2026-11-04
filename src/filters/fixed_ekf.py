"""
Baseline B1: Fixed Extended Kalman Filter (Fixed Q and R).
Uses wheel linear velocity, explicit angular input and position observations.
"""
import numpy as np
from typing import Dict, Any, Tuple
from src.data.dataset import TrajectoryData
from src.models.robot_ode import discrete_transition_matrix, analytic_control_jacobian


class FixedEKF:
    def __init__(self, config: Dict[str, Any], initial_state: np.ndarray = None):
        self.config = config
        self.dt = config.get("simulation", {}).get("dt", 0.05)
        self.angular_input = config.get("filter", {}).get("angular_input", "wheel")

        # 初始状态与协方差
        self.state = np.zeros(3) if initial_state is None else initial_state.copy()

        init_cov = config.get("filter", {}).get("initial_cov", {})
        self.P = np.diag([
            init_cov.get("x", 0.1)**2,
            init_cov.get("y", 0.1)**2,
            init_cov.get("theta", 0.05)**2
        ])

        # 过程噪声与观测噪声
        p_noise = config.get("filter", {}).get("process_noise", {})
        self.q_v = p_noise.get("q_v", 0.1)
        self.q_omega = p_noise.get("q_omega", 0.05)

        m_noise = config.get("filter", {}).get("measurement_noise", {})
        self.r_pos = m_noise.get("r_pos", 0.15)
        self.R = np.eye(2) * (self.r_pos**2)

        self.H = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ])

        self.k_integrity = config.get("integrity", {}).get("k_integrity", 3.0)

    def predict(self, v: float, omega: float, dt: float = None) -> Tuple[np.ndarray, np.ndarray]:
        """EKF 预测步"""
        u = np.array([v, omega])
        theta = self.state[2]
        dt = self.dt if dt is None else float(dt)
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        Phi = discrete_transition_matrix(self.state, u, dt)
        B = analytic_control_jacobian(self.state, u)

        # 1. 状态外推 (航位推算)
        dx = v * np.cos(theta) * dt
        dy = v * np.sin(theta) * dt
        dtheta = omega * dt

        self.state[0] += dx
        self.state[1] += dy
        self.state[2] = (self.state[2] + dtheta + np.pi) % (2 * np.pi) - np.pi

        # 2. 状态转移矩阵 Phi 与输入矩阵 B

        # 3. 过程噪声协方差 Q_k
        cov_u = np.diag([self.q_v**2, self.q_omega**2])
        Q_k = B @ cov_u @ B.T * (dt**2)

        # 4. 协方差传播
        self.P = Phi @ self.P @ Phi.T + Q_k
        return self.state.copy(), self.P.copy()

    def update(self, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        """
        EKF 观测更新步
        z: [x_gnss, y_gnss]
        返回: (更新后状态, 更新后协方差, NIS, 新息向量)
        """
        # 1. 计算新息 (Innovation)
        z_pred = self.H @ self.state
        y = z - z_pred

        # 2. 新息协方差
        S = self.H @ self.P @ self.H.T + self.R

        # 3. 计算卡方 NIS (归一化新息平方)
        S_inv = np.linalg.inv(S)
        nis = float(y.T @ S_inv @ y)

        # 4. 卡尔曼增益
        K = self.P @ self.H.T @ S_inv

        # 5. 后验状态更新
        self.state += K @ y
        self.state[2] = (self.state[2] + np.pi) % (2 * np.pi) - np.pi

        # 6. 后验协方差 (Joseph 稳健形式保证正定对称)
        I_KH = np.eye(3) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

        return self.state.copy(), self.P.copy(), nis, y

    def compute_protection_level(self) -> float:
        """根据当前水平位置协方差最大特征值计算水平保护限 HPL"""
        P_xy = self.P[:2, :2]
        eigvals = np.linalg.eigvalsh(P_xy)
        max_eig = max(0.0, float(np.max(eigvals)))
        return self.k_integrity * np.sqrt(max_eig)

    def run_sequence(self, data: TrajectoryData) -> Dict[str, np.ndarray]:
        """整条时序滤波运行"""
        data.validate()
        angular_velocity = data.angular_velocity(self.angular_input)
        N = len(data)
        est_traj = np.zeros((N, 3))
        cov_hist = np.zeros((N, 3, 3))
        nis_hist = np.full(N, np.nan)
        pl_hist = np.zeros(N)

        est_traj[0] = self.state
        cov_hist[0] = self.P
        pl_hist[0] = self.compute_protection_level()

        for i in range(1, N):
            v = data.wheel_v[i-1]
            omega = angular_velocity[i-1]

            # 预测
            self.predict(v, omega, data.timestamp[i] - data.timestamp[i-1])

            # 更新 (若 GNSS 测量有效)
            if data.gnss_valid[i]:
                z = np.array([data.gnss_x[i], data.gnss_y[i]])
                _, _, nis, _ = self.update(z)
                nis_hist[i] = nis

            est_traj[i] = self.state.copy()
            cov_hist[i] = self.P.copy()
            pl_hist[i] = self.compute_protection_level()

        return {
            "state": est_traj,
            "covariance": cov_hist,
            "nis": nis_hist,
            "pl": pl_hist
        }
