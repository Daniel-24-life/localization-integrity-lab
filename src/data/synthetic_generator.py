"""
Synthetic Trajectory Generator based on continuous-time kinematic ODE.
Used for Gate 1 & Gate 2 verification and analytical bound testing.
"""
import numpy as np
from typing import Dict, Any
from .dataset import TrajectoryData


class SyntheticTrajectoryGenerator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.dt = config.get("simulation", {}).get("dt", 0.05)
        self.duration = config.get("simulation", {}).get("duration", 60.0)
        self.seed = config.get("project", {}).get("random_seed", 42)
        self.rng = np.random.default_rng(self.seed)

        # 传感器噪声参数
        wheel_cfg = config.get("sensors", {}).get("wheel", {})
        gnss_cfg = config.get("sensors", {}).get("gnss", {})
        self.sigma_v = wheel_cfg.get("sigma_v", 0.05)
        self.sigma_omega = wheel_cfg.get("sigma_omega", 0.02)
        self.sigma_pos = gnss_cfg.get("sigma_pos", 0.15)
        self.gnss_rate_hz = gnss_cfg.get("update_rate_hz", 5)

    def generate(self, trajectory_type: str = "figure8") -> TrajectoryData:
        """根据轨迹类型积分运动 ODE 并生成带噪声传感器观测"""
        t = np.arange(0.0, self.duration, self.dt)
        N = len(t)

        gt_x = np.zeros(N)
        gt_y = np.zeros(N)
        gt_theta = np.zeros(N)
        gt_v = np.zeros(N)
        gt_omega = np.zeros(N)

        v_nom = self.config.get("simulation", {}).get("v_nominal", 1.0)
        omega_nom = self.config.get("simulation", {}).get("omega_nominal", 0.2)

        # 定义标称输入曲线
        if trajectory_type == "straight":
            gt_v[:] = v_nom
            gt_omega[:] = 0.0
        elif trajectory_type == "circle":
            gt_v[:] = v_nom
            gt_omega[:] = omega_nom
        elif trajectory_type == "figure8":
            # 8 字形平滑轨迹
            T_period = self.duration / 2.0
            gt_v[:] = v_nom
            gt_omega[:] = omega_nom * np.sin(2 * np.pi * t / T_period)
        else:
            raise ValueError(f"Unknown trajectory_type: {trajectory_type}")

        # 使用连续 ODE 积分生成 Ground Truth (Runge-Kutta 4 阶积分)
        for i in range(1, N):
            dt_step = self.dt
            v_curr = gt_v[i-1]
            omega_curr = gt_omega[i-1]

            theta_prev = gt_theta[i-1]
            x_prev = gt_x[i-1]
            y_prev = gt_y[i-1]

            # RK4 积分
            def odefun(th):
                return np.array([v_curr * np.cos(th), v_curr * np.sin(th), omega_curr])

            k1 = odefun(theta_prev)
            k2 = odefun(theta_prev + 0.5 * dt_step * k1[2])
            k3 = odefun(theta_prev + 0.5 * dt_step * k2[2])
            k4 = odefun(theta_prev + dt_step * k3[2])

            delta_state = (k1 + 2*k2 + 2*k3 + k4) * (dt_step / 6.0)
            gt_x[i] = x_prev + delta_state[0]
            gt_y[i] = y_prev + delta_state[1]
            gt_theta[i] = (theta_prev + delta_state[2] + np.pi) % (2 * np.pi) - np.pi

        # 生成带有加性白噪声的传感器观测
        wheel_v = gt_v + self.rng.normal(0, self.sigma_v, N)
        wheel_omega = gt_omega + self.rng.normal(0, self.sigma_omega, N)
        imu_gz = gt_omega + self.rng.normal(0, self.sigma_omega, N)

        # GNSS 采样 (例如 20Hz 主循环中每 4 个 step 产生一次 GNSS 观测)
        gnss_step_stride = max(1, int(1.0 / (self.dt * self.gnss_rate_hz)))
        gnss_valid = np.zeros(N, dtype=bool)
        gnss_x = np.full(N, np.nan)
        gnss_y = np.full(N, np.nan)

        for i in range(0, N, gnss_step_stride):
            gnss_valid[i] = True
            gnss_x[i] = gt_x[i] + self.rng.normal(0, self.sigma_pos)
            gnss_y[i] = gt_y[i] + self.rng.normal(0, self.sigma_pos)

        return TrajectoryData(
            timestamp=t,
            wheel_v=wheel_v,
            wheel_omega=wheel_omega,
            imu_gz=imu_gz,
            gnss_x=gnss_x,
            gnss_y=gnss_y,
            gnss_valid=gnss_valid,
            gt_x=gt_x,
            gt_y=gt_y,
            gt_theta=gt_theta,
            metadata={"evidence_class": "synthetic", "reference_status": "available",
                      "reference_independent": True, "reference_source": "simulated latent trajectory",
                      "position_observation_source": "simulated reference + Gaussian noise",
                      "seed": self.seed, "trajectory_type": trajectory_type}
        )
