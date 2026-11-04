"""
Baseline B0: wheel linear velocity, explicit angular input, no position correction.
"""
import numpy as np
from src.data.dataset import TrajectoryData


class DeadReckoning:
    def __init__(self, dt: float, initial_state: np.ndarray = None, angular_input="wheel"):
        self.dt = dt
        self.angular_input = angular_input
        self.state = np.zeros(3) if initial_state is None else initial_state.copy()

    def step(self, v: float, omega: float, dt: float = None) -> np.ndarray:
        """单步航位推算"""
        theta = self.state[2]
        dt = self.dt if dt is None else float(dt)
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        dx = v * np.cos(theta) * dt
        dy = v * np.sin(theta) * dt
        dtheta = omega * dt

        self.state[0] += dx
        self.state[1] += dy
        self.state[2] = (self.state[2] + dtheta + np.pi) % (2 * np.pi) - np.pi
        return self.state.copy()

    def run_sequence(self, data: TrajectoryData) -> np.ndarray:
        """对整条时序进行批处理推算"""
        data.validate()
        angular_velocity = data.angular_velocity(self.angular_input)
        N = len(data)
        est_traj = np.zeros((N, 3))
        est_traj[0] = self.state

        for i in range(1, N):
            v = data.wheel_v[i-1]
            omega = angular_velocity[i-1]
            est_traj[i] = self.step(v, omega, data.timestamp[i] - data.timestamp[i-1])

        return est_traj
