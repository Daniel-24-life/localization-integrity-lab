"""
Error Dynamics and Theoretical Boundary Verification.
"""
import numpy as np
from typing import Tuple


def compute_theoretical_failure_time(alert_limit: float, slip_ratio: float, v_nominal: float) -> float:
    """
    理想位置误差首次达到 AL 的时间（不是自动等同于 HMI）:
    零初始误差、恒速直线、航向无误差、无位置修正；v_measured=(1-s)*v_true。
    """
    if not np.isfinite([alert_limit, slip_ratio, v_nominal]).all() or alert_limit <= 0:
        raise ValueError("Parameters must be finite and AL positive")
    if slip_ratio == 0 or v_nominal == 0:
        return float("inf")
    return alert_limit / (abs(slip_ratio) * abs(v_nominal))


def simulate_linear_drift(slip_ratio: float, v_nominal: float, duration: float, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    数值模拟打滑引发的线性漂移演化
    """
    t = np.arange(0, duration, dt)
    error_p = np.abs(slip_ratio) * np.abs(v_nominal) * t
    return t, error_p
