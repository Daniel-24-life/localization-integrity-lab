"""
Mobile Robot Kinematic ODE and Jacobian Formulations.
Provides continuous-time ODE evaluation, analytical Jacobians,
discrete transition matrices, and finite difference verification.
"""
import numpy as np
from typing import Tuple


def robot_kinematic_ode(X: np.ndarray, u: np.ndarray) -> np.ndarray:
    """
    连续时间运动学常微分方程 (ODE):
    dX/dt = f(X, u)
    X = [x, y, theta]^T
    u = [v, omega]^T
    """
    v = u[0]
    omega = u[1]
    theta = X[2]

    x_dot = v * np.cos(theta)
    y_dot = v * np.sin(theta)
    theta_dot = omega

    return np.array([x_dot, y_dot, theta_dot], dtype=float)


def analytic_system_jacobian(X: np.ndarray, u: np.ndarray) -> np.ndarray:
    """
    连续时间系统状态雅可比矩阵 A = df/dX:
    A = [[0, 0, -v*sin(theta)],
         [0, 0,  v*cos(theta)],
         [0, 0,  0          ]]
    """
    v = u[0]
    theta = X[2]

    A = np.zeros((3, 3), dtype=float)
    A[0, 2] = -v * np.sin(theta)
    A[1, 2] = v * np.cos(theta)
    return A


def analytic_control_jacobian(X: np.ndarray, u: np.ndarray) -> np.ndarray:
    """
    连续时间输入雅可比矩阵 B = df/du:
    B = [[cos(theta), 0],
         [sin(theta), 0],
         [0,          1]]
    """
    theta = X[2]

    B = np.zeros((3, 2), dtype=float)
    B[0, 0] = np.cos(theta)
    B[1, 0] = np.sin(theta)
    B[2, 1] = 1.0
    return B


def discrete_transition_matrix(X: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
    """
    离散时间状态转移矩阵 Phi_k = exp(A * dt) ≈ I + A * dt
    """
    A = analytic_system_jacobian(X, u)
    return np.eye(3) + A * dt


def finite_difference_system_jacobian(X: np.ndarray, u: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    中心有限差分计算数值雅可比，用于 Gate 2 单元测试与自动复核
    """
    n = len(X)
    J_num = np.zeros((n, n), dtype=float)

    for j in range(n):
        X_plus = X.copy()
        X_minus = X.copy()
        X_plus[j] += eps
        X_minus[j] -= eps

        f_plus = robot_kinematic_ode(X_plus, u)
        f_minus = robot_kinematic_ode(X_minus, u)

        J_num[:, j] = (f_plus - f_minus) / (2.0 * eps)

    return J_num
