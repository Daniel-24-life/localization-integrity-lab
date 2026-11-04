"""
Spatial Error Diffusion PDE Model (Fokker-Planck / Kolmogorov Forward Equation).
Models the probability density evolution of localization error:
  dp/dt = - div(v_drift * p) + 0.5 * div(D * grad(p))
Experimental density illustration only. Not a validated absorbing-boundary,
first-passage, or HMI probability solver. Clipping/renormalization changes mass.
"""
import numpy as np
from typing import Tuple, Dict


def solve_error_diffusion_pde(
    drift_speed: float,          # 对流漂移速度 s * v (m/s)
    diffusion_coeff: float,      # 扩散系数 D (m^2/s)
    alert_limit: float = 0.5,    # 边界 AL (m)
    t_max: float = 10.0,         # 最大演化时间 (s)
    nx: int = 100,               # 网格空间剖分数
    nt: int = 500                # 时间步数
) -> Dict[str, np.ndarray]:
    """
    求解一维对流-扩散 PDE (沿漂移主轴):
    dp/dt + v_d * dp/dx = 0.5 * D * d^2p/dx^2
    初始条件: p(x, 0) = delta(x) ~ 高斯核 (x0=0, sigma0=0.02)
    历史演示边界：端点值保持，每步截断负值并归一化；不是吸收边界。
    """
    x_min, x_max = -0.2, alert_limit * 1.5
    x = np.linspace(x_min, x_max, nx)
    dx = x[1] - x[0]
    dt = t_max / nt

    # 稳定性条件检查 (CFL 条件)
    cfl_adv = drift_speed * dt / dx
    cfl_diff = 0.5 * diffusion_coeff * dt / (dx**2)

    # 初始概率密度函数 (高斯近似 Dirac)
    sigma0 = 0.03
    p = (1.0 / (np.sqrt(2 * np.pi) * sigma0)) * np.exp(-0.5 * (x / sigma0)**2)
    p /= np.sum(p * dx) # 归一化

    p_history = [p.copy()]
    time_points = [0.0]

    # 一阶迎风差分 + 二阶中心差分时间步进
    for step in range(1, nt + 1):
        p_new = p.copy()
        for i in range(1, nx - 1):
            # 迎风格式处理对流项
            adv = drift_speed * (p[i] - p[i-1]) / dx
            # 中心差分处理扩散项
            diff = 0.5 * diffusion_coeff * (p[i+1] - 2*p[i] + p[i-1]) / (dx**2)
            p_new[i] = p[i] + dt * (-adv + diff)

        p = np.maximum(0.0, p_new) # 保持非负性
        integral = np.sum(p * dx)
        if integral > 0:
            p /= integral # 保持总概率守恒

        if step % (nt // 5) == 0:
            p_history.append(p.copy())
            time_points.append(step * dt)

    return {
        "x": x,
        "time_points": np.array(time_points),
        "p_history": np.array(p_history),
        "alert_limit": alert_limit
    }
