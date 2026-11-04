"""
Unit Tests for Math Derivations, Jacobians, and Baseline Filters.
Validates Gate 1 and Gate 2 criteria automatically.
"""
import pytest
import numpy as np
import yaml
from pathlib import Path

from src.models.robot_ode import (
    robot_kinematic_ode,
    analytic_system_jacobian,
    analytic_control_jacobian,
    finite_difference_system_jacobian
)
from src.models.error_dynamics import compute_theoretical_failure_time
from src.data.synthetic_generator import SyntheticTrajectoryGenerator
from src.filters.dead_reckoning import DeadReckoning
from src.filters.fixed_ekf import FixedEKF
from src.evaluation.metrics import PerformanceEvaluator


def load_test_config():
    config_path = Path(__file__).parent.parent / "configs" / "default_config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_jacobian_finite_difference():
    """Gate 2 核验点 1: 解析雅可比与中心有限差分高精度数值对齐"""
    test_points = [
        (np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0])),
        (np.array([2.5, -1.2, np.pi/4]), np.array([1.5, 0.3])),
        (np.array([-5.0, 3.0, -np.pi/3]), np.array([0.8, -0.5])),
        (np.array([10.0, 10.0, 2.8]), np.array([2.0, 1.2]))
    ]

    for X, u in test_points:
        A_ana = analytic_system_jacobian(X, u)
        A_num = finite_difference_system_jacobian(X, u, eps=1e-7)
        max_diff = np.max(np.abs(A_ana - A_num))
        assert max_diff < 1e-6, f"Jacobian mismatch at X={X}, diff={max_diff}"


def test_gate1_synthetic_data_generation():
    """Gate 1 核验点: 运动学 ODE 生成闭环数据，格式与约束完备"""
    config = load_test_config()
    gen = SyntheticTrajectoryGenerator(config)
    data = gen.generate(trajectory_type="figure8")

    assert len(data) > 0
    assert not np.isnan(data.gt_x).any()
    assert not np.isnan(data.gt_y).any()
    assert (data.timestamp[1:] > data.timestamp[:-1]).all()
    assert data.gnss_valid.sum() > 0


@pytest.mark.parametrize("trajectory_type", ["straight", "circle", "figure8"])
def test_gate2_baseline_ekf_beats_dr(trajectory_type):
    """Gate 2 核验点 2: 正常场景下 Fixed EKF 精度显著优于纯航位推算"""
    config = load_test_config()
    gen = SyntheticTrajectoryGenerator(config)
    data = gen.generate(trajectory_type=trajectory_type)

    gt_xy = np.column_stack([data.gt_x, data.gt_y])

    # 运行 B0: Dead Reckoning
    dr = DeadReckoning(dt=config["simulation"]["dt"])
    traj_b0 = dr.run_sequence(data)
    ate_b0 = PerformanceEvaluator.compute_ate(traj_b0[:, :2], gt_xy)["ate_rmse"]

    # 运行 B1: Fixed EKF
    ekf = FixedEKF(config)
    res_b1 = ekf.run_sequence(data)
    traj_b1 = res_b1["state"]
    ate_b1 = PerformanceEvaluator.compute_ate(traj_b1[:, :2], gt_xy)["ate_rmse"]

    # EKF 应该显著优于纯推算漂移 (至少提高 50% 以上精度)
    assert ate_b1 < 0.5 * ate_b0, f"EKF RMSE ({ate_b1:.3f}) must improve DR ({ate_b0:.3f}) by >50%"


def test_theoretical_t_failure_bound():
    """Gate 2/3 核验点: 恒速直线下 T_failure 仿真与理论值精确吻合"""
    v = 1.0
    s = 0.2
    al = 0.5
    t_theory = compute_theoretical_failure_time(al, s, v)
    assert abs(t_theory - 2.5) < 1e-6

    # 仿真核验
    dt = 0.01
    t = 0.0
    drift = 0.0
    while drift < al:
        t += dt
        drift += s * v * dt

    assert abs(t - t_theory) < dt * 2.0
