"""
Fault Injector Module.
Injects Wheel Slip, GNSS Dropout, and Compound Temporal Faults into TrajectoryData.
"""
import numpy as np
import copy
from src.data.dataset import TrajectoryData


class FaultInjector:
    @staticmethod
    def inject_wheel_slip(data: TrajectoryData, slip_ratio: float, t_start: float, t_end: float) -> TrajectoryData:
        """
        Scale existing measured speed by (1-s) over [start, end).
        Positive s reduces measurement; negative s increases it. This is not a universal physical slip model.
        """
        if not np.isfinite([slip_ratio, t_start, t_end]).all() or t_end < t_start:
            raise ValueError("Invalid wheel bias/window")
        new_data = copy.deepcopy(data)
        mask = (new_data.timestamp >= t_start) & (new_data.timestamp < t_end)
        new_data.wheel_v[mask] = new_data.wheel_v[mask] * (1.0 - slip_ratio)
        return new_data

    @staticmethod
    def inject_gnss_dropout(data: TrajectoryData, t_start: float, duration: float) -> TrajectoryData:
        """Disable observations over [start, start+duration); zero duration is identity."""
        if not np.isfinite([t_start, duration]).all() or duration < 0:
            raise ValueError("Invalid dropout window")
        new_data = copy.deepcopy(data)
        t_end = t_start + duration
        mask = (new_data.timestamp >= t_start) & (new_data.timestamp < t_end)
        new_data.gnss_valid[mask] = False
        return new_data

    @staticmethod
    def inject_compound_fault(
        data: TrajectoryData,
        slip_ratio: float,
        gnss_dropout_duration: float,
        t_start: float = 20.0,
        wheel_slip_duration: float = 10.0
    ) -> TrajectoryData:
        """复合故障注入：在指定时间窗口同时注入打滑与 GNSS 缺失"""
        if not np.isfinite(wheel_slip_duration) or wheel_slip_duration < 0:
            raise ValueError("Invalid wheel bias duration")
        d = FaultInjector.inject_wheel_slip(data, slip_ratio, t_start, t_start + wheel_slip_duration)
        d = FaultInjector.inject_gnss_dropout(d, t_start, gnss_dropout_duration)
        return d
