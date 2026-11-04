"""
Standardized Trajectory Data Container (Decoupled from ROS).
"""
from dataclasses import dataclass, field
import json
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Optional


@dataclass
class TrajectoryData:
    timestamp: np.ndarray        # (N,) 时间戳 (秒)
    wheel_v: np.ndarray          # (N,) 轮速计线速度 (m/s)
    wheel_omega: np.ndarray      # (N,) 轮速计角速度 (rad/s)
    imu_gz: np.ndarray           # (N,) IMU 航向角速度 (rad/s)
    gnss_x: np.ndarray           # (N,) GNSS 水平 X 观测 (m)
    gnss_y: np.ndarray           # (N,) GNSS 水平 Y 观测 (m)
    gnss_valid: np.ndarray       # (N,) GNSS 观测有效标志 (bool)
    gt_x: np.ndarray             # (N,) 真值 X (m) [仅离线评估可用]
    gt_y: np.ndarray             # (N,) 真值 Y (m) [仅离线评估可用]
    gt_theta: np.ndarray         # (N,) 真值航向角 (rad) [仅离线评估可用]
    metadata: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.timestamp)

    def angular_velocity(self, source="wheel"):
        """Select explicitly; IMU requires a recorded rotation into wheel axes."""
        if source == "wheel":
            if self.metadata.get("wheel_angular_velocity_status") == "unavailable":
                raise ValueError("Wheel angular velocity unavailable; select a verified angular source explicitly")
            return self.wheel_omega
        if source == "imu":
            if self.metadata.get("imu_frame") != "wheel" or not self.metadata.get("imu_rotation_verified", False):
                raise ValueError("IMU input requires verified rotation into wheel frame")
            return self.imu_gz
        raise ValueError("angular_input must be wheel or imu")

    def validate(self):
        """Check inputs without changing measurements or reference trajectories."""
        n = len(self)
        if n < 2:
            raise ValueError("At least two samples are required")
        for key in ("timestamp", "wheel_v", "wheel_omega", "imu_gz", "gnss_x", "gnss_y", "gnss_valid", "gt_x", "gt_y", "gt_theta"):
            if np.asarray(getattr(self, key)).shape != (n,):
                raise ValueError(f"Invalid shape: {key}")
        for key in ("timestamp", "wheel_v", "imu_gz"):
            if not np.isfinite(getattr(self, key)).all():
                raise ValueError(f"Non-finite input: {key}")
        if self.metadata.get("wheel_angular_velocity_status") == "unavailable":
            if not np.isnan(self.wheel_omega).all():
                raise ValueError("Unavailable wheel angular velocity must be represented by all NaN")
        elif not np.isfinite(self.wheel_omega).all():
            raise ValueError("Non-finite input: wheel_omega")
        if not np.all(np.diff(self.timestamp) > 0):
            raise ValueError("Timestamps must be strictly increasing")
        if self.gnss_valid.dtype != np.bool_:
            raise ValueError("gnss_valid must be boolean")
        if not (np.isfinite(self.gnss_x[self.gnss_valid]).all() and np.isfinite(self.gnss_y[self.gnss_valid]).all()):
            raise ValueError("Valid position observations must be finite")

    def require_reference(self, allow_dependent: bool = False, require_heading: bool = True):
        self.validate()
        if self.metadata.get("reference_status") != "available":
            raise ValueError("Reference provenance missing/unavailable; accuracy and HMI evaluation blocked")
        fields = ("gt_x", "gt_y", "gt_theta") if require_heading else ("gt_x", "gt_y")
        if not all(np.isfinite(getattr(self, key)).all() for key in fields):
            raise ValueError("Reference contains non-finite values")
        if require_heading and self.metadata.get("reference_heading_status") == "unavailable":
            raise ValueError("Reference heading unavailable; pose/heading evaluation blocked")
        if not self.metadata.get("reference_independent", False) and not allow_dependent:
            raise ValueError("Dependent reference: use explicit exploratory evaluation, never independent validation")

    def save_npz(self, file_path: str):
        """保存为高效紧凑的压缩 npz 格式"""
        file_path = Path(file_path)
        if file_path.suffix != ".npz":
            raise ValueError("Dataset output must have the .npz suffix")
        sidecar = file_path.with_suffix(".metadata.json")
        if file_path.exists() or sidecar.exists():
            raise FileExistsError(f"Refusing to overwrite dataset: {file_path}")
        self.validate()
        file_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            file_path,
            timestamp=self.timestamp,
            wheel_v=self.wheel_v,
            wheel_omega=self.wheel_omega,
            imu_gz=self.imu_gz,
            gnss_x=self.gnss_x,
            gnss_y=self.gnss_y,
            gnss_valid=self.gnss_valid,
            gt_x=self.gt_x,
            gt_y=self.gt_y,
            gt_theta=self.gt_theta
        )
        sidecar.write_text(json.dumps(self.metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load_npz(cls, file_path: str) -> "TrajectoryData":
        """从 npz 加载"""
        file_path = Path(file_path)
        sidecar = file_path.with_suffix(".metadata.json")
        with np.load(file_path, allow_pickle=False) as data:
            result = cls(
            timestamp=data["timestamp"],
            wheel_v=data["wheel_v"],
            wheel_omega=data["wheel_omega"],
            imu_gz=data["imu_gz"],
            gnss_x=data["gnss_x"],
            gnss_y=data["gnss_y"],
            gnss_valid=data["gnss_valid"],
            gt_x=data["gt_x"],
            gt_y=data["gt_y"],
            gt_theta=data["gt_theta"],
            metadata=json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.exists() else {}
            )
        result.validate()
        return result

    def to_dataframe(self) -> pd.DataFrame:
        """转换为 pandas DataFrame 方便审计与画图"""
        return pd.DataFrame({
            "timestamp": self.timestamp,
            "wheel_v": self.wheel_v,
            "wheel_omega": self.wheel_omega,
            "imu_gz": self.imu_gz,
            "gnss_x": self.gnss_x,
            "gnss_y": self.gnss_y,
            "gnss_valid": self.gnss_valid,
            "gt_x": self.gt_x,
            "gt_y": self.gt_y,
            "gt_theta": self.gt_theta
        })
