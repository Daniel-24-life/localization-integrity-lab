"""
M3DGR Rosbag Data Extractor (High-Level Pure-Python implementation).
Specifically designed for the Shanghai Jiao Tong University (SJTU) M3DGR benchmark dataset:
GitHub: https://github.com/sjtuyinjie/M3DGR
Paper: 'Towards Robust Sensor-Fusion Ground SLAM: A Comprehensive Benchmark and a Resilient Framework' (IROS 2025)
Supports both Indoor (Vicon MoCap GT) and Outdoor (u-blox / MAVROS GNSS NavSatFix -> ENU).
"""
import os
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation
from typing import Optional, Dict, Any, List
from .dataset import TrajectoryData


def lla_to_enu(lat_arr: np.ndarray, lon_arr: np.ndarray, lat0: float, lon0: float) -> tuple[np.ndarray, np.ndarray]:
    """WGS84 经纬度投影至局部东北天 (ENU) 平面米制坐标"""
    R_earth = 6378137.0 # 地球长半轴 (米)
    lat_rad = np.radians(lat_arr)
    lon_rad = np.radians(lon_arr)
    lat0_rad = np.radians(lat0)
    lon0_rad = np.radians(lon0)

    dlat = lat_rad - lat0_rad
    dlon = lon_rad - lon0_rad

    x_east = R_earth * dlon * np.cos(lat0_rad)
    y_north = R_earth * dlat
    return x_east, y_north


def extract_m3dgr_bag(bag_path: str, output_npz: Optional[str] = None, simulate_position: bool = False) -> TrajectoryData:
    """
    解析上海交大 M3DGR 数据集包 (ROS1/ROS2) 并输出统一格式 TrajectoryData。
    """
    try:
        from rosbags.highlevel import AnyReader
    except ImportError:
        raise ImportError("请在终端运行: pip install rosbags")

    bag_file = Path(bag_path)
    if not bag_file.exists():
        raise FileNotFoundError(f"找不到 rosbag 文件: {bag_path}")
    if output_npz and (Path(output_npz).exists() or Path(output_npz).with_suffix('.metadata.json').exists()):
        raise FileExistsError(f"Refusing to overwrite dataset: {output_npz}")

    print(f"[M3DGR Data Extractor] 开始读取上交 M3DGR 数据包: {bag_file.name}")
    
    timestamps_wheel = []
    v_wheel = []
    omega_wheel = []

    timestamps_imu = []
    gz_imu = []

    timestamps_gnss = []
    gnss_lat_list = []
    gnss_lon_list = []

    timestamps_gt = []
    gt_x_list = []
    gt_y_list = []
    gt_theta_list = []

    with AnyReader([bag_file]) as reader:
        topic_map = {conn.topic: conn for conn in reader.connections}
        print(f"[M3DGR Data Extractor] 数据包包含 {len(topic_map)} 个话题")

        # 1. 查找轮速计话题
        target_odom = "/odom" if "/odom" in topic_map else None

        # 2. 查找 IMU 话题
        target_imu = None
        for candidate in ["/camera/imu", "/livox/avia/imu", "/livox/mid360/imu"]:
            if candidate in topic_map:
                target_imu = candidate
                break

        # 3. 查找 GNSS 话题
        target_gnss = None
        for candidate in ["/ublox_driver/receiver_lla", "/mavros/global_position/raw/fix"]:
            if candidate in topic_map:
                target_gnss = candidate
                break

        # 4. 查找 Ground Truth 动捕话题
        target_gt = None
        for candidate in ["/vrpn_client_node/UGV/pose", "/vrpn_client_node/robot/pose", "/ground_truth/pose"]:
            if candidate in topic_map:
                target_gt = candidate
                break

        print(f"[M3DGR Data Extractor] 话题映射选择:")
        print(f"  * Wheel Odom : {target_odom}")
        print(f"  * IMU Gyro   : {target_imu}")
        print(f"  * GNSS       : {target_gnss}")
        print(f"  * GT Pose    : {target_gt}")

        target_topics = [t for t in [target_odom, target_imu, target_gnss, target_gt] if t is not None]
        connections_to_read = [topic_map[t] for t in target_topics]

        for conn, timestamp_ns, rawdata in reader.messages(connections=connections_to_read):
            msg = reader.deserialize(rawdata, conn.msgtype)
            t_sec = timestamp_ns * 1e-9

            if conn.topic == target_odom:
                timestamps_wheel.append(t_sec)
                vx = float(msg.twist.twist.linear.x)
                wz = float(msg.twist.twist.angular.z)
                v_wheel.append(vx)
                omega_wheel.append(wz)

            elif conn.topic == target_imu:
                timestamps_imu.append(t_sec)
                gz_imu.append(float(msg.angular_velocity.z))

            elif conn.topic == target_gnss:
                # sensor_msgs/msg/NavSatFix
                if hasattr(msg, "latitude") and hasattr(msg, "longitude"):
                    if not (np.isnan(msg.latitude) or np.isnan(msg.longitude)):
                        timestamps_gnss.append(t_sec)
                        gnss_lat_list.append(float(msg.latitude))
                        gnss_lon_list.append(float(msg.longitude))

            elif conn.topic == target_gt:
                timestamps_gt.append(t_sec)
                pos = msg.pose.position
                rot = msg.pose.orientation
                gt_x_list.append(float(pos.x))
                gt_y_list.append(float(pos.y))
                r = Rotation.from_quat([rot.x, rot.y, rot.z, rot.w])
                yaw = float(r.as_euler("zyx")[0])
                gt_theta_list.append(yaw)

    if not timestamps_wheel:
        raise ValueError(f"在 {bag_path} 中未找到轮速计数据！")

    streams = {"wheel": timestamps_wheel, "imu": timestamps_imu, "gnss": timestamps_gnss, "reference": timestamps_gt}
    coverage = {}
    for name, times in streams.items():
        if times:
            values = np.array(times)
            if not np.isfinite(values).all() or not np.all(np.diff(values) > 0):
                raise ValueError(f"Invalid timestamps in {name}")
            coverage[name] = {"start": float(values[0]), "end": float(values[-1]), "count": len(times)}
    # Crop continuous interpolation streams to their common support; never extrapolate GT/IMU.
    continuous = [timestamps_wheel] + [s for s in [timestamps_imu, timestamps_gt] if s]
    if not timestamps_gt and timestamps_gnss:
        continuous.append(timestamps_gnss)
    lo = max(s[0] for s in continuous)
    hi = min(s[-1] for s in continuous)
    keep = (np.array(timestamps_wheel) >= lo) & (np.array(timestamps_wheel) <= hi)
    if keep.sum() < 2:
        raise ValueError("Insufficient common sensor coverage")
    t_base = np.array(timestamps_wheel)[keep]
    t_start = t_base[0]
    t_rel = t_base - t_start

    v_arr = np.array(v_wheel)[keep]
    omega_arr = np.array(omega_wheel)[keep]
    metadata = {"source_bag": str(bag_file.resolve()), "time_basis": "bag record timestamp seconds, relative to first retained wheel sample",
                "source_coverage_absolute_sec": coverage, "common_support_absolute_sec": [lo, hi],
                "wheel_samples_removed_outside_support": int((~keep).sum()),
                "topics": {"wheel": target_odom, "imu": target_imu, "gnss": target_gnss, "reference": target_gt},
                "imu_fallback_to_wheel": not bool(timestamps_imu),
                "coordinate_alignment_verified": False, "reference_status": "missing", "reference_independent": False}

    # 线性插值对齐 IMU
    gz_arr = np.interp(t_base, timestamps_imu, gz_imu) if timestamps_imu else omega_arr

    # 处理 GT 与 GNSS
    if target_gt and len(gt_x_list) > 0:
        # 室内动捕场景
        gt_x = np.interp(t_base, timestamps_gt, gt_x_list)
        gt_y = np.interp(t_base, timestamps_gt, gt_y_list)
        gt_theta_unwrapped = np.unwrap(gt_theta_list)
        gt_theta_interp = np.interp(t_base, timestamps_gt, gt_theta_unwrapped)
        gt_theta = (gt_theta_interp + np.pi) % (2 * np.pi) - np.pi

        # 室内场景以 5Hz 步长引入真值叠加高斯噪声模拟基准 GNSS
        gnss_valid = np.zeros(len(t_base), dtype=bool)
        gnss_x = np.full(len(t_base), np.nan)
        gnss_y = np.full(len(t_base), np.nan)
        rng = np.random.default_rng(42)
        if simulate_position:
            for i in range(0, len(t_base), 4):
                gnss_valid[i] = True
                gnss_x[i] = gt_x[i] + rng.normal(0, 0.15)
                gnss_y[i] = gt_y[i] + rng.normal(0, 0.15)
        metadata.update({"evidence_class": "semi_synthetic" if simulate_position else "real_wheel_mocap_no_position",
                         "reference_status": "available", "reference_independent": True,
                         "reference_source": "mocap pose interpolated on wheel timestamps",
                         "position_observation_source": "mocap + Gaussian noise" if simulate_position else "none",
                         "seed": 42 if simulate_position else None,
                         "position_sigma_m": 0.15 if simulate_position else None,
                         "position_stride": 4 if simulate_position else None})

    elif target_gnss and len(gnss_lat_list) > 0:
        # 室外真实卫星场景
        lat0 = gnss_lat_list[0]
        lon0 = gnss_lon_list[0]
        enu_x, enu_y = lla_to_enu(np.array(gnss_lat_list), np.array(gnss_lon_list), lat0, lon0)

        # GNSS 插值到主时钟
        gnss_valid = np.zeros(len(t_base), dtype=bool)
        gnss_x = np.full(len(t_base), np.nan)
        gnss_y = np.full(len(t_base), np.nan)

        # 找到每个 GNSS 观测点在 t_base 中最近的时刻
        for t_gnss, x_g, y_g in zip(timestamps_gnss, enu_x, enu_y):
            idx = np.argmin(np.abs(t_base - t_gnss))
            if np.abs(t_base[idx] - t_gnss) < 0.05:
                gnss_valid[idx] = True
                gnss_x[idx] = x_g
                gnss_y[idx] = y_g

        # 室外真值：若无动捕，可用高精度差分 GNSS 插值平滑作为离线参考 GT
        gt_x = np.interp(t_base, timestamps_gnss, enu_x)
        gt_y = np.interp(t_base, timestamps_gnss, enu_y)
        dx = np.gradient(gt_x)
        dy = np.gradient(gt_y)
        gt_theta = np.arctan2(dy, dx)
        metadata.update({"evidence_class": "dependent_reference_exploratory", "reference_status": "available",
                         "reference_independent": False, "reference_source": "same GNSS interpolated; yaw from displacement",
                         "position_observation_source": target_gnss})

    else:
        # 纯相对推算模式
        gt_x = np.full_like(t_base, np.nan)
        gt_y = np.full_like(t_base, np.nan)
        gt_theta = np.full_like(t_base, np.nan)
        gnss_valid = np.zeros(len(t_base), dtype=bool)
        gnss_x = np.full(len(t_base), np.nan)
        gnss_y = np.full(len(t_base), np.nan)

    data = TrajectoryData(
        timestamp=t_rel,
        wheel_v=v_arr,
        wheel_omega=omega_arr,
        imu_gz=gz_arr,
        gnss_x=gnss_x,
        gnss_y=gnss_y,
        gnss_valid=gnss_valid,
        gt_x=gt_x,
        gt_y=gt_y,
        gt_theta=gt_theta,
        metadata=metadata
    )

    if output_npz:
        data.save_npz(output_npz)
        print(f"[M3DGR Data Extractor] 提取成功！共 {len(data)} 帧 (时长 {t_rel[-1]:.2f}s)，已保存至: {output_npz}")

    return data
