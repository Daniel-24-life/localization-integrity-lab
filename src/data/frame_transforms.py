"""Explicit source->target rotations; no inferred axis swaps in production."""
import re
from pathlib import Path
import numpy as np


def load_wheel_to_camera_imu(calibration_path):
    text = Path(calibration_path).read_text(encoding="utf-8")
    match = re.search(r"body_T_wheel:.*?data:\s*\[([^\]]+)\]", text, re.S)
    if not match:
        raise ValueError("Official body_T_wheel matrix not found")
    values = np.fromstring(match.group(1).replace("\n", " "), sep=",")
    if len(values) != 16:
        raise ValueError("Expected 16 homogeneous transform entries")
    transform = values.reshape(4, 4)
    rotation = transform[:3, :3]
    if not np.allclose(transform[3], [0, 0, 0, 1]) or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(rotation), 1, atol=1e-6):
        raise ValueError("Invalid rigid transform")
    return transform


def camera_imu_vectors_to_wheel(vectors, wheel_to_camera_rotation):
    """Row vectors: v_I=R_IW v_W, hence v_W(row)=v_I(row) R_IW."""
    vectors = np.asarray(vectors, dtype=float)
    rotation = np.asarray(wheel_to_camera_rotation, dtype=float)
    if vectors.ndim != 2 or vectors.shape[1] != 3 or rotation.shape != (3, 3):
        raise ValueError("Expected (N,3) vectors and (3,3) rotation")
    if not np.isfinite(vectors).all() or not np.isfinite(rotation).all():
        raise ValueError("Non-finite vectors/rotation")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(rotation), 1, atol=1e-6):
        raise ValueError("Rotation must be proper orthonormal")
    return vectors @ rotation
