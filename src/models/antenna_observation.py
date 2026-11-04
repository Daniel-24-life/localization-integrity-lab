"""Planar antenna observation and analytic Jacobian, independent of any GT."""
import numpy as np


def antenna_observation(state, lever_flu):
    """h=[x,y]+R(theta)*lever_xy; vertical lever has zero effect only if planar.

    state is the pose of the chosen model reference point. The lever must be
    measured from that same point to antenna, in front-left-up coordinates.
    This function alone does not assert that roll/pitch can be neglected.
    """
    x, lever = np.asarray(state, dtype=float), np.asarray(lever_flu, dtype=float)
    if x.shape != (3,) or lever.shape != (3,) or not np.isfinite(x).all() or not np.isfinite(lever).all():
        raise ValueError('Finite 3D pose and lever required')
    c, s = np.cos(x[2]), np.sin(x[2])
    rotated = np.array([c * lever[0] - s * lever[1], s * lever[0] + c * lever[1]])
    H = np.array([[1., 0., -rotated[1]], [0., 1., rotated[0]]])
    return x[:2] + rotated, H
