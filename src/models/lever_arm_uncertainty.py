"""Bounded antenna lever-arm uncertainty propagation model.

This module provides analytical bounds and covariance inflation for unknown
or uncalibrated GNSS antenna lever-arms on mobile robotic platforms.
"""
from typing import Dict, Tuple, Union
import numpy as np


class LeverArmUncertainty:
    """Analytical envelope bounds for uncalibrated sensor lever-arm.
    
    Given a mobile robot body frame with antenna displacement r_ant:
        p_antenna = p_body + R(theta) * r_ant
    For unknown r_ant within an envelope ||r_ant|| <= r_max, this class
    computes worst-case bias bounds, covariance inflation, and adjusted
    protection levels (PL).
    """

    def __init__(self, r_max: float = 0.35, k_integrity: float = 3.0):
        """Initialize lever-arm uncertainty model.
        
        Args:
            r_max: Maximum physical distance (meters) from wheel center to antenna.
                   Default 0.35m is an assumed sensitivity radius, not a measured
                   antenna calibration or a bound certified for this dataset.
            k_integrity: Retained compatibility parameter; no probability guarantee.
        """
        if r_max < 0 or not np.isfinite(r_max):
            raise ValueError(f"r_max must be a non-negative finite float, got {r_max}")
        if k_integrity <= 0 or not np.isfinite(k_integrity):
            raise ValueError(f"k_integrity must be positive finite, got {k_integrity}")
        self.r_max = float(r_max)
        self.k_integrity = float(k_integrity)

    def position_bias_bound(self) -> float:
        """Return the maximum Euclidean position bias caused by unmodeled lever-arm."""
        return self.r_max

    def antenna_position(self, p_body: np.ndarray, theta: float, r_ant: np.ndarray) -> np.ndarray:
        """Compute exact antenna position given body pose and lever arm.
        
        Args:
            p_body: 2D array-like [x, y] in navigation frame.
            theta: Robot heading angle in radians.
            r_ant: 2D array-like [rx, ry] in body frame.
        """
        p_body = np.asarray(p_body, dtype=float)
        r_ant = np.asarray(r_ant, dtype=float)
        if p_body.shape != (2,) or r_ant.shape != (2,):
            raise ValueError("p_body and r_ant must be 2D vectors")
        c = np.cos(theta)
        s = np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        return p_body + R @ r_ant

    def conservative_protection_level(self, nominal_pl: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Compute conservative protection level accounting for worst-case lever-arm bias.
        
        Under unknown lever-arm ||r_ant|| <= r_max, the true error satisfies:
            ||e_body|| <= ||e_antenna|| + r_max
        Conditional on the nominal PL already bounding antenna error, a bound is:
            PL_conservative = PL_nominal + r_max
        A heuristic nominal PL does not become guaranteed by adding a radius.
        """
        pl = np.asarray(nominal_pl, dtype=float)
        if not np.isfinite(pl).all() or np.any(pl < 0):
            raise ValueError("Protection levels must be finite and non-negative")
        return pl + self.r_max

    def inflated_covariance(self, P_pos: np.ndarray) -> np.ndarray:
        """Compute position covariance inflated by bounded lever-arm variance.
        
        Returns an optional heuristic inflation r_max^2 I. A deterministic
        unknown displacement is not independent zero-mean noise. This addition
        alone does not bound total error covariance, its cross terms, or HMI.
        """
        P = np.asarray(P_pos, dtype=float)
        if P.ndim == 2:
            if P.shape != (2, 2):
                raise ValueError("P_pos must be a 2x2 matrix or sequence of 2x2 matrices")
            return P + (self.r_max ** 2) * np.eye(2)
        elif P.ndim == 3:
            if P.shape[1:] != (2, 2):
                raise ValueError("P_pos must be an Nx2x2 tensor")
            inflation = (self.r_max ** 2) * np.eye(2)
            return P + inflation[np.newaxis, :, :]
        else:
            raise ValueError("P_pos must be 2D or 3D")

    def reference_error_bounds(self, estimated_xy, reference_xy):
        """Triangle bounds conditional on ||reference - true position|| <= r_max.

        This is offline geometry; it does not correct an estimator's measurement
        model or supply evidence that the assumed reference bound holds.
        """
        est = np.asarray(estimated_xy, dtype=float)
        ref = np.asarray(reference_xy, dtype=float)
        if est.shape != ref.shape or est.ndim != 2 or est.shape[1] != 2 or not len(est):
            raise ValueError("Matching nonempty (N,2) positions required")
        if not np.isfinite(est).all() or not np.isfinite(ref).all():
            raise ValueError("Finite positions required")
        error = np.linalg.norm(est - ref, axis=1)
        return np.maximum(0., error - self.r_max), error + self.r_max

    def bounds_summary(self, nominal_pl: float, nominal_ate_rmse: float) -> Dict[str, float]:
        """Summarize nominal vs conservative bounds for integrity accounting."""
        return {
            "r_max_m": self.r_max,
            "nominal_pl_m": float(nominal_pl),
            "conservative_pl_m": float(nominal_pl + self.r_max),
            "nominal_ate_rmse_m": float(nominal_ate_rmse),
            "conservative_ate_rmse_bound_m": float(nominal_ate_rmse + self.r_max),
            "lever_arm_variance_m2": float(self.r_max ** 2),
        }
