"""B3: bounded, causal R-only innovation covariance matching baseline.

Matrix extension of Brown & Rutan (1985), Eq. 2.12-2.13; not their complete
chemical-model correction algorithm, nor a new proposed integrity method.
"""
from collections import deque
import numpy as np
from .fixed_ekf import FixedEKF


class AdaptiveEKF(FixedEKF):
    def __init__(self, config, initial_state=None):
        super().__init__(config, initial_state)
        options = config.get('adaptive', {})
        self.window = options.get('window_observations', 20)
        self.minimum = options.get('minimum_observations', 20)
        self.rate = float(options.get('update_rate', .1))
        self.floor = float(options.get('minimum_sigma_m', .05)) ** 2
        self.ceiling = float(options.get('maximum_sigma_m', 3.)) ** 2
        if (type(self.window) is not int or type(self.minimum) is not int
                or not 2 <= self.minimum <= self.window
                or not np.isfinite([self.rate, self.floor, self.ceiling]).all()
                or not 0 <= self.rate <= 1
                or options.get('minimum_sigma_m', .05) <= 0
                or options.get('maximum_sigma_m', 3.) <= 0
                or not 0 < self.floor <= self.r_pos**2 <= self.ceiling):
            raise ValueError('Invalid adaptive window/rate/positive sigma bounds')
        self.innovations = deque(maxlen=self.window)
        self.adaptation_log = []

    def update(self, z):
        z = np.asarray(z, dtype=float)
        if z.shape != (2,) or not np.isfinite(z).all():
            raise ValueError('Expected finite 2D position observation')
        prior_projection = self.H @ self.P @ self.H.T
        used = self.R.copy()
        result = super().update(z)
        self.innovations.append(result[3].copy())
        clipped = False
        if len(self.innovations) >= self.minimum and self.rate > 0:
            samples = np.asarray(self.innovations)
            # Raw second moment: the matching model assumes zero-mean innovations.
            candidate = samples.T @ samples / len(samples) - prior_projection
            candidate = (candidate + candidate.T) / 2
            eigenvalues, vectors = np.linalg.eigh(candidate)
            bounded = np.clip(eigenvalues, self.floor, self.ceiling)
            clipped = bool(np.any(eigenvalues != bounded))
            target = (vectors * bounded) @ vectors.T
            self.R = (1-self.rate) * used + self.rate * target
            self.R = (self.R + self.R.T) / 2
        # Current NIS and Joseph update used the pre-adaptation R. New R is for
        # the next observation, so a sample cannot lower its own diagnostic NIS.
        self.adaptation_log.append((used, self.R.copy(), clipped))
        return result

    def run_sequence(self, data):
        before = len(self.adaptation_log)
        initial_R = self.R.copy()
        result = super().run_sequence(data)
        used = np.full((len(data), 2, 2), np.nan)
        available_next = np.repeat(initial_R[None], len(data), axis=0)
        clipped = np.zeros(len(data), dtype=bool)
        index = before
        for i in range(1, len(data)):
            available_next[i] = available_next[i-1]
            if data.gnss_valid[i]:
                used[i], available_next[i], clipped[i] = self.adaptation_log[index]
                index += 1
        result.update(measurement_covariance_used=used,
                      measurement_covariance_next=available_next, R_projection_clipped=clipped)
        return result
