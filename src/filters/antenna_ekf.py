"""Explicit antenna-observation baselines; original filters remain unchanged."""
from collections import deque
import copy
import numpy as np
from scipy.special import logsumexp
from .fixed_ekf import FixedEKF
from .velocity_bias_ekf import VelocityBiasEKF
from .bias_model_bank import mixture_moments
from src.models.antenna_observation import antenna_observation


class AntennaFixedEKF(FixedEKF):
    def __init__(self, config, initial_state=None):
        super().__init__(config, initial_state)
        self.lever = np.asarray(config['filter']['position_lever_flu_m'], dtype=float)
        antenna_observation(self.state, self.lever)  # Explicit validation.
        self.gate_threshold = None
        self._gated = []

    def update(self, z):
        predicted, H = antenna_observation(self.state, self.lever)
        z = np.asarray(z, dtype=float)
        if z.shape != (2,) or not np.isfinite(z).all():
            raise ValueError('Finite 2D position observation required')
        y = z - predicted
        S = H @ self.P @ H.T + self.R
        nis = float(y @ np.linalg.solve(S, y))
        rejected = self.gate_threshold is not None and nis > self.gate_threshold
        self._gated.append(bool(rejected))
        if not rejected:
            K = np.linalg.solve(S, H @ self.P).T
            self.state += K @ y
            self.state[2] = (self.state[2] + np.pi) % (2 * np.pi) - np.pi
            J = np.eye(3) - K @ H
            self.P = J @ self.P @ J.T + K @ self.R @ K.T
        return self.state.copy(), self.P.copy(), nis, y

    def run_sequence(self, data):
        self._gated = []
        out = super().run_sequence(data)
        if self.gate_threshold is not None:
            flags = np.zeros(len(data), dtype=bool)
            events = np.flatnonzero(data.gnss_valid)[1:] if data.gnss_valid[0] else np.flatnonzero(data.gnss_valid)
            if len(events) != len(self._gated):
                raise AssertionError('Gated observation count mismatch')
            flags[events] = self._gated
            out['gated'] = flags
        return out


class AntennaGatedEKF(AntennaFixedEKF):
    def __init__(self, config, initial_state=None):
        super().__init__(config, initial_state)
        self.gate_threshold = float(config.get('integrity', {}).get('nis_threshold', 9.21))
        if not np.isfinite(self.gate_threshold) or self.gate_threshold <= 0:
            raise ValueError('Positive finite NIS threshold required')


class AntennaAdaptiveEKF(AntennaFixedEKF):
    """B3 with explicit 3D antenna lever-arm observation model."""
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
            raise ValueError('Finite 2D position observation required')
        _, H = antenna_observation(self.state, self.lever)
        prior_projection = H @ self.P @ H.T
        used = self.R.copy()
        result = super().update(z)
        self.innovations.append(result[3].copy())
        clipped = False
        if len(self.innovations) >= self.minimum and self.rate > 0:
            samples = np.asarray(self.innovations)
            candidate = samples.T @ samples / len(samples) - prior_projection
            candidate = (candidate + candidate.T) / 2
            eigenvalues, vectors = np.linalg.eigh(candidate)
            bounded = np.clip(eigenvalues, self.floor, self.ceiling)
            clipped = bool(np.any(eigenvalues != bounded))
            target = (vectors * bounded) @ vectors.T
            self.R = (1 - self.rate) * used + self.rate * target
            self.R = (self.R + self.R.T) / 2
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
            available_next[i] = available_next[i - 1]
            if data.gnss_valid[i]:
                used[i], available_next[i], clipped[i] = self.adaptation_log[index]
                index += 1
        result.update(measurement_covariance_used=used,
                      measurement_covariance_next=available_next, R_projection_clipped=clipped)
        return result


class AntennaVelocityBiasEKF(VelocityBiasEKF):
    def __init__(self, config, initial_state=None):
        super().__init__(config, initial_state)
        self.lever = np.asarray(config.get('filter', {}).get('position_lever_flu_m', [0., 0., 0.]), dtype=float)
        antenna_observation(self.state[:3], self.lever)

    def update(self, z):
        z = np.asarray(z, dtype=float)
        if z.shape != (2,) or not np.isfinite(z).all():
            raise ValueError('Finite 2D position observation required')
        predicted, H3 = antenna_observation(self.state[:3], self.lever)
        H = np.column_stack([H3, np.zeros((2, 1))])
        y = z - predicted
        S = H @ self.P @ H.T + self.R
        nis = float(y @ np.linalg.solve(S, y))
        K = np.linalg.solve(S, H @ self.P).T
        self.state += K @ y
        self.state[2] = (self.state[2] + np.pi) % (2 * np.pi) - np.pi
        I_KH = np.eye(4) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T
        self.P = (self.P + self.P.T) / 2
        return self.state.copy(), self.P.copy(), nis, y


class AntennaBiasModelBank:
    def __init__(self, config, initial_state=None):
        options = config.get('model_bank', {})
        p = float(options.get('initial_bias_probability', .05))
        self.floor = float(options.get('posterior_uniform_floor', .01))
        self.learn = options.get('learn_weights', True)
        if not np.isfinite([p, self.floor]).all() or not 0 < p < 1 or not 0 <= self.floor < .5 or type(self.learn) is not bool:
            raise ValueError('Invalid bank probabilities/settings')
        self.weights = np.array([1 - p, p])
        self.k = config.get('integrity', {}).get('k_integrity', 3.)
        healthy = copy.deepcopy(config)
        healthy['velocity_bias'] = dict(initial_mean_m_s=0., initial_sigma_m_s=0., diffusion_std_m_s_sqrt_s=0.)
        self.models = [AntennaVelocityBiasEKF(healthy, initial_state), AntennaVelocityBiasEKF(config, initial_state)]
        self.angular_input = self.models[0].angular_input
        self.lever = self.models[0].lever

    def moments(self):
        return mixture_moments([m.state for m in self.models], [m.P for m in self.models], self.weights)

    def run_sequence(self, data):
        data.validate()
        omega = data.angular_velocity(self.angular_input)
        n = len(data)
        state = np.zeros((n, 4))
        P = np.zeros((n, 4, 4))
        weights = np.zeros((n, 2))
        nis = np.full(n, np.nan)
        component_nis = np.full((n, 2), np.nan)
        likelihood = np.full((n, 2), np.nan)
        pl = np.zeros(n)
        state[0], P[0] = self.moments()
        weights[0] = self.weights
        pl[0] = self.k * np.sqrt(max(0., np.linalg.eigvalsh(P[0, :2, :2]).max()))
        for i in range(1, n):
            for m in self.models:
                m.predict(data.wheel_v[i - 1], omega[i - 1], data.timestamp[i] - data.timestamp[i - 1])
            if data.gnss_valid[i]:
                z = np.array([data.gnss_x[i], data.gnss_y[i]])
                prior, prior_P = self.moments()
                pred_prior, H_prior3 = antenna_observation(prior[:3], self.lever)
                H_prior = np.column_stack([H_prior3, np.zeros((2, 1))])
                y = z - pred_prior
                S = H_prior @ prior_P @ H_prior.T + self.models[0].R
                nis[i] = float(y @ np.linalg.solve(S, y))
                for j, m in enumerate(self.models):
                    pred_m, H_m3 = antenna_observation(m.state[:3], self.lever)
                    H_m = np.column_stack([H_m3, np.zeros((2, 1))])
                    residual = z - pred_m
                    cov = H_m @ m.P @ H_m.T + m.R
                    sign, logdet = np.linalg.slogdet(cov)
                    if sign <= 0:
                        raise ValueError('Invalid innovation covariance')
                    component_nis[i, j] = float(residual @ np.linalg.solve(cov, residual))
                    likelihood[i, j] = -.5 * (2 * np.log(2 * np.pi) + logdet + component_nis[i, j])
                if self.learn:
                    log_prior = np.full(2, -np.inf)
                    np.log(self.weights, out=log_prior, where=self.weights > 0)
                    logp = log_prior + likelihood[i]
                    posterior = np.exp(logp - logsumexp(logp))
                    self.weights = (1 - 2 * self.floor) * posterior + self.floor
                for m in self.models:
                    m.update(z)
            state[i], P[i] = self.moments()
            weights[i] = self.weights
            pl[i] = self.k * np.sqrt(max(0., np.linalg.eigvalsh(P[i, :2, :2]).max()))
        return dict(state=state[:, :3].copy(), covariance=P[:, :3, :3].copy(), nis=nis, pl=pl,
            augmented_state=state, augmented_covariance=P, model_weights=weights,
            component_nis=component_nis, component_log_likelihood=likelihood)
