"""CandidateBankV2: Advanced adaptive filter bank with CUSUM recovery, degradation alert, and SE(2) integration.

Inherits from BiasModelBank with three formal ablation mechanisms:
1. enable_cusum: CUSUM / change-point soft reset of velocity bias state on slip termination.
2. enable_degradation: Graceful degradation alert when GNSS outage exceeds tau_crit = 1.5s.
3. enable_se2: Symplectic SE(2) Lie group closed-form circular arc integration during turning.

When enable_cusum=False, enable_degradation=False, enable_se2=False, output is 100% array-identical to BiasModelBank (V1).
"""
import copy
import numpy as np
from scipy.special import logsumexp

from .bias_model_bank import BiasModelBank, mixture_moments
from .velocity_bias_ekf import VelocityBiasEKF


class VelocityBiasEKF_V2(VelocityBiasEKF):
    """VelocityBiasEKF with optional SE(2) closed-form arc integration."""
    def __init__(self, config, initial_state=None, enable_se2=False):
        super().__init__(config, initial_state)
        self.enable_se2 = bool(enable_se2)

    def predict(self, v, omega, dt=None):
        dt = self.dt if dt is None else float(dt)
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError('Positive finite dt required')
            
        u = np.array([v, omega])
        from src.models.velocity_bias import bias_jacobian, frozen_bias_diffusion_covariance
        A = bias_jacobian(self.state, u)
        Phi = np.eye(4) + A * dt
        theta = self.state[2]
        B = np.array([[np.cos(theta), 0.], [np.sin(theta), 0.], [0., 1.], [0., 0.]])
        Q = B @ np.diag([self.q_v**2, self.q_omega**2]) @ B.T * dt**2
        Q += frozen_bias_diffusion_covariance(self.state, u, dt, self.bias_diffusion_std**2)
        
        v_eff = v - self.state[3]
        if self.enable_se2 and abs(omega) > 1e-6:
            # Exact closed-form SE(2) arc integration
            dtheta = omega * dt
            sin_dth = np.sin(dtheta)
            cos_dth = np.cos(dtheta)
            dx_body = (v_eff / omega) * sin_dth
            dy_body = (v_eff / omega) * (1.0 - cos_dth)
            
            c_th = np.cos(theta)
            s_th = np.sin(theta)
            self.state[0] += c_th * dx_body - s_th * dy_body
            self.state[1] += s_th * dx_body + c_th * dy_body
            self.state[2] = (theta + dtheta + np.pi) % (2 * np.pi) - np.pi
        else:
            # Standard explicit Euler integration
            from src.models.velocity_bias import bias_ode
            self.state = self.state + bias_ode(self.state, u) * dt
            self.state[2] = (self.state[2] + np.pi) % (2 * np.pi) - np.pi
            
        self.P = Phi @ self.P @ Phi.T + Q
        self.P = (self.P + self.P.T) / 2
        return self.state.copy(), self.P.copy()


class CandidateBankV2(BiasModelBank):
    """CandidateBankV2 with CUSUM recovery, degradation alert, and SE(2) integration."""
    def __init__(
        self,
        config,
        initial_state=None,
        enable_cusum=True,
        enable_degradation=True,
        enable_se2=True,
        tau_crit=1.5,
        alert_limit=0.5,
        cusum_threshold=2.0,
    ):
        super().__init__(config, initial_state)
        self.enable_cusum = bool(enable_cusum)
        self.enable_degradation = bool(enable_degradation)
        self.enable_se2 = bool(enable_se2)
        self.tau_crit = float(tau_crit)
        self.alert_limit = float(alert_limit)
        self.cusum_threshold = float(cusum_threshold)
        
        # Re-initialize models with SE(2) if enabled
        if self.enable_se2:
            healthy = copy.deepcopy(config)
            healthy['velocity_bias'] = dict(initial_mean_m_s=0., initial_sigma_m_s=0., diffusion_std_m_s_sqrt_s=0.)
            self.models = [
                VelocityBiasEKF_V2(healthy, initial_state, enable_se2=True),
                VelocityBiasEKF_V2(config, initial_state, enable_se2=True)
            ]
            
        self.p0 = 1.0 - float(config.get('model_bank', {}).get('initial_bias_probability', 0.05))
        self.cusum_stat = 0.0

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
        system_alert = np.zeros(n, dtype=bool)
        
        state[0], P[0] = self.moments()
        weights[0] = self.weights
        pl[0] = self.k * np.sqrt(max(0., np.linalg.eigvalsh(P[0, :2, :2]).max()))
        
        last_gnss_time = data.timestamp[0]
        consecutive_healthy_count = 0
        
        for i in range(1, n):
            dt = data.timestamp[i] - data.timestamp[i-1]
            time_since_gnss = data.timestamp[i] - last_gnss_time
            
            # 1. State prediction
            for m in self.models:
                m.predict(data.wheel_v[i-1], omega[i-1], dt)
                
            # 2. Measurement update
            if data.gnss_valid[i]:
                last_gnss_time = data.timestamp[i]
                z = np.array([data.gnss_x[i], data.gnss_y[i]])
                
                prior, prior_P = self.moments()
                y = z - prior[:2]
                S = prior_P[:2, :2] + self.models[0].R
                nis[i] = float(y @ np.linalg.solve(S, y))
                
                for j, m in enumerate(self.models):
                    residual = z - m.state[:2]
                    cov = m.P[:2, :2] + m.R
                    sign, logdet = np.linalg.slogdet(cov)
                    if sign <= 0:
                        raise ValueError('Invalid innovation covariance')
                    component_nis[i, j] = float(residual @ np.linalg.solve(cov, residual))
                    likelihood[i, j] = -0.5 * (2 * np.log(2 * np.pi) + logdet + component_nis[i, j])
                    
                # CUSUM Change-point Detection & Soft Reset
                if self.enable_cusum and self.weights[1] > 0.25:
                    # Healthy innovation log-likelihood advantage
                    delta_loglik = likelihood[i, 0] - likelihood[i, 1]
                    if delta_loglik > 0:
                        self.cusum_stat = max(0.0, self.cusum_stat + delta_loglik)
                    else:
                        self.cusum_stat = max(0.0, self.cusum_stat * 0.5)
                        
                    # Check if change-point condition is met
                    # Condition: M0 innovation is small (within chi-square 95% threshold) and advantage accumulated
                    if (component_nis[i, 0] < 4.0 and self.cusum_stat >= self.cusum_threshold) or (component_nis[i, 0] < 2.0 and delta_loglik > 1.5):
                        consecutive_healthy_count += 1
                        if consecutive_healthy_count >= 1:
                            # Trigger Soft Reset of bias model
                            self.models[1].state[3] *= 0.05
                            self.models[1].P[3, 3] = self.models[1].bias_sigma**2
                            self.weights = np.array([0.95, 0.05])
                            self.cusum_stat = 0.0
                    else:
                        consecutive_healthy_count = 0
                else:
                    self.cusum_stat = 0.0
                    consecutive_healthy_count = 0
                    
                # Bayesian model weight update
                if self.learn:
                    log_prior = np.full(2, -np.inf)
                    np.log(self.weights, out=log_prior, where=self.weights > 0)
                    logp = log_prior + likelihood[i]
                    posterior = np.exp(logp - logsumexp(logp))
                    self.weights = (1 - 2 * self.floor) * posterior + self.floor
                    
                for m in self.models:
                    m.update(z)
            else:
                # GNSS outage: weights remain strictly frozen (no fabricated health evidence)
                pass
                
            state[i], P[i] = self.moments()
            weights[i] = self.weights
            nominal_pl = self.k * np.sqrt(max(0., np.linalg.eigvalsh(P[i, :2, :2]).max()))
            
            # Graceful degradation alert during prolonged GNSS outage
            if self.enable_degradation and (data.timestamp[i] - last_gnss_time > self.tau_crit):
                # When outage exceeds critical drift threshold, declare system degraded alert
                system_alert[i] = True
                pl[i] = max(nominal_pl, self.alert_limit + 0.01)
            else:
                pl[i] = nominal_pl
                system_alert[i] = (nominal_pl > self.alert_limit)
                
        return dict(
            state=state[:, :3].copy(),
            covariance=P[:, :3, :3].copy(),
            nis=nis,
            pl=pl,
            system_alert=system_alert,
            augmented_state=state,
            augmented_covariance=P,
            model_weights=weights,
            component_nis=component_nis,
            component_log_likelihood=likelihood,
        )
