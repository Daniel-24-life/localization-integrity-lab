# Mathematical Foundations of Localization Integrity

This document summarizes the analytical models, error propagation mechanisms, and partial differential equations governing mobile robot positioning integrity.

---

## 1. 2D Differential Drive Kinematic ODE

The nominal kinematic model of a planar mobile robot under unicycle kinematics with wheel velocity slip factors $\alpha_v, \alpha_\omega$ is given by:

$$\begin{aligned}
\dot{x}(t) &= (1 - \alpha_v) v(t) \cos \theta(t) \\
\dot{y}(t) &= (1 - \alpha_v) v(t) \sin \theta(t) \\
\dot{\theta}(t) &= (1 - \alpha_\omega) \omega(t) + b_\omega(t)
\end{aligned}$$

where:
- $v(t), \omega(t)$ are the measured wheel linear and angular velocities;
- $\alpha_v, \alpha_\omega \in [0, 1)$ represent longitudinal and rotational wheel slippage;
- $b_\omega(t)$ denotes the time-varying gyroscope bias, typically modeled as an Ornstein-Uhlenbeck (OU) or random walk process:
  $$\dot{b}_\omega(t) = -\frac{1}{\tau_b} b_\omega(t) + w_b(t), \quad w_b(t) \sim \mathcal{N}(0, q_b)$$

Linearizing about the state trajectory $\mathbf{x} = [x, y, \theta, b_\omega]^T$ yields the continuous error transition matrix $\mathbf{F}(t)$:

$$\mathbf{F}(t) = \begin{bmatrix}
0 & 0 & -v(t)\sin\theta(t) & 0 \\
0 & 0 & v(t)\cos\theta(t) & 0 \\
0 & 0 & 0 & 1 \\
0 & 0 & 0 & -1/\tau_b
\end{bmatrix}$$

---

## 2. Extended Kalman Filter (EKF) & Innovation Covariance

At discrete epoch $k$ with measurement interval $\Delta t = t_k - t_{k-1}$:
1. **Prediction**:
   $$\hat{\mathbf{x}}_k^- = \mathbf{f}(\hat{\mathbf{x}}_{k-1}^+, \mathbf{u}_k)$$
   $$\mathbf{P}_k^- = \mathbf{\Phi}_k \mathbf{P}_{k-1}^+ \mathbf{\Phi}_k^T + \mathbf{Q}_k$$
   where $\mathbf{\Phi}_k = \exp(\mathbf{F}(t_k)\Delta t)$ is computed via Matrix Exponential or 2nd-order Taylor expansion.

2. **Measurement Update (with Antenna Lever-Arm $\mathbf{l}^b = [l_x, l_y]^T$)**:
   $$\mathbf{z}_k = \begin{bmatrix} x_k \\ y_k \end{bmatrix} + \mathbf{R}(\theta_k) \mathbf{l}^b + \mathbf{v}_k, \quad \mathbf{v}_k \sim \mathcal{N}(\mathbf{0}, \mathbf{R}_k)$$
   The innovation vector and its theoretical covariance are:
   $$\mathbf{\gamma}_k = \mathbf{z}_k - \mathbf{h}(\hat{\mathbf{x}}_k^-)$$
   $$\mathbf{S}_k = \mathbf{H}_k \mathbf{P}_k^- \mathbf{H}_k^T + \mathbf{R}_k$$

---

## 3. Absorbing Boundary PDE for First-Exit (HMI Risk)

In safety-critical navigation, **Hazardously Misleading Information (HMI)** occurs when the true position error exceeds the Alert Limit ($\text{PE} > \text{AL}$) while the system fails to raise an alarm ($\text{PL} \le \text{AL}$).

Under diffusion approximations, the cumulative probability $u(x, t)$ that a 1D error process $e(t)$ with drift $\mu$ and diffusion $\sigma$ exits the safe interval $[-L, L]$ before time $t$ satisfies the backward Kolmogorov PDE with absorbing boundary conditions:

$$\frac{\partial u}{\partial t} = \mu \frac{\partial u}{\partial x} + \frac{1}{2}\sigma^2 \frac{\partial^2 u}{\partial x^2}, \quad x \in (-L, L)$$

Subject to boundary conditions:
$$u(-L, t) = 1, \quad u(L, t) = 1, \quad \forall t \ge 0$$
and initial condition:
$$u(x, 0) = 0, \quad x \in (-L, L)$$

For pure Brownian motion ($\mu = 0$), the analytical spectral solution via Fourier series is:

$$u(x, t) = 1 - \sum_{n=0}^{\infty} \frac{4}{(2n+1)\pi} \cos\left(\frac{(2n+1)\pi x}{2L}\right) \exp\left(-\frac{(2n+1)^2 \pi^2 \sigma^2 t}{8L^2}\right)$$

This analytical benchmark is implemented in [`models/absorbing_pde.py`](../src/models/absorbing_pde.py) and verified against Monte Carlo simulations with $< 10^{-4}$ tolerance.

---

## 4. Stanford Diagram Geometry & Operational States

Integrity assessment maps each measurement epoch into one of four operational quadrants:

| Operational State | Condition | Definition | Safety Implication |
| :--- | :--- | :--- | :--- |
| **Normal Operation** | $\text{PE} \le \text{PL} \le \text{AL}$ | Error bounded by protection level within alert limit | Optimal operational availability |
| **Safe Alert** | $\text{PL} > \text{AL}$ & $\text{PE} \le \text{AL}$ | Filter correctly flags high uncertainty | System gracefully degrades or halts |
| **System Outage (Misleading Info)** | $\text{PE} > \text{PL}$ but $\text{PL} > \text{AL}$ | Out-of-spec uncertainty with declared alert | Unusable navigation service |
| **Hazardously Misleading Info (HMI)** | $\text{PE} > \text{AL}$ while $\text{PL} \le \text{AL}$ | Critical error hidden from operator | **Hazardous failure mode** |
