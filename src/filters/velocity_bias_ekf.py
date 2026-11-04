"""Candidate additive velocity-bias EKF; fixed R, heuristic marginal PL."""
import numpy as np
from .fixed_ekf import FixedEKF
from src.models.velocity_bias import bias_ode,bias_jacobian,frozen_bias_diffusion_covariance


class VelocityBiasEKF(FixedEKF):
    def __init__(self,config,initial_state=None):
        initial=np.zeros(3) if initial_state is None else np.asarray(initial_state,dtype=float)
        if initial.shape!=(3,) or not np.isfinite(initial).all():raise ValueError('Finite initial pose required')
        super().__init__(config,initial.copy())
        options=config.get('velocity_bias',{})
        self.bias_sigma=float(options.get('initial_sigma_m_s',.1))
        self.bias_diffusion_std=float(options.get('diffusion_std_m_s_sqrt_s',.03))
        bias=float(options.get('initial_mean_m_s',0.))
        if not np.isfinite([self.bias_sigma,self.bias_diffusion_std,bias]).all() or min(self.bias_sigma,self.bias_diffusion_std)<0:
            raise ValueError('Finite bias prior and nonnegative noise scales required')
        self.state=np.r_[initial,bias]
        prior=self.P.copy();self.P=np.zeros((4,4));self.P[:3,:3]=prior;self.P[3,3]=self.bias_sigma**2
        self.H=np.eye(4)[:2]

    def predict(self,v,omega,dt=None):
        dt=self.dt if dt is None else float(dt)
        if not np.isfinite(dt) or dt<=0:raise ValueError('Positive finite dt required')
        u=np.array([v,omega]);A=bias_jacobian(self.state,u)
        Phi=np.eye(4)+A*dt
        theta=self.state[2]
        B=np.array([[np.cos(theta),0.],[np.sin(theta),0.],[0.,1.],[0.,0.]])
        Q=B@np.diag([self.q_v**2,self.q_omega**2])@B.T*dt**2
        Q+=frozen_bias_diffusion_covariance(self.state,u,dt,self.bias_diffusion_std**2)
        self.state=self.state+bias_ode(self.state,u)*dt
        self.state[2]=(self.state[2]+np.pi)%(2*np.pi)-np.pi
        self.P=Phi@self.P@Phi.T+Q;self.P=(self.P+self.P.T)/2
        return self.state.copy(),self.P.copy()

    def update(self,z):
        z=np.asarray(z,dtype=float)
        if z.shape!=(2,) or not np.isfinite(z).all():raise ValueError('Finite 2D observation required')
        y=z-self.H@self.state;S=self.H@self.P@self.H.T+self.R
        solved=np.linalg.solve(S,y);nis=float(y@solved)
        K=np.linalg.solve(S,self.H@self.P).T
        self.state+=K@y;self.state[2]=(self.state[2]+np.pi)%(2*np.pi)-np.pi
        I_KH=np.eye(4)-K@self.H
        self.P=I_KH@self.P@I_KH.T+K@self.R@K.T;self.P=(self.P+self.P.T)/2
        return self.state.copy(),self.P.copy(),nis,y

    def run_sequence(self,data):
        data.validate();angular=data.angular_velocity(self.angular_input)
        n=len(data);states=np.zeros((n,4));cov=np.zeros((n,4,4));nis=np.full(n,np.nan);pl=np.zeros(n)
        states[0]=self.state;cov[0]=self.P;pl[0]=self.compute_protection_level()
        for i in range(1,n):
            self.predict(data.wheel_v[i-1],angular[i-1],data.timestamp[i]-data.timestamp[i-1])
            if data.gnss_valid[i]:
                _,_,nis[i],_=self.update([data.gnss_x[i],data.gnss_y[i]])
            states[i]=self.state;cov[i]=self.P;pl[i]=self.compute_protection_level()
        return dict(state=states[:,:3].copy(),covariance=cov[:,:3,:3].copy(),nis=nis,pl=pl,
            augmented_state=states,augmented_covariance=cov,velocity_bias=states[:,3].copy())
