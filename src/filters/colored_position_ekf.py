"""Research candidate: pose plus correlated ENU position-measurement bias.

Only controls, positions and innovations enter prediction/update. The initial
pose is supplied by the caller; no reference is accessed by this module.
"""
import numpy as np
from src.models.antenna_observation import antenna_observation


class ColoredPositionEKF:
    def __init__(self, config, initial_pose, tau_seconds, bias_density, white_sigma=.15, bias_prior_sigma=1.):
        self.x=np.r_[np.asarray(initial_pose,float),0.,0.]
        self.tau=np.asarray(tau_seconds,float);self.density=np.asarray(bias_density,float)
        if self.x.shape!=(5,) or not np.isfinite(self.x).all():raise ValueError('Finite pose required')
        if self.tau.shape!=(2,) or np.any(self.tau<=0) or np.isnan(self.tau).any():raise ValueError('Positive tau required; infinity denotes random walk')
        if self.density.shape!=(2,2) or not np.isfinite(self.density).all() or not np.allclose(self.density,self.density.T) or np.linalg.eigvalsh(self.density).min()<0:raise ValueError('PSD finite bias density required')
        if not np.isfinite([white_sigma,bias_prior_sigma]).all() or min(white_sigma,bias_prior_sigma)<=0:raise ValueError('Positive noise scales required')
        f=config['filter'];p=f['process_noise'];initial=f['initial_cov']
        self.P=np.diag([initial['x']**2,initial['y']**2,initial['theta']**2,bias_prior_sigma**2,bias_prior_sigma**2])
        self.q=np.diag([p['q_v']**2,p['q_omega']**2]);self.R=np.eye(2)*white_sigma**2
        self.lever=np.array(f['position_lever_flu_m']);self.angular_input=f.get('angular_input','wheel')
        self.k=config['integrity']['k_integrity']

    def predict(self,v,omega,dt):
        if not np.isfinite([v,omega,dt]).all() or dt<=0:raise ValueError('Finite controls and positive dt required')
        theta=self.x[2];c,s=np.cos(theta),np.sin(theta)
        F=np.eye(5);F[0,2]=-v*s*dt;F[1,2]=v*c*dt
        decay=1/self.tau;phi=np.exp(-decay*dt);F[3:,3:]=np.diag(phi)
        Q=np.zeros((5,5));B=np.array([[c,0],[s,0],[0,1]])
        Q[:3,:3]=B@self.q@B.T*dt**2
        rate=decay[:,None]+decay[None,:]
        integral=np.full((2,2),dt);np.divide(-np.expm1(-rate*dt),rate,out=integral,where=rate>0)
        Q[3:,3:]=self.density*integral
        self.x[:2]+=v*np.array([c,s])*dt;self.x[2]=(theta+omega*dt+np.pi)%(2*np.pi)-np.pi
        self.x[3:]*=phi;self.P=F@self.P@F.T+Q;self.P=(self.P+self.P.T)/2

    def update(self,z):
        z=np.asarray(z,float)
        if z.shape!=(2,) or not np.isfinite(z).all():raise ValueError('Finite position required')
        mean,H3=antenna_observation(self.x[:3],self.lever);H=np.c_[H3,np.eye(2)]
        residual=z-mean-self.x[3:];S=H@self.P@H.T+self.R
        K=np.linalg.solve(S,H@self.P).T;nis=float(residual@np.linalg.solve(S,residual))
        self.x+=K@residual;self.x[2]=(self.x[2]+np.pi)%(2*np.pi)-np.pi
        J=np.eye(5)-K@H;self.P=J@self.P@J.T+K@self.R@K.T;self.P=(self.P+self.P.T)/2
        return nis

    def run_sequence(self,data):
        data.validate();angular=data.angular_velocity(self.angular_input);n=len(data)
        states=np.empty((n,5));cov=np.empty((n,5,5));nis=np.full(n,np.nan);pl=np.empty(n)
        for i in range(n):
            if i:
                self.predict(data.wheel_v[i-1],angular[i-1],data.timestamp[i]-data.timestamp[i-1])
                if data.gnss_valid[i]:nis[i]=self.update([data.gnss_x[i],data.gnss_y[i]])
            states[i]=self.x;cov[i]=self.P;pl[i]=self.k*np.sqrt(max(0.,np.linalg.eigvalsh(self.P[:2,:2]).max()))
        return {'state':states[:,:3],'covariance':cov[:,:3,:3],'nis':nis,'pl':pl,'position_bias':states[:,3:],'augmented_covariance':cov}
