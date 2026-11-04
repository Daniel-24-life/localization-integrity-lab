"""Independent likelihood-weighted filter bank, not an IMM or calibrated monitor."""
import copy
import numpy as np
from scipy.special import logsumexp
from .velocity_bias_ekf import VelocityBiasEKF


def mixture_moments(states,covariances,weights):
    states=np.asarray(states,dtype=float);covariances=np.asarray(covariances,dtype=float);w=np.asarray(weights,dtype=float)
    if states.shape!=(2,4) or covariances.shape!=(2,4,4) or w.shape!=(2,) or not np.isfinite(states).all() or not np.isfinite(covariances).all() or not np.isfinite(w).all() or np.any(w<0) or not np.isclose(w.sum(),1.):
        raise ValueError('Two finite 4D moments and normalized weights required')
    sine=w@np.sin(states[:,2]);cosine=w@np.cos(states[:,2])
    if np.hypot(sine,cosine)<1e-10:raise ValueError('Circular heading mixture is ambiguous')
    mean=w@states;mean[2]=np.arctan2(sine,cosine)
    delta=states-mean;delta[:,2]=(delta[:,2]+np.pi)%(2*np.pi)-np.pi
    P=np.einsum('i,ijk->jk',w,covariances)+np.einsum('i,ij,ik->jk',w,delta,delta)
    return mean,(P+P.T)/2


class BiasModelBank:
    def __init__(self,config,initial_state=None):
        options=config.get('model_bank',{})
        p=float(options.get('initial_bias_probability',.05));self.floor=float(options.get('posterior_uniform_floor',.01))
        self.learn=options.get('learn_weights',True)
        if not np.isfinite([p,self.floor]).all() or not 0<p<1 or not 0<=self.floor<.5 or type(self.learn) is not bool:
            raise ValueError('Invalid bank probabilities/settings')
        self.weights=np.array([1-p,p]);self.k=config.get('integrity',{}).get('k_integrity',3.)
        healthy=copy.deepcopy(config);healthy['velocity_bias']=dict(initial_mean_m_s=0.,initial_sigma_m_s=0.,diffusion_std_m_s_sqrt_s=0.)
        self.models=[VelocityBiasEKF(healthy,initial_state),VelocityBiasEKF(config,initial_state)]
        self.angular_input=self.models[0].angular_input

    def moments(self):
        return mixture_moments([m.state for m in self.models],[m.P for m in self.models],self.weights)

    def run_sequence(self,data):
        data.validate();omega=data.angular_velocity(self.angular_input);n=len(data)
        state=np.zeros((n,4));P=np.zeros((n,4,4));weights=np.zeros((n,2));nis=np.full(n,np.nan)
        component_nis=np.full((n,2),np.nan);likelihood=np.full((n,2),np.nan);pl=np.zeros(n)
        state[0],P[0]=self.moments();weights[0]=self.weights
        pl[0]=self.k*np.sqrt(max(0.,np.linalg.eigvalsh(P[0,:2,:2]).max()))
        for i in range(1,n):
            for m in self.models:m.predict(data.wheel_v[i-1],omega[i-1],data.timestamp[i]-data.timestamp[i-1])
            if data.gnss_valid[i]:
                z=np.array([data.gnss_x[i],data.gnss_y[i]])
                prior,prior_P=self.moments();y=z-prior[:2];S=prior_P[:2,:2]+self.models[0].R
                nis[i]=float(y@np.linalg.solve(S,y))
                for j,m in enumerate(self.models):
                    residual=z-m.state[:2];cov=m.P[:2,:2]+m.R
                    sign,logdet=np.linalg.slogdet(cov)
                    if sign<=0:raise ValueError('Invalid innovation covariance')
                    component_nis[i,j]=float(residual@np.linalg.solve(cov,residual))
                    likelihood[i,j]=-.5*(2*np.log(2*np.pi)+logdet+component_nis[i,j])
                if self.learn:
                    log_prior=np.full(2,-np.inf)
                    np.log(self.weights,out=log_prior,where=self.weights>0)
                    logp=log_prior+likelihood[i]
                    posterior=np.exp(logp-logsumexp(logp))
                    self.weights=(1-2*self.floor)*posterior+self.floor
                for m in self.models:m.update(z)
            # Missing position data leaves model weights unchanged: no invented health evidence.
            state[i],P[i]=self.moments();weights[i]=self.weights
            pl[i]=self.k*np.sqrt(max(0.,np.linalg.eigvalsh(P[i,:2,:2]).max()))
        return dict(state=state[:,:3].copy(),covariance=P[:,:3,:3].copy(),nis=nis,pl=pl,
            augmented_state=state,augmented_covariance=P,model_weights=weights,
            component_nis=component_nis,component_log_likelihood=likelihood)
