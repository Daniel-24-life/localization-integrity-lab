"""Bounded synthetic mechanism study; no real-data qualification claims."""
import copy
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.synthetic_generator import SyntheticTrajectoryGenerator
from src.filters.antenna_ekf import AntennaFixedEKF
from src.filters.colored_position_ekf import ColoredPositionEKF
from src.evaluation.metrics import PerformanceEvaluator as E
from src.evaluation.run_artifacts import ROOT, RunSession, load_config, write_json


def main():
    parameter_path=ROOT/'results/runs/20261103T102156_noise_forecast_study_81ecf50c/fitted_parameters.json'
    p=json.loads(parameter_path.read_text(encoding='utf-8'))
    tau=np.asarray(p['effective_ou_tau_seconds'])
    density=np.asarray(p['implied_continuous_covariance_m2_per_s'])
    cfg=load_config()
    cfg['simulation'].update(dt=.1,duration=40.)
    cfg['filter'].update(position_lever_flu_m=[0.,0.,0.],angular_input='wheel')
    cfg['filter']['measurement_noise']['r_pos']=.15
    cfg['sensors']['gnss'].update(sigma_pos=.15,update_rate_hz=5)
    cfg['study']={'seeds':list(range(3100,3108)), 'trajectories':['straight','figure8'],
      'noise':['white_only','matched_OU','unmodelled_ramp'],
      'GT_use':'synthetic generation, common first pose initialization, scoring only',
      'bias_initial_generation':'N(0, 1 m^2 I), nonstationary finite-horizon OU',
      'ramp_m_per_s':[.025,-.015], 'ramp_start_s':10.,
      'scope':'exploratory synthetic mechanism transfer; no real independent validation',
      'candidate_parameter_source':parameter_path.relative_to(ROOT).as_posix()}
    with RunSession('colored_synthetic_study',config=cfg,inputs=[parameter_path]) as run:
        rows=[]
        for seed in cfg['study']['seeds']:
            for trajectory in cfg['study']['trajectories']:
                local=copy.deepcopy(cfg);local['project']['random_seed']=seed
                original=SyntheticTrajectoryGenerator(local).generate(trajectory)
                rng=np.random.default_rng(np.random.SeedSequence([seed,77]))
                dt=local['simulation']['dt'];decay=1/tau;rate=decay[:,None]+decay[None,:]
                q=density*(-np.expm1(-rate*dt))/rate
                chol=np.linalg.cholesky(q);bias=np.zeros((len(original),2))
                bias[0]=rng.normal(size=2)
                for i in range(1,len(original)):
                    bias[i]=np.exp(-decay*dt)*bias[i-1]+chol@rng.normal(size=2)
                for noise in cfg['study']['noise']:
                    data=copy.deepcopy(original)
                    b=bias if noise=='matched_OU' else np.zeros_like(bias)
                    if noise=='unmodelled_ramp':
                        b=np.maximum(data.timestamp-10,0)[:,None]*np.array([.025,-.015])
                    data.gnss_x+=b[:,0];data.gnss_y+=b[:,1]
                    data.metadata.update(noise_mechanism=noise,position_observation_source='synthetic position + white noise + declared bias')
                    ref=np.c_[data.gt_x,data.gt_y,data.gt_theta]
                    models={'fixed_white':AntennaFixedEKF(local,ref[0]),
                      'colored_OU':ColoredPositionEKF(local,ref[0],tau,density),
                      'bias_RW':ColoredPositionEKF(local,ref[0],[np.inf,np.inf],density)}
                    for name,model in models.items():
                        output=model.run_sequence(data)
                        metrics=E.compute_ate(output['state'][:,:2],ref[:,:2])
                        metrics.update(E.evaluate_integrity(output['state'],data,output['pl'],output['nis'],cfg['integrity']['alert_limit'],cfg['integrity']['nis_threshold']))
                        run.save_sequence(f'{seed}_{trajectory}_{noise}_{name}',data,output,metrics)
                        rows.append(dict(seed=seed,trajectory=trajectory,noise=noise,model=name,**{k:v for k,v in metrics.items() if not isinstance(v,np.ndarray)}))
            print('Completed seed',seed,flush=True)
        table=pd.DataFrame(rows);table.to_csv(run.path/'metrics.csv',index=False)
        cols=['ate_rmse','hmi_duration_ratio','safe_availability_duration_ratio','false_alarm_duration_ratio']
        summary=table.groupby(['noise','model'])[cols].mean().reset_index()
        summary.to_csv(run.path/'summary.csv',index=False)
        # Paired seed is replication unit; average the two trajectories within seed.
        paired=table.groupby(['seed','noise','model']).ate_rmse.mean().unstack('model')
        effects=[]
        for noise in cfg['study']['noise']:
            group=paired.xs(noise,level='noise')
            for model in ['colored_OU','bias_RW']:
                delta=group[model]-group.fixed_white
                effects.append(dict(noise=noise,model=model,n_seeds=len(delta),mean_delta_m=delta.mean(),min_delta_m=delta.min(),max_delta_m=delta.max(),better_seeds=int((delta<0).sum())))
        pd.DataFrame(effects).to_csv(run.path/'paired_seed_effects.csv',index=False)
        fig,axes=plt.subplots(1,2,figsize=(12,4),dpi=150)
        summary.pivot(index='noise',columns='model',values='ate_rmse').plot.bar(ax=axes[0],rot=0)
        summary.pivot(index='noise',columns='model',values='safe_availability_duration_ratio').plot.bar(ax=axes[1],rot=0)
        axes[0].set(ylabel='Mean sequence RMSE / m',xlabel='Synthetic noise mechanism')
        axes[1].set(ylabel='Mean safe availability duration ratio',xlabel='Synthetic noise mechanism',ylim=(0,1))
        fig.suptitle('8 seeds x 2 trajectories; fixed parameters; synthetic evidence only')
        fig.tight_layout();fig.savefig(run.path/'synthetic_comparison.png');plt.close(fig)
        write_json(run.path/'scope.json',{'sequences':48,'model_evaluations':144,'independent_seed_units':8,'paired_noise_conditions':True,'matched_model_advantage_disclosed':True,'real_G4_evidence':False,'ratio_aggregation':'equal sequence mean, left-hold duration within each sequence'})
        print(run.path);print(summary.to_string(index=False))


if __name__=='__main__':main()
