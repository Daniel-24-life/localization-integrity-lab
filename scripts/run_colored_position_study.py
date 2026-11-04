"""Online-innovation candidate comparison on fixed post-training time blocks."""
from pathlib import Path
from dataclasses import fields
import sys,json,copy,argparse
import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.data.dataset import TrajectoryData
from src.filters.antenna_ekf import AntennaFixedEKF
from src.filters.colored_position_ekf import ColoredPositionEKF
from src.evaluation.run_artifacts import ROOT,RunSession,write_json
from src.evaluation.metrics import PerformanceEvaluator as E

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ablation',action='store_true',help='Same-R control and fixed bias-prior sensitivity grid')
    args=parser.parse_args()
    prep=ROOT/'results/runs/20261028T233222_i2nav_baseline_preparation_e856265c'
    source=ROOT/'data_processed/versions/20261028T233222_i2nav_baseline_preparation_e856265c/i2nav_playground00.npz'
    parameter_path=ROOT/'results/runs/20261103T102156_noise_forecast_study_81ecf50c/fitted_parameters.json'
    p=json.loads(parameter_path.read_text(encoding='utf-8'))
    cfg=yaml.safe_load((prep/'baseline_config.yaml').read_text(encoding='utf-8'))
    data=TrajectoryData.load_npz(source);data.require_reference()
    quality=json.loads((prep/'quality.json').read_text(encoding='utf-8'))
    if not quality['whole_horizon']['passed']:raise ValueError('Reference/input quality gate failed')
    # Separate research protocol; original baseline protocol is not retroactively changed.
    cfg['colored_candidate_study']={'posthoc':True,'training_parameters_source':parameter_path.relative_to(ROOT).as_posix(),
        'windows_fraction':[[.75,.875],[.875,1.]],'models':['fixed_R','colored_OU','bias_random_walk'],
        'white_sigma_m':.15,'bias_prior_sigma_m':1.,'bias_initial_mean':[0,0],
        'conditions':['observed','artificial_position_dropout_10s_at_30s'],
        'GT_use':'first pose only; all following reference access in scoring only',
        'independent_recordings':1,'original_protocol_result_replacement':False}
    if args.ablation:
        cfg['colored_candidate_study'].update(models=['fixed_R','fixed_R_015','OU_prior_03','OU_prior_10','OU_prior_30','RW_prior_10'],bias_prior_grid_m=[.3,1.,3.],purpose='posthoc mechanism ablation, no winner promotion')
    with RunSession('colored_position_ablation' if args.ablation else 'colored_position_study',config=cfg,inputs=[source,parameter_path,prep/'baseline_config.yaml',prep/'quality.json']) as run:
        rows=[];fig,axes=plt.subplots(2,2,figsize=(13,8),dpi=140)
        for w,(lo,hi) in enumerate(cfg['colored_candidate_study']['windows_fraction']):
            start=int(len(data)*lo);end=int(len(data)*hi)
            arrays={f.name:getattr(data,f.name)[start:end].copy() for f in fields(TrajectoryData) if f.name!='metadata'}
            arrays['timestamp']-=arrays['timestamp'][0]
            for fault in [False,True]:
                seq=TrajectoryData(**{k:v.copy() for k,v in arrays.items()},metadata=dict(data.metadata))
                if fault:seq.gnss_valid[(seq.timestamp>=30)&(seq.timestamp<40)]=False
                ref=np.c_[seq.gt_x,seq.gt_y,seq.gt_theta];initial=ref[0].copy()
                constructors={'fixed_R':lambda:AntennaFixedEKF(cfg,initial),
                  'colored_OU':lambda:ColoredPositionEKF(cfg,initial,p['effective_ou_tau_seconds'],p['implied_continuous_covariance_m2_per_s']),
                  'bias_random_walk':lambda:ColoredPositionEKF(cfg,initial,[np.inf,np.inf],p['implied_continuous_covariance_m2_per_s'])}
                if args.ablation:
                    small_r=copy.deepcopy(cfg)
                    small_r['filter']['measurement_noise']['r_pos']=.15
                    constructors={'fixed_R':lambda:AntennaFixedEKF(cfg,initial),
                      'fixed_R_015':lambda:AntennaFixedEKF(small_r,initial),
                      'RW_prior_10':lambda:ColoredPositionEKF(cfg,initial,[np.inf,np.inf],p['implied_continuous_covariance_m2_per_s'])}
                    for label,sigma in [('03',.3),('10',1.),('30',3.)]:
                        constructors['OU_prior_'+label]=lambda sigma=sigma:ColoredPositionEKF(cfg,initial,p['effective_ou_tau_seconds'],p['implied_continuous_covariance_m2_per_s'],bias_prior_sigma=sigma)
                for name,constructor in constructors.items():
                    out=constructor().run_sequence(seq)
                    metrics=E.compute_ate(out['state'][:,:2],ref[:,:2])
                    metrics.update(E.evaluate_integrity(out['state'],seq,out['pl'],out['nis'],cfg['integrity']['alert_limit'],cfg['integrity']['nis_threshold']))
                    run.save_sequence(f'w{w}_drop{int(fault)}_{name}',seq,out,metrics)
                    scalar={k:v for k,v in metrics.items() if not isinstance(v,np.ndarray)}
                    rows.append(dict(window=w,artificial_dropout=fault,model=name,**scalar))
                    ax=axes[w,int(fault)];ax.plot(seq.timestamp,np.linalg.norm(out['state'][:,:2]-ref[:,:2],axis=1),label=name,lw=.8)
                    ax.set(title=f'Block {w+1}, artificial dropout={fault}',xlabel='Time / s',ylabel='Position error / m');ax.legend(fontsize=8)
                print('Completed block',w,'dropout',fault,flush=True)
        table=pd.DataFrame(rows);table.to_csv(run.path/'metrics.csv',index=False)
        fig.tight_layout();fig.savefig(run.path/'online_comparison.png');plt.close(fig)
        write_json(run.path/'scope.json',{'GT_in_updates':False,'GT_initial_pose':True,'bias_prior':'zero mean, 1m std; no reference mean subtraction',
         'training_geometry_reference_used':True,'single_record_posthoc':True,'PL_calibrated':False,'production_baseline_replaced':False})
        print(run.path);print(table[['window','artificial_dropout','model','ate_rmse']].to_string(index=False))

if __name__=='__main__':main()
