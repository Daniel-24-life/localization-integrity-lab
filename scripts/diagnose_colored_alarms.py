"""Attribute existing alarms to PL and NIS without changing or rerunning filters."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.evaluation.run_artifacts import ROOT,RunSession


def main():
    source=ROOT/'results/runs/20261103T103154_colored_synthetic_study_c3a98f4b'
    cfg=yaml.safe_load((source/'config.yaml').read_text(encoding='utf-8'))
    metrics=pd.read_csv(source/'metrics.csv');inputs=[source/'metrics.csv',source/'config.yaml']
    inputs += [source/f'{r.seed}_{r.trajectory}_{r.noise}_{r.model}_frames.npz' for r in metrics.itertuples()]
    with RunSession('colored_alarm_attribution',config=cfg,inputs=inputs) as run:
        rows=[]
        for row,path in zip(metrics.itertuples(),inputs[2:]):
            with np.load(path) as a:
                t=a['timestamp'];w=np.r_[np.diff(t),0.];total=w.sum()
                pl=a['estimate_pl'];nis=a['estimate_nis']
                error=np.linalg.norm(a['estimate_state'][:,:2]-np.c_[a['gt_x'],a['gt_y']],axis=1)
                p=pl>cfg['integrity']['alert_limit']
                n=np.isfinite(nis)&(nis>cfg['integrity']['nis_threshold'])
                safe=error<=cfg['integrity']['alert_limit']
                ratio=lambda mask:float(w[mask].sum()/total)
                values={'pl_only':ratio(p&~n),'nis_only':ratio(n&~p),'both':ratio(p&n),'neither':ratio(~p&~n),
                  'false_alarm_pl_only':ratio(safe&p&~n),'false_alarm_nis_only':ratio(safe&n&~p),'false_alarm_both':ratio(safe&p&n),
                  'safe_availability':ratio(safe&~p&~n),'error_exceedance':ratio(~safe),
                  'pl_mean_m':float(w@pl/total),'position_error_mean_m':float(w@error/total),
                  'nis_valid_events':int(np.isfinite(nis).sum()),'nis_exceedance_events':int(n.sum())}
                assert np.isclose(sum(values[k] for k in ['pl_only','nis_only','both','neither']),1)
                assert np.isclose(values['safe_availability'],row.safe_availability_duration_ratio)
                assert np.isclose(sum(values[k] for k in ['false_alarm_pl_only','false_alarm_nis_only','false_alarm_both']),row.false_alarm_duration_ratio)
                rows.append(dict(seed=row.seed,trajectory=row.trajectory,noise=row.noise,model=row.model,**values))
        table=pd.DataFrame(rows);table.to_csv(run.path/'attribution.csv',index=False)
        summary=table.groupby(['noise','model']).mean(numeric_only=True).drop(columns='seed').reset_index()
        summary.to_csv(run.path/'summary.csv',index=False)
        fig,ax=plt.subplots(figsize=(11,5),dpi=150)
        white=summary[summary.noise=='white_only'].set_index('model')
        white[['pl_only','nis_only','both','neither']].plot.bar(stacked=True,ax=ax,rot=0)
        ax.set(ylabel='Mean duration fraction',xlabel='Model',title='White-noise negative control: alarm attribution',ylim=(0,1))
        ax.legend(loc='upper left',bbox_to_anchor=(1,1));fig.tight_layout();fig.savefig(run.path/'alarm_attribution.png');plt.close(fig)
        print(run.path);print(white[['pl_only','nis_only','both','neither','false_alarm_pl_only']].to_string())


if __name__=='__main__':main()
