"""Time-ordered noise modelling study and portable HTML research report."""
from pathlib import Path
import sys,json,html,shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.evaluation.run_artifacts import ROOT,RunSession,write_json
from src.evaluation.residual_identification import load_residual_series
from src.evaluation.correlated_noise import fit_models,evaluate_segment,calibrate_covariance

def main():
    source=ROOT/'data_processed/versions/20261028T233222_i2nav_baseline_preparation_e856265c/i2nav_playground00.npz'
    data=load_residual_series(source);n=len(data.timestamp_s)
    cut1=n//2;cut2=3*n//4
    dt=np.diff(data.timestamp_s);median_dt=float(np.median(dt))
    # Event-based AR is accepted only for approximately periodic input.
    if np.max(np.abs(dt-median_dt))>.1*median_dt:raise ValueError('Irregular events require variable-dt model')
    config={'study':'chronological_residual_noise_forecast','fractions':[.5,.25,.25],
            'horizons_events':[1,5,20],'model_selection':'minimum validation RMSE at horizon 5 only',
            'calibration':'common scale max(1, validation q95(mahalanobis2)/chi2_2_95 over horizons)',
            'phi_clip':[.0001,.9999],'covariance_shrinkage':.1,'oracle_initial_residual':True,
            'deployment_claim':False,'data_reuse':'posthoc development of already inspected recording',
            'training_events':[0,cut1],'validation_events':[cut1,cut2],'test_events':[cut2,n]}
    with RunSession('noise_forecast_study',config=config,inputs=[source,source.with_suffix('.metadata.json')]) as run:
        parameters=fit_models(data.residual_m[:cut1])
        parameters['effective_ou_tau_seconds']=-median_dt/np.log(parameters['phi'])
        decay=1/parameters['effective_ou_tau_seconds']
        rate=decay[:,None]+decay[None,:]
        continuous=parameters['innovation_covariance']*rate/(-np.expm1(-rate*median_dt))
        parameters['implied_continuous_covariance_m2_per_s']=continuous
        parameters['ou_embedding_psd']=bool(np.linalg.eigvalsh(continuous).min()>=0)
        validation,_=evaluate_segment(data.residual_m,cut1,cut2,parameters)
        selected=min([r for r in validation if r['horizon_events']==5],key=lambda r:r['rmse_norm_m'])['model']
        parameters['validation_covariance_scale']=calibrate_covariance(data.residual_m,cut1,cut2,parameters)
        test,series=evaluate_segment(data.residual_m,cut2,n,parameters)
        rows=[dict(split='validation',**r) for r in validation]+[dict(split='test',**r) for r in test]
        pd.DataFrame(rows).to_csv(run.path/'forecast_metrics.csv',index=False)
        write_json(run.path/'fitted_parameters.json',parameters)
        write_json(run.path/'selection.json',{'selected_on_validation':selected,'test_used_for_selection':False,'median_event_dt_seconds':median_dt,'independent_recordings':1,'test_is_prospective_unseen':False})
        np.savez_compressed(run.path/'test_predictions.npz',**series)
        plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei','DejaVu Sans']
        fig,axes=plt.subplots(2,1,figsize=(12,7),dpi=160)
        t=data.timestamp_s-data.timestamp_s[0]
        for j,ax in enumerate(axes):
            ax.plot(t,data.residual_m[:,j],lw=.8,label='名义天线残差')
            ax.axvline(t[cut1],color='gray',ls='--');ax.axvline(t[cut2],color='black',ls='--')
            idx=series['target_indices_h5'];ax.plot(t[idx],series[f'{selected}_h5'][:,j],lw=.9,label=f'5事件预测：{selected}')
            ax.set_ylabel(['东向 / m','北向 / m'][j]);ax.legend()
        axes[-1].set_xlabel('记录内时间 / s')
        fig.suptitle('拟合50% → 选择25% → 时间后段25%；使用过去参考残差初始化，非在线定位成绩')
        fig.tight_layout();fig.savefig(run.path/'chronological_forecasts.png');plt.close(fig)
        frame=pd.DataFrame(test);fig,ax=plt.subplots(figsize=(9,4.5),dpi=160)
        for name,g in frame.groupby('model'):ax.plot(g.horizon_events,g.rmse_norm_m,'o-',label=name)
        ax.set(xlabel='预测跨度 / 事件（约秒）',ylabel='残差预测 RMSE / m',title='后25%时间段：四种噪声模型外推比较');ax.legend();fig.tight_layout()
        fig.savefig(run.path/'horizon_comparison.png');plt.close(fig)
        table=frame[['model','horizon_events','rmse_norm_m','mean_nll','ellipse95_empirical_coverage']].to_html(index=False,float_format=lambda v:f'{v:.4f}',border=0)
        report=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>移动机器人：噪声模型与时间外推</title><style>body{{max-width:1080px;margin:40px auto;padding:0 24px;font:17px/1.8 "Microsoft YaHei",sans-serif;color:#203343}}h1,h2{{color:#1f4e79}}img{{max-width:100%}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:9px;border-bottom:1px solid #ccc}}.note{{background:#edf3f8;padding:18px}}.scroll{{overflow:auto}}</style><h1>噪声模型能否跨时间预测？</h1><p>科研增量 · 2026-11-03 · 单条 i2Nav 记录 · {n} 个位置事件</p><p class="note">这是一项名义天线残差预测研究。过去参考残差被用于预测初始化，不能把这里的低误差称为机器人在线定位精度。该记录此前已被分析，本次后段属于时间外推诊断，不是新的独立盲测。</p><h2>数学与拟合</h2><p>比较零均值固定尺度、训练均值＋协方差、随机游走持久性、对角相关噪声四个模型。相关模型为 r[k+1]=μ+diag(φ)(r[k]−μ)+ε；φ仅用前半段估计，限制在[0.0001,0.9999]，创新协方差作10%对角收缩。τ=−Δt/log(φ)只是一种OU时间尺度解释，尚非机器人PDE参数标定。</p><p>前{cut1}事件拟合；中间{cut2-cut1}事件选择；最后{n-cut2}事件只评分。验证段按5事件预测RMSE选出 <b>{html.escape(selected)}</b>，测试段不参与选择。模型参数不在线重拟合；每次预测使用起点的参考残差，因此属于条件预测诊断。</p><img src="chronological_forecasts.png" alt="时间顺序预测与数据切分"><h2>真实计算结果</h2><img src="horizon_comparison.png" alt="不同预测跨度误差"><div class="scroll">{table}</div><p>覆盖率是模型95%椭圆在本后段上的经验命中率，不是风险保证；低RMSE不意味着概率校准良好。重叠预测存在依赖，没有给逐事件独立置信区间。</p><h2>数据分类与局限</h2><p>不同接收机/处理链路但共享RF环境；名义杆臂已投影，滚转俯仰遗漏和物理点验证限制仍在。未减偏置重报原滤波得分、未修改AL/G4、未宣称独立真实验证。原始数据未包含在此分享包中。</p><h2>复现与来源</h2><p>在原项目执行 <code>python scripts/run_noise_forecast_study.py</code>。本次run：{run.path.name}。随包CSV、参数JSON及selection.json可核对图表；精确输入哈希在source_manifest.json。需要原版本化数据才能重跑。整个HTML与两张PNG可离线分享。</p></html>'''
        report=report.replace('<h2>数据分类与局限</h2>',f'<h2>协方差校准</h2><p>仅验证段确定统一协方差放大系数 {parameters["validation_covariance_scale"]:.4f}，取三个跨度的验证Mahalanobis平方95%分位数与二维卡方95%阈值之比的最大值，最小为1。ou_calibrated不改变均值预测，只扩大不确定性。该修订在首次查看后段结果后提出，属于事后开发分析，即使参数只由验证段拟合，也不能称为预注册盲测或分布无关覆盖保证。</p><h2>数据分类与局限</h2>')
        (run.path/'index.html').write_text(report,encoding='utf-8')
        print(run.path);print('Selected:',selected);print(frame.to_string(index=False))
    # Share only aggregate artifacts; do not redistribute raw/reference data.
    out=ROOT/'deliverables'/run.path.name;out.mkdir(exist_ok=False)
    for name in ['index.html','chronological_forecasts.png','horizon_comparison.png','forecast_metrics.csv','fitted_parameters.json','selection.json']:
        shutil.copy2(run.path/name,out/name)
    manifest=json.loads((run.path/'manifest.json').read_text(encoding='utf-8'))
    for item in manifest['inputs']:
        item['path']=Path(item['path']).relative_to(ROOT).as_posix()
    manifest.pop('platform',None);manifest['command']=['python','scripts/run_noise_forecast_study.py']
    write_json(out/'source_manifest.json',manifest)
    print('Share report:',out/'index.html')

if __name__=='__main__':main()
