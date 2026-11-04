"""Quality-first diagnostics; no filter scores or implicit ground-truth heading."""
import numpy as np


def window_quality(stream, start, duration, rules):
    """Evaluate fixed raw/kinematic rules, not estimated-position performance."""
    t=stream['timestamp'];end=start+duration
    rt=stream['reference_time'];xy=stream['reference_xy']
    inside=(rt>=start)&(rt<=end);wheel=(t>=start)&(t<=end)
    reasons=[];result=dict(start_sec=start,end_sec=end)
    if start<max(t[0],rt[0]) or end>min(t[-1],rt[-1]) or inside.sum()<2 or wheel.sum()<2:
        return dict(**result,candidate=False,reasons=['outside common support'])
    grid=np.arange(start,end+1e-8,1.)
    ref=np.column_stack([np.interp(grid,rt,xy[:,k]) for k in [0,1]])
    speeds=np.linalg.norm(np.diff(ref,axis=0),axis=1)
    v=np.interp((grid[1:]+grid[:-1])/2,t,stream['wheel_v'])
    omega=np.abs(stream['imu_gz'][wheel])
    # Include bracketing intervals at both endpoints, not only interior samples.
    # A long leading/trailing gap must not disappear from the quality check.
    left=max(0,np.searchsorted(rt,start,side='right')-1)
    right=min(len(rt),np.searchsorted(rt,end,side='left')+1)
    selected_rt=rt[left:right];selected_xy=xy[left:right]
    result.update(reference_max_gap_sec=float(np.diff(selected_rt).max()),
        reference_max_step_m=float(np.linalg.norm(np.diff(selected_xy,axis=0),axis=1).max()),
        reference_speed_median_mps=float(np.median(speeds)),reference_speed_max_mps=float(speeds.max()),
        wheel_reference_speed_difference_p95_mps=float(np.quantile(np.abs(v-speeds),.95)),
        absolute_imu_yaw_rate_p95_radps=float(np.quantile(omega,.95)))
    limits=[('reference_max_gap_sec','max_reference_gap_sec'),('reference_max_step_m','max_reference_step_m'),
            ('reference_speed_max_mps','max_reference_speed_mps'),
            ('wheel_reference_speed_difference_p95_mps','max_speed_disagreement_p95_mps'),
            ('absolute_imu_yaw_rate_p95_radps','max_yaw_rate_p95_radps')]
    for metric,limit in limits:
        if result[metric]>rules[limit]:reasons.append(metric+' exceeds '+limit)
    if result['reference_speed_median_mps']<rules['min_reference_median_speed_mps']:
        reasons.append('insufficient translational motion')
    return dict(**result,candidate=not reasons,reasons=reasons,
        qualification='kinematic candidate only; device/point/reference quality not certified')
