"""Offline, arrival-aligned response diagnostics, not an exact executed-command trace."""
import numpy as np


def candidate_commands(sample_times, command_times, commands, timeout=.5):
    """Causal last-message association. Receipt clock uncertainty must be reported externally."""
    t=np.asarray(sample_times,float);ct=np.asarray(command_times,float);u=np.asarray(commands,float)
    if not len(ct) or u.shape!=(len(ct),3) or np.any(np.diff(ct)<0):
        raise ValueError('Nonempty ordered command stream [N,3] required')
    index=np.searchsorted(ct,t,side='right')-1
    known=index>=0;index=np.maximum(index,0);age=t-ct[index]
    fresh=known&(age<=timeout)
    selected=u[index].copy()
    selected[~fresh]=0
    return selected,fresh,age


def continuous_duration(times, mask, max_gap=.05):
    t=np.asarray(times,float);mask=np.asarray(mask,bool);duration=np.zeros(len(t))
    for i in range(1,len(t)):
        dt=t[i]-t[i-1]
        if mask[i] and mask[i-1] and 0<dt<=max_gap:
            duration[i]=duration[i-1]+dt
    return duration


def conditioned_speed(times,velocities,candidate,fresh,yaw_limit=.1,dwell=1.):
    t=np.asarray(times,float);v=np.asarray(velocities,float);u=np.asarray(candidate,float)
    condition=np.asarray(fresh)&(np.abs(u[:,0]-.6)<1e-6)&(np.abs(u[:,1])<.01)&(np.abs(u[:,2])<=yaw_limit)
    selected=condition&(continuous_duration(t,condition)>=dwell)
    dt=np.r_[0,np.diff(t)];dt=np.where((dt>0)&(dt<=.05),dt,0)
    weights=dt[selected];values=v[selected,0]
    if not len(values) or not weights.sum():
        return dict(yaw_command_limit=yaw_limit,minimum_continuous_condition_s=dwell,samples=0,duration_s=0,mean_vx=None)
    return dict(yaw_command_limit=yaw_limit,minimum_continuous_condition_s=dwell,
        samples=int(selected.sum()),duration_s=float(weights.sum()),
        mean_vx=float(np.average(values,weights=weights)),sample_p05_p50_p95=np.percentile(values,[5,50,95]).tolist())
