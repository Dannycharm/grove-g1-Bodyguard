"""Read existing robot-state export plus raw command/log channels from an MCAP bag. No ROS node."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import numpy as np
from escort_rl.contract import sha256,load_contract
from escort_rl.measurements import rotation_wxyz
from escort_rl.moving_reference import candidate_commands,continuous_duration,conditioned_speed


def analyze(root, motion_log, out):
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    source_files=[p for p in (root/'bag').iterdir() if p.is_file()]
    source_files += [root/'export/reference_events.jsonl',motion_log]
    before={str(p):sha256(p) for p in source_files}
    out.mkdir(parents=True,exist_ok=False)
    raw=[];logs=[]
    reader=rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(root/'bag'),storage_id=''),rosbag2_py.ConverterOptions('',''))
    inventory={t.name:t.type for t in reader.get_all_topics_and_types()}
    classes={t:get_message(inventory[t]) for t in ('/cmd_vel_raw','/rosout') if t in inventory}
    while reader.has_next():
        topic,data,stamp=reader.read_next()
        if topic not in classes:continue
        m=deserialize_message(data,classes[topic])
        if topic=='/cmd_vel_raw':raw.append(dict(time_ns=int(stamp),command=[m.linear.x,m.linear.y,m.angular.z]))
        else:logs.append(dict(time_ns=int(stamp),level=int(m.level),node=m.name,message=m.msg))
    with open(out/'extra_channels.json','x') as f:json.dump(dict(raw=raw,rosout=logs),f)
    events=[json.loads(line) for line in (root/'export/reference_events.jsonl').open()]
    base=[e for e in events if e['kind']=='base_state'];post=[e for e in events if e['kind']=='command_received'];joints=[e for e in events if e['kind']=='joint_state']
    origin=min(e['bag_timestamp_ns'] for e in events)
    t=(np.array([e['header_stamp_ns'] for e in base],np.int64)-origin)/1e9
    arrival=(np.array([e['bag_timestamp_ns'] for e in base],np.int64)-origin)/1e9
    ct=(np.array([e['bag_timestamp_ns'] for e in post],np.int64)-origin)/1e9
    rt=(np.array([e['time_ns'] for e in raw],np.int64)-origin)/1e9
    u=np.array([e['requested'] for e in post]);ru=np.array([e['command'] for e in raw])
    v=np.array([e['linear_velocity_pelvis'] for e in base]);w=np.array([e['angular_velocity_pelvis'] for e in base]);position=np.array([e['position'] for e in base]);q=np.array([e['quaternion_wxyz'] for e in base])
    candidate,fresh,age=candidate_commands(t,ct,u)
    conditioned=[conditioned_speed(t,v,candidate,fresh,yaw,dwell) for yaw in (.05,.1,.2,.3) for dwell in (0.,.5,1.)]
    dt=np.diff(t)
    timing=dict(base_header_dt_p50_p95_p99_max_ms=(np.percentile(dt,[50,95,99,100])*1000).tolist(),
                nonincreasing_header_times=int(np.sum(dt<=0)),base_gaps_over_50ms=int(np.sum(dt>.05)),
                base_arrival_minus_header_p50_p95_max_ms=(np.percentile(arrival-t,[50,95,100])*1000).tolist())
    matches=[];latency=[]
    for time,cmd in zip(ct,u):
        idx=np.flatnonzero((rt<=time)&(rt>=time-.2))
        exact=idx[np.all(np.isclose(ru[idx],cmd,rtol=0,atol=1e-8),axis=1)]
        matches.append(bool(len(exact)))
        if len(exact):latency.append(time-rt[exact[-1]])
    matching=dict(post_messages=len(post),raw_messages=len(raw),post_with_exact_prior_raw_within_200ms=int(sum(matches)),
        unmatched_post=int(len(matches)-sum(matches)),matched_arrival_lag_p50_p95_max_ms=(np.percentile(latency,[50,95,100])*1000).tolist(),
        qualification='causal content matching only; no publisher attribution or exact controller receipt/execution guarantee')
    upright=1-2*(q[:,1]**2+q[:,2]**2)
    stability=dict(min_height_m=float(position[:,2].min()),min_upright_cosine=float(upright.min()),
        sampled_fall_count=int(np.sum((position[:,2]<.45)|(upright<.5))),
        emergency_latch_observed=False,emergency_status='not recorded; do not interpret as no emergency',
        measured_vx_peak=float(v[:,0].max()),measured_planar_speed_peak=float(np.linalg.norm(v[:,:2],axis=1).max()))
    contract=load_contract();indices=[contract['observation_joints'].index(n) for n in contract['action_joints']]
    dq=np.array([e['joint_velocity'] for e in joints]);speed=np.abs(dq[:,indices])
    stability.update(max_gait_joint_speed_rad_s=float(speed.max()),max_mean_gait_joint_speed_rad_s=float(speed.mean(axis=1).max()))
    misses=[e for e in logs if 'Control loop missed' in e['message']]
    rate_values=[float(re.search(r'Current loop rate is ([0-9.]+)',e['message']).group(1).rstrip('.')) for e in misses]
    log_report=dict(nav2_control_loop_missed_warnings=len(misses),nav2_reported_missed_loop_rate_p05_p50_p95=np.percentile(rate_values,[5,50,95]).tolist(),
        collision_ahead_warnings=sum('detected collision ahead' in e['message'] for e in logs),
        controller_patience_errors=sum('patience exceeded' in e['message'] for e in logs),
        follow_path_aborts=sum('Aborting handle' in e['message'] for e in logs),
        navigation_success_logs=sum(e['node']=='bt_navigator' and 'Goal succeeded' in e['message'] for e in logs))
    receives=[e for e in logs if e['node']=='controller_server' and 'Received a goal' in e['message']]
    successes=[e for e in logs if e['node']=='bt_navigator' and 'Goal succeeded' in e['message']]
    per_goal=[];last=-np.inf
    for i,end in enumerate(successes):
        finish=(end['time_ns']-origin)/1e9
        starts=[(e['time_ns']-origin)/1e9 for e in receives if last<(e['time_ns']-origin)/1e9<=finish]
        begin=min(starts) if starts else last
        mask=(t>=begin)&(t<=finish)
        per_goal.append(dict(index=i+1,start_s=begin,end_s=finish,controller_goal_segments=len(starts),
            forward_conditioned=conditioned_speed(t[mask],v[mask],candidate[mask],fresh[mask],.1,.5)))
        last=finish
    stops=[]
    zero=np.all(np.abs(u)<1e-8,axis=1)
    for i in range(1,len(ct)):
        if not zero[i] or zero[i-1]:continue
        following=np.flatnonzero((ct>ct[i])&(~zero))
        end=min(ct[following[0]] if len(following) else ct[i]+5,ct[i]+5)
        valid=(t>=ct[i])&(t<end);ii=np.flatnonzero(valid)
        if not len(ii):continue
        stopped=(np.linalg.norm(v[:,:2],axis=1)<.05)&(np.abs(w[:,2])<.05)
        duration=continuous_duration(t,stopped&valid)
        hit=np.flatnonzero(valid&(duration>=.5))
        if len(hit):
            confirmation=int(hit[0]);begin=int(np.searchsorted(t,t[confirmation]-.5))
            path=float(np.linalg.norm(np.diff(position[ii[0]:begin+1,:2],axis=0),axis=1).sum())
            stops.append(dict(zero_request_s=float(ct[i]),dwell_start_latency_s=float(t[begin]-ct[i]),
                confirmation_latency_s=float(t[confirmation]-ct[i]),path_to_dwell_m=path,
                qualification='navigation zero-request episode; may already be stopped or turning, not a standardized full-speed stop'))
        else:stops.append(dict(zero_request_s=float(ct[i]),settled=False,window_s=float(end-ct[i])))
    result=dict(source=str(root),motion_log=str(motion_log),goal_results_succeeded=motion_log.read_text().count('Goal finished with status: SUCCEEDED'),
        command_conditioning='base-state mapped header vs recorded post-monitor Twist arrival; freshness <=0.5s; no exact executed-command claim',
        conditioned_forward=conditioned,timing=timing,raw_post_matching=matching,stability=stability,rosout=log_report,
        per_navigation_goal=per_goal,zero_request_episodes=stops,
        full_dynamics_parity_established=False,collision_monitor_state_messages=0)
    with open(out/'analysis.json','x') as f:json.dump(result,f,indent=2)
    after={str(p):sha256(p) for p in source_files}
    if before!=after:raise RuntimeError('Input evidence changed')
    with open(out/'provenance.json','x') as f:json.dump(dict(input_sha256=before,input_unchanged=True,analyzer_sha256=sha256(__file__)),f,indent=2)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,1,figsize=(12,9),sharex=True)
    axes[0].step(rt,ru[:,0],where='post',label='raw vx',alpha=.6);axes[0].step(ct,u[:,0],where='post',label='post-monitor vx');axes[0].plot(t,v[:,0],label='pelvis vx',linewidth=.6)
    axes[1].step(ct,np.clip(u[:,2],-.9,.9),where='post',label='post request clipped to gait cap');axes[1].plot(t,w[:,2],label='pelvis angular z',linewidth=.6)
    axes[2].plot(t,position[:,2],label='pelvis height');axes[2].plot(t,upright,label='upright cosine')
    for ax in axes:ax.legend();ax.grid(alpha=.3)
    axes[0].set_ylabel('m/s');axes[1].set_ylabel('rad/s');axes[2].set_xlabel('seconds relative to first exported record (mixed mapped/arrival clocks)')
    fig.tight_layout();fig.savefig(out/'time_series.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots();ax.plot(position[:,0],position[:,1]);ax.scatter(position[0,0],position[0,1],label='recording start');ax.set(xlabel='native world x (m)',ylabel='native world y (m)',aspect='equal');ax.legend();fig.tight_layout();fig.savefig(out/'route.png',dpi=150);plt.close(fig)
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--reference',required=True);p.add_argument('--motion-log',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    analyze(Path(a.reference),Path(a.motion_log),Path(a.output))
