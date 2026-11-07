"""Generate, validate and export evaluator-only Stage 1 trajectories; no simulator required."""
import argparse,csv,json
from dataclasses import asdict
from pathlib import Path
import numpy as np
from escort_rl.stage1_scenarios import load_config,sample_episode,DEFAULT_CONFIG,FAMILIES


def export_episode(episode,config,out):
    out.mkdir(parents=True,exist_ok=False)
    times=np.unique(np.r_[np.arange(0,episode.duration_s,config['scenario']['sample_dt_s']),episode.duration_s])
    rows=[asdict(episode.client_pose_at(float(t))) for t in times]
    with open(out/'client_evaluation_trajectory.csv','x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    with open(out/'metadata.json','x') as f:json.dump(episode.scenario_metadata(),f,indent=2)
    with open(out/'validation.json','x') as f:json.dump(episode.validate(config),f,indent=2)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots();ax.plot([r['x'] for r in rows],[r['y'] for r in rows],label='client')
    ax.scatter(*episode.robot_pose[:2],label='G1 reset spawn');ax.set(xlabel='world x (m)',ylabel='world y (m)',aspect='equal');ax.legend()
    fig.savefig(out/'xy.png');plt.close(fig)
    fig,axes=plt.subplots(3,1,sharex=True)
    axes[0].plot(times,[[r['x'],r['y']] for r in rows]);axes[0].set_ylabel('position (m)')
    axes[1].plot(times,[np.hypot(r['vx'],r['vy']) for r in rows]);axes[1].set_ylabel('speed (m/s)')
    axes[2].plot(times,[r['yaw'] for r in rows],label='heading');axes[2].plot(times,[r['yaw_rate'] for r in rows],label='yaw rate');axes[2].legend();axes[2].set_xlabel('scenario elapsed seconds')
    fig.tight_layout();fig.savefig(out/'time_series.png');plt.close(fig)
    return dict(family=episode.scenario_metadata()['family'],duration_s=episode.duration_s,distance_m=episode.length,
        peak_speed_m_s=max(np.hypot(r['vx'],r['vy']) for r in rows),pause_s=episode.scenario_metadata()['pause_s'],evaluator_only=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default=str(DEFAULT_CONFIG));p.add_argument('--family',choices=FAMILIES,required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--output',required=True);a=p.parse_args()
    c=load_config(a.config);episode=sample_episode(a.seed,a.family,c)
    print(json.dumps(export_episode(episode,c,Path(a.output)),indent=2))
