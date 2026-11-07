"""Plots existing training evidence; never trains or rewrites experiment records."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    run=Path(a.run);out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    episodes=[json.loads(s) for s in (run/'training_episodes.jsonl').read_text().splitlines()]
    steps=[json.loads(s) for s in (run/'training_steps.jsonl').read_text().splitlines()]
    with (run/'logs/progress.csv').open() as f:optimization=list(csv.DictReader(f))
    fig,axes=plt.subplots(2,2,figsize=(11,8));t=[e['timestep'] for e in episodes]
    axes[0,0].plot(t,[e['return_total'] for e in episodes],'.-',label='episode return');axes[0,0].legend()
    for key in ('formation_mean_m','formation_p95_m'):axes[0,1].plot(t,[e[key] for e in episodes],label=key)
    axes[0,1].legend()
    axes[1,0].bar(*zip(*Counter(e['family'] for e in episodes).items()));axes[1,0].set_title('Completed training episodes by family')
    actions=np.array([s['action'] for s in steps])
    for i,label in enumerate(('vx','vy','yaw')):axes[1,1].hist(actions[:,i],bins=40,alpha=.4,label=label)
    axes[1,1].legend();fig.tight_layout();fig.savefig(out/'training_task_metrics.png');plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(13,7))
    for ax,key in zip(axes.flat,('train/explained_variance','train/value_loss','train/policy_gradient_loss','train/entropy_loss','train/approx_kl','train/clip_fraction')):
        points=[(float(r.get('time/total_timesteps') or 0),float(r[key])) for r in optimization if r.get(key)]
        if points:ax.plot(*zip(*points))
        ax.set_title(key)
    fig.tight_layout();fig.savefig(out/'ppo_optimization.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5))
    if episodes:
        for key in episodes[0]['reward_components']:ax.plot(t,[e['reward_components'].get(key,0) for e in episodes],label=key)
    ax.legend();ax.set_xlabel('collected transitions');fig.tight_layout();fig.savefig(out/'reward_components.png');plt.close(fig)
    stats=dict(completed_episode_families=dict(Counter(e['family'] for e in episodes)),
        failures=dict(Counter(e['reason'] for e in episodes)),
        policy_unbounded_action_fraction=float(np.mean(np.any(np.abs(actions)>1,axis=1))),
        observation_clipped_feature_count=sum(s['observation_clipped'] for s in steps),
        mean_action=actions.mean(axis=0).tolist(),std_action=actions.std(axis=0).tolist())
    (out/'diagnostics.json').write_text(json.dumps(stats,indent=2))


if __name__=='__main__':main()
