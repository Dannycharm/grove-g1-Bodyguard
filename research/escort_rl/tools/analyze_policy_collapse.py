"""Read-only checkpoint/trace audit plus optional isolated zero-command stop probes.

Never trains. Route metadata belongs to this evaluator, never policy.predict().
"""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from escort_rl.env import G1EscortEnv
from escort_rl.stage1_scenarios import load_config,sample_episode,FAMILIES


def trace_audit(folder,checkpoint):
    model=PPO.load(checkpoint,device='cpu');config=load_config();ends=[];post=[];means=[];stds=[];nearzero=[];family={}
    for directory in sorted(Path(folder).glob('*/scenario_evaluator_metadata.json')):
        base=directory.parent;meta=json.loads(directory.read_text())
        rows=[json.loads(s) for s in (base/'evaluation.jsonl').read_text().splitlines()]
        actor=[json.loads(s) for s in (base/'observations.jsonl').read_text().splitlines()]
        obs=np.array([s['observation'] for s in actor],np.float32)
        with torch.no_grad():
            distribution=model.policy.get_distribution(torch.as_tensor(obs)).distribution
            means.extend(distribution.mean.numpy().tolist());stds.extend(distribution.stddev.numpy().tolist())
        scenario=sample_episode(meta['seed'],meta['family'],config)
        assert scenario.scenario_metadata()['scenario_id']==meta['scenario_id']
        end=rows[-1];ends.append(end['evaluation']['formation_error_client_frame'])
        for row in rows:
            if np.linalg.norm(row['applied_command'])<.03:nearzero.append(row)
            if row['elapsed_s']>=meta['duration_s']:post.append(row)
        family.setdefault(meta['family'],[]).append(end['evaluation']['formation_error_client_frame'][0])
    def average(rows,key):return np.mean([r[key] for r in rows],axis=0).tolist() if rows else None
    summary=json.loads((Path(folder)/'summary.json').read_text())
    return dict(success=summary['success_rate'],per_family=summary['per_family'],failures=summary['failures'],
        terminal_signed_error_mean=np.mean(ends,axis=0).tolist(),
        terminal_error_norm_mean=float(np.mean(np.linalg.norm(ends,axis=1))),
        terminal_signed_longitudinal_by_family={f:float(np.mean(a)) for f,a in family.items()},
        episodes_ahead_of_target=int(sum(x[0]>0 for x in ends)),episodes_ahead_of_client=int(sum(x[0]>1.5 for x in ends)),
        policy_raw_mean=np.mean(means,axis=0).tolist(),policy_std=np.mean(stds,axis=0).tolist(),
        deterministic_mean_outside_box_fraction=float(np.mean(np.any(np.abs(means)>1,axis=1))),
        postarrival_steps=len(post),postarrival_command_mean=average(post,'applied_command'),
        postarrival_velocity_mean=average(post,'body_velocity'),
        postarrival_forward_command_fraction=float(np.mean([r['applied_command'][0]>.03 for r in post])) if post else None,
        nearzero_command_steps=len(nearzero),nearzero_command_velocity_mean=average(nearzero,'body_velocity'),
        mean_reward_components={k:float(np.mean([e['reward_components'].get(k,0) for e in summary['episodes_detail']])) for k in summary['episodes_detail'][0]['reward_components']})


def stop_probes(checkpoint,out):
    """Same original policy until arrival, then evaluator forces zero. Not policy success."""
    model=PPO.load(checkpoint,device='cpu');results=[]
    for family in FAMILIES:
        env=G1EscortEnv();rows=[];receipt=None;origin=None
        try:
            obs,_=env.reset(options={'family':family,'episode_seed':1000001});elapsed=0.
            arrival=env.scenario_metadata()['duration_s'];done=False
            while not done:
                if elapsed>=arrival:
                    action=np.zeros(3,np.float32)
                    if receipt is None:
                        receipt=elapsed;origin=np.array(env._runner.measure()['position'][:2])
                else:action,_=model.predict(obs,deterministic=True)
                obs,reward,t,tr,info=env.step(action);elapsed=info['elapsed_s'];done=t or tr
                rows.append(info)
            after=[r for r in rows if receipt is not None and r['elapsed_s']>receipt]
            settle=None
            for j in range(len(after)-4):
                if all(np.linalg.norm(r['body_velocity'][:2])<.03 for r in after[j:j+5]):
                    settle=after[j];break
            result=dict(family=family,zero_receipt_s=receipt,client_arrival_s=arrival,
                stopping_time_s=None if settle is None else settle['elapsed_s']-receipt,
                stopping_displacement_m=None if settle is None else float(np.linalg.norm(np.array(settle['robot_position'][:2])-origin)),
                criterion='planar speed <.03m/s for five consecutive 5Hz samples; time resolution .2s',
                censored=settle is None,reason=rows[-1]['reason'],postzero_steps=len(after))
            if after:
                first=next((r for r in after if np.linalg.norm(r['applied_command'])<1e-7),None)
                result['applied_zero_observation_latency_s']=None if first is None else first['elapsed_s']-receipt
            results.append(result);(out/f'stop_probe_{family}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        finally:env.close()
    return results


def plots(evidence,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    config=load_config();fig,axes=plt.subplots(4,2,figsize=(13,13));representative='040_stop_start_1000001'
    for name in ('10k','20k','70k','100k'):
        d=evidence/f'eval_{name}'/representative;m=json.loads((d/'scenario_evaluator_metadata.json').read_text())
        scenario=sample_episode(m['seed'],m['family'],config)
        rows=[json.loads(x) for x in (d/'evaluation.jsonl').read_text().splitlines()];t=[r['elapsed_s'] for r in rows]
        axes[0,0].plot(t,[r['evaluation']['formation_error_client_frame'][0] for r in rows],label=name)
        axes[0,1].plot(t,[r['evaluation']['formation_error_client_frame'][1] for r in rows],label=name)
        axes[1,0].plot(t,[np.linalg.norm(r['body_velocity'][:2]) for r in rows],label=name+' G1')
        if name=='100k':axes[1,0].plot(t,[np.hypot(scenario.client_pose_at(x).vx,scenario.client_pose_at(x).vy) for x in t],color='black',ls='--',label='client')
        for ax,i in ((axes[1,1],0),(axes[2,0],1),(axes[2,1],2)):
            ax.plot(t,[r['applied_command'][i] for r in rows],label=name+' command')
            ax.plot(t,[r['body_velocity'][i] for r in rows],ls=':',alpha=.6,label=name+' measured')
        axes[3,0].plot(t,[sum(r['reward_components'].values()) for r in rows],label=name)
        xy=np.array([r['robot_position'][:2] for r in rows]);axes[3,1].plot(xy[:,0],xy[:,1],label=name+' G1')
        if name=='100k':
            xy=np.array([r['client_position'][:2] for r in rows]);axes[3,1].plot(xy[:,0],xy[:,1],ls='--',color='black',label='client')
    labels=('Signed longitudinal error (m)','Signed lateral error (m)','Planar speed (m/s)',
        'vx command / measured (m/s)','vy command / measured (m/s)','Angular z command / measured (rad/s)',
        'Per-transition total reward','World XY (m)')
    for ax,title in zip(axes.flat,labels):ax.set_title(title);ax.legend(fontsize=6);ax.set_xlabel('Active time (s)');ax.grid(alpha=.2)
    axes[3,1].set_xlabel('World x (m)');axes[3,1].set_ylabel('World y (m)');axes[3,1].axis('equal')
    fig.tight_layout();fig.savefig(out/'checkpoint_stop_comparison.png',dpi=130);plt.close(fig)


def main():
    p=argparse.ArgumentParser();p.add_argument('--evidence',required=True);p.add_argument('--source-run',required=True);p.add_argument('--pilot',required=True);p.add_argument('--output',required=True);p.add_argument('--stop-probe',action='store_true');a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    evidence=Path(a.evidence);run=Path(a.source_run)
    checkpoints={'10k':Path(a.pilot)/'final.zip','20k':run/'round_01/final.zip','70k':run/'round_06/final.zip','100k':run/'round_09/final.zip'}
    results={n:trace_audit(evidence/f'eval_{n}',cp) for n,cp in checkpoints.items()}
    # Check all original checkpoints' evaluated trajectories, not just selected examples.
    trend=[]
    for d in sorted(run.glob('round_??')):
        errors=[]
        for pth in (d/'validation_final').glob('*/evaluation.jsonl'):
            last=json.loads(pth.read_text().splitlines()[-1]);errors.append(last['evaluation']['formation_error_client_frame'][0])
        result=json.loads((d/'result.json').read_text())
        trend.append(dict(timesteps=result['end_timesteps'],positive_terminal_error_fraction=float(np.mean(np.array(errors)>0)),mean_terminal_longitudinal_error_m=float(np.mean(errors))))
    results['overshoot_trend']=trend
    if a.stop_probe:results['forced_zero_diagnostics']=stop_probes(checkpoints['100k'],out)
    (out/'summary.json').write_text(json.dumps(results,indent=2));plots(evidence,out)
    print(json.dumps(results,indent=2))


if __name__=='__main__':main()
