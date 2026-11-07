"""Full-G1 verification only: zero/random/causal diagnostic follower; never PPO or ROS."""
import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import time
import numpy as np
from escort_rl.env import G1EscortEnv
from escort_rl.stage1_scenarios import DEFAULT_CONFIG,FAMILIES


def diagnostic_action(observation,config):
    """Uses only the same normalized actor vector/config as a future PPO actor.

    PD-like position tracking with a fixed dead-zone compensation, not Reactive Baseline A.
    No route object, family, progress, endpoint or evaluator info is accepted.
    """
    o=config['observation'];obs=np.asarray(observation)
    relative=obs[0:2]*o['position_scale_m']
    relative_v=obs[2:4]*o['relative_velocity_scale_m_s']
    delta=np.arctan2(obs[4],obs[5]);R=np.array([[np.cos(delta),-np.sin(delta)],[np.sin(delta),np.cos(delta)]])
    offset=obs[6:8]*o['offset_scale_m'];target=relative+R@offset
    measured=obs[10:13]*o['robot_velocity_scales']
    desired=relative_v+measured[:2]+.65*target
    commands=np.zeros(3)
    for i in (0,1):
        if abs(desired[i])>.025:
            commands[i]=desired[i]/.8+np.sign(desired[i])*(.08 if i==0 else .10)
    commands[2]=1.8*delta+.15*target[1]
    return np.clip(commands/np.array(config['environment']['command_caps']),-1,1).astype(np.float32)


def plot_episode(rows,out,config):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if not rows:return
    t=[r['elapsed_s'] for r in rows]
    fig,axes=plt.subplots(2,2,figsize=(11,8))
    ax=axes[0,0];robot=np.array([r['robot_position'] for r in rows]);client=np.array([r['client_position'] for r in rows])
    ax.plot(robot[:,0],robot[:,1],label='G1');ax.plot(client[:,0],client[:,1],label='client');ax.axis('equal');ax.set(xlabel='world x (m)',ylabel='world y (m)');ax.legend()
    axes[0,1].plot(t,[r['evaluation']['formation_error_m'] for r in rows],label='formation error norm')
    for axis,label in enumerate(['longitudinal error','lateral error']):
        axes[0,1].plot(t,[r['evaluation']['formation_error_client_frame'][axis] for r in rows],label=label,alpha=.6)
    axes[0,1].axhline(0.,color='black',linestyle=':',label='desired error')
    axes[0,1].axhline(config['outcomes']['final_formation_tolerance_m'],linestyle='--',label='final tolerance');axes[0,1].set_ylabel('client-frame error (m)');axes[0,1].legend(fontsize=7)
    for axis,label in enumerate(['vx','vy','angular z']):
        axes[1,0].plot(t,[r['applied_command'][axis] for r in rows],linestyle='--',label=label+' applied')
        axes[1,0].plot(t,[r['body_velocity'][axis] for r in rows],alpha=.75,label=label+' measured')
    axes[1,0].legend(fontsize=7);axes[1,0].set_xlabel('active episode seconds')
    for key in rows[0]['reward_components']:
        axes[1,1].plot(t,[r['reward_components'][key] for r in rows],label=key)
    axes[1,1].legend(fontsize=7);axes[1,1].set_xlabel('active episode seconds')
    fig.tight_layout();fig.savefig(out/'diagnostics.png',dpi=120);plt.close(fig)


def reproduce(env,seed,family):
    traces=[]
    for _ in range(2):
        obs,info=env.reset(seed=seed,options={} if family=='mixed' else {'family':family})
        trace=[obs.tolist()]
        for _ in range(8):
            obs,reward,term,trunc,info=env.step(diagnostic_action(obs,env.config))
            trace.append([obs.tolist(),reward,term,trunc,info])
            if term or trunc:break
        traces.append(json.dumps(trace,sort_keys=True,allow_nan=False))
    if traces[0]!=traces[1]:raise AssertionError('Seed/action traces are not bitwise reproducible')
    return True


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default=str(DEFAULT_CONFIG));p.add_argument('--family',choices=(*FAMILIES,'mixed'),default='mixed')
    p.add_argument('--seeds',nargs='+',type=int,default=[101]);p.add_argument('--policy',choices=['zero','random','follower'],default='follower')
    p.add_argument('--suite',action='store_true');p.add_argument('--reproducibility',action='store_true');p.add_argument('--output',required=True);a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    env=G1EscortEnv(a.config)
    try:
        from gymnasium.utils.env_checker import check_env
        check_env(env,skip_render_check=True)
        checks={'gymnasium':'passed','stable_baselines3':'skipped: not installed'}
        if importlib.util.find_spec('stable_baselines3') is not None:
            from stable_baselines3.common.env_checker import check_env as sb3_check
            sb3_check(env,warn=True);checks['stable_baselines3']='passed'
        if a.reproducibility:checks['deterministic_trace']=reproduce(env,a.seeds[0],a.family)
        with open(out/'manifest.json','x') as f:json.dump(dict(config=env.config,runtime=env._runner.runtime_manifest(),checks=checks),f,indent=2)
        if a.suite:
            jobs=[(family,100+index*10+i,['zero','random','follower','follower','follower'][i])
                  for index,family in enumerate(FAMILIES) for i in range(5)]
            jobs += [('mixed',200+i,('zero' if i==0 else 'random' if i==1 else 'follower')) for i in range(10)]
        else:jobs=[(a.family,seed,a.policy) for seed in a.seeds]
        summaries=[];wall_total=0.;total_steps=0;physics_total=0;gait_total=0;reset_latencies=[];step_latencies=[]
        for index,(family,seed,policy) in enumerate(jobs):
            episode_dir=out/f'episode_{index:03d}_{family}_{policy}_{seed}';episode_dir.mkdir()
            begin=time.perf_counter()
            obs,reset_info=env.reset(seed=seed,options={} if family=='mixed' else {'family':family})
            reset_latencies.append(time.perf_counter()-begin)
            metadata=env.scenario_metadata()
            with open(episode_dir/'scenario_evaluator_metadata.json','x') as f:json.dump(metadata,f,indent=2)
            rng=np.random.default_rng(seed);rows=[];return_total=0.;terminated=truncated=False
            with open(episode_dir/'observations.jsonl','x') as actor_log,open(episode_dir/'evaluation.jsonl','x') as evaluator_log:
                while not (terminated or truncated):
                    action=np.zeros(3,np.float32) if policy=='zero' else rng.uniform(-1,1,3).astype(np.float32) if policy=='random' else diagnostic_action(obs,env.config)
                    previous=obs.copy();tick=time.perf_counter()
                    obs,reward,terminated,truncated,info=env.step(action)
                    step_latencies.append(time.perf_counter()-tick);return_total+=reward;total_steps+=1
                    actor_log.write(json.dumps(dict(observation=previous.tolist(),action=action.tolist(),next_observation=obs.tolist()))+'\n');actor_log.flush()
                    evaluator_log.write(json.dumps(dict(reward=reward,terminated=terminated,truncated=truncated,**info),allow_nan=False)+'\n');evaluator_log.flush()
                    if 'robot_position' in info:rows.append(info)
            elapsed=time.perf_counter()-begin;wall_total+=elapsed;physics_total+=env._runner.physics_steps;gait_total+=env._runner.gait.calls
            summary=dict(seed=seed,family=metadata['family'],requested_family=family,policy=policy,reason=info['reason'],
                terminated=terminated,truncated=truncated,active_seconds=info['elapsed_s'],return_total=return_total,
                wall_s=elapsed,formation_mean_m=float(np.mean([r['evaluation']['formation_error_m'] for r in rows])) if rows else None,
                formation_p95_m=float(np.percentile([r['evaluation']['formation_error_m'] for r in rows],95)) if rows else None,
                warmup_simulation_s=reset_info['warmup_simulation_s'],physics_steps=env._runner.physics_steps,gait_calls=env._runner.gait.calls)
            summaries.append(summary)
            with open(episode_dir/'summary.json','x') as f:json.dump(summary,f,indent=2)
            plot_episode(rows,episode_dir,env.config)
            print(f"{index+1}/{len(jobs)} {family} {policy}: {info['reason']} ({info['elapsed_s']:.1f}s)",flush=True)
        results=dict(episodes=summaries,outcomes=dict(Counter(r['reason'] for r in summaries)),
            throughput=dict(transitions=total_steps,wall_s=wall_total,transitions_s=total_steps/wall_total,
                physics_steps=physics_total,physics_steps_s=physics_total/wall_total,gait_calls=gait_total,gait_calls_s=gait_total/wall_total,
                simulated_seconds_s=physics_total*.002/wall_total,reset_latency_p50_p95_s=np.percentile(reset_latencies,[50,95]).tolist(),
                step_latency_p50_p95_s=np.percentile(step_latencies,[50,95]).tolist(),
                scope='one full-G1 instance; scenarios/obs/rewards/warmup/reset/flushed logs included; model load, checkers and plotting excluded'),checks=checks)
        with open(out/'summary.json','x') as f:json.dump(results,f,indent=2)
        print(json.dumps({k:v for k,v in results.items() if k!='episodes'},indent=2))
    finally:env.close()

if __name__=='__main__':main()
