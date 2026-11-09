"""Velocity-obstacle teacher using current actor observations only.
Constant-velocity extrapolation is a controller assumption, never scripted futures.
"""
import numpy as np
from escort_rl.capacity_diagnostic import teacher_function
PARAMETERS=dict(version='causal_velocity_candidates_v2',horizon_s=4.,centre_margin_m=.85,response_lag_s=.4,velocity_grid_m_s=[-.3,-.2,-.1,0.,.1,.2,.3])

class Teacher:
    def __init__(self,config):
        self.config=config;self.follow=teacher_function();self.previous=np.zeros(2);self.active=False
    def act(self,observation):
        obs=np.asarray(observation,float);action=self.follow(obs,self.config)
        if obs[27]<.5:return action
        o=self.config['observation'];v=obs[10:12]*np.asarray(o['robot_velocity_scales'][:2]);client=obs[2:4]*o['relative_velocity_scale_m_s']+v
        r=obs[22:24]*o['position_scale_m'];ped=obs[24:26]*o['relative_velocity_scale_m_s']+v
        a=np.arctan2(obs[4],obs[5]);rot=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
        target=obs[:2]*o['position_scale_m']+rot@(obs[6:8]*o['offset_scale_m']);desired=np.clip(client+.65*target,-.35,.35)
        grid=PARAMETERS['velocity_grid_m_s'];candidates=np.array([[x,y] for x in grid for y in grid]+[desired.tolist()])
        times=np.linspace(0,PARAMETERS['horizon_s'],25)
        # Conservative lag: initially retain measured robot motion before a new velocity.
        lag=PARAMETERS['response_lag_s'];motion=v[None,None,:]*np.minimum(times,lag)[None,:,None]+candidates[:,None,:]*np.maximum(times-lag,0)[None,:,None]
        distances=np.linalg.norm(r[None,None,:]+ped[None,None,:]*times[None,:,None]-motion,axis=2)
        clearance=distances.min(axis=1);risk=np.maximum(PARAMETERS['centre_margin_m']-clearance,0)
        cost=200*risk**2+np.sum((candidates-desired)**2,axis=1)+.1*np.sum((candidates-self.previous)**2,axis=1)
        chosen=candidates[np.argmin(cost)];self.previous=chosen.copy();self.active=bool(np.linalg.norm(chosen-desired)>.03)
        command=np.zeros(3)
        for i in (0,1):
            if abs(chosen[i])>.025:command[i]=chosen[i]/.8+np.sign(chosen[i])*(.08 if i==0 else .10)
        command[2]=action[2]*self.config['environment']['command_caps'][2]
        return np.clip(command/np.asarray(self.config['environment']['command_caps']),-1,1).astype(np.float32)


def audit(output,count=20):
    """Fixed validation only; never exports a fitting dataset."""
    import json,time
    from pathlib import Path
    from escort_rl.stage2.scenarios import load_config
    from escort_rl.stage2.env import Stage2Env
    from escort_rl.stage2.evaluation import cases,aggregate
    from escort_rl.ppo_evaluation import EpisodeMetrics
    from escort_rl.contract import sha256
    out=Path(output);out.mkdir(parents=True,exist_ok=False);c=load_config();selected=cases(c,count);env=Stage2Env();rows=[];start=time.perf_counter()
    write=lambda p,x:p.write_text(json.dumps(x,indent=2))
    write(out/'manifest.json',dict(protocol='causal_teacher_v2_validation',parameters=PARAMETERS,teacher_sha256=sha256(__file__),cases=selected,final_tests_used=False,fitting_data=False))
    try:
        for i,case in enumerate(selected):
            obs,_=env.reset(options=dict(family=case['family'],episode_seed=case['episode_seed']));teacher=Teacher(env.config);metrics=EpisodeMetrics()
            with (out/f'trace_{i:03d}.jsonl').open('x') as f:
                while True:
                    old=obs.copy();action=teacher.act(obs);obs,r,t,tr,info=env.step(action);metrics.add(r,info)
                    f.write(json.dumps(dict(observation=old.tolist(),action=action.tolist(),info=info))+'\n')
                    if t or tr:break
            row=metrics.summary(case['family'],case['episode_seed']);row.update(geometric_intrusion=info['pedestrian_intrusion'],minimum_surface_clearance_m=info['minimum_geometric_surface_clearance_m']);rows.append(row)
            write(out/'episodes.json',rows);print(f'validation {i+1}/{len(selected)} {case["family"]}: {row["reason"]}',flush=True)
    finally:env.close()
    result=aggregate(rows);result.update(wall_s=time.perf_counter()-start,teacher_version=PARAMETERS['version'],final_tests_used=False);write(out/'summary.json',result);return result

if __name__=='__main__':
    import argparse,json
    p=argparse.ArgumentParser(description='Causal teacher fixed validation; no fitting or PPO');p.add_argument('--output',required=True);p.add_argument('--episodes-per-encounter',type=int,default=20);a=p.parse_args()
    if not 1<=a.episodes_per_encounter<=20:p.error('Use 1..20 cases per encounter')
    print(json.dumps(audit(a.output,a.episodes_per_encounter),indent=2))
