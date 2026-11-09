"""Matched reward audit with causal diagnostic controllers; NEVER teacher training.

Outside Stage 2's source globs: adding diagnostics does not invalidate checkpoints.
No rewards, low-level controls, dynamics, bounds, observations or scenarios change.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from escort_rl.contract import PACKAGE,ROOT,sha256
from escort_rl.capacity_diagnostic import teacher_function
from escort_rl.stage2.env import Stage2Env
from escort_rl.stage2.scenarios import load_config,sample_episode,ENCOUNTERS
from escort_rl.stage2.evaluation import cases
from escort_rl.ppo_evaluation import reserved,EpisodeMetrics
from escort_rl.stage2.training import manifest

PARAMETERS=dict(version='causal_yield_reward_audit_v2',horizon_s=4.5,risk_centre_distance_m=.95,
                release_centre_distance_m=1.05,release_dwell_s=.4,sidestep_offset_m=.65,
                sidestep_command_cap_m_s=.30,nominal_actual_speed_cap_m_s=.35)


class Controller:
    def __init__(self,strategy,config,parameters=None):
        if strategy not in ('follow','yield','sidestep','stop'):raise ValueError(strategy)
        self.strategy=strategy;self.config=config;self.p=dict(PARAMETERS if parameters is None else parameters)
        self.follow=teacher_function();self.active=False;self.clear_s=0.;self.side=1.;self.activations=0;self.active_s=0.
    def act(self,observation):
        obs=np.asarray(observation,float);command=self.follow(obs,self.config).astype(float)
        if self.strategy=='stop':return np.zeros(3,np.float32)
        if self.strategy=='follow' or obs[27]<.5:return command.astype(np.float32)
        o=self.config['observation'];robot_v=obs[10:12]*np.asarray(o['robot_velocity_scales'][:2]);client_v=obs[2:4]*o['relative_velocity_scale_m_s']+robot_v
        r=obs[22:24]*o['position_scale_m'];ped_v=obs[24:26]*o['relative_velocity_scale_m_s']+robot_v
        angle=np.arctan2(obs[4],obs[5]);R=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        target=obs[:2]*o['position_scale_m']+R@(obs[6:8]*o['offset_scale_m'])
        nominal=np.clip(client_v+.65*target,-self.p['nominal_actual_speed_cap_m_s'],self.p['nominal_actual_speed_cap_m_s'])
        # Constant-velocity prediction from CURRENT observations, not scripted future.
        relative_v=ped_v-nominal;vv=float(np.dot(relative_v,relative_v));ttc=0. if vv<1e-12 else float(-np.dot(r,relative_v)/vv)
        closest=float(np.linalg.norm(r+np.clip(ttc,0,self.p['horizon_s'])*relative_v));distance=float(np.linalg.norm(r))
        risk=distance<self.p['risk_centre_distance_m'] or (ttc>=0 and closest<self.p['risk_centre_distance_m'])
        clear=(distance>self.p['release_centre_distance_m'] and (ttc<0 or closest>self.p['release_centre_distance_m']))
        if not self.active and risk:
            self.active=True;self.activations+=1;self.clear_s=0.;self.side=-1. if r[1]>=0 else 1.
        if self.active:
            self.clear_s=self.clear_s+self.config['environment']['dt_s'] if clear else 0.
            if self.clear_s>=self.p['release_dwell_s']:
                self.active=False;self.clear_s=0.
            else:
                self.active_s+=self.config['environment']['dt_s'];command[0]=0.
                if self.strategy=='yield':command[1]=0.
                else:
                    shifted=obs.copy();shifted[7]+=self.side*self.p['sidestep_offset_m']/o['offset_scale_m']
                    lateral=self.follow(shifted,self.config)[1]
                    cap=self.p['sidestep_command_cap_m_s']/self.config['environment']['command_caps'][1]
                    command[1]=np.clip(lateral,-cap,cap)
        return np.clip(command,-1,1).astype(np.float32)


def selected_cases(c,suite,count,seed_start):
    if suite=='validation':return cases(c,count)
    split=json.loads((Path(c['stage1_config']).parent/'stage1_splits.json').read_text());out=[]
    for family in ENCOUNTERS:
        seed=seed_start
        while sum(x['family']==family for x in out)<count:
            m=sample_episode(seed,family,c).scenario_metadata()
            if not reserved(m,split):out.append(dict(family=family,episode_seed=seed,metadata=m))
            seed+=4
    return out


def run(strategy,output,suite='development',count=2,seed_start=640000):
    c=load_config();selected=selected_cases(c,suite,count,seed_start);out=Path(output);out.mkdir(parents=True,exist_ok=False)
    record=dict(experiment='stage2_reward_audit_no_training',strategy=strategy,parameters=PARAMETERS,stage2_contract=manifest(PACKAGE/'stage2_config/stage2.yaml'),diagnostic_source_sha256=sha256(__file__),gamma=.99,suite=suite,cases=selected,controller_inputs='current 40-vector and fixed configuration only',new_optimizer_updates=0)
    (out/'manifest.json').write_text(json.dumps(record,indent=2));env=Stage2Env();results=[];wall=time.perf_counter()
    try:
        for i,case in enumerate(selected):
            obs,_=env.reset(options=dict(family=case['family'],episode_seed=case['episode_seed']));controller=Controller(strategy,env.config);metrics=EpisodeMetrics();discounted=Counter();intervals={};factor=1.;step=0
            folder=out/f'{i:03d}_{case["family"]}_{case["episode_seed"]}';folder.mkdir()
            with (folder/'trace.jsonl').open('x') as trace:
                while True:
                    before=obs.copy();action=controller.act(obs);obs,r,t,tr,info=env.step(action);metrics.add(r,info)
                    assert abs(r-sum(info['reward_components'].values()))<1e-8
                    for k,v in info['reward_components'].items():discounted[k]+=factor*v
                    start=int(max(0,round(info['elapsed_s']/.2)-1)//25)*5
                    if start not in intervals:intervals[start]=Counter()
                    intervals[start].update(info['reward_components'])
                    trace.write(json.dumps(dict(observation=before.tolist(),action=action.tolist(),reward=r,discount_factor=factor,terminated=t,truncated=tr,diagnostic_active=controller.active,info=info))+'\n')
                    factor*=.99;step+=1
                    if t or tr:break
            result=metrics.summary(case['family'],case['episode_seed']);result.update(strategy=strategy,discounted_return=sum(discounted.values()),discounted_reward_components=dict(discounted),five_second_reward_components={str(k):dict(v) for k,v in intervals.items()},minimum_surface_clearance_m=info['minimum_geometric_surface_clearance_m'],intrusion=info['pedestrian_intrusion'],diagnostic_activations=controller.activations,diagnostic_active_s=controller.active_s)
            results.append(result);(folder/'summary.json').write_text(json.dumps(result,indent=2));print(f'{strategy} {i+1}/{len(selected)} {case["family"]}: {info["reason"]} return={result["return_total"]:.2f} discounted={result["discounted_return"]:.2f}',flush=True)
    finally:env.close()
    summary=dict(strategy=strategy,episodes=len(results),successes=sum(r['success'] for r in results),reasons=dict(Counter(r['reason'] for r in results)),per_family={f:dict(episodes=sum(r['family']==f for r in results),successes=sum(r['success'] for r in results if r['family']==f),mean_return=float(np.mean([r['return_total'] for r in results if r['family']==f])),mean_discounted_return=float(np.mean([r['discounted_return'] for r in results if r['family']==f]))) for f in ENCOUNTERS},mean_reward_components={k:float(np.mean([r['reward_components'].get(k,0.) for r in results])) for k in results[0]['reward_components']},wall_s=time.perf_counter()-wall,episodes_detail=results)
    (out/'summary.json').write_text(json.dumps(summary,indent=2));return summary


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--strategy',choices=['follow','yield','sidestep','stop'],required=True);p.add_argument('--output',required=True);p.add_argument('--suite',choices=['development','validation'],default='development');p.add_argument('--cases-per-family',type=int,default=2);p.add_argument('--seed-start',type=int,default=640000)
    a=p.parse_args()
    if not 1<=a.cases_per_family<=20:p.error('Use 1..20 cases/family')
    if a.suite=='development' and a.seed_start%4:p.error('Development seeds must use training namespace residue zero')
    print(json.dumps(run(a.strategy,a.output,a.suite,a.cases_per_family,a.seed_start),indent=2))


if __name__=='__main__':main()
