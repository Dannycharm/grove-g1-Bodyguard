"""Synthetic evaluator transitions, NOT physics/locomotion capability evidence."""
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
from .rewards import EpisodeEvaluator
from .stage1_scenarios import load_config,sample_episode

CASES=('walking_correct','walking_falls_behind','walking_closes_gap','stopped_correct',
       'stopped_recovery','stopped_crosses_front','resumes_recovers','robot_stationary',
       'races_ahead_waits','oscillates')


def states(case,t):
    client=min(.2*t,4.)
    robot=client-1.5
    if case=='walking_falls_behind':robot=-1.5+.1*t
    elif case=='walking_closes_gap':robot=client-1.5+min(1.5,.15*t)
    elif case=='stopped_correct':client=0.;robot=-1.5
    elif case=='stopped_recovery':client=0.;robot=-2.+min(.5,.05*t)
    elif case=='stopped_crosses_front':client=0.;robot=min(.1,-1.5+.2*t)
    elif case=='resumes_recovers':client=min(max(0.,t-5)*.2,4.);robot=client-2.+min(.5,.05*t)
    elif case=='robot_stationary':robot=-1.5
    elif case=='races_ahead_waits':robot=min(2.5,-1.5+t)
    elif case=='oscillates':client=0.;robot=-1.5+.25*np.sin(2*np.pi*t/4)
    return float(client),float(robot)


def audit_case(case,version='legacy_v1'):
    c=load_config();template=sample_episode(1,'straight').client_pose_at(0)
    _,initial=states(case,0.)
    if version=='legacy_v1':e=EpisodeEvaluator(c,initial,4.)
    else:
        from .reward_variants import FormationProgressEvaluator
        e=FormationProgressEvaluator(c,initial,4.,lambda xy:float(xy[0]))
    totals={};rows=[];discounted=0.;done=False
    for i in range(1,126):
        t=i*.2;client,robot=states(case,t);pc,pr=states(case,t-.2)
        if done:
            rows.append(dict(elapsed_s=t,absorbing=True,reward=0.));continue
        actor=replace(template,x=client,y=0.,yaw=0.,vx=(client-pc)/.2,vy=0.)
        action=np.array([np.clip((robot-pr)/.2/.6,-1,1),0.,0.])
        reward,term,trunc,info=e.update(elapsed=t,dt=.2,robot_xy=[robot,0.],client=actor,
            robot_progress=robot,client_progress=client,client_arrived=20. if client>=4. and case!='resumes_recovers' else (25. if client>=4. else None),action=action)
        for k,v in info['reward_components'].items():totals[k]=totals.get(k,0.)+v
        discounted+=.99**(i-1)*reward
        rows.append(dict(elapsed_s=t,client_progress=client,robot_progress=robot,reward=reward,**info))
        done=term or trunc
    return dict(case=case,version=version,horizon_s=25.,discounted_return_gamma_099=discounted,
        total_reward=sum(totals.values()),reward_components=totals,transitions=rows)


def export_audit(output,versions=('legacy_v1',)):
    out=Path(output);out.mkdir(parents=True,exist_ok=False);results=[]
    for version in versions:
        for case in CASES:
            result=audit_case(case,version);results.append(result)
            (out/f'{version}_{case}.json').write_text(json.dumps(result,indent=2))
    (out/'summary.json').write_text(json.dumps([{k:v for k,v in r.items() if k!='transitions'} for r in results],indent=2))
    return results
