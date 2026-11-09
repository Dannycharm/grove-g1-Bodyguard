"""Separate fixed single-pedestrian validation and EXACT original Stage 1 regression."""
import json
from pathlib import Path
import numpy as np
from ..stage1_scenarios import load_config as load_stage1
from ..ppo_evaluation import reserved,validation_cases,EpisodeMetrics
from .scenarios import sample_episode,ENCOUNTERS,load_config
from .env import Stage2Env


def cases(config,per_encounter=None):
    c=config;split=json.loads((Path(c['stage1_config']).parent/'stage1_splits.json').read_text());result=[];seen=set()
    for family in ENCOUNTERS:
        seed=c['validation']['candidate_seed_start'];count=0
        while count<(per_encounter or c['validation']['episodes_per_encounter']):
            m=sample_episode(seed,family,c).scenario_metadata()
            if not reserved(m,split):
                if m['scenario_id'] in seen:raise ValueError('Repeated scenario identity')
                result.append(dict(family=family,episode_seed=seed,metadata=m));seen.add(m['scenario_id']);count+=1
            seed+=4
    return result


def aggregate(episodes):
    from collections import Counter
    return dict(episodes=len(episodes),success_rate=float(np.mean([e['success'] for e in episodes])),
        per_family={f:dict(episodes=sum(e['family']==f for e in episodes),success_rate=float(np.mean([e['success'] for e in episodes if e['family']==f]))) for f in sorted({e['family'] for e in episodes})},
        reasons=dict(Counter(e['reason'] for e in episodes)),formation_mean_m=float(np.mean([e['formation_mean_m'] for e in episodes])),
        geometric_intrusions=sum(e['geometric_intrusion'] for e in episodes),episodes_detail=episodes)


def evaluate(model,config,output,*,suite='pedestrian',per_encounter=None):
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    env=Stage2Env(config.get('_config_path',Path(__file__).resolve().parents[2]/'stage2_config/stage2.yaml'))
    base=load_stage1(config['stage1_config']);split=json.loads((Path(config['stage1_config']).parent/'stage1_splits.json').read_text())
    selected=validation_cases(base,split) if suite=='stage1' else cases(config,per_encounter)
    rows=[]
    try:
        for i,case in enumerate(selected):
            obs,_=env.reset(options=dict(family=case['family'],episode_seed=case['episode_seed']))
            folder=out/f'{i:03d}_{case["family"]}_{case["episode_seed"]}';folder.mkdir();(folder/'scenario_evaluator_metadata.json').write_text(json.dumps(env.scenario_metadata(),indent=2))
            metrics=EpisodeMetrics();last={}
            with (folder/'trace.jsonl').open('x') as f:
                while True:
                    action,_=model.predict(obs,deterministic=True);old=obs.copy();obs,r,t,tr,info=env.step(action);metrics.add(r,info);last=info
                    f.write(json.dumps(dict(observation=old.tolist(),action=action.tolist(),reward=r,terminated=t,truncated=tr,info=info))+'\n')
                    if t or tr:break
            row=metrics.summary(case['family'],case['episode_seed']);row.update(geometric_intrusion=last.get('pedestrian_intrusion',False),minimum_surface_clearance_m=last.get('minimum_geometric_surface_clearance_m'),formation_recovery_durations_s=last.get('formation_recovery_durations_s',[]))
            rows.append(row);(folder/'summary.json').write_text(json.dumps(row,indent=2));print(f'{suite} {i+1}/{len(selected)} {case["family"]}: {row["reason"]}',flush=True)
    finally:env.close()
    result=aggregate(rows);result.update(suite=suite,full_suite=len(selected)==(60 if suite=='stage1' else config['validation']['episodes_per_encounter']*len(ENCOUNTERS)),final_test='Not evaluated',reward_version='stage2_tracking_clearance_v1',clearance_metric='Swept pelvis footprint disc vs pedestrian disc; not physical collision')
    (out/'summary.json').write_text(json.dumps(result,indent=2));return result


def gate(pedestrian,stage1,config,parent_stage1):
    v=config['validation'];why=[]
    if not pedestrian.get('full_suite') or not stage1.get('full_suite'):why.append('incomplete_validation_suite')
    if pedestrian['success_rate']<v['pedestrian_success']:why.append('pedestrian_success')
    if any(pedestrian['per_family'].get(f,{}).get('success_rate',0)<v['each_encounter_success'] for f in ENCOUNTERS):why.append('encounter_family_success')
    if stage1['success_rate']<v['stage1_success'] or any(x['success_rate']<v['stage1_each_family_success'] for x in stage1['per_family'].values()):why.append('stage1_success_regression')
    limit=min(v['stage1_max_formation_mean_m'],parent_stage1['formation_mean_m']+v['stage1_max_mean_error_increase_m'])
    if stage1['formation_mean_m']>limit:why.append('stage1_formation_regression')
    if sum(x['reasons'].get(k,0) for x in (pedestrian,stage1) for k in ('fall','supervisor_failure','physics_failure'))>v['maximum_critical_failures']:why.append('critical_failures')
    if pedestrian['geometric_intrusions']>v['maximum_geometric_intrusions']:why.append('geometric_intrusions')
    return dict(round_pass=not why,reasons=why,stage1_mean_error_limit_m=limit,promotion_approved=False,required_consecutive_rounds=v['consecutive_rounds'])
