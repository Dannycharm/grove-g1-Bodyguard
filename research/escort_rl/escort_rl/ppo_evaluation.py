"""Offline deterministic evaluation. Privileged diagnostics never enter predict()."""
from collections import Counter
import json
from pathlib import Path
import numpy as np
from .env import G1EscortEnv
from .stage1_scenarios import sample_episode, FAMILIES


def reserved(metadata, split):
    r=split['final_test_reserved_combination']
    return metadata['peak_speed_limit_m_s']>=r['min_speed_m_s'] and metadata['route_length_m']>=r['min_route_length_m']


def validation_cases(config, split):
    """Resolve validation only; final-test scenarios remain untouched."""
    cases=[];seen=set()
    for family in FAMILIES:
        seed=split['validation_candidate_start'];count=0
        while count<split['episodes_per_family']:
            m=sample_episode(seed,family,config).scenario_metadata()
            if not reserved(m,split):
                signature=json.dumps({k:m[k] for k in ('family','route_length_m','peak_speed_limit_m_s',
                    'start_xy_m','robot_spawn_xy_yaw','heading_rad','curvature_m_inv','pause_s','stop_fraction')},sort_keys=True)
                if signature in seen:raise ValueError('Duplicate validation geometry')
                seen.add(signature);cases.append(dict(family=family,episode_seed=seed,metadata=m));count+=1
            seed+=split['seed_modulus']
    return cases


class EpisodeMetrics:
    def __init__(self):self.rows=[];self.total=0.
    def add(self,reward,info):self.rows.append(info);self.total+=float(reward)
    def summary(self,family,seed):
        rows=self.rows;last=rows[-1];ev=last.get('evaluation',{})
        formation=[r['evaluation']['formation_error_m'] for r in rows if 'evaluation' in r]
        rewards=Counter()
        for r in rows:rewards.update(r.get('reward_components',{}))
        return dict(family=family,episode_seed=int(seed),reason=last['reason'],success=last['reason']=='success',
            return_total=self.total,steps=len(rows),duration_s=last['elapsed_s'],
            formation_mean_m=float(np.mean(formation)) if formation else None,
            formation_p95_m=float(np.percentile(formation,95)) if formation else None,
            accompanied_progress_m=ev.get('accompanied_robot_progress_m',0.),robot_progress_m=ev.get('robot_advance_m',0.),
            mean_body_velocity=np.mean([r.get('body_velocity',[0,0,0]) for r in rows],axis=0).tolist(),
            mean_abs_command=np.mean(np.abs([r.get('applied_command',[0,0,0]) for r in rows]),axis=0).tolist(),
            action_clipped_fraction=float(np.mean([np.any(np.abs(r.get('raw_action',[0,0,0]))>1) for r in rows])),
            command_saturation_fraction=float(np.mean([np.any(np.abs(r.get('clipped_action',[0,0,0]))>=.999) for r in rows])),
            observation_clipped_features=int(sum(r.get('observation_saturated_features',0) for r in rows)),
            reward_components=dict(rewards))


def aggregate(episodes):
    reasons=dict(Counter(e['reason'] for e in episodes))
    return dict(episodes=len(episodes),success_rate=float(np.mean([e['success'] for e in episodes])),
        per_family={f:dict(episodes=sum(e['family']==f for e in episodes),
            success_rate=float(np.mean([e['success'] for e in episodes if e['family']==f]))) for f in FAMILIES if any(e['family']==f for e in episodes)},
        failures=reasons,formation_mean_m=float(np.mean([e['formation_mean_m'] for e in episodes if e['formation_mean_m'] is not None])),
        mean_return=float(np.mean([e['return_total'] for e in episodes])),episodes_detail=episodes)


def selection_key(summary):
    """Success first, then absence of critical failures, formation, return."""
    failures=summary['failures'];critical=sum(failures.get(k,0) for k in ('fall','supervisor_failure','physics_failure'))
    return (summary['success_rate'],-critical,-summary['formation_mean_m'],summary['mean_return'])


def promotion_pass(summary,criteria):
    details=summary['episodes_detail']
    return (summary['success_rate']>=criteria['overall_success'] and
        all(summary['per_family'][f]['episodes']>=20 and summary['per_family'][f]['success_rate']>=criteria['per_family_success'] for f in FAMILIES) and
        not any(summary['failures'].get(k,0) for k in ('fall','supervisor_failure','physics_failure')) and
        summary['formation_mean_m']<=criteria['max_mean_formation_error_m'] and
        np.mean([e['accompanied_fraction'] for e in details])>=criteria['min_mean_accompanied_fraction'])


def evaluate(policy,config,cases,output,*,kind='ppo',plots=True,reward_version='legacy_v1'):
    from .reward_variants import make_environment
    out=Path(output);out.mkdir(parents=True,exist_ok=False);env=make_environment(config=config,reward_version=reward_version);episodes=[]
    # Existing diagnostic follower/plot utility, reused without exposing evaluator state.
    from importlib.util import spec_from_file_location,module_from_spec
    spec=spec_from_file_location('escort_stage1_verifier',Path(__file__).resolve().parents[1]/'tools/verify_stage1_env.py')
    verifier=module_from_spec(spec);spec.loader.exec_module(verifier)
    try:
        for i,case in enumerate(cases):
            obs,_=env.reset(seed=case['episode_seed'],options={'family':case['family'],'episode_seed':case['episode_seed']})
            folder=out/f'{i:03d}_{case["family"]}_{case["episode_seed"]}';folder.mkdir()
            (folder/'scenario_evaluator_metadata.json').write_text(json.dumps(env.scenario_metadata(),indent=2))
            metrics=EpisodeMetrics();done=False
            with (folder/'observations.jsonl').open('x') as actors,(folder/'evaluation.jsonl').open('x') as evaluator:
                while not done:
                    if kind=='zero':action=np.zeros(3,np.float32)
                    elif kind=='follower':action=verifier.diagnostic_action(obs,config)
                    else:action,_=policy.predict(obs,deterministic=True)
                    actors.write(json.dumps(dict(observation=obs.tolist(),action=np.asarray(action).tolist()))+'\n')
                    obs,reward,term,trunc,info=env.step(action);done=term or trunc;metrics.add(reward,info)
                    evaluator.write(json.dumps(dict(info,reward=reward,terminated=term,truncated=trunc))+'\n')
            result=metrics.summary(case['family'],case['episode_seed'])
            result['accompanied_fraction']=result['accompanied_progress_m']/case['metadata']['route_length_m']
            episodes.append(result);(folder/'summary.json').write_text(json.dumps(result,indent=2))
            if plots:verifier.plot_episode([r for r in metrics.rows if 'evaluation' in r],folder,config)
            print(f'evaluation {kind} {i+1}/{len(cases)} {case["family"]}: {result["reason"]}',flush=True)
    finally:env.close()
    summary=aggregate(episodes);summary['reward_version']=reward_version
    (out/'summary.json').write_text(json.dumps(summary,indent=2));return summary
