"""Training-only causal teacher audit and gated Stage 2 actor fine-tuning.
No PPO learn, future-route inputs, reward edits or live command authority.
"""
import argparse, copy, json, sys
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from escort_rl.contract import ROOT, PACKAGE, sha256
from escort_rl.stage2.env import Stage2Env
from escort_rl.stage2.scenarios import DEFAULT, FAMILIES, load_config, sample_episode
from escort_rl.stage2.training import load_stage2, make_vec, manifest
from escort_rl.stage2.evaluation import evaluate, gate
from escort_rl.ppo_evaluation import reserved, EpisodeMetrics
from escort_rl.imitation_ppo import ACTOR_PREFIXES
from escort_rl.ppo_training import checkpoint
from escort_rl.sb3_restart import restore_checkpoint_rng
sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage2_causal_teacher import Teacher, PARAMETERS

PROTOCOL='stage2_causal_teacher_bootstrap_v2'
PARENT=PACKAGE/'runs/stage2_readiness_20261107/transition_ready/initial.zip'
STRATEGIES={f:'causal_velocity_candidates_v2' for f in FAMILIES}

def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False))

def teacher_gate(rows, minimum=.8):
    reasons=[]
    for family in FAMILIES:
        selected=[r for r in rows if r['family']==family]
        if not selected or np.mean([r['success'] for r in selected])<minimum:reasons.append('teacher_success_'+family)
    if any(r['reason'] in ('fall','supervisor_failure','physics_failure') for r in rows):reasons.append('critical_failure')
    return reasons

def episode_split(ids,families):
    """Whole episodes, one or more unseen examples per family; never random rows."""
    hold=[]
    for f in FAMILIES:
        es=sorted(set(ids[families==f].tolist()))
        if len(es)<3:raise ValueError('Need at least three successful teacher episodes: '+f)
        hold.extend(es[-max(1,len(es)//5):])
    h=np.isin(ids,hold);return np.flatnonzero(~h),np.flatnonzero(h)

def collect(output,count=10,seed=852000,config=DEFAULT):
    if count<3 or seed%4:raise ValueError('At least 3 episodes/family and training-residue seed divisible by four required')
    out=Path(output);out.mkdir(parents=True,exist_ok=False);c=load_config(config)
    split=json.loads((Path(c['stage1_config']).parent/'stage1_splits.json').read_text())
    record=dict(protocol=PROTOCOL,stage2_contract=manifest(config),source_sha256=sha256(__file__),teacher_sha256=sha256(Path(__file__).with_name('stage2_causal_teacher.py')),parameters=PARAMETERS,strategies=STRATEGIES,episodes_per_family=count,seed=seed,actor_inputs='current causal 40-vector and previous selected velocity only; no future trajectories or family-specific control',reward_unchanged=True)
    write(out/'manifest.json',record);rows=[];obs_all=[];actions=[];ids=[];families=[]
    env=Stage2Env(config)
    try:
        for family in FAMILIES:
            accepted=0;candidate=seed
            while accepted<count:
                m=sample_episode(candidate,family,c).scenario_metadata();candidate+=4
                if reserved(m,split):continue
                obs,_=env.reset(options=dict(family=family,episode_seed=m['seed']));teacher=Teacher(env.config);metrics=EpisodeMetrics();xs=[];ys=[];episode=len(rows)
                with (out/f'trace_{episode:03d}.jsonl').open('x') as trace:
                    while True:
                        action=teacher.act(obs);xs.append(obs.copy());ys.append(action.copy());before=obs.copy()
                        obs,r,t,tr,info=env.step(action);metrics.add(r,info)
                        trace.write(json.dumps(dict(observation=before.tolist(),action=action.tolist(),reward=r,info=info))+'\n')
                        if t or tr:break
                row=metrics.summary(family,m['seed']);row['scenario']=m;row['geometric_intrusion']=info.get('pedestrian_intrusion',False);rows.append(row);accepted+=1
                # Failed trajectories remain in traces/reports, never expert labels.
                if row['success']:
                    obs_all.extend(xs);actions.extend(ys);ids.extend([episode]*len(xs));families.extend([family]*len(xs))
                write(out/'episodes.json',rows)
                print(f'teacher {len(rows)}/{count*len(FAMILIES)} {family}: {row["reason"]}',flush=True)
    finally:env.close()
    np.savez_compressed(out/'demonstrations.npz',observations=np.asarray(obs_all,np.float32).reshape(-1,40),actions=np.asarray(actions,np.float32).reshape(-1,3),episode_ids=np.asarray(ids),families=np.asarray(families))
    reasons=teacher_gate(rows)
    result=dict(protocol=PROTOCOL,teacher_ready=not reasons,reasons=reasons,per_family={f:dict(episodes=sum(r['family']==f for r in rows),successes=sum(r['success'] for r in rows if r['family']==f)) for f in FAMILIES},demonstration_transitions=len(ids),all_outcomes=dict(Counter(r['reason'] for r in rows)),final_tests_used=False)
    result['data_sha256']=sha256(out/'demonstrations.npz')
    result['episodes_sha256']=sha256(out/'episodes.json')
    write(out/'summary.json',result);return result

def fit_actor(model,x,y,ids,families,updates=3000):
    if x.shape!=(len(ids),40) or y.shape!=(len(ids),3) or not np.isfinite(x).all() or not np.isfinite(y).all() or np.max(np.abs(y),initial=0)>1:raise ValueError('Invalid demonstration arrays')
    train,hold=episode_split(ids,families);rng=np.random.default_rng(852001)
    original={n:v.detach().clone() for n,v in model.policy.state_dict().items()}
    flags={n:p.requires_grad for n,p in model.policy.named_parameters()}
    params=[]
    for n,p in model.policy.named_parameters():
        p.requires_grad_(n.startswith(ACTOR_PREFIXES))
        if p.requires_grad:params.append(p)
    opt=torch.optim.Adam(params,lr=1e-4);best=float('inf');chosen=None;history=[]
    def loss(index):
        xt=torch.as_tensor(x[index],device=model.device);yt=torch.as_tensor(y[index],device=model.device)
        return ((model.policy.get_distribution(xt).distribution.mean-yt)**2).mean()
    try:
        for step in range(1,updates+1):
            # Equal family contribution: half Stage 1 replay, half encounters.
            idx=np.concatenate([rng.choice(train[families[train]==f],size=32,replace=True) for f in FAMILIES])
            opt.zero_grad();value=loss(idx);value.backward();torch.nn.utils.clip_grad_norm_(params,.5);opt.step()
            if step%100==0 or step==updates:
                with torch.no_grad():v=float(np.mean([loss(hold[families[hold]==f]).item() for f in FAMILIES]))
                history.append(dict(update=step,balanced_holdout_mse=v))
                if v<best:best=v;chosen={n:p.detach().clone() for n,p in model.policy.state_dict().items() if n.startswith(ACTOR_PREFIXES)}
        model.policy.load_state_dict(chosen,strict=False)
    finally:
        for n,p in model.policy.named_parameters():p.requires_grad_(flags[n])
    for n,v in original.items():
        if not n.startswith(ACTOR_PREFIXES) and not torch.equal(v,model.policy.state_dict()[n]):raise RuntimeError('Non-actor tensor changed: '+n)
    # Explicit offline actor fine-tune boundary: clear stale actor Adam moments ONLY.
    reset=[]
    for n,p in model.policy.named_parameters():
        if n.startswith(ACTOR_PREFIXES):model.policy.optimizer.state.pop(p,None);reset.append(n)
    return dict(history=history,selected_balanced_holdout_mse=best,actor_optimizer_moments_reset=reset,critic_and_log_std_unchanged=True,ppo_timesteps_unchanged=model.num_timesteps)

def fit(data,output,parent=PARENT,updates=3000,config=DEFAULT):
    if updates<1:raise ValueError('Positive fitting updates required')
    data=Path(data);record=json.loads((data/'manifest.json').read_text());rows=json.loads((data/'episodes.json').read_text());summary=json.loads((data/'summary.json').read_text())
    if record['protocol']!=PROTOCOL or record['stage2_contract']!=manifest(config):raise ValueError('Dataset provenance mismatch')
    for filename,key in [('demonstrations.npz','data_sha256'),('episodes.json','episodes_sha256')]:
        if summary.get(key)!=sha256(data/filename):raise ValueError('Dataset integrity mismatch: '+filename)
    if any(r['episode_seed']%4 for r in rows):raise ValueError('Non-training seed in dataset')
    reasons=teacher_gate(rows)
    if reasons:raise ValueError('Teacher not ready; no fitting performed: '+', '.join(reasons))
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    prior=json.loads(Path(str(Path(parent).with_suffix(''))+'.manifest.json').read_text());vec=make_vec(str(Path(config).resolve()),prior['workers'],prior['root_seed'],out)
    try:
        model,m=load_stage2(parent,config,vec);state=torch.load(str(Path(parent).with_suffix(''))+'.rng.pt',weights_only=False)
        with np.load(data/'demonstrations.npz',allow_pickle=False) as d:report=fit_actor(model,d['observations'],d['actions'],d['episode_ids'],d['families'],updates)
        write(out/'fit_report.json',report);c=load_config(config);c['_config_path']=str(Path(config).resolve())
        ped=evaluate(model,c,out/'validation_pedestrian');s1=evaluate(model,c,out/'validation_stage1',suite='stage1');decision=gate(ped,s1,c,m['parent_stage1_validation'])
        # Strict gates before exposing a resumable initialization.
        saved=False
        if decision['round_pass']:
            m=copy.deepcopy(m);m['teacher_bootstrap']=dict(protocol=PROTOCOL,parent=str(Path(parent).resolve()),parent_sha256=sha256(parent),data_manifest_sha256=sha256(data/'manifest.json'),data_sha256=sha256(data/'demonstrations.npz'),source_sha256=sha256(__file__),fit_report=report,method='actor-only supervised fine-tuning; critic/log_std unchanged; actor Adam moments reset; no PPO learn')
            restore_checkpoint_rng(model,vec,state);checkpoint(model,out/'prepared',m);saved=True
            registry=ROOT/'docs/bodyguard/MODEL_WEIGHTS.md'
            with registry.open('a') as f:f.write(f'\n## Gated Stage 2 teacher bootstrap\n\n- Weights: `{out.resolve()}/prepared.zip`\n- SHA-256: `{sha256(out/"prepared.zip")}`\n- Parent: `{parent}`; PPO counter {model.num_timesteps}, unchanged by supervised fitting.\n- Actor-only imitation fine-tune, original reward; both fixed validation suites passed one round, no promotion.\n- Preserve adjacent manifest/RNG sidecars and fit/validation reports.\n')
        result=dict(saved=saved,promotion=decision,ppo_learn_called=False,report=report);write(out/'result.json',result);return result
    finally:vec.close()

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['collect','fit']);p.add_argument('--output',required=True);p.add_argument('--data');p.add_argument('--checkpoint',default=str(PARENT));p.add_argument('--config',default=str(DEFAULT));p.add_argument('--episodes-per-family',type=int,default=10);p.add_argument('--seed',type=int,default=852000);p.add_argument('--updates',type=int,default=3000);a=p.parse_args()
    if a.mode=='fit' and not a.data:p.error('fit requires --data')
    result=collect(a.output,a.episodes_per_family,a.seed,a.config) if a.mode=='collect' else fit(a.data,a.output,a.checkpoint,a.updates,a.config)
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
