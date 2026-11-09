"""Opt-in intervention-balanced actor fitting and causal student-state labeling.
No PPO learning. Diagnostic actors are parameter-only and cannot be PPO resumes.
"""
import argparse,ast,copy,fcntl,json,sys,time
from pathlib import Path
import numpy as np
import torch
import yaml
from escort_rl.contract import ROOT,PACKAGE,sha256
from escort_rl.imitation_ppo import ACTOR_PREFIXES
from escort_rl.stage1_scenarios import load_config as stage1_config
from escort_rl.stage2.scenarios import DEFAULT,FAMILIES,ENCOUNTERS,load_config,sample_episode
from escort_rl.stage2.env import Stage2Env
from escort_rl.stage2.training import load_stage2,make_vec,manifest
from escort_rl.stage2.evaluation import evaluate,gate
from escort_rl.ppo_evaluation import reserved,EpisodeMetrics
from escort_rl.ppo_training import checkpoint
from escort_rl.sb3_restart import restore_checkpoint_rng
sys.path.insert(0,str(Path(__file__).resolve().parent))
from stage2_causal_teacher import Teacher,PARAMETERS
from stage2_teacher_bootstrap import episode_split,teacher_gate,PARENT

CONFIG=Path(__file__).with_name('stage2_distillation_v3.yaml')
DAGGER='stage2_student_state_labels_v1'
BASE=PACKAGE/'runs/stage2_teacher_v2_readiness_20261108'

def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False))
def read_config(path):
    c=yaml.safe_load(Path(path).read_text())
    if c['protocol']!='stage2_intervention_balanced_distillation_v3':raise ValueError('Unknown distillation protocol')
    if c['stage1_replay_fraction']!=.5 or c['intervention_fraction']!=.5 or c['samples_per_family']%2:raise ValueError('Require half Stage 1 replay, equal within-encounter strata and even family batch')
    for k in ('updates','samples_per_family','score_interval','dagger_episodes_per_family'):
        if type(c[k]) is not int or c[k]<1:raise ValueError(k)
    for k in ('learning_rate','max_grad_norm','intervention_action_l2_threshold'):
        if not np.isfinite(c[k]) or c[k]<=0:raise ValueError(k)
    for k in ('fit_seed','dagger_seed'):
        if type(c[k]) is not int or c[k]<0:raise ValueError(k)
    if not 0<=c['dagger_teacher_probability']<=1 or c['dagger_seed']%4:raise ValueError('Invalid DAgger config')
    return c

def teacher_ast(source):
    return ast.dump(next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name=='Teacher'),include_attributes=False)

def verify_teacher(record,folder):
    current=Path(__file__).with_name('stage2_causal_teacher.py')
    if record.get('parameters')!=PARAMETERS:raise ValueError('Teacher parameter mismatch')
    if record['teacher_sha256']==sha256(current):return 'exact_source'
    # v2 collection predates appending the audit CLI. Require its exact archived
    # source hash AND identical Teacher AST/parameters, never ignore a hash mismatch.
    snapshot=folder/'teacher_source_snapshot.py'
    if not snapshot.exists() or sha256(snapshot)!=record['teacher_sha256'] or teacher_ast(snapshot.read_text())!=teacher_ast(current.read_text()):raise ValueError('Teacher source mismatch')
    return 'exact_archived_source_and_identical_teacher_class'

def load_data(paths,config):
    arrays=[];train=[];hold=[];offset=0;row_offset=0;seen=set();provenance=[]
    c=load_config(config);split=json.loads((Path(c['stage1_config']).parent/'stage1_splits.json').read_text())
    for path in paths:
        folder=Path(path);record=json.loads((folder/'manifest.json').read_text());summary=json.loads((folder/'summary.json').read_text());rows=json.loads((folder/'episodes.json').read_text())
        if record['protocol'] not in ('stage2_causal_teacher_bootstrap_v2',DAGGER) or record['stage2_contract']!=manifest(config):raise ValueError('Dataset provenance mismatch')
        compatibility=verify_teacher(record,folder)
        for filename,key in [('demonstrations.npz','data_sha256'),('episodes.json','episodes_sha256')]:
            if summary.get(key)!=sha256(folder/filename):raise ValueError('Dataset integrity mismatch: '+filename)
        if record['protocol']!='stage2_causal_teacher_bootstrap_v2' and record.get('label_semantics')!='teacher commands queried on student-visited states; NOT successful expert returns':raise ValueError('Unknown student-state labels')
        if record['protocol']=='stage2_causal_teacher_bootstrap_v2' and teacher_gate(rows):raise ValueError('Teacher-readiness failure')
        for row in rows:
            key=(row['family'],row['episode_seed'])
            if row['episode_seed']%4 or key in seen or reserved(row['scenario'],split):raise ValueError('Partition violation or duplicate scenario')
            seen.add(key)
        with np.load(folder/'demonstrations.npz',allow_pickle=False) as d:
            x,y,ids,fs=(d[k].copy() for k in ('observations','actions','episode_ids','families'))
        if x.dtype!=np.float32 or y.dtype!=np.float32 or not np.issubdtype(ids.dtype,np.integer) or set(fs)!=set(FAMILIES) or x.shape!=(len(ids),40) or y.shape!=(len(ids),3) or fs.shape!=(len(ids),) or not np.isfinite(x).all() or not np.isfinite(y).all() or np.max(np.abs(x),initial=0)>1 or np.max(np.abs(y),initial=0)>1:raise ValueError('Invalid causal arrays')
        for i,f in zip(ids,fs):
            if not 0<=i<len(rows) or rows[int(i)]['family']!=f:raise ValueError('Row/episode identity mismatch')
        ti,hi=episode_split(ids,fs)  # Split EACH source separately: previous holdout stays held out.
        train.extend((ti+row_offset).tolist());hold.extend((hi+row_offset).tolist())
        arrays.append((x,y,ids+offset,fs));row_offset+=len(ids);offset+=len(rows)
        provenance.append(dict(path=str(folder.resolve()),manifest_sha256=sha256(folder/'manifest.json'),data_sha256=sha256(folder/'demonstrations.npz'),teacher_compatibility=compatibility,train_episode_ids=sorted(set(ids[ti].tolist())),holdout_episode_ids=sorted(set(ids[hi].tolist()))))
    if not arrays:raise ValueError('No datasets')
    result=tuple(np.concatenate([a[i] for a in arrays]) for i in range(4))
    return (*result,np.asarray(train),np.asarray(hold),provenance)

def intervention_mask(x,y,base_config,threshold):
    teacher=Teacher(base_config)
    follow=np.asarray([teacher.follow(obs,base_config) for obs in x])
    return np.linalg.norm(y-follow,axis=1)>threshold

def pools(indices,families,intervention):
    result=[]
    for f in FAMILIES:
        selected=indices[families[indices]==f]
        groups=[selected[intervention[selected]],selected[~intervention[selected]]] if f in ENCOUNTERS else [selected]
        if any(not len(g) for g in groups):raise ValueError('Missing fitting stratum: '+f)
        result.append(groups)
    return result

def sample_batch(groups,rng,size):
    return np.concatenate([rng.choice(g,size=size//len(family),replace=True) for family in groups for g in family])

def metrics(pred,y,indices,fs,active,caps):
    result={}
    for f in FAMILIES:
        fi=indices[fs[indices]==f];row={}
        for name,ii in [('all',fi),('intervention',fi[active[fi]]),('ordinary',fi[~active[fi]])]:
            if not len(ii):row[name]=dict(samples=0,mse=None,command_mae=None);continue
            error=pred[ii]-y[ii];row[name]=dict(samples=len(ii),mse=float(np.mean(error**2)),command_mae=np.mean(np.abs(error)*caps,axis=0).tolist())
        if not len(fi):raise ValueError('Missing holdout family: '+f)
        # Each family has equal weight; each encounter stratum equal within family.
        result[f]=row
    family_scores=[]
    for f,r in result.items():
        parts=[r[n]['mse'] for n in ('intervention','ordinary') if r[n]['samples']] if f in ENCOUNTERS else [r['all']['mse']]
        family_scores.append(float(np.mean(parts)))
    return dict(balanced_selection_mse=float(np.mean(family_scores)),per_family=result)

def predictions(model,x):
    out=[]
    with torch.no_grad():
        for i in range(0,len(x),1024):out.append(model.policy.get_distribution(torch.as_tensor(x[i:i+1024],device=model.device)).distribution.mean.cpu().numpy())
    return np.concatenate(out)

def fit_actor(model,data,c,base_config):
    x,y,ids,fs,train,hold,_=data;active=intervention_mask(x,y,base_config,c['intervention_action_l2_threshold']);groups=pools(train,fs,active);rng=np.random.default_rng(c['fit_seed'])
    original={n:v.detach().clone() for n,v in model.policy.state_dict().items()};flags={n:p.requires_grad for n,p in model.policy.named_parameters()};params=[]
    for n,p in model.policy.named_parameters():
        p.requires_grad_(n.startswith(ACTOR_PREFIXES))
        if p.requires_grad:params.append(p)
    opt=torch.optim.Adam(params,lr=c['learning_rate']);history=[];chosen=None;best=float('inf');selected=0
    initial=metrics(predictions(model,x),y,hold,fs,active,np.array(base_config['environment']['command_caps']))
    begin=time.perf_counter()
    try:
        for step in range(1,c['updates']+1):
            ii=sample_batch(groups,rng,c['samples_per_family']);pred=model.policy.get_distribution(torch.as_tensor(x[ii],device=model.device)).distribution.mean
            loss=((pred-torch.as_tensor(y[ii],device=model.device))**2).mean();opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(params,c['max_grad_norm']);opt.step()
            if step%c['score_interval']==0 or step==c['updates']:
                score=metrics(predictions(model,x),y,hold,fs,active,np.array(base_config['environment']['command_caps']));history.append(dict(update=step,**score))
                if score['balanced_selection_mse']<best:best=score['balanced_selection_mse'];selected=step;chosen={n:v.detach().clone() for n,v in model.policy.state_dict().items() if n.startswith(ACTOR_PREFIXES)}
        model.policy.load_state_dict(chosen,strict=False)
    finally:
        for n,p in model.policy.named_parameters():p.requires_grad_(flags[n])
    for n,v in original.items():
        if not n.startswith(ACTOR_PREFIXES) and not torch.equal(v,model.policy.state_dict()[n]):raise RuntimeError('Protected tensor changed: '+n)
    reset=[]
    for n,p in model.policy.named_parameters():
        if n.startswith(ACTOR_PREFIXES):model.policy.optimizer.state.pop(p,None);reset.append(n)
    final=metrics(predictions(model,x),y,hold,fs,active,np.array(base_config['environment']['command_caps']))
    return dict(initial=initial,final=final,history=history,selected_update=selected,updates_executed=c['updates'],fit_wall_s=time.perf_counter()-begin,actor_adam_moments_reset=reset,critic_and_variance_unchanged=True,ppo_timesteps=model.num_timesteps,intervention_rows=int(active.sum()),total_rows=len(x),train_episode_ids=sorted(set(ids[train].tolist())),holdout_episode_ids=sorted(set(ids[hold].tolist())))

def register(out,parent,model,decision):
    registry=ROOT/'docs/bodyguard/MODEL_WEIGHTS.md';rel=out.resolve().relative_to(ROOT)
    text=f'\n## Intervention-balanced imitation candidate — {rel}\n\n- Diagnostic actor only: `{rel}/candidate_actor.npz`, SHA-256 `{sha256(out/"candidate_actor.npz")}`; metadata `candidate.json`. NOT a PPO resume archive.\n- Parent `{parent}`; PPO counter {model.num_timesteps} unchanged, actor-only supervised updates.\n- Both validation reports and fitting/data provenance are in this directory. Round pass: {decision["round_pass"]}; no promotion.\n'
    if decision['round_pass']:text+=f'- Gated continuation archive: `{rel}/prepared.zip`, SHA-256 `{sha256(out/"prepared.zip")}`; retain `.manifest.json` and `.rng.pt` sidecars. Review before PPO.\n'
    with registry.open('a') as f:fcntl.flock(f,fcntl.LOCK_EX);f.write(text)

def plot_fit(report,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4));history=report['history'];steps=[r['update'] for r in history]
    axes[0].plot(steps,[r['balanced_selection_mse'] for r in history]);axes[0].set(xlabel='Supervised optimizer updates',ylabel='Balanced holdout action MSE')
    for family in ENCOUNTERS:
        values=[r['per_family'][family]['intervention']['mse'] for r in history]
        axes[1].plot(steps,values,label=family)
    axes[1].set(xlabel='Supervised optimizer updates',ylabel='Intervention-only holdout MSE');axes[1].legend();fig.tight_layout();fig.savefig(out/'fitting_curves.png',dpi=140);plt.close(fig)

def fit(paths,output,parent=PARENT,config=DEFAULT,experiment=CONFIG,updates=None):
    c=read_config(experiment)
    if updates is not None:
        if updates<1:raise ValueError('Positive updates required')
        c['updates']=updates
    data=load_data(paths,config);out=Path(output);out.resolve().relative_to(ROOT);out.mkdir(parents=True,exist_ok=False)
    prior=json.loads(Path(str(Path(parent).with_suffix(''))+'.manifest.json').read_text());torch.set_num_threads(1)
    vec=make_vec(str(Path(config).resolve()),prior['workers'],prior['root_seed'],out)
    try:
        model,m=load_stage2(parent,config,vec);state=torch.load(str(Path(parent).with_suffix(''))+'.rng.pt',weights_only=False)
        report=fit_actor(model,data,c,stage1_config(load_config(config)['stage1_config']));write(out/'fit_report.json',report);plot_fit(report,out)
        np.savez_compressed(out/'candidate_actor.npz',**{n:v.detach().cpu().numpy() for n,v in model.policy.state_dict().items() if n.startswith(ACTOR_PREFIXES)})
        candidate=dict(protocol=c['protocol'],diagnostic_only=True,parent=str(Path(parent).resolve()),parent_sha256=sha256(parent),stage2_contract=manifest(config),config=c,source_sha256=sha256(__file__),data=data[-1],actor_sha256=sha256(out/'candidate_actor.npz'),fit_report_sha256=sha256(out/'fit_report.json'),use='Only load through this tool for deterministic evaluation/student-state labeling; NOT a PPO checkpoint')
        write(out/'candidate.json',candidate)
        sc=load_config(config);sc['_config_path']=str(Path(config).resolve());ped=evaluate(model,sc,out/'validation_pedestrian');s1=evaluate(model,sc,out/'validation_stage1',suite='stage1');decision=gate(ped,s1,sc,m['parent_stage1_validation'])
        if decision['round_pass']:
            m=copy.deepcopy(m);m['distillation_v3']=candidate;restore_checkpoint_rng(model,vec,state);checkpoint(model,out/'prepared',m)
        result=dict(saved_diagnostic_actor=True,saved_prepared=decision['round_pass'],promotion=decision,ppo_learn_called=False,ppo_timesteps=model.num_timesteps);write(out/'result.json',result);register(out,parent,model,decision);return result
    finally:vec.close()

def load_student(folder,config):
    folder=Path(folder);c=json.loads((folder/'candidate.json').read_text())
    if c['protocol']!='stage2_intervention_balanced_distillation_v3' or c.get('diagnostic_only') is not True or c['stage2_contract']!=manifest(config) or c['parent_sha256']!=sha256(c['parent']) or c['actor_sha256']!=sha256(folder/'candidate_actor.npz'):raise ValueError('Diagnostic candidate provenance mismatch')
    model,_=load_stage2(c['parent'],config)
    with np.load(folder/'candidate_actor.npz',allow_pickle=False) as data:
        expected={n:v for n,v in model.policy.state_dict().items() if n.startswith(ACTOR_PREFIXES)}
        if set(data.files)!=set(expected):raise ValueError('Actor keys mismatch')
        weights={}
        for n,v in expected.items():
            if data[n].shape!=tuple(v.shape) or not np.isfinite(data[n]).all() or data[n].dtype!=v.cpu().numpy().dtype:raise ValueError('Actor tensor mismatch')
            weights[n]=torch.as_tensor(data[n].copy(),device=model.device)
    model.policy.load_state_dict(weights,strict=False);return model,c

def dagger(candidate,output,config=DEFAULT,experiment=CONFIG,count=None,beta=None):
    c=read_config(experiment);count=c['dagger_episodes_per_family'] if count is None else count;beta=c['dagger_teacher_probability'] if beta is None else beta
    if count<3 or not 0<=beta<=1:raise ValueError('Require >=3 episodes/family and beta in [0,1]')
    model,student=load_student(candidate,config);out=Path(output);out.mkdir(parents=True,exist_ok=False);sc=load_config(config);split=json.loads((Path(sc['stage1_config']).parent/'stage1_splits.json').read_text());rng=np.random.default_rng(c['dagger_seed']);env=Stage2Env(config)
    record=dict(protocol=DAGGER,stage2_contract=manifest(config),teacher_sha256=sha256(Path(__file__).with_name('stage2_causal_teacher.py')),parameters=PARAMETERS,label_semantics='teacher commands queried on student-visited states; NOT successful expert returns',candidate=str(Path(candidate).resolve()),candidate_sha256=student['actor_sha256'],teacher_execution_probability=beta,seed=c['dagger_seed'],source_sha256=sha256(__file__),final_tests_used=False)
    write(out/'manifest.json',record);xs=[];ys=[];ids=[];fs=[];rows=[];executed=0;physics=0;gait=0;begin=time.perf_counter()
    try:
        for family in FAMILIES:
            seed=c['dagger_seed'];finished=0
            while finished<count:
                scenario=sample_episode(seed,family,sc).scenario_metadata();seed+=4
                if reserved(scenario,split):continue
                obs,_=env.reset(options=dict(family=family,episode_seed=scenario['seed']));teacher=Teacher(env.config);metrics_=EpisodeMetrics();episode=len(rows)
                with (out/f'trace_{episode:03d}.jsonl').open('x') as log:
                    while True:
                        student_action,_=model.predict(obs,deterministic=True);label=teacher.act(obs);use_teacher=bool(rng.random()<beta);action=label if use_teacher else student_action
                        xs.append(obs.copy());ys.append(label);ids.append(episode);fs.append(family);executed+=int(use_teacher);before=obs.copy();obs,r,t,tr,info=env.step(action);metrics_.add(r,info);physics+=info.get('physics_steps',0);gait+=info.get('gait_calls',0)
                        log.write(json.dumps(dict(observation=before.tolist(),student_action=student_action.tolist(),teacher_label=label.tolist(),teacher_executed=use_teacher,executed_action=action.tolist(),info=info))+'\n')
                        if t or tr:break
                row=metrics_.summary(family,scenario['seed']);row['scenario']=scenario;rows.append(row);finished+=1;write(out/'episodes.json',rows);print(f'DAgger {len(rows)}/{count*len(FAMILIES)} {family}: {row["reason"]}',flush=True)
                if row['reason'] in ('fall','supervisor_failure','physics_failure'):raise RuntimeError('Critical failure: collection aborted; do not fit incomplete data')
    finally:env.close()
    np.savez_compressed(out/'demonstrations.npz',observations=np.asarray(xs,np.float32),actions=np.asarray(ys,np.float32),episode_ids=np.asarray(ids),families=np.asarray(fs))
    result=dict(protocol=DAGGER,episodes=len(rows),transitions=len(xs),active_physics_steps=physics,active_gait_calls=gait,teacher_executed_steps=executed,student_executed_steps=len(xs)-executed,wall_s=time.perf_counter()-begin,data_sha256=sha256(out/'demonstrations.npz'),episodes_sha256=sha256(out/'episodes.json'),teacher_ready='Not applicable: these are teacher labels on student rollouts, not successful teacher demonstrations',ppo_learn_called=False)
    write(out/'summary.json',result);return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['fit','dagger']);p.add_argument('--output',required=True);p.add_argument('--data',action='append');p.add_argument('--checkpoint',default=str(PARENT));p.add_argument('--candidate');p.add_argument('--config',default=str(DEFAULT));p.add_argument('--experiment',default=str(CONFIG));p.add_argument('--updates',type=int);p.add_argument('--episodes-per-family',type=int);p.add_argument('--teacher-probability',type=float);a=p.parse_args()
    if a.mode=='dagger' and not a.candidate:p.error('dagger requires --candidate')
    result=fit(a.data or [BASE],a.output,a.checkpoint,a.config,a.experiment,a.updates) if a.mode=='fit' else dagger(a.candidate,a.output,a.config,a.experiment,a.episodes_per_family,a.teacher_probability)
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
