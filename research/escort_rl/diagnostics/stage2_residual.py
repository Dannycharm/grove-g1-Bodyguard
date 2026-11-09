"""Frozen Stage 1 mean + exactly mask-gated residual; no PPO or live interfaces.
Reuse Stage2Env, existing teacher and unchanged fixed evaluation/gates.
"""
import argparse,copy,fcntl,json,sys,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import yaml
from escort_rl.contract import ROOT,PACKAGE,sha256
from escort_rl.ppo_training import TimedPPO
from escort_rl.stage1_scenarios import load_config as load_stage1
from escort_rl.stage2.scenarios import DEFAULT,load_config,sample_episode
from escort_rl.stage2.training import manifest
from escort_rl.stage2.env import Stage2Env
from escort_rl.stage2.evaluation import evaluate,aggregate,gate
from escort_rl.ppo_evaluation import EpisodeMetrics,reserved
sys.path.insert(0,str(Path(__file__).resolve().parent))
from stage2_causal_teacher import Teacher,PARAMETERS

CONFIG=Path(__file__).with_name('stage2_residual_v1.yaml')
PROTOCOL='frozen_stage1_residual_avoidance_v1'
FEATURES=51

def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False))
def config(path=CONFIG):
    c=yaml.safe_load(Path(path).read_text())
    if c['protocol']!=PROTOCOL or c['teacher_mode'] not in ('observable_previous_command_v1','original_history_v2'):raise ValueError('Unknown protocol or teacher mode')
    if c['residual_limit']!=2. or c['hidden_sizes']!=[128,128]:raise ValueError('Version 1 architecture/bounds mismatch')
    if c['batch_size']<1 or c['iterations']<1 or c['updates_per_iteration']<1 or not 0<=c['daggers_teacher_probability']<=1:raise ValueError('Invalid budget')
    return c

def base_observation(x):
    y=np.asarray(x,np.float32).copy();y[...,22:40]=0.;return y

def gate_mask(x):return np.any(np.asarray(x)[..., [27,33,39]]>=.5,axis=-1)

def load_base(c,stage2=DEFAULT):
    path=ROOT/c['base_checkpoint'];old=json.loads(path.with_suffix('.manifest.json').read_text());current=manifest(stage2)['stage1_contract']
    if sha256(path)!=c['base_sha256']:raise ValueError('Selected base checkpoint hash mismatch')
    for k in ('source_hashes','dependencies','config','reference_manifest_sha256','frozen_gait_sha256','reward_contract_sha256'):
        if old.get(k)!=current[k]:raise ValueError('Base runtime provenance mismatch: '+k)
    if old.get('initialization_protocol')!='imitation_pretrained_ppo_v1':raise ValueError('Require selected guarded Stage 1 policy')
    model=TimedPPO.load(path,device='cpu');model.policy.set_training_mode(False)
    for p in model.policy.parameters():p.requires_grad_(False)
    return model

def components(obs,bc):
    x=np.atleast_2d(np.asarray(obs,np.float32));o=bc['observation']
    v=x[:,10:12]*np.asarray(o['robot_velocity_scales'][:2]);client=x[:,2:4]*o['relative_velocity_scale_m_s']+v
    r=x[:,22:24]*o['position_scale_m'];rel=x[:,24:26]*o['relative_velocity_scale_m_s'];ped=rel+v
    angle=np.arctan2(x[:,4],x[:,5]);off=x[:,6:8]*o['offset_scale_m'];co=np.cos(angle);si=np.sin(angle)
    target=x[:,:2]*o['position_scale_m']+np.stack([co*off[:,0]-si*off[:,1],si*off[:,0]+co*off[:,1]],axis=1)
    desired=np.clip(client+.65*target,-.35,.35)
    return x,v,client,r,rel,ped,target,desired

def risk_features(obs,bc,horizon=4.):
    x,v,client,r,rel,ped,target,desired=components(obs,bc)
    distance=np.linalg.norm(r,axis=1);closing=-np.sum(r*rel,axis=1)/np.maximum(distance,1e-8)
    def closest(w):
        t=np.clip(-np.sum(r*w,axis=1)/np.maximum(np.sum(w*w,axis=1),1e-8),0,horizon)
        d=np.linalg.norm(r+t[:,None]*w,axis=1);return t,d
    t,d=closest(rel);tn,dn=closest(ped-desired)
    return np.clip(np.stack([distance/4,closing,t/horizon,d/2,tn/horizon,dn/2,np.maximum(.85-dn,0)/.85,np.linalg.norm(client,axis=1)/.5],axis=1),-4,4).astype(np.float32)

def features(obs,base_action,bc,c):
    x=np.atleast_2d(np.asarray(obs,np.float32)).copy();o=bc['observation']
    x[:,:2]*=o['position_scale_m']/3.;x[:,2:4]*=o['relative_velocity_scale_m_s']/.5
    x[:,10:12]*=np.asarray(o['robot_velocity_scales'][:2])/.5
    x[:,22:24]*=o['position_scale_m']/3.;x[:,24:26]*=o['relative_velocity_scale_m_s']/.75
    return np.concatenate([np.clip(x,-4,4),risk_features(obs,bc,c['risk_horizon_s']),np.atleast_2d(base_action)],axis=1).astype(np.float32)

class ResidualPolicy:
    def __init__(self,c=None,stage2=DEFAULT):
        self.c=config() if c is None else copy.deepcopy(c);self.stage2=str(Path(stage2).resolve());self.bc=load_stage1(load_config(stage2)['stage1_config']);self.base=load_base(self.c,stage2)
        with torch.random.fork_rng():
            torch.manual_seed(self.c['seed']);self.net=nn.Sequential(nn.Linear(FEATURES,128),nn.Tanh(),nn.Linear(128,128),nn.Tanh(),nn.Linear(128,3))
            nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
        self.base_before={n:v.detach().clone() for n,v in self.base.policy.state_dict().items()};self.updates=0
    def base_action(self,obs):
        a,_=self.base.predict(base_observation(obs),deterministic=True);return np.asarray(a,np.float32)
    def residual_tensor(self,z):return self.c['residual_limit']*torch.tanh(self.net(z))
    def predict(self,observation,deterministic=True):
        if not deterministic:raise ValueError('Residual PPO/stochastic law not implemented; deterministic imitation only')
        obs=np.asarray(observation,np.float32)
        if obs.shape[-1:]!=(40,) or not np.isfinite(obs).all():raise ValueError('Require finite external 40-vector')
        if np.any(obs[..., [33,39]]>=.5):raise ValueError('Only Stage 2 single pedestrian is supported')
        single=obs.ndim==1;x=np.atleast_2d(obs);base=np.atleast_2d(self.base_action(x));out=base.copy();active=gate_mask(x)
        if active.any():
            with torch.no_grad():delta=self.residual_tensor(torch.as_tensor(features(x[active],base[active],self.bc,self.c))).numpy()
            out[active]=np.clip(base[active]+delta,-1,1)
        # Bypass inactive rows altogether: no 0*NaN or numeric perturbation of base.
        return (out[0] if single else out).astype(np.float32),None
    def assert_frozen(self):
        if any(p.requires_grad for p in self.base.policy.parameters()):raise RuntimeError('Base unexpectedly trainable')
        if any(not torch.equal(v,self.base.policy.state_dict()[n]) for n,v in self.base_before.items()):raise RuntimeError('Frozen base tensor changed')
        if sha256(ROOT/self.c['base_checkpoint'])!=self.c['base_sha256']:raise RuntimeError('Base archive changed')

class SupervisionTeacher:
    """Explicit observable anchoring of original teacher's previous-command cost."""
    def __init__(self,bc,mode):self.bc=bc;self.mode=mode;self.inner=Teacher(bc)
    def act(self,obs):
        if self.mode=='observable_previous_command_v1':
            command=np.asarray(obs[19:21])*np.asarray(self.bc['environment']['command_caps'][:2]);dead=np.array([.08,.10])
            self.inner.previous=np.where(np.abs(command)>dead,.8*(command-np.sign(command)*dead),0.)
        return self.inner.act(obs)

def audit_teacher(output,stage2=DEFAULT):
    out=Path(output);out.mkdir(parents=True,exist_ok=False);bc=load_stage1(load_config(stage2)['stage1_config']);folder=PACKAGE/'runs/stage2_teacher_v2_readiness_20261108'
    record=json.loads((folder/'manifest.json').read_text());summary=json.loads((folder/'summary.json').read_text())
    if summary['data_sha256']!=sha256(folder/'demonstrations.npz') or record['parameters']!=PARAMETERS:raise ValueError('Audit dataset integrity mismatch')
    with np.load(folder/'demonstrations.npz',allow_pickle=False) as d:x=d['observations'];y=d['actions'];ids=d['episode_ids']
    indices=np.flatnonzero(x[:,27]>=.5);differences=[];observable=[];replayed=[];expert=Teacher(bc);last=None
    for i in indices:
        if ids[i]!=last:expert=Teacher(bc);last=ids[i]
        replayed.append(float(np.max(np.abs(expert.act(x[i])-y[i]))))
        anchored=SupervisionTeacher(bc,'observable_previous_command_v1').act(x[i]);observable.append(float(np.linalg.norm(anchored-y[i])))
        actions=[]
        for previous in ([0,0],[.3,.3],[-.3,-.3],[.3,-.3],[-.3,.3]):
            t=Teacher(bc);t.previous=np.array(previous,float);actions.append(t.act(x[i]))
        differences.append(max(float(np.linalg.norm(a-b)) for a in actions for b in actions))
    # Exact duplicate observations with different labels are unambiguous contradictions;
    # near-neighbour ambiguity requires additional scale/metric assumptions.
    groups={};contradictions=0
    for i in indices:
        k=x[i].tobytes()
        if k in groups and np.linalg.norm(y[i]-groups[k])>.05:contradictions+=1
        groups[k]=y[i]
    result=dict(samples=len(indices),original_sequence_replay_max_abs=float(max(replayed)),hidden_previous_changes_action_fraction=float(np.mean(np.array(differences)>.05)),hidden_previous_max_action_l2=float(max(differences)),observable_anchor_changed_fraction=float(np.mean(np.array(observable)>.05)),observable_anchor_max_action_l2=float(max(observable)),exact_duplicate_label_conflicts=contradictions,interpretation='History dependence tested at identical observations; observable teacher is a recorded variant requiring closed-loop revalidation, not silently relabeled old demonstrations',teacher_sha256=sha256(Path(__file__).with_name('stage2_causal_teacher.py')),data_sha256=sha256(folder/'demonstrations.npz'))
    write(out/'audit.json',result);return result

def checked_cases(c,stage2):
    split=json.loads((Path(load_config(stage2)['stage1_config']).parent/'stage1_splits.json').read_text());result=[]
    for case in c['development_cases']:
        if case['family'] not in ('crossing','diagonal') or case['episode_seed']%4:raise ValueError('Use development crossing/diagonal training partition')
        meta=sample_episode(case['episode_seed'],case['family'],load_config(stage2)).scenario_metadata()
        if reserved(meta,split):raise ValueError('Reserved final-test parameters')
        result.append(dict(case,metadata=meta))
    return result

def rollout(policy,cases,out,*,teacher_execution=False,stage2=DEFAULT,collect=False,beta=0.,iteration=0):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);env=Stage2Env(stage2);rows=[];xs=[];ys=[];ids=[];phases=[];physics=0;gait=0;teacher_steps=0;rng=np.random.default_rng(policy.c['seed']+iteration)
    try:
        for ep,case in enumerate(cases):
            obs,_=env.reset(options=dict(family=case['family'],episode_seed=case['episode_seed']));expert=SupervisionTeacher(env.config,policy.c['teacher_mode']);metrics=EpisodeMetrics();phase=[]
            with (out/f'trace_{ep:03d}.jsonl').open('x') as f:
                while True:
                    base=policy.base_action(obs);target=expert.act(obs);student=policy.predict(obs)[0];use=teacher_execution or bool(rng.random()<beta);action=target if use else student
                    x,v,client,r,rel,ped,error,desired=components(obs,env.config)
                    avoidance=np.linalg.norm(target-base)>policy.c['intervention_threshold']
                    stopping=np.linalg.norm(client[0])<policy.c['stopping_speed_m_s'];recovery=np.linalg.norm(error[0])>policy.c['recovery_error_m']
                    risk=risk_features(obs,env.config,policy.c['risk_horizon_s'])[0]
                    anticipation=bool(risk[4]>0 and risk[5]*2<policy.c['anticipation_separation_m'])
                    label='avoidance' if avoidance else 'anticipation' if anticipation else 'stopping' if stopping else 'recovery' if recovery else 'ordinary';phase.append(label)
                    if collect:xs.append(obs.copy());ys.append(target.copy());ids.append(ep)
                    before=obs.copy();obs,reward,term,trunc,info=env.step(action);metrics.add(reward,info);physics+=info['physics_steps'];gait+=info['gait_calls'];teacher_steps+=int(use)
                    f.write(json.dumps(dict(observation=before.tolist(),base_action=base.tolist(),student_action=student.tolist(),teacher_action=target.tolist(),teacher_executed=use,executed_action=action.tolist(),phase=label,info=info))+'\n')
                    if term or trunc:break
            if collect:phases.extend(phase)
            row=metrics.summary(case['family'],case['episode_seed']);row.update(geometric_intrusion=info.get('pedestrian_intrusion',False),minimum_surface_clearance_m=info.get('minimum_geometric_surface_clearance_m'));rows.append(row);print(f'{out.name} {case["family"]} {case["episode_seed"]}: {row["reason"]}',flush=True)
            if row['reason'] in ('fall','supervisor_failure','physics_failure'):raise RuntimeError('Critical diagnostic failure')
    finally:env.close()
    result=aggregate(rows);result.update(active_physics_steps=physics,active_gait_calls=gait,teacher_executed_steps=teacher_steps,student_executed_steps=sum(r['steps'] for r in rows)-teacher_steps,teacher_probability=beta,teacher_mode=policy.c['teacher_mode']);write(out/'summary.json',result)
    if collect:
        # Phases were computed from CURRENT causal observations, never future labels.
        phases=np.asarray(phases,dtype='<U16');ids=np.asarray(ids)
        np.savez_compressed(out/'labels.npz',observations=np.asarray(xs,np.float32),teacher_actions=np.asarray(ys,np.float32),episode_ids=ids,phases=phases)
    return result

def fitting_loss(base,delta,target,c):
    raw=base+delta;final=torch.clamp(raw,-1,1);residual_target=target-base
    # Raw residual MSE supplies gradients where action clipping itself is flat.
    executed=((final-target)**2).mean(dim=1);raw_error=((delta-residual_target)**2).mean(dim=1);excess=(raw-final).square().mean(dim=1)
    return executed+c['residual_target_weight']*raw_error+c['saturation_excess_weight']*excess

def fit_iteration(policy,optimizer,paths,c,rng):
    xx=[];yy=[];ww=[];source=[]
    for path in paths:
        with np.load(Path(path)/'labels.npz',allow_pickle=False) as d:
            xx.append(d['observations']);yy.append(d['teacher_actions']);ww.extend(c['phase_weights'][str(p)] for p in d['phases']);source.append(dict(path=str(path),sha256=sha256(Path(path)/'labels.npz')))
    x=np.concatenate(xx);target=np.concatenate(yy);weights=np.asarray(ww,float);weights/=weights.sum();base=policy.base_action(x);z=features(x,base,policy.bc,c)
    initial=policy.predict(x)[0];history=[]
    for step in range(c['updates_per_iteration']):
        ii=rng.choice(len(x),c['batch_size'],replace=True,p=weights);delta=policy.residual_tensor(torch.as_tensor(z[ii]));loss=fitting_loss(torch.as_tensor(base[ii]),delta,torch.as_tensor(target[ii]),c).mean();optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(policy.net.parameters(),c['max_grad_norm']);optimizer.step();policy.updates+=1
        if (step+1)%500==0:history.append(dict(update=policy.updates,loss=float(loss.detach())))
    final=policy.predict(x)[0];policy.assert_frozen()
    return dict(rows=len(x),sources=source,updates=policy.updates,history=history,initial_action_mse=float(np.mean((initial-target)**2)),final_action_mse=float(np.mean((final-target)**2)),clipped_fraction=float(np.mean(np.any(np.abs(final)>=.999,axis=1))))

def save(policy,path,optimizer=None,rng=None,extra=None):
    policy.assert_frozen();path=Path(path)
    if path.exists():raise FileExistsError(path)
    state=dict(net=policy.net.state_dict(),optimizer=None if optimizer is None else optimizer.state_dict(),torch_rng=torch.get_rng_state(),numpy_rng=None if rng is None else rng.bit_generator.state,updates=policy.updates,config=policy.c)
    torch.save(state,path)
    base=ROOT/policy.c['base_checkpoint'];record=dict(protocol=PROTOCOL,base=str(base),base_sha256=sha256(base),base_manifest_sha256=sha256(base.with_suffix('.manifest.json')),base_rng_sha256=sha256(str(base.with_suffix(''))+'.rng.pt'),stage2_contract=manifest(policy.stage2),source_sha256=sha256(__file__),teacher_sha256=sha256(Path(__file__).with_name('stage2_causal_teacher.py')),residual_sha256=sha256(path),external_observation_size=40,internal_features=FEATURES,base_frozen=True,ppo_implemented=False,extra=extra or {})
    write(str(path)+'.manifest.json',record)

def load(path):
    path=Path(path);record=json.loads(Path(str(path)+'.manifest.json').read_text())
    if record['protocol']!=PROTOCOL or sha256(path)!=record['residual_sha256'] or sha256(__file__)!=record['source_sha256'] or sha256(Path(__file__).with_name('stage2_causal_teacher.py'))!=record['teacher_sha256'] or record['stage2_contract']!=manifest(DEFAULT):raise ValueError('Residual provenance mismatch')
    state=torch.load(path,weights_only=False);policy=ResidualPolicy(state['config']);base=ROOT/policy.c['base_checkpoint']
    if sha256(base.with_suffix('.manifest.json'))!=record['base_manifest_sha256'] or sha256(str(base.with_suffix(''))+'.rng.pt')!=record['base_rng_sha256']:raise ValueError('Base sidecar mismatch')
    policy.net.load_state_dict(state['net']);policy.updates=state['updates'];return policy,record

def register(policy,path,extra):
    registry=ROOT/'docs/bodyguard/MODEL_WEIGHTS.md';rel=Path(path).resolve().relative_to(ROOT)
    text=f'\n## Frozen-base residual diagnostic — {rel}\n\n- Residual weights `{rel}`, SHA-256 `{sha256(path)}`; preserve adjacent `.manifest.json`.\n- Frozen base `{policy.c["base_checkpoint"]}`, SHA-256 `{policy.c["base_sha256"]}`; inherited base PPO counter {policy.base.num_timesteps}; residual PPO updates zero.\n- Residual BC updates {policy.updates}; architecture 51→128→128→3, external 40-vector unchanged; teacher mode `{policy.c["teacher_mode"]}`.\n- Diagnostic only, no Stage 2 promotion or PPO support. Evidence: `{str(Path(path).resolve().parent.relative_to(ROOT))}`; {extra}\n'
    with registry.open('a') as f:fcntl.flock(f,fcntl.LOCK_EX);f.write(text)

def diagnostic(output,cpath=CONFIG,stage2=DEFAULT):
    c=config(cpath);out=Path(output);out.resolve().relative_to(ROOT);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1);policy=ResidualPolicy(c,stage2);cases=checked_cases(c,stage2);write(out/'resolved_config.json',c);write(out/'cases.json',cases)
    # No teacher/student labels are collected until history audit is saved.
    audit=audit_teacher(out/'teacher_audit',stage2)
    teacher_result=rollout(policy,cases,out/'teacher_development',teacher_execution=True,stage2=stage2,collect=True)
    if teacher_result['success_rate']<1:write(out/'result.json',dict(status='teacher_development_failed',audit=audit));return
    baseline=rollout(policy,cases,out/'initial_student',stage2=stage2)
    paths=[out/'teacher_development'];optimizer=torch.optim.Adam(policy.net.parameters(),lr=c['learning_rate']);rng=np.random.default_rng(c['seed']+1);rounds=[]
    for iteration in range(c['iterations']):
        fit=fit_iteration(policy,optimizer,paths,c,rng);write(out/f'fit_{iteration}.json',fit)
        result=rollout(policy,cases,out/f'student_{iteration}',stage2=stage2);rounds.append(result)
        path=out/f'residual_{iteration}.pt';save(policy,path,optimizer,rng,dict(iteration=iteration,case_suite='two development training cases, NOT validation',data=fit['sources'],closed_loop=result));register(policy,path,f'two-case development success {result["success_rate"]}; fixed suites pending')
        if iteration+1<c['iterations']:
            folder=out/f'dagger_{iteration}';rollout(policy,cases,folder,stage2=stage2,collect=True,beta=c['daggers_teacher_probability'],iteration=iteration);paths.append(folder)
    write(out/'result.json',dict(status='diagnostic_complete',initial=baseline,rounds=rounds,teacher_audit=audit,final_checkpoint=str(path),base_frozen=True,ppo_updates=0,interpretation='Two-case overfit only; must still evaluate original fixed 60+60 suites'))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['audit','diagnostic','evaluate']);p.add_argument('--output',required=True);p.add_argument('--experiment',default=str(CONFIG));p.add_argument('--checkpoint');p.add_argument('--suite',choices=['pedestrian','stage1'],default='pedestrian');a=p.parse_args()
    if a.mode=='audit':result=audit_teacher(a.output)
    elif a.mode=='diagnostic':result=diagnostic(a.output,a.experiment)
    else:
        if not a.checkpoint:p.error('evaluate requires --checkpoint')
        policy,_=load(a.checkpoint);sc=load_config();sc['_config_path']=str(DEFAULT);result=evaluate(policy,sc,a.output,suite=a.suite)
    if result is not None:print(json.dumps(result,indent=2))
if __name__=='__main__':main()
