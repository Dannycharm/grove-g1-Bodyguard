"""Bounded residual-only PPO research pilot. NOT Stage 2 promotion authorization."""
import argparse,copy,functools,hashlib,json,os,sys,time,shutil
from collections import Counter
from pathlib import Path
import numpy as np
import torch
import yaml
from stable_baselines3.common.vec_env import DummyVecEnv,SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from escort_rl.contract import ROOT,PACKAGE,sha256
from escort_rl.ppo_training import TimedPPO,streams,checkpoint,exact_learn
from escort_rl.sb3_restart import clear_pending_resets,restore_checkpoint_rng
from escort_rl.stage2.env import Stage2Env
from escort_rl.stage2.training import Sampler,manifest as runtime_manifest
from escort_rl.stage2.scenarios import DEFAULT,FAMILIES,load_config,sample_episode
from escort_rl.stage2.evaluation import evaluate,gate
from escort_rl.ppo_evaluation import EpisodeMetrics,reserved
from escort_rl.imitation_ppo import returns
sys.path.insert(0,str(Path(__file__).resolve().parent))
from residual_ppo_policy import ResidualActorCriticPolicy,LatentResidualEnv,ExecutedPolicy,LATENT_LIMIT
import stage2_residual as core

CONFIG=Path(__file__).with_name('residual_ppo_pilot_v1.yaml');PROTOCOL='residual_latent_ppo_pilot_v1'
PRE_LOGGING_FIX_DRIVER_SHA256='d6e96a96a40e5486a418e33b47a8237c66ec92ac87fd2279416a28b455d02bf1'

def json_numpy(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    raise TypeError(f'Unsupported JSON value: {type(value).__name__}')

class ResidualPPO(TimedPPO):
    def train(self):
        super().train()
        # Read-only monitoring: no extra shuffled buffer.get() or RNG consumption.
        obs=self.rollout_buffer.observations.reshape(-1,40);actions=self.rollout_buffer.actions.reshape(-1,3);old=self.rollout_buffer.log_probs.reshape(-1);total=count=0
        with torch.no_grad():
            for i in range(0,len(obs),512):
                x=torch.as_tensor(obs[i:i+512],device=self.device);a=torch.as_tensor(actions[i:i+512],device=self.device);_,logp,_=self.policy.evaluate_actions(x,a);mask=torch.as_tensor(core.gate_mask(obs[i:i+512]),device=self.device);ratio=logp-torch.as_tensor(old[i:i+512],device=self.device)
                total+=float(((torch.exp(ratio)-1)-ratio)[mask].sum());count+=int(mask.sum())
        self.logger.record('train/active_latent_approx_kl',total/count if count else 0.);self.policy.assert_frozen()

def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False))
def read_config(path=CONFIG):
    c=yaml.safe_load(Path(path).read_text())
    if c['protocol']!=PROTOCOL or c['workers'] not in (1,2) or c['max_total_ppo_transitions']!=5000 or not 0<c['latent_sigma']<=.02:raise ValueError('Require versioned <=5k small-noise pilot')
    if c['ppo']['learning_rate']>1e-5 or not 0<c['ppo']['target_kl']<=.01:raise ValueError('Conservative learning-rate/KL guard required')
    if c['critic']['episodes']<18 or c['critic']['episodes']%len(FAMILIES) or c['critic']['scenario_seed_start']%4:raise ValueError('Balanced complete critic episodes in training partition required')
    if sha256(ROOT/c['source'])!=c['source_sha256']:raise ValueError('Residual BC source hash mismatch')
    return c

def provenance(c):
    files=[Path(__file__).resolve(),Path(__file__).with_name('residual_ppo_policy.py').resolve()]
    return dict(stage='stage2_residual_ppo',protocol=PROTOCOL,config=c,source_hashes={str(p.relative_to(ROOT)):sha256(p) for p in files},runtime=runtime_manifest(DEFAULT),transition='BC residual imported exactly; new value network and fresh PPO optimizer; BC optimizer not a PPO resume',action_law='Masked raw Gaussian u stored/scored before SB3 latent clip +/-20; applied=clip(base+mask*2*tanh(clipped_u)); inactive u/logprob/entropy=0',experiment_authorization='Bounded research pilot before promotion; original geometric-intrusion gate remains unchanged',promotion_approved=False,final_test='Not generated or evaluated')

def worker(source,split,seed,log):
    os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1';torch.set_num_threads(1)
    return Monitor(Sampler(LatentResidualEnv(Stage2Env(),source),split,seed,log))

def make_vec(c,out):
    sc=load_config();split=json.loads((Path(sc['stage1_config']).parent/'stage1_splits.json').read_text());ss=streams(c['seed'],c['workers'])
    factories=[functools.partial(worker,str(ROOT/c['source']),split,ss['workers'][i],str(Path(out)/f'worker_{i}_scenarios.jsonl')) for i in range(c['workers'])]
    return DummyVecEnv(factories) if len(factories)==1 else SubprocVecEnv(factories,start_method='spawn')

def build(c,vec):
    ss=streams(c['seed'],c['workers']);model=ResidualPPO(ResidualActorCriticPolicy,vec,device='cpu',seed=ss['model'],policy_kwargs=dict(residual_source=str(ROOT/c['source']),latent_sigma=c['latent_sigma']),**c['ppo'])
    clear_pending_resets(vec)  # Constructor must not overwrite independent scenario streams.
    model.policy.assert_frozen();return model

def checked_load(path,c,vec=None):
    path=Path(path).with_suffix('');prior=json.loads(Path(str(path)+'.manifest.json').read_text());current=provenance(c)
    old_sources=prior.get('source_hashes',{});expected=dict(current['source_hashes']);expected[str(Path(__file__).resolve().relative_to(ROOT))]=PRE_LOGGING_FIX_DRIVER_SHA256
    logging_migration=old_sources==expected and old_sources!=current['source_hashes'] and prior.get('num_timesteps')==0 and prior.get('preparation',{}).get('fit',{}).get('accepted',False)
    for key in ('stage','protocol','config','source_hashes','runtime'):
        if key=='source_hashes' and logging_migration:continue
        if prior.get(key)!=current[key]:raise ValueError('Residual PPO resume mismatch: '+key)
    if prior.get('archive_sha256')!=sha256(str(path)+'.zip'):raise ValueError('PPO archive hash mismatch')
    model=ResidualPPO.load(str(path)+'.zip',env=vec,device='cpu',force_reset=True)
    if logging_migration:
        if model.num_timesteps!=0 or model._n_updates!=0 or model.policy.optimizer.state:raise ValueError('Logging-fix migration requires a prepared checkpoint with zero PPO updates')
        prior=dict(prior,logging_fix_migration=dict(from_source_hashes=old_sources,to_source_hashes=current['source_hashes'],reason='NumPy callback serialization fix; prepared zero-update archive only'))
    if vec is not None:restore_checkpoint_rng(model,vec,torch.load(str(path)+'.rng.pt',weights_only=False))
    model.policy.assert_frozen();return model,prior

def save(model,path,record):
    model.policy.assert_frozen();checkpoint(model,path,record);file=Path(str(path)+'.manifest.json');m=json.loads(file.read_text());m['archive_sha256']=sha256(str(path)+'.zip');m['num_timesteps']=model.num_timesteps;write(file,m)

def verify_import(model,c,out):
    source,_=core.load(ROOT/c['source']);all_exact=True;no_ped_exact=True;max_error=0.;count=0;reference={}
    # Offline observation probes only: fixed validation traces never enter gradients.
    for suite in ('pedestrian','stage1'):
        folder=(ROOT/c['source']).parent/f'validation_{suite}';summary=json.loads((folder/'summary.json').read_text());reference[suite]=dict(summary_sha256=sha256(folder/'summary.json'),summary={k:v for k,v in summary.items() if k!='episodes_detail'})
        for episode in folder.iterdir():
            if not episode.is_dir():continue
            for line in open(episode/'trace.jsonl'):
                row=json.loads(line);obs=np.asarray(row['observation'],np.float32);old=np.asarray(row['action'],np.float32);new=ExecutedPolicy(model).predict(obs)[0]
                error=float(np.max(np.abs(new-old)));max_error=max(max_error,error);all_exact &= np.array_equal(new,old);count+=1
                if not core.gate_mask(obs):no_ped_exact &= np.array_equal(new,old)
    if max_error>1e-6 or not no_ped_exact:raise RuntimeError('Deterministic residual import changed behavior')
    result=dict(probes=count,max_abs_error=max_error,all_exact=bool(all_exact),inactive_exact=bool(no_ped_exact),baseline=reference,evidence='Read-only same-observation import check; no fresh initial full-dynamics rollout')
    write(Path(out)/'import_verification.json',result);return result

def collect_critic(model,c,out):
    cfg=c['critic'];sc=load_config();split=json.loads((Path(sc['stage1_config']).parent/'stage1_splits.json').read_text());x=[];targets=[];ids=[];rows=[];phys=gait=0
    # Entire construction/collection is isolated from the later PPO action RNG.
    with torch.random.fork_rng():
        env=LatentResidualEnv(Stage2Env(),str(ROOT/c['source']));torch.manual_seed(cfg['action_seed'])
        try:
            for j,family in enumerate(FAMILIES):
                seed=cfg['scenario_seed_start']+j*1000;complete=0
                while complete<cfg['episodes']//len(FAMILIES):
                    meta=sample_episode(seed,family,sc).scenario_metadata();seed+=4
                    if reserved(meta,split):continue
                    obs,_=env.reset(options=dict(family=family,episode_seed=meta['seed']));episode=len(rows);rewards=[];xx=[];metrics=EpisodeMetrics()
                    while True:
                        latent,_=model.predict(obs,deterministic=False);xx.append(obs.copy());obs,r,t,tr,info=env.step(latent);rewards.append(r);metrics.add(r,info);phys+=info['physics_steps'];gait+=info['gait_calls']
                        if t or tr:break
                    if info['reason'] in ('fall','supervisor_failure','physics_failure'):raise RuntimeError('Critical failure during stochastic critic collection')
                    bootstrap=0.
                    if tr:
                        with torch.no_grad():bootstrap=float(model.policy.predict_values(torch.as_tensor(obs[None])).item())
                    x.extend(xx);targets.extend(returns(rewards,model.gamma,bootstrap));ids.extend([episode]*len(xx));row=metrics.summary(family,meta['seed']);row['bootstrap']=bootstrap;rows.append(row);complete+=1
                    write(Path(out)/'critic_episodes.json',rows);print(f'critic {len(rows)}/{cfg["episodes"]} {family}: {row["reason"]}',flush=True)
        finally:env.close()
    np.savez_compressed(Path(out)/'critic_data.npz',observations=np.asarray(x,np.float32),returns=np.asarray(targets,np.float32),episode_ids=np.asarray(ids))
    return np.asarray(x,np.float32),np.asarray(targets,np.float32),np.asarray(ids),dict(transitions=len(x),active_physics_steps=phys,active_gait_calls=gait,episodes=len(rows))

def fit_critic(model,x,y,ids,c):
    cfg=c['critic'];original={k:v.detach().clone() for k,v in model.policy.residual_net.state_dict().items()};std=model.policy.log_std.detach().clone()
    each=cfg['episodes']//len(FAMILIES);hold=np.isin(ids,np.arange(0,cfg['episodes'],each));train=np.flatnonzero(~hold);test=np.flatnonzero(hold)
    if not len(train) or not len(test):raise ValueError('Whole-episode calibration holdout missing')
    z=model.policy.inputs(torch.as_tensor(x))[0].detach();target=torch.as_tensor(y);opt=torch.optim.Adam(model.policy.value_net.parameters(),lr=cfg['learning_rate']);rng=np.random.default_rng(cfg['fit_seed'])
    def mse(index):
        with torch.no_grad():return float(((model.policy.value_net(z[index]).flatten()-target[index])**2).mean())
    initial=mse(test);constant=float(np.mean((y[test]-y[train].mean())**2));best=float('inf');chosen=None;history=[]
    for step in range(1,cfg['updates']+1):
        ii=rng.choice(train,cfg['batch_size'],replace=True);loss=((model.policy.value_net(z[ii]).flatten()-target[ii])**2).mean();opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.policy.value_net.parameters(),.5);opt.step()
        if step%100==0 or step==cfg['updates']:
            error=mse(test);history.append(dict(update=step,holdout_mse=error))
            if error<best:best=error;chosen=copy.deepcopy(model.policy.value_net.state_dict())
    model.policy.value_net.load_state_dict(chosen)
    if not all(torch.equal(v,model.policy.residual_net.state_dict()[k]) for k,v in original.items()) or not torch.equal(std,model.policy.log_std):raise RuntimeError('Critic preparation changed residual actor/variance')
    if model.policy.optimizer.state:raise RuntimeError('PPO optimizer must remain fresh during calibration')
    return dict(initial_mse=initial,final_mse=best,constant_mse=constant,accepted=best<initial and best<constant,history=history,holdout_episode_ids=np.arange(0,cfg['episodes'],each).tolist(),actor_and_variance_unchanged=True)

class PilotCallback(BaseCallback):
    def __init__(self,out,c,record):
        super().__init__();self.out=Path(out);self.c=c;self.record=record;self.next=None;self.logs=None;self.metrics={};self.physics=0;self.gait=0;self.completed=[];self.steps=0;self.saturated=0;self.latent_clipped=0
    def _on_training_start(self):
        if self.logs is None:self.logs=(self.out/'training_steps.jsonl').open('x');self.episodes=(self.out/'training_episodes.jsonl').open('x')
        if self.next is None:self.next=(self.model.num_timesteps//self.c['checkpoint_interval']+1)*self.c['checkpoint_interval']
    def _on_step(self):
        self.model.policy.assert_frozen()
        for i,info in enumerate(self.locals['infos']):
            info['ppo_raw_latent']=self.locals['actions'][i].tolist();info['ppo_latent_clipped']=bool(np.any(np.abs(self.locals['actions'][i])>LATENT_LIMIT))
            self.steps+=1;self.saturated+=int(info['ppo_action_saturated']);self.latent_clipped+=int(info['ppo_latent_clipped'])
            self.physics+=info['physics_steps'];self.gait+=info['gait_calls'];metric=self.metrics.setdefault(i,EpisodeMetrics());metric.add(float(self.locals['rewards'][i]),info)
            self.logs.write(json.dumps(dict(timestep=self.model.num_timesteps,worker=i,info=info),default=json_numpy)+'\n')
            if self.locals['dones'][i]:
                case=info['scenario_diagnostics'];row=metric.summary(case['family'],case['episode_seed']);self.episodes.write(json.dumps(row,default=json_numpy)+'\n');self.metrics[i]=EpisodeMetrics();self.completed.append(row);recent=self.completed[-100:]
                self.logger.record('escort/episode_reward_mean',float(np.mean([r['return_total'] for r in recent])));self.logger.record('escort/success_rate',float(np.mean([r['success'] for r in recent])));self.logger.record('escort/formation_mean_m',float(np.mean([r['formation_mean_m'] for r in recent])))
                for family in FAMILIES:
                    group=[r for r in recent if r['family']==family]
                    if group:self.logger.record('escort/'+family+'_success',float(np.mean([r['success'] for r in group])))
                for reason,count in Counter(r['reason'] for r in self.completed).items():self.logger.record('failures/'+reason,count)
                for name in row['reward_components']:self.logger.record('reward_components/'+name,float(np.mean([r['reward_components'].get(name,0.) for r in recent])))
            if info['reason'] in ('fall','supervisor_failure','physics_failure'):raise RuntimeError('Critical training failure; pilot stopped')
        if self.model.num_timesteps>=self.next:
            save(self.model,self.out/f'checkpoint_{self.model.num_timesteps:07d}',self.record);self.next+=self.c['checkpoint_interval']
        return True
    def close(self):
        if self.logs:self.logs.close();self.episodes.close()

def register(out,model,status):
    p=ROOT/'docs/bodyguard/MODEL_WEIGHTS.md';rel=Path(out).resolve().relative_to(ROOT)
    with p.open('a') as f:f.write(f'\n## Residual PPO pilot — {rel}\n\n- Status: {status}; research only, no automatic promotion.\n- Checkpoint directory `{rel}`; preserve ZIP, manifest and RNG sidecars.\n- New residual PPO transitions {model.num_timesteps}; source residual BC 28k and frozen base PPO 5k are separate counts.\n- Use only residual PPO CLI, not generic Stage 1/Stage 2 resume. Training/final validation reports in directory.\n')
    with p.open('a') as f:
        for archive in sorted(Path(out).glob('*.zip')):
            sidecar=archive.with_suffix('.manifest.json');meta=json.loads(sidecar.read_text());rng=archive.with_suffix('.rng.pt')
            f.write(f'- `{archive.relative_to(ROOT)}` SHA-256 `{sha256(archive)}`; counter {meta.get("num_timesteps",meta.get("timesteps","see manifest"))}; manifest `{sidecar.relative_to(ROOT)}`, RNG `{rng.relative_to(ROOT)}`.\n')
        f.write(f'- Full parent hash, source/configuration hashes, seed, reward/runtime identity and transition method: `{rel}/experiment.json` and checkpoint manifests. Actual validation: `{rel}/validation_pedestrian/summary.json` and `{rel}/validation_stage1/summary.json` when completed; interrupted/calibration-failed archives are diagnostic only.\n')

def validation_key(ped,stage1):
    critical=sum(d['reasons'].get(k,0) for d in (ped,stage1) for k in ('fall','supervisor_failure','physics_failure'))
    return (-critical,-ped['geometric_intrusions'],stage1['success_rate'],ped['success_rate'],min(v['success_rate'] for v in ped['per_family'].values()),-stage1['formation_mean_m'],-ped['formation_mean_m'])

def plot_training(out):
    import csv,matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=list(csv.DictReader(open(Path(out)/'logs/progress.csv')));fig,axes=plt.subplots(2,3,figsize=(12,7))
    for ax,key in zip(axes.flat,['rollout/ep_rew_mean','train/active_latent_approx_kl','train/value_loss','train/explained_variance','train/clip_fraction','escort/success_rate']):
        rr=[r for r in rows if r.get(key) and r.get('time/total_timesteps')];ax.plot([float(r['time/total_timesteps']) for r in rr],[float(r[key]) for r in rr]);ax.set(xlabel='Residual PPO transitions',title=key)
    fig.tight_layout();fig.savefig(Path(out)/'training_curves.png',dpi=140);plt.close(fig)

def run(output,config_path=CONFIG,timesteps=5000,resume=None):
    c=read_config(config_path)
    if timesteps<2*c['workers'] or timesteps%c['workers'] or timesteps>5000:raise ValueError('Require <=5,000 pilot transitions, >=2 per worker and divisible by workers')
    out=Path(output).resolve();out.relative_to(ROOT);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1);torch.use_deterministic_algorithms(True);record=provenance(c);write(out/'experiment.json',record);vec=make_vec(c,out);callback=None;model=None
    try:
        if resume:
            model,prior=checked_load(resume,c,vec)
            if not prior.get('preparation',{}).get('fit',{}).get('accepted',False):raise ValueError('Resume requires a passed stochastic critic calibration; incomplete preparation cannot be bypassed')
            record.update(resume=str(Path(resume).resolve()),preparation=prior['preparation'])
            if 'logging_fix_migration' in prior:record['logging_fix_migration']=prior['logging_fix_migration']
        else:
            model=build(c,vec);verify_import(model,c,out);data=collect_critic(model,c,out);fit=fit_critic(model,*data[:3],c);write(out/'critic_fit.json',fit);record['preparation']=dict(collection=data[3],fit={k:v for k,v in fit.items() if k!='history'},data_sha256=sha256(out/'critic_data.npz'))
            if not fit['accepted']:save(model,out/'calibration_failed',record);write(out/'result.json',dict(status='critic_calibration_failed',ppo_transitions=0));register(out,model,'critic calibration failed; no PPO');return
        if timesteps<2*c['workers'] or timesteps%c['workers'] or model.num_timesteps+timesteps>5000:raise ValueError('Pilot permits <=5,000 cumulative residual PPO transitions and >=2/worker')
        save(model,out/'initial',record);model.set_logger(configure(str(out/'logs'),['stdout','csv','tensorboard']));callback=PilotCallback(out,c,record);start=model.num_timesteps;begin=time.perf_counter();exact_learn(model,timesteps,callback);wall=time.perf_counter()-begin;model.logger.dump(step=model.num_timesteps);save(model,out/'final',record)
        stats=dict(active_physics_steps=callback.physics,active_gait_calls=callback.gait,completed_episodes=len(callback.completed),outcomes=dict(Counter(r['reason'] for r in callback.completed)),action_saturation_fraction=callback.saturated/max(1,callback.steps),latent_clip_fraction=callback.latent_clipped/max(1,callback.steps),optimizer_steps=getattr(model,'gradient_steps',0),optimizer_wall_s=getattr(model,'optimizer_wall_s',0.));callback.close();callback=None;plot_training(out)
        sc=load_config();sc['_config_path']=str(DEFAULT);ped=evaluate(ExecutedPolicy(model),sc,out/'validation_pedestrian');s1=evaluate(ExecutedPolicy(model),sc,out/'validation_stage1',suite='stage1');base_summary=json.loads(((ROOT/c['source']).parent/'validation_stage1/summary.json').read_text());decision=gate(ped,s1,sc,base_summary)
        source_ped=json.loads(((ROOT/c['source']).parent/'validation_pedestrian/summary.json').read_text());source_s1=base_summary
        # On resume, best selection is segment-initial versus segment-final, not global.
        if resume:
            initial_model,_=checked_load(out/'initial',c);initial_ped=evaluate(ExecutedPolicy(initial_model),sc,out/'validation_segment_initial_pedestrian');initial_s1=evaluate(ExecutedPolicy(initial_model),sc,out/'validation_segment_initial_stage1',suite='stage1')
        else:initial_ped,initial_s1=source_ped,source_s1
        chosen='final' if validation_key(ped,s1)>validation_key(initial_ped,initial_s1) else 'initial'
        for suffix in ('.zip','.manifest.json','.rng.pt'):shutil.copyfile(str(out/chosen)+suffix,str(out/'best_validation')+suffix)
        result=dict(status='pilot_complete',start_ppo_timesteps=start,end_ppo_timesteps=model.num_timesteps,new_transitions=model.num_timesteps-start,training_wall_s=wall,transitions_per_second=timesteps/wall,training=stats,best_validation=chosen,selected_residual_ppo_timesteps=model.num_timesteps if chosen=='final' else start,promotion_gate=decision,promotion_approved=False,note='Research pilot, not stage promotion; two consecutive full passing rounds remain required');write(out/'result.json',result);register(out,model,'completed pilot; inspect validation, no automatic promotion');return result
    except (KeyboardInterrupt,RuntimeError,TypeError,ValueError):
        if model is not None:save(model,out/f'interrupted_{model.num_timesteps:07d}',record);register(out,model,'interrupted; inspect failure before resume')
        raise
    finally:
        if callback:callback.close()
        vec.close()

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['train','evaluate']);p.add_argument('--output',required=True);p.add_argument('--config',default=str(CONFIG));p.add_argument('--timesteps',type=int,default=5000);p.add_argument('--resume');p.add_argument('--checkpoint');p.add_argument('--suite',choices=['pedestrian','stage1'],default='pedestrian');a=p.parse_args()
    if a.mode=='train':result=run(a.output,a.config,a.timesteps,a.resume)
    else:
        if not a.checkpoint:p.error('evaluate requires --checkpoint')
        c=read_config(a.config);torch.set_num_threads(1);model,_=checked_load(a.checkpoint,c);sc=load_config();sc['_config_path']=str(DEFAULT);result=evaluate(ExecutedPolicy(model),sc,a.output,suite=a.suite)
    if result is not None:print(json.dumps({k:v for k,v in result.items() if k!='episodes_detail'},indent=2))
if __name__=='__main__':main()
