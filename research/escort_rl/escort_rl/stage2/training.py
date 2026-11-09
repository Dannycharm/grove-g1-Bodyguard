"""Explicit opt-in stage transition and guarded small PPO pilot. No automatic training."""
import functools
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from stable_baselines3.common.vec_env import DummyVecEnv,SubprocVecEnv
from stable_baselines3.common.logger import configure
from ..contract import ROOT,PACKAGE,sha256
from ..ppo_training import TrainingScenarios,TimedPPO,read_config,provenance,streams,checkpoint,exact_learn
from ..sb3_restart import restore_checkpoint_rng
from ..imitation_ppo import PreservationCallback
from ..ppo_evaluation import reserved
from .scenarios import load_config,sample_episode,DEFAULT,ENCOUNTERS
from .env import Stage2Env
from .evaluation import evaluate,gate

PROTOCOL='stage2_single_pedestrian_transition_v1'


def manifest(config_path):
    c=load_config(config_path);base=provenance(read_config(c['stage1_ppo_config']))
    files=list((PACKAGE/'escort_rl/stage2').glob('*.py'))+list((PACKAGE/'stage2_tools').glob('*.py'))+list((PACKAGE/'stage2_config').glob('*.yaml'))
    return dict(stage='stage2',protocol=PROTOCOL,stage2_config=c,stage1_contract=base,stage2_sources={str(p.relative_to(ROOT)):sha256(p) for p in files},observation_change='Activate slot 0 at 22:28; other 40-vector semantics unchanged',reward_version='stage2_tracking_clearance_v1',final_test='Not generated or evaluated')


class Sampler(TrainingScenarios):
    def reset(self,*,seed=None,options=None):
        if seed is not None and self._restored_stream:raise RuntimeError('Cannot reseed restored Stage 2 stream')
        if seed is not None:self.rng=np.random.default_rng(seed)
        if options:raise ValueError('Sampler owns scenario options')
        for _ in range(10000):
            episode_seed=int(self.rng.integers(0,2**60))*4
            m=sample_episode(episode_seed,config=self.env.unwrapped.stage2_config).scenario_metadata()
            if not reserved(m,self.split):break
        else:raise RuntimeError('Cannot sample Stage 2 training partition')
        obs,info=self.env.reset(options={'episode_seed':episode_seed});self.current=dict(self.env.scenario_metadata(),episode_seed=episode_seed);self.count+=1
        if self.log_path:
            with open(self.log_path,'a') as f:f.write(json.dumps(self.current)+'\n')
        return obs,info


def worker(path,split,seed,log):return Sampler(Stage2Env(path),split,seed,log)


def make_vec(path,workers,seed,out):
    c=load_config(path);split=json.loads((Path(c['stage1_config']).parent/'stage1_splits.json').read_text());s=streams(seed,workers)
    factories=[functools.partial(worker,path,split,s['workers'][i],str(Path(out)/f'worker_{i}_scenarios.jsonl')) for i in range(workers)]
    return DummyVecEnv(factories) if workers==1 else SubprocVecEnv(factories,start_method='spawn')


def activate_slot(model):
    """Zero only six never-active input columns and their Adam moments; no refit."""
    tensors=[]
    for prefix in ('policy_net','value_net'):
        layer=getattr(model.policy.mlp_extractor,prefix)[0]
        if layer.in_features!=40:raise ValueError('Require original 40-input network')
        before=layer.weight.detach().clone()
        with torch.no_grad():layer.weight[:,22:28]=0.
        state=model.policy.optimizer.state.get(layer.weight,{})
        for k in ('exp_avg','exp_avg_sq','max_exp_avg_sq'):
            if k in state:state[k][:,22:28]=0.
        assert torch.equal(layer.weight[:,:22],before[:,:22]) and torch.equal(layer.weight[:,28:],before[:,28:])
        tensors.append('mlp_extractor.'+prefix+'.0.weight[:,22:28]')
    return tensors


def transition(parent,config_path,output):
    out=Path(output);out.mkdir(parents=True,exist_ok=False);m=manifest(config_path);p=Path(parent).resolve().with_suffix('');old=json.loads(Path(str(p)+'.manifest.json').read_text())
    for k in ('source_hashes','dependencies','config','reference_manifest_sha256','frozen_gait_sha256','reward_contract_sha256'):
        if old.get(k)!=m['stage1_contract'][k]:raise ValueError('Stage 1 parent mismatch: '+k)
    if old.get('initialization_protocol')!='imitation_pretrained_ppo_v1':raise ValueError('Require verified guarded imitation-PPO parent')
    workers=old['workers'];seed=old['seed'];torch.set_num_threads(1);vec=make_vec(str(Path(config_path).resolve()),workers,seed,out)
    try:
        model=TimedPPO.load(str(p)+'.zip',env=vec,device='cpu',force_reset=True)
        state=torch.load(str(p)+'.rng.pt',weights_only=False);restore_checkpoint_rng(model,vec,state)
        # The existing Stage 1 policy always observes zeros in these columns.
        probe=np.load(ROOT/old['config']['imitation']['source_data'])['observations'][:256]
        with torch.no_grad():old_mean=model.policy.get_distribution(torch.as_tensor(probe)).distribution.mean.clone();old_value=model.policy.predict_values(torch.as_tensor(probe)).clone()
        changed=activate_slot(model)
        with torch.no_grad():assert torch.equal(old_mean,model.policy.get_distribution(torch.as_tensor(probe)).distribution.mean) and torch.equal(old_value,model.policy.predict_values(torch.as_tensor(probe)))
        m.update(parent_checkpoint=str(p)+'.zip',parent_sha256=sha256(str(p)+'.zip'),parent_manifest_sha256=sha256(str(p)+'.manifest.json'),root_seed=seed,workers=workers,stage2_start_timesteps=model.num_timesteps,parent_stage1_validation=json.loads((p.parent/'validation_final/summary.json').read_text()),slot_initialization=changed,stage1_probe_outputs_exact=True,stage2_training_transitions=0)
        checkpoint(model,out/'initial',m);(out/'manifest.json').write_text(json.dumps(m,indent=2));return m
    finally:vec.close()


def load_stage2(path,config_path,vec=None):
    p=Path(path).with_suffix('');saved=json.loads(Path(str(p)+'.manifest.json').read_text());current=manifest(config_path)
    for k in ('stage','protocol','stage2_config','stage1_contract','stage2_sources'):
        if saved.get(k)!=current[k]:raise ValueError('Stage 2 provenance mismatch: '+k)
    model=TimedPPO.load(str(p)+'.zip',env=vec,device='cpu',force_reset=True)
    if vec is not None:restore_checkpoint_rng(model,vec,torch.load(str(p)+'.rng.pt',weights_only=False))
    return model,saved


class Callback(PreservationCallback):
    def _on_training_start(self):
        super()._on_training_start()
        if not hasattr(self,'ped_log'):self.ped_log=(self.output/'stage2_diagnostics.jsonl').open('x')
    def _on_step(self):
        result=super()._on_step()
        for worker,info in enumerate(self.locals['infos']):
            keys=('reason','elapsed_s','pedestrian_count','geometric_surface_clearance_m','minimum_geometric_surface_clearance_m','pedestrian_intrusion','formation_recovery_durations_s','pending_formation_recovery_s','applied_command','body_velocity','pedestrian_position','pedestrian_velocity_world')
            self.ped_log.write(json.dumps(dict(timestep=self.num_timesteps,worker=worker,**{k:info.get(k) for k in keys}))+'\n')
        return result
    def close(self):
        super().close()
        if hasattr(self,'ped_log'):self.ped_log.close()
    def _on_rollout_end(self):
        super()._on_rollout_end()
        for family in ENCOUNTERS:
            subset=[x for x in self.episodes[-100:] if x['family']==family]
            if subset:self.logger.record('escort/'+family+'_success',np.mean([e['success'] for e in subset]))


def train(resume,config_path,output,timesteps=5000):
    if not 4<=timesteps<=5000 or timesteps%2:raise ValueError('Bounded Stage 2 pilot: even 4..5000 NEW transitions')
    prior=json.loads(Path(str(Path(resume).with_suffix(''))+'.manifest.json').read_text());workers=prior['workers'];seed=prior['root_seed'];out=Path(output);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);vec=make_vec(str(Path(config_path).resolve()),workers,seed,out);callback=None
    try:
        model,m=load_stage2(resume,config_path,vec);m=dict(m,resume_checkpoint=str(Path(resume).resolve()))
        def save(model,path):
            checkpoint(model,path,dict(m,stage2_training_transitions=model.num_timesteps-m['stage2_start_timesteps']))
        model.set_logger(configure(str(out/'logs'),['stdout','csv','tensorboard']));save(model,out/'initial')
        callback=Callback(out,workers,2048,save);start=model.num_timesteps;begin=time.perf_counter();exact_learn(model,timesteps,callback);wall=time.perf_counter()-begin;save(model,out/'final')
        c=load_config(config_path);c['_config_path']=str(Path(config_path).resolve())
        ped=evaluate(model,c,out/'validation_pedestrian');stage1=evaluate(model,c,out/'validation_stage1',suite='stage1');decision=gate(ped,stage1,c,m['parent_stage1_validation'])
        result=dict(start_timesteps=start,end_timesteps=model.num_timesteps,new_transitions=model.num_timesteps-start,stage2_transitions=model.num_timesteps-m['stage2_start_timesteps'],physics_steps=callback.physics,gait_calls=callback.gait,training_wall_s=wall,transitions_per_second=timesteps/wall,promotion=decision)
        (out/'result.json').write_text(json.dumps(result,indent=2));return result
    except (KeyboardInterrupt,RuntimeError):
        if 'model' in locals():checkpoint(model,out/f'stopped_{model.num_timesteps:09d}',dict(m,stage2_training_transitions=model.num_timesteps-m['stage2_start_timesteps']))
        raise
    finally:
        if callback:callback.close()
        vec.close()
