"""SB3 orchestration around the verified environment; no low-level control changes."""
import copy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import time
import functools
import numpy as np
import gymnasium as gym
import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv,SubprocVecEnv
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.logger import configure
from .env import G1EscortEnv
from .contract import PACKAGE,ROOT,sha256,load_contract
from .stage1_scenarios import load_config,sample_episode
from .ppo_evaluation import reserved,validation_cases,evaluate,selection_key,promotion_pass
from .training_callbacks import ResearchCallback
from .sb3_restart import PROTOCOL,clear_pending_resets,restore_checkpoint_rng,validate_stream
from .reward_variants import make_environment,VERSIONS

DEFAULT_PPO_CONFIG=PACKAGE/'configs/ppo_stage1.yaml'


class TimedPPO(PPO):
    """Unchanged SB3 optimizer with wall-time accounting only."""
    def train(self):
        begin=time.perf_counter()
        def counted_step(optimizer,args,kwargs):
            self.gradient_steps=getattr(self,'gradient_steps',0)+1
        hook=self.policy.optimizer.register_step_post_hook(counted_step)
        try:super().train()
        finally:hook.remove()
        self.optimizer_wall_s=getattr(self,'optimizer_wall_s',0.)+time.perf_counter()-begin
        self.logger.record('train/gradient_steps_total',self.gradient_steps)


def process_stats(vec):
    pids=[os.getpid()]+[p.pid for p in getattr(vec,'processes',[])]
    rss=cpu=0
    for pid in pids:
        status=Path(f'/proc/{pid}/status').read_text().splitlines()
        rss+=int(next(line.split()[1] for line in status if line.startswith('VmRSS:')))
        fields=Path(f'/proc/{pid}/stat').read_text().split()
        cpu+=(int(fields[13])+int(fields[14]))/os.sysconf('SC_CLK_TCK')
    return dict(rss_kib=rss,cpu_seconds=cpu)


def read_config(path=DEFAULT_PPO_CONFIG):
    path=Path(path).resolve();c=yaml.safe_load(path.read_text())
    c['environment_path']=str(path.parent/c['environment_config']);c['split_path']=str(path.parent/c['split_config'])
    c.setdefault('reward_version','legacy_v1')
    if c['reward_version'] not in VERSIONS:raise ValueError('Unknown reward version')
    if c['schema_version']!=1 or c['device']!='cpu':raise ValueError('This milestone requires schema 1 and CPU')
    return c


def streams(seed,workers):
    children=np.random.SeedSequence(seed).spawn(workers+2)
    return dict(model=int(children[0].generate_state(1)[0]),workers=[int(x.generate_state(1)[0]) for x in children[1:1+workers]],
                evaluation=int(children[-1].generate_state(1)[0]))


class TrainingScenarios(gym.Wrapper):
    """Disjoint seed namespace + held-out combination rejection, no observation changes."""
    def __init__(self,env,split,seed,log_path=None):
        super().__init__(env);self.split=split;self.rng=np.random.default_rng(seed);self.count=0;self.log_path=log_path;self.current=None
        self._restored_stream=False
    def reset(self,*,seed=None,options=None):
        if seed is not None and self._restored_stream:
            raise RuntimeError('Attempt to reseed a restored scenario stream')
        if seed is not None:self.rng=np.random.default_rng(seed)
        if options:raise ValueError('Training wrapper owns scenario selection')
        for _ in range(10000):
            scenario_seed=int(self.rng.integers(0,2**60))*self.split['seed_modulus']+self.split['train_residue']
            m=sample_episode(scenario_seed,config=self.env.unwrapped.config).scenario_metadata()
            if not reserved(m,self.split):break
        else:raise RuntimeError('Cannot sample training split')
        obs,info=self.env.reset(options={'episode_seed':scenario_seed})
        self.current=dict(self.env.unwrapped.scenario_metadata(),episode_seed=scenario_seed)
        self.count+=1
        if self.log_path:
            with open(self.log_path,'a') as f:f.write(json.dumps(self.current)+'\n')
        return obs,info
    def step(self,action):
        obs,r,t,tr,info=self.env.step(action)
        info['scenario_diagnostics']={k:self.current[k] for k in ('family','episode_seed','route_length_m')}
        return obs,r,t,tr,info
    def rng_state(self):return dict(rng=copy.deepcopy(self.rng.bit_generator.state),episodes=self.count)
    def set_rng_state(self,state):
        state=validate_stream(state)
        self.rng.bit_generator.state=state['rng'];self.count=state['episodes'];self._restored_stream=True


def make_worker(env_path,split,seed,log_path,reward_version='legacy_v1'):
    # Native ORT sessions already enforce one intra/inter-op thread each.
    os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
    return TrainingScenarios(make_environment(env_path,reward_version=reward_version),split,seed,log_path)


def make_vec(c,workers,seed,output=None):
    split=json.loads(Path(c['split_path']).read_text());s=streams(seed,workers)
    factories=[functools.partial(make_worker,c['environment_path'],split,s['workers'][i],
        None if output is None else str(Path(output)/f'worker_{i}_scenarios.jsonl'),c['reward_version']) for i in range(workers)]
    if workers==1:return DummyVecEnv(factories)
    return SubprocVecEnv(factories,start_method='spawn')


def provenance(c):
    contract=load_contract()
    reward_details=None
    if c['reward_version']=='tracking_stop_v3':
        from .tracking_objective import settings
        reward_details=settings()
    files=list((PACKAGE/'escort_rl').glob('*.py'))+list((PACKAGE/'configs').glob('*.yaml'))+list((PACKAGE/'configs').glob('*.json'))
    files+=list((PACKAGE/'tools').glob('*.py'))
    files+=list((ROOT/'research/escort_scenarios/escort_scenarios').glob('*.py'))
    return dict(config=c,reward_version=c['reward_version'],reward_details=reward_details,restart_protocol=PROTOCOL,source_hashes={str(p.relative_to(ROOT)):sha256(p) for p in files},
        reference_manifest_sha256=sha256(PACKAGE/'configs/gait_reference.json'),
        frozen_gait_checkpoint=contract['checkpoint'],frozen_gait_sha256=contract['checkpoint_sha256'],
        dependencies={name:importlib.metadata.version(name) for name in ('torch','stable-baselines3','tensorboard','gymnasium','numpy','mujoco','onnxruntime')},
        observation_normalizer='none; unchanged static environment scales',reward_contract_sha256=sha256(PACKAGE/'escort_rl/rewards.py'))


def checkpoint(model,path,manifest=None):
    path=Path(path)
    if path.with_suffix('.zip').exists():raise FileExistsError(path)
    model.save(path)
    state=dict(torch=torch.get_rng_state(),numpy=np.random.get_state(),python=random.getstate(),
        environment_streams=model.get_env().env_method('rng_state'),timesteps=model.num_timesteps,
        resume_limitation='Policy/optimizer and RNG streams restored. Partial rollout, robot physics, active episode and schedule phase are NOT restored; resume starts new episodes from saved next-scenario RNG states.')
    torch.save(state,str(path)+'.rng.pt')
    if manifest is not None:Path(str(path)+'.manifest.json').write_text(json.dumps(manifest,indent=2))


def exact_learn(model,transitions,callback):
    """Finish full PPO updates without SB3 rounding beyond the authorized budget."""
    workers=model.n_envs
    if transitions%workers:raise ValueError('Transition budget must be divisible by workers')
    original_steps=model.n_steps
    remaining=transitions
    while remaining:
        chunk=min(original_steps*workers,remaining);steps=chunk//workers
        if steps<2:raise ValueError('At least two samples per worker required for PPO')
        if model.n_steps!=steps:
            model.n_steps=steps
            model.rollout_buffer=RolloutBuffer(steps,model.observation_space,model.action_space,device=model.device,
                gamma=model.gamma,gae_lambda=model.gae_lambda,n_envs=workers)
        model.learn(total_timesteps=chunk,reset_num_timesteps=False,callback=callback,log_interval=1,tb_log_name='stage1')
        remaining-=chunk
    model.n_steps=original_steps
    model.rollout_buffer=RolloutBuffer(original_steps,model.observation_space,model.action_space,device=model.device,
        gamma=model.gamma,gae_lambda=model.gae_lambda,n_envs=workers)


def run_training(config_path,seed,workers,timesteps,output,resume=None,*,validation=True):
    if timesteps>10000:raise ValueError('Milestone CLI permits at most 10,000 new training transitions')
    if workers not in (1,2) or timesteps<2*workers:raise ValueError('Require 1 or 2 workers and >=2 samples/worker')
    c=read_config(config_path);out=Path(output);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(c['torch_threads']);torch.use_deterministic_algorithms(True)
    manifest=provenance(c);manifest.update(seed=seed,workers=workers,seed_streams=streams(seed,workers),requested_new_transitions=timesteps)
    split=json.loads(Path(c['split_path']).read_text());env_config=load_config(c['environment_path'])
    cases=validation_cases(env_config,split) if validation else []
    manifest['validation_cases']=cases;manifest['final_test']='Not generated or evaluated'
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    vec=make_vec(c,workers,seed,out);callback=None
    try:
        if resume:
            path=Path(resume);prior=json.loads(Path(str(path.with_suffix(''))+'.manifest.json').read_text())
            for key in ('restart_protocol','source_hashes','reward_contract_sha256','reference_manifest_sha256','config','dependencies'):
                if prior.get(key)!=manifest[key]:raise ValueError(f'Resume provenance mismatch: {key}; legacy archives require a new experiment, not a hash bypass')
            if prior['workers']!=workers or prior['seed']!=seed:raise ValueError('Resume requires same workers and root seed')
            model=TimedPPO.load(path,env=vec,device='cpu',force_reset=True)
            state=torch.load(str(path.with_suffix(''))+'.rng.pt',weights_only=False)
            restore_checkpoint_rng(model,vec,state)
            manifest['resume']=dict(checkpoint=str(path.resolve()),state_restoration=state['resume_limitation'])
        else:
            model=TimedPPO('MlpPolicy',vec,device='cpu',seed=manifest['seed_streams']['model'],**c['ppo'],verbose=1)
            # PPO.seed() may reseed vector workers on first reset; seed with independent worker streams instead.
            clear_pending_resets(vec)
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
        model.set_logger(configure(str(out/'logs'),['stdout','csv','tensorboard']))
        save=functools.partial(checkpoint,manifest=manifest)
        save(model,out/'initial')
        initial=evaluate(model,env_config,cases,out/'validation_initial',plots=False,reward_version=c['reward_version']) if validation else None
        callback=ResearchCallback(out,workers,c['checkpoint_interval'],save)
        start=model.num_timesteps;begin=time.perf_counter();resources_before=process_stats(vec)
        optimizer_before=getattr(model,'optimizer_wall_s',0.)
        exact_learn(model,timesteps,callback)
        model.logger.dump(step=model.num_timesteps)
        elapsed=time.perf_counter()-begin;resources_after=process_stats(vec)
        save(model,out/'final')
        final=evaluate(model,env_config,cases,out/'validation_final',reward_version=c['reward_version']) if validation else None
        if validation:
            chosen='final' if selection_key(final)>selection_key(initial) else 'initial'
            # Independent preserved SB3 archive, RNG/provenance copied; no symlink ambiguity.
            import shutil
            for suffix in ('.zip','.rng.pt','.manifest.json'):shutil.copyfile(out/(chosen+suffix),out/('best_validation'+suffix))
            passed=promotion_pass(final,c['promotion'])
        else:chosen=None;passed=False
        result=dict(start_timesteps=start,end_timesteps=model.num_timesteps,new_transitions=model.num_timesteps-start,
            sb3_optimization_epochs=model._n_updates,gradient_steps_total=getattr(model,'gradient_steps',0),
            optimizer_updates=model._n_updates,training_wall_s=elapsed,transitions_per_second=(model.num_timesteps-start)/elapsed,
            active_physics_steps=callback.physics,active_gait_calls=callback.gait,training_episode_count=len(callback.episodes),
            optimizer_wall_s=getattr(model,'optimizer_wall_s',0.)-optimizer_before,
            cpu_seconds=resources_after['cpu_seconds']-resources_before['cpu_seconds'],
            resident_memory_kib_parent_and_workers=resources_after['rss_kib'],
            best_validation=chosen,promotion_round_pass=bool(passed),promotion_approved=False,
            promotion_note='Two consecutive full passing validation rounds required; no curriculum promotion in this milestone.')
        (out/'result.json').write_text(json.dumps(result,indent=2));return result
    except KeyboardInterrupt:
        if 'model' in locals():checkpoint(model,out/f'interrupted_{model.num_timesteps:09d}',manifest)
        raise
    finally:
        if callback:callback.close()
        vec.close()


def benchmark(config_path,output,steps=256,trials=2):
    c=read_config(config_path);out=Path(output);out.mkdir(parents=True,exist_ok=False);results=[]
    torch.set_num_threads(1)
    for workers in (1,2):
        for trial in range(trials):
            v=make_vec(c,workers,700+trial);rng=np.random.default_rng(800+trial)
            try:
                start=time.perf_counter();v.reset();resets=time.perf_counter()-start
                begin=time.perf_counter();physics=gait=0
                resource_start=process_stats(v)
                for _ in range(steps//workers):
                    _,_,_,infos=v.step(rng.uniform(-.25,.25,(workers,3)).astype(np.float32))
                    physics+=sum(i['physics_steps'] for i in infos);gait+=sum(i['gait_calls'] for i in infos)
                wall=time.perf_counter()-begin
                resource_end=process_stats(v)
                results.append(dict(workers=workers,trial=trial,transitions=steps,wall_s=wall,transitions_per_second=steps/wall,
                    physics_steps=physics,gait_calls=gait,initial_reset_wall_s=resets,includes_automatic_episode_resets=True))
                results[-1]['resident_memory_kib_parent_and_workers']=resource_end['rss_kib']
                results[-1]['cpu_seconds']=resource_end['cpu_seconds']-resource_start['cpu_seconds']
            finally:v.close()
    (out/'benchmark.json').write_text(json.dumps(results,indent=2));return results
