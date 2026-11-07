"""Explicit actor initialization, training-only critic preparation and bounded PPO.

This is imitation-pretrained PPO, never a resume of the diagnostic BC archive.
The existing reward, observation, dynamics and SB3 restart protocol are unchanged.
"""
import copy
import functools
import json
from pathlib import Path
import time
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from .contract import ROOT, sha256
from .ppo_training import (read_config, provenance, make_vec, streams, TimedPPO,
    TrainingScenarios, checkpoint, exact_learn, process_stats)
from .sb3_restart import clear_pending_resets, restore_checkpoint_rng
from .ppo_evaluation import (EpisodeMetrics, aggregate, reserved, validation_cases,
    evaluate, selection_key)
from .stage1_scenarios import FAMILIES, load_config, sample_episode
from .reward_variants import make_environment
from .training_callbacks import ResearchCallback

PROTOCOL = 'imitation_pretrained_ppo_v1'
ACTOR_PREFIXES = ('mlp_extractor.policy_net.', 'action_net.')
CRITIC_PREFIXES = ('mlp_extractor.value_net.', 'value_net.')


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2))


def subset(policy, prefixes):
    return {k: v.detach().clone() for k, v in policy.state_dict().items() if k.startswith(prefixes)}


def actor_mean(model, obs):
    with torch.no_grad():
        tensor, _ = model.policy.obs_to_tensor(obs)
        return model.policy.get_distribution(tensor).distribution.mean.cpu().numpy()[0]


def checked_source(c):
    cfg = c['imitation']
    if cfg['protocol'] != PROTOCOL or c['reward_version'] != 'tracking_stop_v3':
        raise ValueError('Require versioned imitation protocol and released objective C')
    for path_key, hash_key in [('actor_source','actor_sha256'), ('source_manifest','source_manifest_sha256'), ('source_data','source_data_sha256')]:
        if sha256(ROOT/cfg[path_key]) != cfg[hash_key]:
            raise ValueError('Initialization source hash mismatch: '+path_key)
    old = json.loads((ROOT/cfg['source_manifest']).read_text())
    current = provenance(c)
    protected = ['env.py','observations.py','mujoco_runner.py','frozen_gait.py',
                 'controller_reference.py','measurements.py','stage1_scenarios.py','rewards.py']
    paths = ['research/escort_rl/escort_rl/'+name for name in protected]
    paths += ['research/escort_rl/configs/'+name for name in ('stage1.yaml','stage1_splits.json','gait_reference.json')]
    paths += [k for k in old['source_hashes'] if k.startswith('research/escort_scenarios/')]
    for path in paths:
        if old['source_hashes'].get(path) != current['source_hashes'].get(path):
            raise ValueError('Protected initialization contract mismatch: '+path)
    if old['frozen_gait_sha256'] != current['frozen_gait_sha256']:
        raise ValueError('Frozen gait mismatch')
    return PPO.load(ROOT/cfg['actor_source'], device='cpu')


def initialize_actor(target, source, sigma):
    if not 0 < sigma <= 1 or target.num_timesteps or target.policy.optimizer.state:
        raise ValueError('Actor import requires positive sigma and a fresh PPO model/optimizer')
    for name in ('observation_space','action_space'):
        a,b = getattr(target,name),getattr(source,name)
        if a.shape != b.shape or a.dtype != b.dtype or not np.array_equal(a.low,b.low) or not np.array_equal(a.high,b.high):
            raise ValueError('Actor space mismatch: '+name)
    before = subset(target.policy, CRITIC_PREFIXES)
    src, dst = subset(source.policy, ACTOR_PREFIXES), subset(target.policy, ACTOR_PREFIXES)
    if src.keys() != dst.keys() or any(src[k].shape != dst[k].shape or src[k].dtype != dst[k].dtype for k in src):
        raise ValueError('Actor tensor contract mismatch')
    if source.policy.activation_fn != target.policy.activation_fn:
        raise ValueError('Activation mismatch')
    target.policy.load_state_dict(src, strict=False)
    with torch.no_grad(): target.policy.log_std.fill_(np.log(sigma))
    assert all(torch.equal(v, subset(target.policy, CRITIC_PREFIXES)[k]) for k,v in before.items())
    assert not target.policy.optimizer.state


def development_cases(c):
    config = load_config(c['environment_path']); split = json.loads(Path(c['split_path']).read_text()); cases=[]
    for j,family in enumerate(FAMILIES):
        seed = (80000+1000*j)*split['seed_modulus']+split['train_residue']
        while sum(x['family']==family for x in cases) < c['imitation']['noise_cases_per_family']:
            m = sample_episode(seed, family, config).scenario_metadata()
            if not reserved(m,split): cases.append(dict(family=family,episode_seed=seed,metadata=m))
            seed += split['seed_modulus']
    return cases


def noise_pass(summary, cfg):
    return (summary['success_rate'] >= cfg['noise_min_success'] and
            all(x['success_rate'] >= cfg['noise_min_family_success'] for x in summary['per_family'].values()) and
            summary['formation_mean_m'] <= cfg['noise_max_formation_m'] and
            not any(summary['failures'].get(k,0) for k in ('fall','supervisor_failure','physics_failure')))


def noise_sweep(config_path, output):
    c=read_config(config_path); cfg=c['imitation']; out=Path(output); out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(c['torch_threads']); model=checked_source(c); cases=development_cases(c)
    manifest=provenance(c); write(out/'manifest.json',manifest); write(out/'cases.json',cases)
    env=make_environment(c['environment_path'],reward_version=c['reward_version']); results=[]
    try:
        for sigma in cfg['noise_sigmas']:
            folder=out/f'sigma_{sigma:g}'; folder.mkdir(); episodes=[]; clipped=total=0
            for i,case in enumerate(cases):
                obs,_=env.reset(options={'family':case['family'],'episode_seed':case['episode_seed']})
                rng=np.random.default_rng(np.random.SeedSequence([cfg['noise_seed'],case['episode_seed'],i]))
                metrics=EpisodeMetrics(); done=False
                with (folder/f'{i:03d}.jsonl').open('x') as f:
                    while not done:
                        mean=actor_mean(model,obs); raw=mean+sigma*rng.standard_normal(3)
                        action=np.clip(raw,-1,1).astype(np.float32); clipped+=int(np.any(np.abs(raw)>1));total+=1
                        obs,r,t,tr,info=env.step(action);done=t or tr;metrics.add(r,info)
                        f.write(json.dumps(dict(mean=mean.tolist(),raw_action=raw.tolist(),evaluation=info,reward=r))+'\n')
                result=metrics.summary(case['family'],case['episode_seed']);episodes.append(result)
                print(f'noise sigma={sigma:g} {i+1}/{len(cases)} {case["family"]}: {result["reason"]}',flush=True)
            summary=aggregate(episodes);summary.update(sigma=sigma,raw_action_clipped_fraction=clipped/total)
            summary['passes']=noise_pass(summary,cfg);results.append(summary);write(folder/'summary.json',summary)
    finally:env.close()
    passing=[r['sigma'] for r in results if r['passes']]
    result=dict(protocol=PROTOCOL,actor_sha256=cfg['actor_sha256'],config=c,source_hashes=manifest['source_hashes'],
                selected_sigma=max(passing) if passing else None,results=results,
                selection='Largest tested positive sigma passing prespecified development gates; common random noise per case')
    write(out/'noise_report.json',result);return result


def returns(rewards, gamma, bootstrap=0.):
    values=[]; value=float(bootstrap)
    for reward in reversed(rewards):
        value=float(reward)+gamma*value;values.append(value)
    return np.array(values[::-1],dtype=np.float32)


def collect_critic(model,c,out):
    cfg=c['imitation'];split=json.loads(Path(c['split_path']).read_text())
    env=TrainingScenarios(make_environment(c['environment_path'],reward_version=c['reward_version']),split,cfg['critic_scenario_seed'])
    rng=np.random.default_rng(cfg['critic_action_seed']); observations=[]; targets=[]; ids=[]; summaries=[]
    sigma=model.policy.log_std.detach().exp().cpu().numpy()
    try:
        for i in range(cfg['critic_episodes']):
            obs,_=env.reset();epobs=[];rewards=[];metrics=EpisodeMetrics();done=False
            while not done:
                epobs.append(obs.copy());action=np.clip(actor_mean(model,obs)+sigma*rng.standard_normal(3),-1,1).astype(np.float32)
                obs,r,t,tr,info=env.step(action);done=t or tr;rewards.append(r);metrics.add(r,info)
            bootstrap=0.
            if tr and not t:
                with torch.no_grad():
                    x,_=model.policy.obs_to_tensor(obs);bootstrap=float(model.policy.predict_values(x).item())
            observations.extend(epobs);targets.extend(returns(rewards,c['ppo']['gamma'],bootstrap));ids.extend([i]*len(epobs))
            summary=metrics.summary(env.current['family'],env.current['episode_seed'])
            summary.update(scenario=env.current,bootstrap=bootstrap,truncated=bool(tr));summaries.append(summary)
            print(f'critic rollout {i+1}/{cfg["critic_episodes"]}: {summary["reason"]}',flush=True)
    finally:env.close()
    np.savez_compressed(out/'critic_training_data.npz',observations=np.asarray(observations,np.float32),returns=np.asarray(targets,np.float32),episode_ids=np.asarray(ids))
    write(out/'critic_episodes.json',summaries)
    return np.asarray(observations,np.float32),np.asarray(targets,np.float32),np.asarray(ids)


def fit_critic(model, observations, targets, episode_ids, cfg):
    """Only value tensors receive gradients; independent Adam never replaces PPO Adam."""
    unique=np.unique(episode_ids)
    if len(unique)<5: raise ValueError('Need at least five whole episodes for critic holdout')
    hold=np.isin(episode_ids,unique[::5]);train=~hold
    before=subset(model.policy,ACTOR_PREFIXES+('log_std',)); rng=np.random.default_rng(cfg['critic_fit_seed'])
    params=[p for k,p in model.policy.named_parameters() if k.startswith(CRITIC_PREFIXES)]
    flags={k:p.requires_grad for k,p in model.policy.named_parameters()}
    for k,p in model.policy.named_parameters():p.requires_grad_(k.startswith(CRITIC_PREFIXES))
    optimizer=torch.optim.Adam(params,lr=cfg['critic_learning_rate'])
    x=torch.as_tensor(observations);y=torch.as_tensor(targets).reshape(-1,1)
    def scores():
        with torch.no_grad():pred=model.policy.predict_values(x).flatten().numpy()
        return dict(train_mse=float(np.mean((pred[train]-targets[train])**2)),holdout_mse=float(np.mean((pred[hold]-targets[hold])**2)),
            holdout_explained_variance=float(1-np.var(targets[hold]-pred[hold])/max(np.var(targets[hold]),1e-12)))
    history=[dict(update=0,**scores())];best=history[0]['holdout_mse'];best_state=subset(model.policy,CRITIC_PREFIXES)
    try:
        for update in range(1,cfg['critic_updates']+1):
            index=rng.choice(np.flatnonzero(train),cfg['critic_batch_size'],replace=True)
            loss=torch.mean((model.policy.predict_values(x[index])-y[index])**2)
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(params,.5);optimizer.step()
            if update%100==0 or update==cfg['critic_updates']:
                score=dict(update=update,**scores());history.append(score)
                if score['holdout_mse']<best:best=score['holdout_mse'];best_state=subset(model.policy,CRITIC_PREFIXES)
        model.policy.load_state_dict(best_state,strict=False)
    finally:
        for k,p in model.policy.named_parameters():p.requires_grad_(flags[k])
    assert all(torch.equal(v,subset(model.policy,ACTOR_PREFIXES+('log_std',))[k]) for k,v in before.items())
    assert not model.policy.optimizer.state
    constant=float(np.mean((targets[hold]-np.mean(targets[train]))**2))
    final=scores();return dict(initial=history[0],final=final,constant_holdout_mse=constant,passes=final['holdout_mse']<constant,
        history=history,train_episodes=unique[~np.isin(unique,unique[::5])].tolist(),holdout_episodes=unique[::5].tolist(),
        train_samples=int(train.sum()),holdout_samples=int(hold.sum()),actor_unchanged=True,
        ppo_optimizer_fresh=True,bootstrap='zero at termination; initial critic at time-limit truncation',
        selection='Best whole-training-episode holdout MSE; no validation gradients')


class PreservationCallback(ResearchCallback):
    def _on_step(self):
        result=super()._on_step()
        if any(i['reason'] in ('fall','supervisor_failure','physics_failure') for i in self.locals['infos']):
            raise RuntimeError('Critical training failure; preservation run stopped')
        return result


def preservation_pass(summary):
    return summary['success_rate']>=.9 and all(x['success_rate']>=.8 for x in summary['per_family'].values()) and summary['formation_mean_m']<=.15 and not any(summary['failures'].get(k,0) for k in ('fall','supervisor_failure','physics_failure'))


def train(config_path,output,noise_report=None,seed=42,workers=2,timesteps=5000,resume=None):
    c=read_config(config_path);cfg=c['imitation']
    if not 2*workers<=timesteps<=cfg['max_new_ppo_transitions'] or timesteps%workers or workers not in (1,2):
        raise ValueError('Only 1/2 workers and at most 5k new transitions per explicit call')
    out=Path(output);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(c['torch_threads']);torch.use_deterministic_algorithms(True)
    manifest=provenance(c);manifest.update(initialization_protocol=PROTOCOL,seed=seed,workers=workers,seed_streams=streams(seed,workers),requested_new_transitions=timesteps)
    envconfig=load_config(c['environment_path']);split=json.loads(Path(c['split_path']).read_text());cases=validation_cases(envconfig,split)
    manifest.update(validation_cases=cases,final_test='Not generated or evaluated')
    source=None
    if not resume:
        if not noise_report:raise ValueError('Fresh initialization requires completed noise report')
        report=json.loads(Path(noise_report).read_text())
        if report['config']!=c or report['source_hashes']!=manifest['source_hashes'] or report['actor_sha256']!=cfg['actor_sha256']:
            raise ValueError('Noise report provenance mismatch')
        sigma=report['selected_sigma']
        if sigma is None:raise ValueError('No exploration level passed; do not train')
        passing=[r['sigma'] for r in report['results'] if noise_pass(r,cfg)]
        if not passing or sigma!=max(passing):raise ValueError('Noise selection does not satisfy declared gates')
        source=checked_source(c)
        manifest['initialization']=dict(actor_sha256=cfg['actor_sha256'],noise_report_sha256=sha256(noise_report),sigma=sigma,
            method='Only actor tensors copied; fresh critic, PPO optimizer, counters and RNG. Critic prepared with training-only MC returns.')
    vec=make_vec(c,workers,seed,out);callback=None
    try:
        if resume:
            path=Path(resume).with_suffix('');prior=json.loads(Path(str(path)+'.manifest.json').read_text())
            for key in ('initialization_protocol','restart_protocol','source_hashes','config','dependencies','workers','seed','reference_manifest_sha256','reward_contract_sha256'):
                if prior.get(key)!=manifest[key]:raise ValueError('Resume provenance mismatch: '+key)
            manifest['initialization']=prior['initialization'];manifest['resume']=str(path.resolve())
            model=TimedPPO.load(str(path)+'.zip',env=vec,device='cpu',force_reset=True)
            restore_checkpoint_rng(model,vec,torch.load(str(path)+'.rng.pt',weights_only=False))
        else:
            model=TimedPPO('MlpPolicy',vec,device='cpu',seed=manifest['seed_streams']['model'],**c['ppo'],verbose=1)
            clear_pending_resets(vec);initialize_actor(model,source,sigma)
            data=collect_critic(model,c,out);fit=fit_critic(model,*data,cfg);write(out/'critic_fit.json',fit)
            if not fit['passes']:raise RuntimeError('Critic did not beat constant holdout baseline; no PPO updates')
        write(out/'manifest.json',manifest);model.set_logger(configure(str(out/'logs'),['stdout','csv','tensorboard']))
        save=functools.partial(checkpoint,manifest=manifest);save(model,out/'initial')
        initial=evaluate(model,envconfig,cases,out/'validation_initial',plots=False,reward_version=c['reward_version'])
        if not preservation_pass(initial):raise RuntimeError('Initial actor failed preservation gate; no PPO updates')
        with np.load(ROOT/cfg['source_data']) as data:
            # Diagnostic action drift only; these labels never enter PPO loss.
            key='observations' if 'observations' in data.files else 'obs'
            probe=data[key][:2048].astype(np.float32)
        def probe_means():
            with torch.no_grad():return model.policy.get_distribution(torch.as_tensor(probe)).distribution.mean.numpy()
        before=probe_means();std_before=model.policy.log_std.detach().exp().tolist();start=model.num_timesteps
        callback=PreservationCallback(out,workers,c['checkpoint_interval'],save)
        begin=time.perf_counter();resources=process_stats(vec);optimizer_before=getattr(model,'optimizer_wall_s',0.)
        exact_learn(model,timesteps,callback);elapsed=time.perf_counter()-begin;model.logger.dump(step=model.num_timesteps)
        save(model,out/'final');final=evaluate(model,envconfig,cases,out/'validation_final',plots=True,reward_version=c['reward_version'])
        chosen='final' if selection_key(final)>selection_key(initial) else 'initial'
        import shutil
        for suffix in ('.zip','.rng.pt','.manifest.json'):shutil.copyfile(out/(chosen+suffix),out/('best_validation'+suffix))
        afterres=process_stats(vec)
        result=dict(method='imitation-pretrained PPO',start_timesteps=start,end_timesteps=model.num_timesteps,new_transitions=model.num_timesteps-start,
            training_wall_s=elapsed,transitions_per_second=timesteps/elapsed,physics_steps=callback.physics,gait_calls=callback.gait,
            training_episodes=len(callback.episodes),training_successes=sum(e['success'] for e in callback.episodes),
            optimizer_wall_s=getattr(model,'optimizer_wall_s',0.)-optimizer_before,gradient_steps_total=getattr(model,'gradient_steps',0),
            actor_mean_drift_mse=float(np.mean((probe_means()-before)**2)),std_before=std_before,std_after=model.policy.log_std.detach().exp().tolist(),
            initial_success=initial['success_rate'],final_success=final['success_rate'],preservation_pass=preservation_pass(final),
            best_validation=chosen,promotion_approved=False,cpu_seconds=afterres['cpu_seconds']-resources['cpu_seconds'],rss_kib=afterres['rss_kib'])
        write(out/'result.json',result);return result
    except (KeyboardInterrupt,RuntimeError):
        if 'model' in locals():checkpoint(model,out/f'stopped_{model.num_timesteps:09d}',manifest)
        raise
    finally:
        if callback:callback.close()
        vec.close()
