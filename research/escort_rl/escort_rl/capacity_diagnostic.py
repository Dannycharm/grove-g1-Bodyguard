"""Supervised actor-capacity diagnostic; NEVER PPO training or final Reactive Baseline A.

Teacher accepts only the same causal 40-vector and static config as the actor.
All labels come from training-only full-G1 trajectories, including learner-visited states.
"""
import importlib.util
import json
from pathlib import Path
import time
import numpy as np
import torch
from stable_baselines3 import PPO
from .contract import PACKAGE
from .ppo_training import read_config,TrainingScenarios,provenance
from .stage1_scenarios import load_config
from .env import G1EscortEnv
from .ppo_evaluation import evaluate,validation_cases


def teacher_function():
    spec=importlib.util.spec_from_file_location('capacity_teacher',PACKAGE/'tools/verify_stage1_env.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.diagnostic_action


def dataset_partition(seeds):
    """Whole training scenarios, not adjacent rows, define supervised holdout."""
    seeds=np.asarray(seeds,np.int64)
    holdout=(seeds//4)%5==0
    if not holdout.any() or holdout.all():raise ValueError('Need both supervised train/holdout scenarios')
    return ~holdout,holdout


def actor_parameters(model):
    return list(model.policy.mlp_extractor.policy_net.parameters())+list(model.policy.action_net.parameters())


def fit_actor(model,observations,targets,seeds,steps,seed):
    train,holdout=dataset_partition(seeds)
    x=torch.as_tensor(observations,dtype=torch.float32);y=torch.as_tensor(targets,dtype=torch.float32)
    indices=np.flatnonzero(train);rng=np.random.default_rng(seed)
    optimizer=torch.optim.Adam(actor_parameters(model),lr=3e-4)
    history=[];start=time.perf_counter()
    for step in range(steps):
        batch=rng.choice(indices,min(256,len(indices)),replace=False)
        predicted=model.policy.get_distribution(x[batch]).distribution.mean
        loss=torch.nn.functional.mse_loss(predicted,y[batch])
        optimizer.zero_grad();loss.backward()
        torch.nn.utils.clip_grad_norm_(actor_parameters(model),1.0);optimizer.step()
        if step%500==0 or step==steps-1:
            with torch.no_grad():
                p=model.policy.get_distribution(x).distribution.mean
                history.append(dict(update=step+1,train_mse=float(((p[train]-y[train])**2).mean()),
                    holdout_mse=float(((p[holdout]-y[holdout])**2).mean())))
    with torch.no_grad():p=model.policy.get_distribution(x).distribution.mean.numpy()
    return dict(updates=steps,wall_s=time.perf_counter()-start,train_rows=int(train.sum()),holdout_rows=int(holdout.sum()),
        train_episodes=len(set(np.asarray(seeds)[train].tolist())),holdout_episodes=len(set(np.asarray(seeds)[holdout].tolist())),
        train_mse=float(np.mean((p[train]-targets[train])**2)),holdout_mse=float(np.mean((p[holdout]-targets[holdout])**2)),
        holdout_rmse_per_axis=np.sqrt(np.mean((p[holdout]-targets[holdout])**2,axis=0)).tolist(),history=history)


def run_capacity(output,*,seed=314,scenario_seed=91001,updates=4000,episode_counts=(18,12,12)):
    out=Path(output);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    c=read_config();config=load_config(c['environment_path']);split=json.loads(Path(c['split_path']).read_text())
    manifest=provenance(c);manifest.update(experiment='diagnostic_imitation_only',model_seed=seed,scenario_seed=scenario_seed,
        episodes=list(episode_counts),updates_per_stage=updates,supervised_holdout='whole training scenarios where (episode_seed//4)%5==0',
        final_test='not generated',normalization='unchanged environment static 40-vector; no extra normalizer')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    env=TrainingScenarios(G1EscortEnv(config=config),split,scenario_seed,str(out/'training_scenarios.jsonl'))
    model=PPO('MlpPolicy',env,device='cpu',seed=seed,**c['ppo'])
    # No PPO learn()/vector reset is invoked here. The raw wrapper owns scenario RNG.
    teacher=teacher_function();observations=[];targets=[];seeds=[];stages=[];episodes=[]
    protected={k:v.clone() for k,v in model.policy.state_dict().items() if k.startswith('mlp_extractor.value_net') or k.startswith('value_net') or k=='log_std'}
    try:
        for stage,count in enumerate(episode_counts):
            for _ in range(count):
                obs,_=env.reset();meta=env.current;done=False;steps=0
                while not done:
                    label=teacher(obs,config)
                    observations.append(obs.copy());targets.append(label);seeds.append(meta['episode_seed'])
                    action=label if stage==0 else model.predict(obs,deterministic=True)[0]
                    obs,_,term,trunc,info=env.step(action);done=term or trunc;steps+=1
                episodes.append(dict(stage=stage,family=meta['family'],episode_seed=meta['episode_seed'],steps=steps,reason=info['reason']))
                print('capacity data',stage,len(episodes),meta['family'],info['reason'],flush=True)
            result=fit_actor(model,np.asarray(observations,np.float32),np.asarray(targets,np.float32),np.asarray(seeds,np.int64),updates,seed+stage)
            stages.append(result);print('capacity fit',stage,json.dumps({k:v for k,v in result.items() if k!='history'}),flush=True)
        assert all(torch.equal(value,model.policy.state_dict()[key]) for key,value in protected.items())
        assert model.num_timesteps==0 and model._n_updates==0
        model.save(out/'diagnostic_imitation_actor')
        np.savez_compressed(out/'causal_training_data.npz',observations=np.asarray(observations,np.float32),targets=np.asarray(targets,np.float32),episode_seeds=np.asarray(seeds,np.int64))
        (out/'supervised_fit.json').write_text(json.dumps(dict(stages=stages,episodes=episodes,critic_and_log_std_unchanged=True,
            trained_actor_parameters=sum(p.numel() for p in actor_parameters(model)),ppo_transitions=0),indent=2))
        cases=validation_cases(config,split)
        summary=evaluate(model,config,cases,out/'heldout_closed_loop',kind='imitation_diagnostic')
        (out/'result.json').write_text(json.dumps(dict(fit=stages[-1],closed_loop=summary,
            interpretation='Tests representability/control under supervised diagnostic labels; not PPO or deployment evidence'),indent=2))
        return summary
    finally:env.close()
